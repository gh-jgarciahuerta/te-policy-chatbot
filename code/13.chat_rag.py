# Reason for this code:
# This script runs the chat-based RAG interface for the policy system.
# Retrieval is hybrid: a FAISS vector search (semantic meaning) and a BM25 keyword search
# (exact terms like dollar amounts, section numbers, "per diem") are fused so that both
# kinds of match surface. The index only embeds each atomic rule and its Q&A; the full
# policy section text is stored as metadata and stitched into the model's context here,
# once per section, so the model sees the authoritative wording without it having skewed
# retrieval. The chat keeps a rolling window of recent turns so follow-up questions work:
# a follow-up is first rewritten into a standalone question (using the history) before
# retrieval, and the history is also passed to the model when it answers.
# The answer is printed along with the supporting policy references.

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "vectorstore"

# Upper bound on answer length. Multi-part answers (rules with several exceptions) were
# being cut off at 300; 1024 leaves room while still capping runaway output and cost.
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# Conversation memory: how many previous turns (one turn = question + answer) to keep.
# Older turns are dropped so the prompt stays bounded.
MEMORY_TURNS = 5

# How many candidates each retriever returns, and how many survive fusion.
RETRIEVER_K = 8
TOP_K = 5

# Relative weight of vector vs keyword results when fusing (must sum to 1).
# Keep these equal. With Reciprocal Rank Fusion, an unequal split means a hit that only
# the weaker retriever found can never outrank the stronger retriever's 5th result,
# which silently disables keyword matches for section numbers and dollar amounts.
VECTOR_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5

AWS_REGION = os.getenv("AWS_REGION")
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL")

WELCOME_MESSAGE = (
    "RAG chatbot ready.\n"
    "Type 'quit' to exit.\n"
    "Type 'new' to start a new conversation (forgets previous questions).\n"
    "Type 'clear' to reset the screen and start a new conversation.\n"
)

# =========================
# CLEAR CONSOLE
# =========================


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# =========================
# PROMPTS
# =========================

SYSTEM_PROMPT = (
    "You answer T&E policy questions using only the provided context.\n"
    "Be concise and action-oriented.\n"
    "Focus on what the user should do.\n"
    "Use the conversation so far to understand follow-up questions, but base the "
    "answer only on the context below.\n"
    "If insufficient info, say exactly: 'I don't have enough information to answer that question.'\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

# Rewrites a follow-up ("what about international?") into a standalone question
# ("Is alcohol reimbursable on international trips?") so retrieval has the full meaning.
# Only used when there is prior history.
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rewrite the follow-up "
            "as a single standalone question that includes any subject it refers back to. "
            "If it is already standalone, return it unchanged. "
            "Return only the rewritten question, with no explanation.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


# =========================
# POLICY REFERENCE
# =========================


# Pull a display-friendly (section_id, title, body) from a retrieved document.
# The full policy section lives in metadata, not in the embedded page_content.
def extract_policy_reference(doc):
    content = doc.metadata.get("policy_text_markdown", "") or ""

    header_match = re.match(r"^\s*#+\s*([\d\.]+)\s+(.*)", content)

    if header_match:
        section_id = header_match.group(1).strip()
        title = header_match.group(2).strip()
        body = content[header_match.end() :].strip()
    else:
        section_id = doc.metadata.get("source_id")
        title = doc.metadata.get("section_title") or None
        body = content

    body = re.sub(r"\s+", " ", body)
    return section_id, title, body


# =========================
# DEDUPE
# =========================


def dedupe_sources_by_section(docs):
    seen = set()
    unique = []

    for doc in docs:
        sid = doc.metadata.get("source_id")
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(doc)

    return unique


# =========================
# CONTEXT
# =========================


# Assemble the context the model sees.
# Retrieved rules are grouped by policy section. Each section's full markdown appears
# once, followed by every retrieved rule (and its Q&A) that belongs to it. This keeps
# the authoritative wording in front of the model without repeating it per rule.
def build_context(docs):
    sections = {}
    order = []

    for doc in docs:
        sid = doc.metadata.get("source_id") or "Unknown"
        if sid not in sections:
            sections[sid] = {
                "markdown": doc.metadata.get("policy_text_markdown", "") or "",
                "rules": [],
            }
            order.append(sid)
        sections[sid]["rules"].append(doc.page_content)

    blocks = []
    for sid in order:
        section = sections[sid]
        block = [f"=== Policy Section {sid} ===", section["markdown"] or "(no section text)", ""]
        for rule_text in section["rules"]:
            block.append(rule_text)
            block.append("")
        blocks.append("\n".join(block).rstrip())

    return "\n\n".join(blocks)


# =========================
# RETRIEVAL
# =========================


# Tokenizer for the BM25 keyword retriever.
# Lowercases and strips punctuation so "reimbursable?" matches "reimbursable.", while
# keeping section numbers ("7.17") and dollar amounts ("$150.00") as single tokens.
def bm25_tokenize(text):
    return re.findall(r"\$?\d+(?:\.\d+)*|\w+", text.lower())


def load_vectorstore(index_dir: Path):
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )

    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# Build the hybrid retriever.
# - Vector retriever comes straight from the FAISS index.
# - BM25 keyword retriever is built in memory from the same documents stored in the
#   FAISS docstore, so there is no second index to maintain on disk.
# - EnsembleRetriever fuses the two ranked lists with Reciprocal Rank Fusion.
def build_retriever(vectorstore):
    documents = list(vectorstore.docstore._dict.values())

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    keyword_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=bm25_tokenize
    )
    keyword_retriever.k = RETRIEVER_K

    return EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[VECTOR_WEIGHT, KEYWORD_WEIGHT],
    )


# =========================
# LLM
# =========================


def build_llm():
    kwargs = {
        "model": MODEL_ID,
        "region_name": AWS_REGION,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    if AWS_BEARER_TOKEN_BEDROCK:
        kwargs["bedrock_api_key"] = AWS_BEARER_TOKEN_BEDROCK

    return ChatBedrockConverse(**kwargs)


def message_text(response):
    return response.content if isinstance(response.content, str) else str(response.content)


# =========================
# CONVERSATION MEMORY
# =========================


class ConversationMemory:
    # Rolling window of the last MEMORY_TURNS question/answer pairs as LangChain messages.
    def __init__(self, max_turns=MEMORY_TURNS):
        self.max_turns = max_turns
        self.messages = []

    def add_turn(self, question, answer):
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=answer))
        # Two messages per turn
        self.messages = self.messages[-(self.max_turns * 2) :]

    def history(self):
        return list(self.messages)

    def reset(self):
        self.messages = []


# Turn a follow-up into a standalone question for retrieval. With no history, the
# question is used as-is and no extra model call is made.
def rewrite_query(query, history, llm):
    if not history:
        return query

    messages = REWRITE_PROMPT.format_messages(history=history, input=query)
    rewritten = message_text(llm.invoke(messages)).strip()

    return rewritten or query


# =========================
# PROCESS QUERY
# =========================


def process_query(query, retriever, llm, memory):
    history = memory.history()

    # Resolve references to earlier turns before searching
    search_query = rewrite_query(query, history, llm)

    # Hybrid retrieval, then keep the top fused results
    docs = retriever.invoke(search_query)[:TOP_K]

    context = build_context(docs)
    messages = PROMPT.format_messages(context=context, history=history, input=query)

    answer = message_text(llm.invoke(messages)).strip() or "No answer returned."

    memory.add_turn(query, answer)

    return answer, dedupe_sources_by_section(docs), search_query


# =========================
# PRINT SOURCES
# =========================


def print_sources(sources):
    print("\nPolicy Reference(s):")

    if not sources:
        print("No policy references were retrieved.")
        return

    for i, s in enumerate(sources, 1):
        sid, title, body = extract_policy_reference(s)

        sid = sid or s.metadata.get("source_id", "Unknown")
        title = title or "Policy Section"

        if len(title) > 80:
            title = title[:77] + "..."

        preview = body[:200] + ("..." if len(body) > 200 else "")

        print(f"{i}. Section {sid} — {title}")
        print(f"   {preview}")


# =========================
# MAIN
# =========================


def main():
    vectorstore = load_vectorstore(INDEX_DIR)
    retriever = build_retriever(vectorstore)
    llm = build_llm()
    memory = ConversationMemory()

    clear_console()
    print(WELCOME_MESSAGE)

    while True:
        query = input("\nQ: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        if query.lower() == "new":
            memory.reset()
            print("Started a new conversation.")
            continue

        if query.lower() in {"clear", "cls"}:
            memory.reset()
            clear_console()
            print(WELCOME_MESSAGE)
            continue

        answer, sources, search_query = process_query(query, retriever, llm, memory)

        # Show the user how their follow-up was interpreted
        if search_query != query:
            print(f"\n(Searching for: {search_query})")

        print(f"\nA: {answer}")

        print_sources(sources)


if __name__ == "__main__":
    main()
