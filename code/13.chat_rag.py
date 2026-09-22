# Reason for this code:
# This script runs the chat-based RAG interface for the policy system.
# It loads the FAISS index built from atomic policy rules, retrieves the chunks most relevant
# to each question, sends them to the Bedrock chat model, and prints both the answer and the
# supporting policy references.

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "vectorstore"

MAX_TOKENS = 300
TEMPERATURE = 0.4
TOP_K = 5

AWS_REGION = os.getenv("AWS_REGION")
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL")

WELCOME_MESSAGE = (
    "RAG chatbot ready.\n"
    "Type 'quit' to exit.\n"
    "Type 'clear' to reset the screen.\n"
)

# =========================
# CLEAR CONSOLE
# =========================


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = (
    "You answer T&E policy questions using only the provided context.\n"
    "Be concise and action-oriented.\n"
    "Focus on what the user should do.\n"
    "If insufficient info, say exactly: 'I don't have enough information to answer that question.'\n\n"
    "Context:\n{context}"
)


# =========================
# PARSE POLICY
# =========================


def extract_policy_reference(text: str):
    match = re.search(
        r"Policy Section:\s*(.*?)(?:\nAtomic Rule:|\Z)",
        text,
        flags=re.DOTALL,
    )

    content = match.group(1).strip() if match else text.strip()

    header_match = re.match(r"^\s*#+\s*([\d\.]+)\s+(.*)", content)

    if header_match:
        section_id = header_match.group(1).strip()
        title = header_match.group(2).strip()
        body = content[header_match.end() :].strip()
    else:
        section_id = None
        title = None
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
# VECTORSTORE
# =========================


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


# =========================
# QA CHAIN
# =========================


def setup_qa_chain(vectorstore):
    llm = build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return create_retrieval_chain(retriever, doc_chain), retriever


# =========================
# PROCESS QUERY
# =========================


def process_query(query, qa_chain, retriever):
    docs = retriever.invoke(query)
    docs = dedupe_sources_by_section(docs)

    result = qa_chain.invoke({"input": query})
    answer = result.get("answer", "No answer returned.")

    return answer, docs


# =========================
# PRINT SOURCES
# =========================


def print_sources(sources):
    print("\nPolicy Reference(s):")

    if not sources:
        print("No policy references were retrieved.")
        return

    for i, s in enumerate(sources, 1):
        sid, title, body = extract_policy_reference(s.page_content)

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
    qa_chain, retriever = setup_qa_chain(vectorstore)

    clear_console()
    print(WELCOME_MESSAGE)

    while True:
        query = input("\nQ: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        if query.lower() in {"clear", "cls"}:
            clear_console()
            print(WELCOME_MESSAGE)
            continue

        answer, sources = process_query(query, qa_chain, retriever)

        print(f"\nA: {answer}")

        print_sources(sources)


if __name__ == "__main__":
    main()
