# Reason for this code:
# This script runs the chat-based RAG interface for the policy system.
# It loads two separate FAISS indexes:
# 1) preparer index for guidance on what should be done
# 2) approver index for identifying violations and rejection reasoning
# At runtime, it first asks whether the user is a preparer or approver,
# then routes each question to the correct index, retrieves relevant chunks,
# sends them to the Bedrock chat model, and prints both the answer and supporting policy references.

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
INDEX_ROOT_DIR = PROJECT_ROOT / "vectorstore"
PREPARER_INDEX_DIR = INDEX_ROOT_DIR / "preparer"
APPROVER_INDEX_DIR = INDEX_ROOT_DIR / "approver"

MAX_TOKENS = 300
TEMPERATURE = 0.4
TOP_K = 5

AWS_REGION = os.getenv("AWS_REGION")
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL")

# =========================
# CLEAR CONSOLE
# =========================


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# =========================
# PROMPTS
# =========================


def build_system_prompt(mode: str) -> str:
    if mode == "approver":
        return (
            "You answer approver policy questions using only the provided context.\n"
            "Answer in 1-2 sentences maximum.\n"
            "Prefer the shortest complete answer.\n"
            "You may make simple policy-preserving inferences.\n"
            "Use '<rule> unless <exception>' when applicable.\n"
            "Do not restate the question.\n"
            "If insufficient info, say exactly: 'I don't have enough information to answer that question.'\n\n"
            "Context:\n{context}"
        )

    return (
        "You answer preparer policy questions using only the provided context.\n"
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
        r"Policy Section:\s*(.*?)(?:\n(?:Atomic Rule|Violation Scenario):|\Z)",
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
# MODE PROMPT
# =========================


def prompt_for_mode():
    while True:
        role = input("Are you a preparer or approver? ").strip().lower()
        if role in {"preparer", "p"}:
            return "preparer"
        if role in {"approver", "a"}:
            return "approver"
        print("Please enter 'preparer' or 'approver'.")


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


def setup_qa_chain(vectorstore, mode):
    llm = build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", build_system_prompt(mode)), ("human", "{input}")]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return create_retrieval_chain(retriever, doc_chain), retriever


# =========================
# PROCESS QUERY
# =========================


def process_query(query, mode, qa_chain, retriever):
    retrieval_query = f"{mode}: {query}"

    docs = retriever.invoke(retrieval_query)
    docs = dedupe_sources_by_section(docs)

    result = qa_chain.invoke({"input": retrieval_query})
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
    preparer_vs = load_vectorstore(PREPARER_INDEX_DIR)
    approver_vs = load_vectorstore(APPROVER_INDEX_DIR)

    clear_console()

    print(
        "RAG chatbot ready.\n"
        "Type 'quit' to exit.\n"
        "Type 'clear' to reset the screen.\n"
        "Type 'switch' to change between preparer and approver.\n"
    )

    current_mode = prompt_for_mode()

    while True:
        query = input(f"\n[{current_mode}] Q: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        if query.lower() in {"clear", "cls"}:
            clear_console()
            print(
                "RAG chatbot ready.\n"
                "Type 'quit' to exit.\n"
                "Type 'clear' to reset the screen.\n"
                "Type 'switch' to change between preparer and approver.\n"
            )
            continue

        if query.lower() in {"switch", "change role", "change mode"}:
            clear_console()
            print(
                "RAG chatbot ready.\n"
                "Type 'quit' to exit.\n"
                "Type 'clear' to reset the screen.\n"
                "Type 'switch' to change between preparer and approver.\n"
            )
            current_mode = prompt_for_mode()
            continue

        vectorstore = approver_vs if current_mode == "approver" else preparer_vs
        qa_chain, retriever = setup_qa_chain(vectorstore, current_mode)

        answer, sources = process_query(query, current_mode, qa_chain, retriever)

        print(f"\nMode: {current_mode}")
        print(f"\nA: {answer}")

        print_sources(sources)


if __name__ == "__main__":
    main()
