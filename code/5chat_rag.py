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
# Configuration
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

missing = [
    name for name, value in {
        "AWS_REGION": AWS_REGION,
        "BEDROCK_CHAT_MODEL": MODEL_ID,
        "BEDROCK_EMBED_MODEL": EMBEDDING_MODEL_ID,
    }.items() if not value
]

if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def extract_policy_reference(text: str) -> str:
    match = re.search(
        r"Policy Section:\s*(?:##+\s*)?(.+?)\s*Atomic Rule:",
        text,
        flags=re.DOTALL
    )

    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = text.strip()

    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def dedupe_sources_by_section(docs):
    unique_docs = []
    seen_source_ids = set()

    for doc in docs:
        source_id = doc.metadata.get("source_id")
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        unique_docs.append(doc)

    return unique_docs


def load_vectorstore():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore directory not found: {INDEX_DIR}\n"
            f"Run build_index.py first."
        )

    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def setup_qa_chain(vectorstore):
    llm_kwargs = {
        "model": MODEL_ID,
        "region_name": AWS_REGION,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    if AWS_BEARER_TOKEN_BEDROCK:
        llm_kwargs["bedrock_api_key"] = AWS_BEARER_TOKEN_BEDROCK

    llm = ChatBedrockConverse(**llm_kwargs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant answering questions about the loaded policy retrieval chunks.\n"
            "Use only the provided context.\n"
            "If the answer is not in the context, say exactly: "
            "'I don't have enough information to answer that question.'\n\n"
            "Context:\n{context}"
        ),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain, retriever


def process_query(query, qa_chain, retriever):
    retrieved_docs = retriever.invoke(query)
    deduped_docs = dedupe_sources_by_section(retrieved_docs)

    result = qa_chain.invoke({"input": query})
    answer = result.get("answer", "No answer returned.")

    return answer, deduped_docs


def main():
    print("Loading saved vector index...")
    vectorstore = load_vectorstore()

    print("Setting up QA chain...")
    qa_chain, retriever = setup_qa_chain(vectorstore)

    print("\nRAG chatbot ready.")
    print("Enter 'quit' to exit the program.")

    while True:
        query = input("\nQ: How can I help? ").strip()

        if query.lower() == "quit":
            print("Exiting the program. Goodbye!")
            break

        if not query:
            print("Please enter a non-empty question.")
            continue

        try:
            answer, sources = process_query(query, qa_chain, retriever)

            print(f"\nA: {answer}")
            print("\nPolicy Reference(s):")

            if not sources:
                print("No policy references were retrieved.")
                continue

            for i, source in enumerate(sources, 1):
                reference_text = extract_policy_reference(source.page_content)
                preview = reference_text[:300]
                if len(reference_text) > 300:
                    preview += "..."
                print(f"{i}. {preview}")

        except Exception as e:
            print(f"\nAn error occurred while processing your query: {e}")
            print("Check your AWS credentials, Bedrock model access, region, model IDs, and saved index.")


if __name__ == "__main__":
    main()