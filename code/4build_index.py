import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "outputCSV" / "retrieval_chunks.json"
INDEX_DIR = PROJECT_ROOT / "vectorstore"

AWS_REGION = os.getenv("AWS_REGION")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL")

missing = [
    name for name, value in {
        "AWS_REGION": AWS_REGION,
        "BEDROCK_EMBED_MODEL": EMBEDDING_MODEL_ID,
    }.items() if not value
]

if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def load_json_chunks(file_path: Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of chunk objects.")

    documents = []
    for item in data:
        text = item.get("text")
        if not text:
            continue

        item_metadata = item.get("metadata", {})

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": item.get("id"),
                    "source_id": item_metadata.get("source_id"),
                    "rule_tag": item_metadata.get("rule_tag"),
                    "source": str(file_path),
                },
            )
        )

    if not documents:
        raise ValueError("The JSON file was loaded, but no documents were created.")

    print(f"Loaded {len(documents)} retrieval chunks from JSON")
    return documents


def build_and_save_index():
    print(f"Loading chunks from: {DATA_PATH}")
    documents = load_json_chunks(DATA_PATH)

    print("Initializing embeddings...")
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    manifest = {
        "data_path": str(DATA_PATH),
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "aws_region": AWS_REGION,
        "document_count": len(documents),
    }

    manifest_path = INDEX_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved vector index to: {INDEX_DIR}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    build_and_save_index()