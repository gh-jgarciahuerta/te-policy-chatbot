# Reason for this code:
# This script builds the FAISS vector index for the RAG system using retrieval-ready chunks.
# Every chunk (one atomic rule plus its related Q&A and policy section) is embedded and stored
# in a single index, saved alongside a manifest for traceability and reproducibility.

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env (AWS credentials, model IDs, etc.)
load_dotenv()

# =========================
# Configuration
# =========================

# Resolve project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input retrieval chunks (already formatted for embedding)
CODE_DIR = Path(__file__).resolve().parent
DATA_PATH = CODE_DIR / "11.retrieval_chunks.json"

# Output directory where the FAISS index will be stored
INDEX_DIR = PROJECT_ROOT / "vectorstore"

# Required AWS / Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL")

# Validate required environment variables early to fail fast
missing = [
    name
    for name, value in {
        "AWS_REGION": AWS_REGION,
        "BEDROCK_EMBED_MODEL": EMBEDDING_MODEL_ID,
    }.items()
    if not value
]

if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Load retrieval chunks JSON and convert them into LangChain Document objects.
def load_json_chunks(file_path: Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of chunk objects.")

    documents = []

    # Iterate through each retrieval chunk and convert to Document
    for item in data:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if not text:
            continue

        item_metadata = item.get("metadata", {})

        # Create LangChain Document with content + metadata
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

    # Fail if no usable documents were created
    if not documents:
        raise ValueError(
            "The JSON file was loaded, but no valid documents were created."
        )

    print(f"Loaded {len(documents)} retrieval chunks from JSON")

    return documents


# Save a manifest file alongside the index.
# This captures metadata about how the index was built for debugging and reproducibility.
def save_manifest(index_dir: Path, document_count: int):
    manifest = {
        "data_path": str(DATA_PATH),
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "aws_region": AWS_REGION,
        "document_count": document_count,
    }

    manifest_path = index_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")


# Main pipeline:
# 1. Load retrieval chunks
# 2. Initialize embedding model
# 3. Build the FAISS index
# 4. Save the index and its manifest
def build_and_save_index():
    print(f"Loading chunks from: {DATA_PATH}")

    documents = load_json_chunks(DATA_PATH)

    print("Initializing embeddings...")

    # Initialize Bedrock embedding model
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )

    print("Building FAISS index...")

    # Convert documents into vector embeddings and build FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Ensure output directory exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index to disk
    vectorstore.save_local(str(INDEX_DIR))

    # Save metadata manifest for this index
    save_manifest(index_dir=INDEX_DIR, document_count=len(documents))

    print(f"Saved vector index to: {INDEX_DIR}")


# Script entry point
if __name__ == "__main__":
    build_and_save_index()
