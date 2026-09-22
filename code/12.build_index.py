# Reason for this code:
# This script builds FAISS vector indexes for the RAG system using retrieval-ready chunks.
# It splits the dataset into two independent embedding spaces:
# 1) preparer index (rules + Q&A for guidance)
# 2) approver index (violations + rejection reasoning)
# Each index is saved separately along with metadata manifests for traceability and reproducibility.

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

# Output directory where FAISS indexes will be stored
INDEX_ROOT_DIR = PROJECT_ROOT / "vectorstore"

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
# Also split documents into preparer vs approver groups for separate indexes.
def load_json_chunks(file_path: Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of chunk objects.")

    preparer_documents = []
    approver_documents = []

    # Iterate through each retrieval chunk and convert to Document
    for item in data:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if not text:
            continue

        item_metadata = item.get("metadata", {})
        chunk_type = item_metadata.get("chunk_type")

        # Create LangChain Document with content + metadata
        doc = Document(
            page_content=text,
            metadata={
                "id": item.get("id"),
                "source_id": item_metadata.get("source_id"),
                "chunk_type": chunk_type,
                "rule_tag": item_metadata.get("rule_tag"),
                "violation_tag": item_metadata.get("violation_tag"),
                "source": str(file_path),
            },
        )

        # Route document into correct index group
        if chunk_type == "preparer":
            preparer_documents.append(doc)
        elif chunk_type == "approver":
            approver_documents.append(doc)

    # Fail if no usable documents were created
    if not preparer_documents and not approver_documents:
        raise ValueError(
            "The JSON file was loaded, but no valid preparer/approver documents were created."
        )

    print(f"Loaded {len(preparer_documents)} preparer retrieval chunks from JSON")
    print(f"Loaded {len(approver_documents)} approver retrieval chunks from JSON")

    return preparer_documents, approver_documents


# Save a manifest file alongside each index.
# This captures metadata about how the index was built for debugging and reproducibility.
def save_manifest(index_dir: Path, chunk_type: str, document_count: int):
    manifest = {
        "chunk_type": chunk_type,
        "data_path": str(DATA_PATH),
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "aws_region": AWS_REGION,
        "document_count": document_count,
    }

    manifest_path = index_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")


# Build and persist a single FAISS index for one chunk type (preparer or approver).
def build_and_save_single_index(
    documents, index_dir: Path, chunk_type: str, embeddings
):
    # Skip index creation if no documents exist for this type
    if not documents:
        print(f"No {chunk_type} documents found. Skipping index build.")
        return

    print(f"Building {chunk_type} FAISS index...")

    # Convert documents into vector embeddings and build FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Ensure output directory exists
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index to disk
    vectorstore.save_local(str(index_dir))

    # Save metadata manifest for this index
    save_manifest(
        index_dir=index_dir, chunk_type=chunk_type, document_count=len(documents)
    )

    print(f"Saved {chunk_type} vector index to: {index_dir}")


# Main pipeline:
# 1. Load retrieval chunks
# 2. Initialize embedding model
# 3. Build separate FAISS indexes for preparer and approver
# 4. Save indexes and metadata manifests
def build_and_save_indexes():
    print(f"Loading chunks from: {DATA_PATH}")

    # Load and split documents
    preparer_documents, approver_documents = load_json_chunks(DATA_PATH)

    print("Initializing embeddings...")

    # Initialize Bedrock embedding model
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )

    # Define output directories for each index
    preparer_index_dir = INDEX_ROOT_DIR / "preparer"
    approver_index_dir = INDEX_ROOT_DIR / "approver"

    # Build preparer index
    build_and_save_single_index(
        documents=preparer_documents,
        index_dir=preparer_index_dir,
        chunk_type="preparer",
        embeddings=embeddings,
    )

    # Build approver index
    build_and_save_single_index(
        documents=approver_documents,
        index_dir=approver_index_dir,
        chunk_type="approver",
        embeddings=embeddings,
    )

    # Build a root-level manifest summarizing both indexes
    root_manifest = {
        "data_path": str(DATA_PATH),
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "aws_region": AWS_REGION,
        "indexes": {
            "preparer": {
                "path": str(preparer_index_dir),
                "document_count": len(preparer_documents),
            },
            "approver": {
                "path": str(approver_index_dir),
                "document_count": len(approver_documents),
            },
        },
    }

    # Ensure root directory exists
    INDEX_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Save root manifest
    root_manifest_path = INDEX_ROOT_DIR / "manifest.json"
    with root_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2)

    print(f"Saved root manifest to: {root_manifest_path}")


# Script entry point
if __name__ == "__main__":
    build_and_save_indexes()
