# Reason for this code:
# This script converts the structured policy JSON into retrieval-ready text chunks for the RAG system.
# It takes the parsed preparer and approver records and turns each one into a formatted text block
# plus lightweight metadata. The output is used downstream for embedding and FAISS index creation.
# output: 11.retrieval_chunks.json

import json
from pathlib import Path

# Input JSON from the parsing step and output JSON for retrieval chunk generation
CODE_DIR = Path(__file__).resolve().parent
INPUT_FILE = CODE_DIR / "8.policyInJson.json"
OUTPUT_FILE = CODE_DIR / "11.retrieval_chunks.json"


# Load the source JSON file from disk and verify the top-level structure.
# The script expects a list of record objects.
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list.")

    return data


# Normalize text values before building retrieval strings.
# - Convert None to empty string
# - Standardize line endings
# - Trim leading/trailing whitespace
def normalize_text(text):
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


# Build the formatted Q&A section for preparer retrieval chunks.
# This turns a list of question/answer objects into a readable text block.
# If no valid Q&A exists, return a default "None" section.
def build_related_qa_text(qa_items):
    if not qa_items:
        return "Related Q&A:\nNone"

    lines = ["Related Q&A:"]
    for qa in qa_items:
        if not isinstance(qa, dict):
            continue

        question = normalize_text(qa.get("question", ""))
        answer = normalize_text(qa.get("answer", ""))

        if question or answer:
            lines.append(f"Q: {question if question else 'None'}")
            lines.append(f"A: {answer if answer else 'None'}")
            lines.append("")

    # Remove any trailing blank lines from the assembled text
    while lines and lines[-1] == "":
        lines.pop()

    # If nothing valid was added beyond the section header, return a default empty section
    if len(lines) == 1:
        return "Related Q&A:\nNone"

    return "\n".join(lines)


# Build the retrieval text for a preparer chunk.
# This includes:
# - chunk type
# - section ID
# - rule tag
# - policy section
# - atomic rule
# - related Q&A
def build_preparer_chunk_text(record):
    source_id = normalize_text(record.get("source_id", ""))
    rule_tag = normalize_text(record.get("rule_tag", ""))
    policy_text_markdown = normalize_text(record.get("policy_text_markdown", ""))
    atomic_rule = normalize_text(record.get("atomic_rule", ""))
    qa_items = record.get("qa", [])

    parts = [
        f"Chunk Type: preparer",
        f"Section ID: {source_id or 'None'}",
        f"Rule Tag: {rule_tag or 'None'}",
        "",
        "Policy Section:",
        policy_text_markdown if policy_text_markdown else "None",
        "",
        "Atomic Rule:",
        atomic_rule if atomic_rule else "None",
        "",
        build_related_qa_text(qa_items),
    ]

    return "\n".join(parts).strip()


# Build the retrieval text for an approver chunk.
# This includes:
# - chunk type
# - section ID
# - violation tag
# - policy section
# - violation scenario
# - rejection response
def build_approver_chunk_text(record):
    source_id = normalize_text(record.get("source_id", ""))
    violation_tag = normalize_text(record.get("violation_tag", ""))
    policy_text_markdown = normalize_text(record.get("policy_text_markdown", ""))
    violation_scenario = normalize_text(record.get("violation_scenario", ""))
    rejection_response = normalize_text(record.get("rejection_response", ""))

    parts = [
        f"Chunk Type: approver",
        f"Section ID: {source_id or 'None'}",
        f"Violation Tag: {violation_tag or 'None'}",
        "",
        "Policy Section:",
        policy_text_markdown if policy_text_markdown else "None",
        "",
        "Violation Scenario:",
        violation_scenario if violation_scenario else "None",
        "",
        "Rejection Response:",
        rejection_response if rejection_response else "None",
    ]

    return "\n".join(parts).strip()


# Route each record to the correct text builder based on chunk_type.
# Includes a backward-compatible fallback for older records that may not
# explicitly include chunk_type but still contain recognizable fields.
def build_chunk_text(record):
    chunk_type = normalize_text(record.get("chunk_type", "")).lower()

    if chunk_type == "preparer":
        return build_preparer_chunk_text(record)
    if chunk_type == "approver":
        return build_approver_chunk_text(record)

    # Backward compatibility fallback for older record shapes
    if "atomic_rule" in record or "qa" in record:
        return build_preparer_chunk_text(record)
    if "violation_scenario" in record or "rejection_response" in record:
        return build_approver_chunk_text(record)

    raise ValueError(
        f"Unknown or unsupported chunk_type in record: {record.get('id', '<missing id>')}"
    )


# Build lightweight metadata for the retrieval chunk.
# This metadata is used downstream for filtering, indexing, and source tracing.
def build_metadata(record):
    chunk_type = normalize_text(record.get("chunk_type", "")).lower()
    metadata = {
        "source_id": normalize_text(record.get("source_id", "")),
        "chunk_type": chunk_type or "unknown",
    }

    # Attach the tag that matches the record type
    if chunk_type == "preparer":
        metadata["rule_tag"] = normalize_text(record.get("rule_tag", ""))
    elif chunk_type == "approver":
        metadata["violation_tag"] = normalize_text(record.get("violation_tag", ""))
    else:
        # Backward compatibility fallback if chunk_type is missing
        if "rule_tag" in record:
            metadata["rule_tag"] = normalize_text(record.get("rule_tag", ""))
        if "violation_tag" in record:
            metadata["violation_tag"] = normalize_text(record.get("violation_tag", ""))

    return metadata


# Convert one parsed JSON record into the final retrieval chunk format:
# {
#   "id": "...",
#   "text": "...",
#   "metadata": {...}
# }
def build_output_record(record):
    chunk_id = normalize_text(record.get("id", ""))
    if not chunk_id:
        raise ValueError("Record is missing required field: id")

    return {
        "id": chunk_id,
        "text": build_chunk_text(record),
        "metadata": build_metadata(record),
    }


# Main pipeline:
# 1. Load the parsed JSON dataset
# 2. Convert each valid record into a retrieval chunk
# 3. Skip invalid records with logging
# 4. Save the final retrieval_chunks.json file
def main():
    data = load_json(INPUT_FILE)

    output = []
    skipped = 0

    # Process each record from the parsed JSON dataset
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            skipped += 1
            continue

        try:
            output.append(build_output_record(record))
        except Exception as e:
            skipped += 1
            print(f"Skipping record at index {i}: {e}")

    # Ensure the output directory exists before writing the file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save retrieval-ready chunks to disk
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(output)} retrieval chunks to {OUTPUT_FILE}")
    if skipped:
        print(f"Skipped {skipped} invalid record(s).")


# Script entry point
if __name__ == "__main__":
    main()
