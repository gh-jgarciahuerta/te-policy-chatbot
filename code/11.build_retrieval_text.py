# Reason for this code:
# This script converts the structured policy JSON into retrieval-ready chunks for the RAG system.
# Each atomic-rule record becomes one chunk with two parts:
#   - "text": the compact content that gets EMBEDDED and searched. This is only the section
#     title, the atomic rule, and its related Q&A. The full policy section markdown is
#     deliberately left out so that sibling rules from the same section do not embed
#     almost identically and crowd each other out of the top-k results.
#   - "metadata": everything needed at ANSWER time, including the full policy section
#     markdown, so the chat step can show the user the exact policy text without it
#     having influenced retrieval.
# output: 12.retrieval_chunks.json

import json
import re
from pathlib import Path

# Input JSON from the parsing step and output JSON for retrieval chunk generation
CODE_DIR = Path(__file__).resolve().parent
INPUT_FILE = CODE_DIR / "9.policyInJson.json"
OUTPUT_FILE = CODE_DIR / "12.retrieval_chunks.json"


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


# Pull the section title out of the policy markdown.
# The markdown starts with a heading like "## 9.6.1 Meals While Traveling".
# Returns just the title text ("Meals While Traveling"), or "" if no heading is found.
def extract_section_title(policy_text_markdown):
    match = re.match(r"^\s*#+\s*[\d\.]*\s*(.*)", policy_text_markdown)
    if not match:
        return ""
    return match.group(1).strip()


# Build the formatted Q&A section for a retrieval chunk.
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


# Build the text that gets embedded for one record.
# Intentionally compact: section title, the atomic rule, and related Q&A only.
# The full policy section is NOT included here (see the note at the top of this file).
def build_chunk_text(record, section_title):
    source_id = normalize_text(record.get("source_id", ""))
    atomic_rule = normalize_text(record.get("atomic_rule", ""))
    qa_items = record.get("qa", [])

    parts = [
        f"Section: {source_id or 'None'} {section_title}".strip(),
        "",
        "Rule:",
        atomic_rule if atomic_rule else "None",
        "",
        build_related_qa_text(qa_items),
    ]

    return "\n".join(parts).strip()


# Build metadata for the retrieval chunk.
# This carries the full policy section markdown so it can be shown at answer time,
# plus the identifiers used for grouping, deduplication, and source tracing.
def build_metadata(record, section_title):
    return {
        "source_id": normalize_text(record.get("source_id", "")),
        "rule_tag": normalize_text(record.get("rule_tag", "")),
        "section_title": section_title,
        "policy_text_markdown": normalize_text(record.get("policy_text_markdown", "")),
    }


# Convert one parsed JSON record into the final retrieval chunk format:
# {
#   "id": "...",
#   "text": "...",        <- embedded
#   "metadata": {...}     <- attached at answer time
# }
def build_output_record(record):
    chunk_id = normalize_text(record.get("id", ""))
    if not chunk_id:
        raise ValueError("Record is missing required field: id")

    section_title = extract_section_title(
        normalize_text(record.get("policy_text_markdown", ""))
    )

    return {
        "id": chunk_id,
        "text": build_chunk_text(record, section_title),
        "metadata": build_metadata(record, section_title),
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
