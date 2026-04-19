# Reason for this code:
# This script transforms the source T&E policy CSV into a structured JSON dataset for RAG.
# It parses each row into two independent chunk families:
# 1) preparer chunks built from atomic rules and related Q&A
# 2) approver chunks built from violation scenarios and rejection responses
# The output JSON is then used downstream for validation, retrieval chunk creation, and vector indexing.
# Output: output.json

import csv
import json
import re
from pathlib import Path

# Input CSV source file and output JSON destination
INPUT_FILE = Path("data/T&E_RAG_V1.6.csv")
OUTPUT_FILE = Path("data/outputCSV/output.json")


# Normalize text so downstream parsing is more reliable.
# - Convert None to empty string
# - Standardize line endings
# - Strip leading/trailing whitespace
def normalize_text(text):
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


# Clean CSV header keys.
# This removes BOM characters and extra whitespace so column lookups work consistently.
def normalize_row_keys(row):
    out = {}
    for k, v in row.items():
        clean_key = str(k).replace("\ufeff", "").strip()
        out[clean_key] = v
    return out


# Return the first matching value from a list of possible column names.
# This helps support alternate header names across CSV versions.
def get_value(row, possible_keys):
    for key in possible_keys:
        if key in row:
            return row[key] or ""
    return ""


# Open the CSV using a list of fallback encodings.
# This prevents the script from failing when the file was saved with different encodings.
def open_csv(input_file):
    encodings_to_try = ["utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        f = None
        try:
            f = open(input_file, "r", encoding=enc, newline="")
            reader = csv.DictReader(f)

            # Force header read to catch encoding issues early
            _ = reader.fieldnames

            print(f"Opened CSV with encoding: {enc}")
            return f, reader
        except UnicodeDecodeError as e:
            last_error = e
            if f is not None:
                f.close()

    raise (
        last_error
        if last_error
        else ValueError("Could not open CSV with supported encodings.")
    )


# Split a tagged text field into logical blocks.
# Expected formats look like:
#
# [a]
# text...
#
# [b]
# text...
#
# [c,d]
# Q: ...
# A: ...
#
# Returns a list of dicts like:
# [
#   {"tags": ["a"], "body": "..."},
#   {"tags": ["c", "d"], "body": "..."}
# ]
def split_blocks_by_tag(text):
    text = normalize_text(text)
    if not text:
        return []

    # Split whenever a new tag header begins at the start of a line
    parts = re.split(
        r"(?=^\s*\[[a-z](?:\s*,\s*[a-z])*\]\s*$)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    blocks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        first_line = lines[0].strip()

        # Capture one or more tags from the header line, e.g. [a] or [a,b]
        m = re.match(r"^\[([a-z](?:\s*,\s*[a-z])*)\]$", first_line, flags=re.IGNORECASE)
        if not m:
            continue

        tags = [t.strip().lower() for t in m.group(1).split(",")]
        body = "\n".join(lines[1:]).strip()

        if body:
            blocks.append({"tags": tags, "body": body})

    return blocks


# Parse atomic rule blocks.
# Only single-tag blocks are accepted here because each atomic rule should map to one local rule tag.
# Returns:
# [
#   {"tag": "a", "atomic_rule": "..."},
#   {"tag": "b", "atomic_rule": "..."}
# ]
def parse_atomic_rule_tagged(text):
    blocks = split_blocks_by_tag(text)
    rules = []

    for block in blocks:
        if len(block["tags"]) != 1:
            continue
        rules.append({"tag": block["tags"][0], "atomic_rule": block["body"]})

    return rules


# Parse tagged Q&A blocks.
# Each block can belong to one or multiple tags, and must contain both:
#   Q: ...
#   A: ...
#
# Returns:
# [
#   {"tags": ["a"], "question": "...", "answer": "..."},
#   {"tags": ["b", "c"], "question": "...", "answer": "..."}
# ]
def parse_qa_tagged(text):
    blocks = split_blocks_by_tag(text)
    qa_items = []

    for block in blocks:
        body = block["body"]

        # Capture question content from Q: up to A: or end of block
        q_match = re.search(
            r"Q:\s*(.*?)(?=\n\s*A:|\Z)", body, flags=re.DOTALL | re.IGNORECASE
        )

        # Capture answer content from A: to end of block
        a_match = re.search(r"A:\s*(.*)$", body, flags=re.DOTALL | re.IGNORECASE)

        if not q_match or not a_match:
            continue

        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()

        qa_items.append({"tags": block["tags"], "question": question, "answer": answer})

    return qa_items


# Parse simple tagged text blocks for approver fields such as:
# - Violation_Scenario
# - Rejection_Response
#
# Example input:
# [a]
# Missing banking information
#
# [b]
# Invalid expense type used
#
# Returns:
# [
#   {"tag": "a", "body": "Missing banking information"},
#   {"tag": "b", "body": "Invalid expense type used"}
# ]
#
# If a block has multiple tags like [a,b], duplicate the same body for both tags.
def parse_simple_tagged_blocks(text):
    blocks = split_blocks_by_tag(text)
    out = []

    for block in blocks:
        for tag in block["tags"]:
            out.append({"tag": tag, "body": block["body"]})

    return out


# Build a stable output chunk ID.
# Example:
# - source_id=7.10.1, chunk_type=rule, tag=a -> 7.10.1-rule-a
# - source_id=7.10.1, chunk_type=violation, tag=b -> 7.10.1-violation-b
def build_chunk_id(source_id, chunk_type, tag):
    if source_id:
        return f"{source_id}-{chunk_type}-{tag}"
    return f"{chunk_type}-{tag}"


# Main pipeline:
# 1. Read the CSV
# 2. Parse each row
# 3. Build preparer and approver chunk records
# 4. Save final JSON output
def main():
    output = []

    # Ensure the output directory exists before writing the JSON file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    f, reader = open_csv(INPUT_FILE)
    try:
        # Print detected headers to help diagnose CSV schema mismatches
        print("Detected headers:")
        print(reader.fieldnames)
        print("-" * 80)

        # Process each CSV row into structured output records
        for row_num, raw_row in enumerate(reader, start=1):
            row = normalize_row_keys(raw_row)

            # Pull core row fields
            source_id = normalize_text(get_value(row, ["ID"]))
            markdown = normalize_text(get_value(row, ["Format-Only Markdown - Edited"]))

            # Pull tagged source columns, allowing for alternate header names where needed
            atomic_tagged = get_value(row, ["Atomic Rule Tagged", "Atomic Rule"])
            qa_tagged = get_value(row, ["Q&A Tagged", "Q&A"])
            violation_tagged = get_value(row, ["Violation_Scenario"])
            response_tagged = get_value(row, ["Rejection_Response"])

            # Print previews for the first few rows for debugging and schema validation
            if row_num <= 3:
                print(f"ROW {row_num} source_id={source_id!r}")
                print("Atomic cell preview:")
                print(repr(atomic_tagged[:500]))
                print("Q&A cell preview:")
                print(repr(qa_tagged[:500]))
                print("Violation_Scenario preview:")
                print(repr(violation_tagged[:500]))
                print("Rejection_Response preview:")
                print(repr(response_tagged[:500]))
                print("-" * 80)

            # ----------------------------
            # PREPARER PARSING
            # ----------------------------
            # Parse atomic rules and tagged Q&A from the row
            atomic_rules = parse_atomic_rule_tagged(atomic_tagged)
            qa_items = parse_qa_tagged(qa_tagged)

            # Backward compatibility fallback:
            # if the atomic rule field uses an older "||" delimited format,
            # generate synthetic local tags a, b, c, ...
            if not atomic_rules and "||" in atomic_tagged:
                old_rules = [x.strip() for x in atomic_tagged.split("||") if x.strip()]
                atomic_rules = [
                    {"tag": chr(ord("a") + i), "atomic_rule": rule}
                    for i, rule in enumerate(old_rules)
                ]

            # Group Q&A items by tag so each preparer chunk can attach only its relevant examples
            qa_by_tag = {}
            for qa in qa_items:
                for tag in qa["tags"]:
                    qa_by_tag.setdefault(tag, []).append(
                        {"question": qa["question"], "answer": qa["answer"]}
                    )

            # ----------------------------
            # APPROVER PARSING
            # ----------------------------
            # Parse violation scenarios and rejection responses as simple tagged blocks
            violation_blocks = parse_simple_tagged_blocks(violation_tagged)
            response_blocks = parse_simple_tagged_blocks(response_tagged)

            # Build lookup dictionaries by tag for quick chunk construction
            violation_by_tag = {item["tag"]: item["body"] for item in violation_blocks}
            response_by_tag = {item["tag"]: item["body"] for item in response_blocks}

            # Collect all approver tags seen in either violations or responses
            approver_tags = sorted(
                set(violation_by_tag.keys()) | set(response_by_tag.keys())
            )

            # Print per-row parse summary for debugging
            print(
                f"Row {row_num}: source_id={source_id!r}, "
                f"preparer_rules={len(atomic_rules)}, "
                f"qa_items={len(qa_items)}, "
                f"approver_tags={len(approver_tags)}"
            )

            # Warn if approver columns contain content but failed to parse into tagged blocks
            if violation_tagged and not violation_blocks:
                print(
                    f"WARNING Row {row_num}: Violation_Scenario exists but no tagged blocks were parsed."
                )

            if response_tagged and not response_blocks:
                print(
                    f"WARNING Row {row_num}: Rejection_Response exists but no tagged blocks were parsed."
                )

            # Warn about partial approver pairs where one side is missing
            for tag in approver_tags:
                if tag not in violation_by_tag:
                    print(
                        f"WARNING Row {row_num}: approver tag [{tag}] has response but no violation."
                    )
                if tag not in response_by_tag:
                    print(
                        f"WARNING Row {row_num}: approver tag [{tag}] has violation but no response."
                    )

            # ----------------------------
            # EMIT PREPARER CHUNKS
            # ----------------------------
            # Each atomic rule becomes one preparer chunk.
            # Related Q&A is attached by local tag.
            for rule in atomic_rules:
                tag = rule["tag"]

                output.append(
                    {
                        "id": build_chunk_id(source_id, "rule", tag),
                        "chunk_type": "preparer",
                        "source_id": source_id,
                        "rule_tag": tag,
                        "policy_text_markdown": markdown,
                        "atomic_rule": rule["atomic_rule"],
                        "qa": qa_by_tag.get(tag, []),
                    }
                )

            # ----------------------------
            # EMIT APPROVER CHUNKS
            # ----------------------------
            # Each approver tag becomes one approver chunk.
            # Violation and rejection fields are joined by local tag.
            for tag in approver_tags:
                output.append(
                    {
                        "id": build_chunk_id(source_id, "violation", tag),
                        "chunk_type": "approver",
                        "source_id": source_id,
                        "violation_tag": tag,
                        "policy_text_markdown": markdown,
                        "violation_scenario": violation_by_tag.get(tag, ""),
                        "rejection_response": response_by_tag.get(tag, ""),
                    }
                )

    finally:
        # Always close the CSV file handle, even if parsing fails
        f.close()

    # Fail fast if nothing was generated, since that usually means header or parsing issues
    if not output:
        raise ValueError(
            "No output records were generated. Check the printed header names and cell previews above."
        )

    # Write final structured JSON output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(output)} records to {OUTPUT_FILE}")


# Script entry point
if __name__ == "__main__":
    main()
