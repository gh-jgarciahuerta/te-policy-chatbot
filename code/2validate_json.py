# Reason for this code:
# This script validates the structured JSON output generated for the RAG pipeline.
# It checks that each record matches the expected schema for its chunk type:
# 1) preparer chunks with atomic rules and Q&A
# 2) approver chunks with violation scenarios and rejection responses
# The validator helps catch missing fields, invalid values, duplicate IDs, and weak data quality
# before the dataset is used for retrieval chunk generation or FAISS indexing.

import json
from pathlib import Path
from collections import Counter

# Input JSON file produced by the CSV-to-JSON transformation step
INPUT_FILE = Path("Data/outputCSV/output.json")


# Load the JSON dataset from disk and verify the top-level structure.
# The validator expects a list of record objects.
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of objects.")

    return data


# Helper function to verify that a value is a non-empty string.
# This is used throughout validation to enforce required text fields.
def is_non_empty_string(value):
    return isinstance(value, str) and value.strip() != ""


# Validate one preparer record.
# Expected schema:
# {
#   "id": "...",
#   "chunk_type": "preparer",
#   "source_id": "...",
#   "rule_tag": "...",
#   "policy_text_markdown": "...",
#   "atomic_rule": "...",
#   "qa": [...]
# }
#
# Returns:
# - errors: critical issues that should block downstream use
# - warnings: non-critical quality issues worth reviewing
def validate_preparer_record(record, index):
    errors = []
    warnings = []

    # Required fields for preparer chunks
    required_fields = [
        "id",
        "chunk_type",
        "source_id",
        "rule_tag",
        "policy_text_markdown",
        "atomic_rule",
        "qa",
    ]

    # Check that all required fields exist
    for field in required_fields:
        if field not in record:
            errors.append(f"Record {index}: missing '{field}'")

    # If required fields are missing, stop deeper validation for this record
    if errors:
        return errors, warnings

    # Enforce correct chunk type
    if record["chunk_type"] != "preparer":
        errors.append(f"Record {index}: invalid chunk_type for preparer record")

    # Validate required text fields
    if not is_non_empty_string(record["id"]):
        errors.append(f"Record {index}: invalid 'id'")

    if not is_non_empty_string(record["source_id"]):
        errors.append(f"Record {index}: invalid 'source_id'")

    if not is_non_empty_string(record["rule_tag"]):
        errors.append(f"Record {index}: invalid 'rule_tag'")

    if not is_non_empty_string(record["atomic_rule"]):
        errors.append(f"Record {index}: empty 'atomic_rule'")

    # Markdown is useful context, but allowed to be empty as a warning rather than an error
    if not is_non_empty_string(record["policy_text_markdown"]):
        warnings.append(f"Record {index} ({record['id']}): empty markdown")

    # Q&A must be a list so it can be safely iterated downstream
    if not isinstance(record["qa"], list):
        errors.append(f"Record {index} ({record['id']}): 'qa' must be a list")
        return errors, warnings

    # Validate each Q&A item and detect duplicate question-answer pairs
    seen_qa = set()
    for i, qa in enumerate(record["qa"], start=1):
        if not isinstance(qa, dict):
            errors.append(f"{record['id']} qa[{i}] not an object")
            continue

        q = qa.get("question")
        a = qa.get("answer")

        if not is_non_empty_string(q):
            errors.append(f"{record['id']} qa[{i}] invalid question")

        if not is_non_empty_string(a):
            errors.append(f"{record['id']} qa[{i}] invalid answer")

        # Track duplicate Q&A pairs as warnings
        if is_non_empty_string(q) and is_non_empty_string(a):
            key = (q.strip(), a.strip())
            if key in seen_qa:
                warnings.append(f"{record['id']} duplicate QA pair")
            seen_qa.add(key)

    # These are not fatal, but can indicate sparse or noisy data
    if len(record["qa"]) == 0:
        warnings.append(f"{record['id']} has no Q&A")

    if len(record["qa"]) > 5:
        warnings.append(f"{record['id']} has many Q&A ({len(record['qa'])})")

    return errors, warnings


# Validate one approver record.
# Expected schema:
# {
#   "id": "...",
#   "chunk_type": "approver",
#   "source_id": "...",
#   "violation_tag": "...",
#   "policy_text_markdown": "...",
#   "violation_scenario": "...",
#   "rejection_response": "..."
# }
#
# Returns:
# - errors: critical issues that should block downstream use
# - warnings: non-critical quality issues worth reviewing
def validate_approver_record(record, index):
    errors = []
    warnings = []

    # Required fields for approver chunks
    required_fields = [
        "id",
        "chunk_type",
        "source_id",
        "violation_tag",
        "policy_text_markdown",
        "violation_scenario",
        "rejection_response",
    ]

    # Check that all required fields exist
    for field in required_fields:
        if field not in record:
            errors.append(f"Record {index}: missing '{field}'")

    # If required fields are missing, stop deeper validation for this record
    if errors:
        return errors, warnings

    # Enforce correct chunk type
    if record["chunk_type"] != "approver":
        errors.append(f"Record {index}: invalid chunk_type for approver record")

    # Validate required text fields
    if not is_non_empty_string(record["id"]):
        errors.append(f"Record {index}: invalid 'id'")

    if not is_non_empty_string(record["source_id"]):
        errors.append(f"Record {index}: invalid 'source_id'")

    if not is_non_empty_string(record["violation_tag"]):
        errors.append(f"Record {index}: invalid 'violation_tag'")

    if not is_non_empty_string(record["violation_scenario"]):
        errors.append(f"Record {index}: empty 'violation_scenario'")

    if not is_non_empty_string(record["rejection_response"]):
        errors.append(f"Record {index}: empty 'rejection_response'")

    # Markdown is useful context, but allowed to be empty as a warning rather than an error
    if not is_non_empty_string(record["policy_text_markdown"]):
        warnings.append(f"Record {index} ({record['id']}): empty markdown")

    return errors, warnings


# Route record validation based on chunk_type.
# This allows the dataset to contain multiple record schemas in one file.
def validate_record(record, index):
    errors = []
    warnings = []

    if "chunk_type" not in record:
        errors.append(f"Record {index}: missing 'chunk_type'")
        return errors, warnings

    chunk_type = record["chunk_type"]

    if chunk_type == "preparer":
        return validate_preparer_record(record, index)

    if chunk_type == "approver":
        return validate_approver_record(record, index)

    errors.append(f"Record {index}: unknown chunk_type '{chunk_type}'")
    return errors, warnings


# Validate the full dataset.
# This function:
# - checks duplicate IDs
# - counts preparer vs approver chunks
# - runs record-level validation on every item
def validate_dataset(data):
    errors = []
    warnings = []

    # Count record IDs to detect duplicates
    ids = [item.get("id") for item in data if isinstance(item, dict)]
    counts = Counter(ids)

    for i, c in counts.items():
        if i and c > 1:
            errors.append(f"Duplicate id: {i}")

    preparer_count = 0
    approver_count = 0

    # Validate each record individually
    for idx, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            errors.append(f"Record {idx} not an object")
            continue

        chunk_type = record.get("chunk_type")
        if chunk_type == "preparer":
            preparer_count += 1
        elif chunk_type == "approver":
            approver_count += 1

        e, w = validate_record(record, idx)
        errors.extend(e)
        warnings.extend(w)

    return errors, warnings, preparer_count, approver_count


# Print a readable validation summary to the console.
# This includes totals, counts by chunk type, and detailed error/warning lists.
def print_summary(data, errors, warnings, preparer_count, approver_count):
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total records: {len(data)}")
    print(f"Preparer chunks: {preparer_count}")
    print(f"Approver chunks: {approver_count}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}\n")

    if errors:
        print("ERRORS:")
        for e in errors:
            print("-", e)
        print()

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("-", w)
        print()

    if not errors:
        print("✅ No critical errors found")
    else:
        print("❌ Fix errors before proceeding")


# Main validation flow:
# 1. Load the JSON dataset
# 2. Validate all records
# 3. Print a summary of results
def main():
    data = load_json(INPUT_FILE)
    errors, warnings, preparer_count, approver_count = validate_dataset(data)
    print_summary(data, errors, warnings, preparer_count, approver_count)


# Script entry point
if __name__ == "__main__":
    main()