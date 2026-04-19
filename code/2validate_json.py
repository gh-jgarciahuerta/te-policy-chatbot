import json
from pathlib import Path
from collections import Counter

# ✅ UPDATED PATH
INPUT_FILE = Path("Data/outputCSV/output.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of objects.")

    return data


def is_non_empty_string(value):
    return isinstance(value, str) and value.strip() != ""


def validate_record(record, index):
    errors = []
    warnings = []

    required_fields = [
        "id",
        "source_id",
        "rule_tag",
        "policy_text_markdown",
        "atomic_rule",
        "qa",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(f"Record {index}: missing '{field}'")

    if errors:
        return errors, warnings

    if not is_non_empty_string(record["id"]):
        errors.append(f"Record {index}: invalid 'id'")

    if not is_non_empty_string(record["source_id"]):
        errors.append(f"Record {index}: invalid 'source_id'")

    if not is_non_empty_string(record["rule_tag"]):
        errors.append(f"Record {index}: invalid 'rule_tag'")

    if not is_non_empty_string(record["atomic_rule"]):
        errors.append(f"Record {index}: empty 'atomic_rule'")

    if not is_non_empty_string(record["policy_text_markdown"]):
        warnings.append(f"Record {index} ({record['id']}): empty markdown")

    if not isinstance(record["qa"], list):
        errors.append(f"Record {index} ({record['id']}): 'qa' must be a list")
        return errors, warnings

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

        key = (q.strip(), a.strip())
        if key in seen_qa:
            warnings.append(f"{record['id']} duplicate QA pair")
        seen_qa.add(key)

    if len(record["qa"]) == 0:
        warnings.append(f"{record['id']} has no Q&A")

    if len(record["qa"]) > 5:
        warnings.append(f"{record['id']} has many Q&A ({len(record['qa'])})")

    return errors, warnings


def validate_dataset(data):
    errors = []
    warnings = []

    ids = [item.get("id") for item in data]
    counts = Counter(ids)

    for i, c in counts.items():
        if i and c > 1:
            errors.append(f"Duplicate id: {i}")

    for idx, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            errors.append(f"Record {idx} not an object")
            continue

        e, w = validate_record(record, idx)
        errors.extend(e)
        warnings.extend(w)

    return errors, warnings


def print_summary(data, errors, warnings):
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total records: {len(data)}")
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


def main():
    data = load_json(INPUT_FILE)
    errors, warnings = validate_dataset(data)
    print_summary(data, errors, warnings)


if __name__ == "__main__":
    main()