import csv
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/T&E_RAG_V1.4.csv")
OUTPUT_FILE = Path("data/outputCSV/output.json")


def normalize_text(text):
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_row_keys(row):
    out = {}
    for k, v in row.items():
        clean_key = str(k).replace("\ufeff", "").strip()
        out[clean_key] = v
    return out


def get_value(row, possible_keys):
    for key in possible_keys:
        if key in row:
            return row[key] or ""
    return ""


def open_csv(input_file):
    encodings_to_try = ["utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        f = None
        try:
            f = open(input_file, "r", encoding=enc, newline="")
            reader = csv.DictReader(f)

            # Force header read now so decode errors happen here
            _ = reader.fieldnames

            print(f"Opened CSV with encoding: {enc}")
            return f, reader
        except UnicodeDecodeError as e:
            last_error = e
            if f is not None:
                f.close()

    raise last_error if last_error else ValueError("Could not open CSV with supported encodings.")


def split_blocks_by_tag(text):
    """
    Parses blocks of the form:

    [a]
    text...

    [b]
    text...

    [c,d]
    Q: ...
    A: ...
    """
    text = normalize_text(text)
    if not text:
        return []

    parts = re.split(
        r'(?=^\s*\[[a-z](?:\s*,\s*[a-z])*\]\s*$)',
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    blocks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        first_line = lines[0].strip()

        m = re.match(r'^\[([a-z](?:\s*,\s*[a-z])*)\]$', first_line, flags=re.IGNORECASE)
        if not m:
            continue

        tags = [t.strip().lower() for t in m.group(1).split(",")]
        body = "\n".join(lines[1:]).strip()

        if body:
            blocks.append({
                "tags": tags,
                "body": body
            })

    return blocks


def parse_atomic_rule_tagged(text):
    blocks = split_blocks_by_tag(text)
    rules = []

    for block in blocks:
        if len(block["tags"]) != 1:
            continue
        rules.append({
            "tag": block["tags"][0],
            "atomic_rule": block["body"]
        })

    return rules


def parse_qa_tagged(text):
    blocks = split_blocks_by_tag(text)
    qa_items = []

    for block in blocks:
        body = block["body"]

        q_match = re.search(r'Q:\s*(.*?)(?=\n\s*A:|\Z)', body, flags=re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'A:\s*(.*)$', body, flags=re.DOTALL | re.IGNORECASE)

        if not q_match or not a_match:
            continue

        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()

        qa_items.append({
            "tags": block["tags"],
            "question": question,
            "answer": answer
        })

    return qa_items


def build_chunk_id(source_id, tag):
    return f"{source_id}-{tag}" if source_id else tag


def main():
    output = []

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    f, reader = open_csv(INPUT_FILE)
    try:
        print("Detected headers:")
        print(reader.fieldnames)
        print("-" * 80)

        for row_num, raw_row in enumerate(reader, start=1):
            row = normalize_row_keys(raw_row)

            source_id = normalize_text(get_value(row, ["ID"]))
            markdown = get_value(row, ["Format-Only Markdown - Edited"])

            atomic_tagged = get_value(row, ["Atomic Rule Tagged", "Atomic Rule"])
            qa_tagged = get_value(row, ["Q&A Tagged", "Q&A"])

            if row_num <= 3:
                print(f"ROW {row_num} source_id={source_id!r}")
                print("Atomic cell preview:")
                print(repr(atomic_tagged[:500]))
                print("Q&A cell preview:")
                print(repr(qa_tagged[:500]))
                print("-" * 80)

            atomic_rules = parse_atomic_rule_tagged(atomic_tagged)
            qa_items = parse_qa_tagged(qa_tagged)

            # Fallback for old format
            if not atomic_rules and "||" in atomic_tagged:
                old_rules = [x.strip() for x in atomic_tagged.split("||") if x.strip()]
                atomic_rules = [
                    {"tag": chr(ord("a") + i), "atomic_rule": rule}
                    for i, rule in enumerate(old_rules)
                ]

            qa_by_tag = {}
            for qa in qa_items:
                for tag in qa["tags"]:
                    qa_by_tag.setdefault(tag, []).append({
                        "question": qa["question"],
                        "answer": qa["answer"]
                    })

            print(
                f"Row {row_num}: source_id={source_id!r}, "
                f"rules_found={len(atomic_rules)}, qa_found={len(qa_items)}"
            )

            for rule in atomic_rules:
                output.append({
                    "id": build_chunk_id(source_id, rule["tag"]),
                    "source_id": source_id,
                    "rule_tag": rule["tag"],
                    "policy_text_markdown": markdown,
                    "atomic_rule": rule["atomic_rule"],
                    "qa": qa_by_tag.get(rule["tag"], [])
                })

    finally:
        f.close()

    if not output:
        raise ValueError(
            "No output records were generated. Check the printed header names and cell previews above."
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(output)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()