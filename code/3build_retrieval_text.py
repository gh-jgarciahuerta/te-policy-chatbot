import json
from pathlib import Path

INPUT_FILE = Path("Data/outputCSV/output.json")
OUTPUT_FILE = Path("Data/outputCSV/retrieval_chunks.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list.")

    return data


def normalize_text(text):
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def build_related_qa_text(qa_items):
    if not qa_items:
        return "Related Q&A:\nNone"

    lines = ["Related Q&A:"]
    for qa in qa_items:
        question = normalize_text(qa.get("question", ""))
        answer = normalize_text(qa.get("answer", ""))
        if question or answer:
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}")
            lines.append("")

    # Remove trailing blank line if present
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def build_chunk_text(record):
    source_id = normalize_text(record.get("source_id", ""))
    rule_tag = normalize_text(record.get("rule_tag", ""))
    policy_text_markdown = normalize_text(record.get("policy_text_markdown", ""))
    atomic_rule = normalize_text(record.get("atomic_rule", ""))
    qa_items = record.get("qa", [])

    parts = [
        f"Section ID: {source_id}",
        f"Rule Tag: {rule_tag}",
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


def build_output_record(record):
    chunk_id = normalize_text(record.get("id", ""))
    source_id = normalize_text(record.get("source_id", ""))
    rule_tag = normalize_text(record.get("rule_tag", ""))

    return {
        "id": chunk_id,
        "text": build_chunk_text(record),
        "metadata": {
            "source_id": source_id,
            "rule_tag": rule_tag,
        }
    }


def main():
    data = load_json(INPUT_FILE)

    output = []
    for record in data:
        if not isinstance(record, dict):
            continue
        output.append(build_output_record(record))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(output)} retrieval chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()