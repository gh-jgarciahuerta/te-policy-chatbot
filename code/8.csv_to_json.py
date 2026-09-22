# Reason for this code:
# This script transforms the source T&E policy CSV into a structured JSON dataset for RAG.
# Each CSV row holds one policy section. The section's "Atomic Rule" column contains one or
# more tagged rules ([a], [b], ...), and the "Q&A" column contains tagged question/answer
# examples that reference those same tags. Every atomic rule becomes one JSON record, with
# its matching Q&A attached.
#
# Tagging problems are collected rather than silently skipped. Anything that would cause
# text to be dropped or a record to be wrong is an ERROR; anything that merely looks
# suspicious is a WARNING. A report is printed at the end. If there are any errors the
# JSON is NOT written and the script exits with code 1, so a stray "[a]" typo in Excel
# stops the pipeline instead of quietly losing a rule. Pass --allow-errors to write anyway.
#
# The output JSON is then used downstream for validation, retrieval chunk creation, and vector indexing.
# Output: 9.policyInJson.json

import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Input CSV source file and output JSON destination
CODE_DIR = Path(__file__).resolve().parent
INPUT_FILE = CODE_DIR / "7.T&E_RAG_V1.4.csv"
OUTPUT_FILE = CODE_DIR / "9.policyInJson.json"

# A tag header line: [a] or [a,b] or [a, b, c]
TAG_HEADER_RE = r"^\s*\[[a-z](?:\s*,\s*[a-z])*\]\s*$"

# Lines that look like someone tried to type a tag but got it slightly wrong,
# e.g. "(a)", "[a", "a]", "[A ]", "[ab]". Used only to give a helpful hint.
SUSPICIOUS_TAG_RE = r"^[\[\(\{]?\s*[a-z]{1,2}\s*[\]\)\}]?\s*$"


# =========================
# Issue collection
# =========================


class IssueLog:
    # Collects ERROR / WARNING findings per row so they can be reported together.
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, row_num, source_id, message):
        self.errors.append((row_num, source_id, message))

    def warning(self, row_num, source_id, message):
        self.warnings.append((row_num, source_id, message))

    def print_report(self):
        print()
        print("=" * 80)
        print("PARSE REPORT")
        print("=" * 80)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}) - these drop or corrupt data:")
            for row_num, sid, msg in self.errors:
                print(f"  row {row_num} [{sid or '?'}]: {msg}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}) - worth a look:")
            for row_num, sid, msg in self.warnings:
                print(f"  row {row_num} [{sid or '?'}]: {msg}")

        if not self.errors and not self.warnings:
            print("\nNo issues found.")

        print()


# =========================
# Text helpers
# =========================


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


# =========================
# Tag parsing
# =========================


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
# Returns (blocks, problems):
#   blocks   = [{"tags": ["a"], "body": "..."}, {"tags": ["c", "d"], "body": "..."}]
#   problems = list of human-readable strings for text that could not be used
def split_blocks_by_tag(text):
    text = normalize_text(text)
    if not text:
        return [], []

    # Split whenever a new tag header begins at the start of a line
    parts = re.split(f"(?={TAG_HEADER_RE})", text, flags=re.MULTILINE | re.IGNORECASE)

    blocks = []
    problems = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        first_line = lines[0].strip()

        # Capture one or more tags from the header line, e.g. [a] or [a,b]
        m = re.match(r"^\[([a-z](?:\s*,\s*[a-z])*)\]$", first_line, flags=re.IGNORECASE)
        if not m:
            # Text that is not under any tag header. It would be lost.
            preview = first_line[:60] + ("..." if len(first_line) > 60 else "")
            hint = ""
            if re.match(SUSPICIOUS_TAG_RE, first_line, flags=re.IGNORECASE):
                hint = " (looks like a mistyped tag; expected e.g. [a])"
            problems.append(f"untagged text will be dropped: {preview!r}{hint}")
            continue

        tags = [t.strip().lower() for t in m.group(1).split(",")]
        body = "\n".join(lines[1:]).strip()

        if not body:
            problems.append(f"tag [{','.join(tags)}] has no text under it")
            continue

        # A mistyped tag line inside the body (e.g. "(b)") would be swallowed as text.
        for line in lines[1:]:
            stripped = line.strip()
            if stripped and re.match(SUSPICIOUS_TAG_RE, stripped, flags=re.IGNORECASE):
                problems.append(
                    f"line {stripped!r} under tag [{','.join(tags)}] looks like a mistyped tag"
                )

        blocks.append({"tags": tags, "body": body})

    return blocks, problems


# Parse atomic rule blocks.
# Only single-tag blocks are accepted here because each atomic rule should map to one local rule tag.
# Returns (rules, problems):
#   rules = [{"tag": "a", "atomic_rule": "..."}, {"tag": "b", "atomic_rule": "..."}]
def parse_atomic_rule_tagged(text):
    blocks, problems = split_blocks_by_tag(text)
    rules = []
    seen = set()

    for block in blocks:
        if len(block["tags"]) != 1:
            problems.append(
                f"atomic rule block [{','.join(block['tags'])}] has multiple tags; "
                "each rule needs exactly one tag, so this rule is dropped"
            )
            continue

        tag = block["tags"][0]
        if tag in seen:
            problems.append(
                f"duplicate rule tag [{tag}]; the second rule would overwrite the first"
            )
            continue

        seen.add(tag)
        rules.append({"tag": tag, "atomic_rule": block["body"]})

    return rules, problems


# Parse tagged Q&A blocks.
# Each block can belong to one or multiple tags, and must contain both:
#   Q: ...
#   A: ...
#
# Returns (qa_items, problems):
#   qa_items = [{"tags": ["a"], "question": "...", "answer": "..."}, ...]
def parse_qa_tagged(text):
    blocks, problems = split_blocks_by_tag(text)
    qa_items = []

    for block in blocks:
        body = block["body"]
        tags_label = ",".join(block["tags"])

        # Capture question content from Q: up to A: or end of block
        q_match = re.search(
            r"Q:\s*(.*?)(?=\n\s*A:|\Z)", body, flags=re.DOTALL | re.IGNORECASE
        )

        # Capture answer content from A: to end of block
        a_match = re.search(r"A:\s*(.*)$", body, flags=re.DOTALL | re.IGNORECASE)

        if not q_match or not a_match:
            missing = " and ".join(
                label for label, ok in (("Q:", q_match), ("A:", a_match)) if not ok
            )
            problems.append(f"Q&A block [{tags_label}] is missing {missing}; block dropped")
            continue

        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()

        if not question or not answer:
            problems.append(f"Q&A block [{tags_label}] has an empty question or answer; block dropped")
            continue

        qa_items.append({"tags": block["tags"], "question": question, "answer": answer})

    return qa_items, problems


# Build a stable output chunk ID.
# Example:
# - source_id=7.10.1, tag=a -> 7.10.1-rule-a
def build_chunk_id(source_id, tag):
    if source_id:
        return f"{source_id}-rule-{tag}"
    return f"rule-{tag}"


# Pull the section number out of the markdown heading, e.g. "## 9.6.1 Meals" -> "9.6.1".
def heading_section_number(markdown):
    m = re.match(r"^\s*#+\s*([\d\.]+)\s", markdown)
    return m.group(1).rstrip(".") if m else ""


# =========================
# Row processing
# =========================


# Parse one CSV row into records, logging any problems against that row.
def process_row(row_num, row, issues):
    source_id = normalize_text(get_value(row, ["ID"]))
    markdown = normalize_text(get_value(row, ["Format-Only Markdown - Edited"]))

    # Pull tagged source columns, allowing for alternate header names where needed
    atomic_tagged = normalize_text(get_value(row, ["Atomic Rule Tagged", "Atomic Rule"]))
    qa_tagged = normalize_text(get_value(row, ["Q&A Tagged", "Q&A"]))

    if not source_id:
        issues.error(row_num, source_id, "row has no ID")

    # Sanity check: the ID column should match the section number in the heading.
    heading_num = heading_section_number(markdown)
    if source_id and heading_num and heading_num != source_id:
        issues.warning(
            row_num, source_id, f"ID does not match heading number {heading_num!r} in markdown"
        )

    atomic_rules, rule_problems = parse_atomic_rule_tagged(atomic_tagged)
    qa_items, qa_problems = parse_qa_tagged(qa_tagged)

    for msg in rule_problems:
        issues.error(row_num, source_id, f"Atomic Rule: {msg}")
    for msg in qa_problems:
        issues.error(row_num, source_id, f"Q&A: {msg}")

    # Backward compatibility fallback:
    # if the atomic rule field uses an older "||" delimited format,
    # generate synthetic local tags a, b, c, ...
    if not atomic_rules and "||" in atomic_tagged:
        old_rules = [x.strip() for x in atomic_tagged.split("||") if x.strip()]
        atomic_rules = [
            {"tag": chr(ord("a") + i), "atomic_rule": rule} for i, rule in enumerate(old_rules)
        ]
        issues.warning(row_num, source_id, "Atomic Rule uses legacy '||' format; tags were generated")

    # A row with tagged content but no usable rules is almost certainly a tagging mistake.
    if not atomic_rules and (atomic_tagged or qa_tagged):
        issues.error(row_num, source_id, "cell has content but no atomic rules were parsed")

    # Tags should run a, b, c, ... without gaps. A gap usually means a typo in one header.
    rule_tags = [rule["tag"] for rule in atomic_rules]
    expected = [chr(ord("a") + i) for i in range(len(rule_tags))]
    if rule_tags and rule_tags != expected:
        issues.warning(
            row_num,
            source_id,
            f"rule tags are {rule_tags}, expected {expected} (out of order or gap)",
        )

    # Group Q&A items by tag so each record can attach only its relevant examples
    qa_by_tag = {}
    for qa in qa_items:
        for tag in qa["tags"]:
            qa_by_tag.setdefault(tag, []).append(
                {"question": qa["question"], "answer": qa["answer"]}
            )

    # A Q&A tag with no matching rule means the Q&A would be dropped.
    for tag in sorted(set(qa_by_tag) - set(rule_tags)):
        issues.error(
            row_num, source_id, f"Q&A tag [{tag}] has no matching atomic rule; its Q&A is dropped"
        )

    # Each atomic rule becomes one record. Related Q&A is attached by local tag.
    records = []
    for rule in atomic_rules:
        tag = rule["tag"]
        records.append(
            {
                "id": build_chunk_id(source_id, tag),
                "source_id": source_id,
                "rule_tag": tag,
                "policy_text_markdown": markdown,
                "atomic_rule": rule["atomic_rule"],
                "qa": qa_by_tag.get(tag, []),
            }
        )

    return records


# =========================
# Main
# =========================


# Main pipeline:
# 1. Read the CSV
# 2. Parse each row's atomic rules and Q&A, collecting issues
# 3. Print the parse report
# 4. Write the JSON only if there are no errors (or --allow-errors was passed)
def main():
    allow_errors = "--allow-errors" in sys.argv

    issues = IssueLog()
    output = []
    row_count = 0

    f, reader = open_csv(INPUT_FILE)
    try:
        # Print detected headers to help diagnose CSV schema mismatches
        print("Detected headers:")
        print(reader.fieldnames)
        print("-" * 80)

        for row_num, raw_row in enumerate(reader, start=1):
            row = normalize_row_keys(raw_row)

            # Skip completely blank rows (Excel often leaves a few at the bottom)
            if not any(normalize_text(v) for v in row.values()):
                continue

            row_count += 1
            output.extend(process_row(row_num, row, issues))
    finally:
        # Always close the CSV file handle, even if parsing fails
        f.close()

    # Duplicate IDs across rows would collide in the index.
    seen_ids = set()
    for record in output:
        if record["id"] in seen_ids:
            issues.error(0, record["source_id"], f"duplicate record id {record['id']!r}")
        seen_ids.add(record["id"])

    print(f"Rows read: {row_count}")
    print(f"Records built: {len(output)}")
    issues.print_report()

    if issues.errors and not allow_errors:
        print(
            f"FAILED: {len(issues.errors)} error(s). Fix the CSV/Excel and rerun, "
            "or pass --allow-errors to write the JSON anyway."
        )
        sys.exit(1)

    # Fail fast if nothing was generated, since that usually means header or parsing issues
    if not output:
        print("FAILED: no output records were generated. Check the printed header names above.")
        sys.exit(1)

    # Write final structured JSON output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(output)} records to {OUTPUT_FILE}")


# Script entry point
if __name__ == "__main__":
    main()
