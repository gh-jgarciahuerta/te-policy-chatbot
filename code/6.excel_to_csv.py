# Reason for this code:
# This script exports the RAG data sheet from the Excel workbook to the CSV that the rest of
# the pipeline reads. The Excel is the single source of truth: the T&E team edits rules and
# Q&A there, and this step replaces the manual "Save As CSV" that used to be done by hand.
# Only the four columns the pipeline needs are exported. Blank rows are skipped, line
# endings inside cells are normalised, and a short summary of what changed compared to the
# previous CSV is printed so drift is visible.
# Output: 7.T&E_RAG_V1.4.csv

import csv
import sys
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Input workbook and output CSV
CODE_DIR = Path(__file__).resolve().parent
INPUT_FILE = CODE_DIR / "1.T&E_RAG_V1.4.xlsx"
OUTPUT_FILE = CODE_DIR / "7.T&E_RAG_V1.4.csv"

# Sheet holding the policy rows
SHEET_NAME = "T&E_Policy_RAG_Data"

# Columns the downstream parser (8.csv_to_json.py) needs, in output order
EXPORT_COLUMNS = ["ID", "Format-Only Markdown - Edited", "Atomic Rule", "Q&A"]


# Normalize a cell value into the text the CSV should hold.
# - None becomes ""
# - Excel stores in-cell newlines as LF, but be defensive about CRLF and CR
# - Strip leading/trailing whitespace
def normalize_cell(value):
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


# Read the data sheet and return a list of rows (each a list of 4 strings).
# Fails clearly if the sheet or any expected column is missing.
def read_rows(workbook_path: Path):
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    # read_only + data_only keeps this fast and returns computed values, not formulas
    wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)

    if SHEET_NAME not in wb.sheetnames:
        raise ValueError(f"Sheet {SHEET_NAME!r} not found. Sheets: {wb.sheetnames}")

    ws = wb[SHEET_NAME]
    rows_iter = ws.iter_rows(values_only=True)

    header = [normalize_cell(v) for v in next(rows_iter)]
    missing = [c for c in EXPORT_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Missing column(s) in {SHEET_NAME!r}: {missing}. Header is: {header}")

    col_index = [header.index(c) for c in EXPORT_COLUMNS]
    id_index = header.index("ID")

    rows = []
    skipped_blank = 0
    for raw in rows_iter:
        # Rows can be shorter than the header if trailing cells are empty
        raw = list(raw) + [None] * (len(header) - len(raw))

        if not normalize_cell(raw[id_index]):
            skipped_blank += 1
            continue

        rows.append([normalize_cell(raw[i]) for i in col_index])

    wb.close()
    return rows, skipped_blank


# Load the previous CSV (if any) keyed by ID so we can report what changed.
def load_previous(csv_path: Path):
    if not csv_path.is_file():
        return None

    for enc in ["utf-8-sig", "cp1252", "latin-1"]:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return {
                    (r.get("ID") or "").strip(): r
                    for r in reader
                    if (r.get("ID") or "").strip()
                }
        except UnicodeDecodeError:
            continue

    return None


# Compare the new export with the previous CSV and print a short summary.
def report_changes(rows, previous):
    if previous is None:
        print("No previous CSV found; nothing to compare against.")
        return

    new_by_id = {r[0]: dict(zip(EXPORT_COLUMNS, r)) for r in rows}

    added = sorted(set(new_by_id) - set(previous))
    removed = sorted(set(previous) - set(new_by_id))

    changed = []
    for sid in sorted(set(new_by_id) & set(previous)):
        diff_cols = [
            c
            for c in EXPORT_COLUMNS
            if normalize_cell(new_by_id[sid].get(c)) != normalize_cell(previous[sid].get(c))
        ]
        if diff_cols:
            changed.append((sid, diff_cols))

    print(f"Compared to previous CSV: {len(added)} added, {len(removed)} removed, {len(changed)} changed")
    for sid in added:
        print(f"  + {sid}")
    for sid in removed:
        print(f"  - {sid}")
    for sid, cols in changed:
        print(f"  ~ {sid}: {', '.join(cols)}")


# Main pipeline:
# 1. Read the data sheet from the workbook
# 2. Compare with the previous CSV and print what changed
# 3. Write the CSV (UTF-8 with BOM so Excel opens it correctly; the parser accepts it)
def main():
    print(f"Reading: {INPUT_FILE}")
    rows, skipped_blank = read_rows(INPUT_FILE)

    if not rows:
        raise ValueError("No rows with an ID were found in the data sheet.")

    previous = load_previous(OUTPUT_FILE)
    report_changes(rows, previous)

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(EXPORT_COLUMNS)
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_FILE}")
    if skipped_blank:
        print(f"Skipped {skipped_blank} blank row(s).")


# Script entry point
if __name__ == "__main__":
    main()
