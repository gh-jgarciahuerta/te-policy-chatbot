from pathlib import Path
import pandas as pd


# =============================================================================
# File settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

INPUT_FILE = CODE_DIR / "1.T&E_RAG_V1.4.xlsx"
OUTPUT_FILE = CODE_DIR / "3.T&E_RAG_V1.4.md"

SHEET_NAME = "T&E_Policy_RAG_Data"
COLUMN_INDEX = 3  # Column D = zero-based index 3


# =============================================================================
# Helper functions
# =============================================================================


def clean_markdown_cell(value) -> str:
    """Clean one Excel cell containing markdown text."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove wrapping quotes if the full cell is quoted
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Convert Excel-style escaped quotes
    text = text.replace('""', '"')

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text.strip()


# =============================================================================
# Main process
# =============================================================================


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input Excel file not found: {INPUT_FILE}")

    print(f"Reading Excel file: {INPUT_FILE}")
    print(f"Sheet: {SHEET_NAME}")
    print("Column: D")

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    if df.shape[1] <= COLUMN_INDEX:
        raise ValueError(
            f"Column D was not found. Sheet only has {df.shape[1]} columns."
        )

    markdown_blocks = []

    for value in df.iloc[:, COLUMN_INDEX]:
        cleaned = clean_markdown_cell(value)
        if cleaned:
            markdown_blocks.append(cleaned)

    if not markdown_blocks:
        raise ValueError("No markdown content found in column D.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        "\n\n".join(markdown_blocks) + "\n",
        encoding="utf-8",
    )

    print(f"Markdown rows exported: {len(markdown_blocks)}")
    print(f"Created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
