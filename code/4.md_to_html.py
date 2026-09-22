from pathlib import Path
import re
import html
import markdown


# =============================================================================
# File settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_INPUT_FILE = CODE_DIR / "3.T&E_RAG_V1.4.md"
DEFAULT_OUTPUT_FILE = CODE_DIR / "5.T&E_RAG_V1.4.html"
DEFAULT_DOCUMENT_TITLE = "GL Power Query Build"


# =============================================================================
# Helper functions
# =============================================================================


def load_env(env_path: Path) -> dict:
    """
    Load simple KEY=VALUE pairs from a .env file.

    Blank lines, comments, and malformed lines are ignored.
    """
    env = {}

    if not env_path.exists():
        return env

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

    return env


def slugify(text: str) -> str:
    """
    Convert heading text into a URL-friendly ID.

    Example:
    "Phase 1: Load & Clean Data" -> "phase-1-load-and-clean-data"
    """
    text = text.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)

    return text.strip("-")


def sanitize_markdown(md_text: str) -> str:
    """
    Clean up markdown before conversion.

    This fixes malformed code fences like:

        ``` id="abc"

    and converts them into plain code fences:

        ```
    """
    cleaned_lines = []

    for line in md_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            cleaned_lines.append("```")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def add_ids_to_headings(html_text: str) -> str:
    """
    Add unique id attributes to headings that do not already have one.

    This makes headings linkable and supports table-of-contents navigation.
    """
    used_ids = set()

    def replace_heading(match):
        level = match.group(1)
        attrs = match.group(2) or ""
        inner_html = match.group(3)

        # Do not overwrite an existing heading id.
        if "id=" in attrs:
            return match.group(0)

        # Remove any nested HTML tags so the ID is based on visible text only.
        plain_text = re.sub(r"<.*?>", "", inner_html)
        heading_id = slugify(plain_text)

        if not heading_id:
            return match.group(0)

        # Avoid duplicate IDs by appending -2, -3, etc.
        base_id = heading_id
        counter = 2

        while heading_id in used_ids:
            heading_id = f"{base_id}-{counter}"
            counter += 1

        used_ids.add(heading_id)

        return f'<h{level}{attrs} id="{heading_id}">{inner_html}</h{level}>'

    return re.sub(
        r"<h([1-6])([^>]*)>(.*?)</h\1>",
        replace_heading,
        html_text,
        flags=re.DOTALL,
    )


def enhance_sections(html_text: str) -> str:
    """
    Apply custom HTML wrappers/classes for styling.

    This function:
    - Wraps the first H1 in a title block.
    - Adds a phase-label span around PHASE headings.
    - Converts CHECKPOINT paragraphs into styled checkpoint boxes.
    """
    html_text = re.sub(
        r"<h1([^>]*)>(.*?)</h1>",
        r'<section class="title-block"><h1\1>\2</h1></section>',
        html_text,
        count=1,
        flags=re.DOTALL,
    )

    html_text = re.sub(
        r"<h2([^>]*)>(PHASE .*?)</h2>",
        r'<h2\1><span class="phase-label">\2</span></h2>',
        html_text,
        flags=re.DOTALL,
    )

    html_text = re.sub(
        r"<p>✅ CHECKPOINT(.*?)</p>",
        r'<div class="checkpoint"><strong>✅ CHECKPOINT\1</strong></div>',
        html_text,
        flags=re.DOTALL,
    )

    return html_text


# =============================================================================
# CSS used in the generated HTML document
# =============================================================================

CSS = """
:root {
    --blue: #1f4e79;
    --blue-light: #eaf3fb;
    --border: #d9e2f3;
    --text: #222;
    --muted: #666;
    --code-bg: #f6f8fa;
    --page-bg: #f3f6fa;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--page-bg);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
    line-height: 1.65;
}

.document {
    max-width: 980px;
    margin: 32px auto;
    padding: 44px 56px;
    background: white;
    border: 1px solid #e3e8ef;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(31, 78, 121, 0.10);
}

.title-block {
    margin-bottom: 28px;
    padding-bottom: 18px;
    border-bottom: 3px solid var(--border);
}

h1 {
    margin: 0;
    color: var(--blue);
    font-size: 32px;
    line-height: 1.2;
}

h2 {
    margin-top: 42px;
    padding: 12px 16px;
    background: var(--blue-light);
    border-left: 6px solid var(--blue);
    border-radius: 8px;
    color: var(--blue);
    font-size: 22px;
}

h3 {
    margin-top: 28px;
    color: var(--blue);
    font-size: 18px;
}

p {
    margin: 10px 0;
}

ul, ol {
    padding-left: 24px;
}

li {
    margin: 4px 0;
}

code {
    padding: 2px 5px;
    background: var(--code-bg);
    border: 1px solid #e6e8eb;
    border-radius: 5px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 0.95em;
}

pre {
    margin: 16px 0;
    padding: 16px 18px;
    background: #0f172a;
    color: #e5e7eb;
    border-radius: 10px;
    overflow-x: auto;
    line-height: 1.5;
}

pre code {
    padding: 0;
    background: transparent;
    border: none;
    color: inherit;
}

blockquote {
    margin: 18px 0;
    padding: 12px 18px;
    background: #fafafa;
    border-left: 5px solid #c8d6e5;
    color: var(--muted);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 14px;
}

th {
    background: var(--blue);
    color: white;
}

th, td {
    padding: 10px 12px;
    border: 1px solid #d7dde5;
    text-align: left;
}

tr:nth-child(even) td {
    background: #fafcff;
}

a {
    color: #0563c1;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

img {
    max-width: 100%;
    height: auto;
    margin: 18px 0;
    padding: 6px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
}

hr {
    margin: 32px 0;
    border: none;
    border-top: 1px solid #dce3ec;
}

.toc {
    margin: 28px 0 36px 0;
    padding: 18px 22px;
    background: #f8fbff;
    border: 1px solid var(--border);
    border-radius: 12px;
}

.toc ul {
    margin: 8px 0;
}

.toc a {
    color: var(--blue);
}

.checkpoint {
    margin: 20px 0;
    padding: 14px 18px;
    background: #f0f9f4;
    border: 1px solid #b7e4c7;
    border-left: 6px solid #2d8a4e;
    border-radius: 8px;
    color: #1f5130;
}

.phase-label {
    display: inline-block;
}

@media print {
    body {
        background: white;
    }

    .document {
        margin: 0;
        padding: 24px;
        border: none;
        box-shadow: none;
        max-width: none;
    }

    h2 {
        break-after: avoid;
    }

    pre, table {
        break-inside: avoid;
    }
}
"""


# =============================================================================
# Main process
# =============================================================================


def main() -> None:
    """
    Convert a markdown process document into a styled HTML file.
    """

    # Load optional overrides from the .env file.
    env = load_env(ENV_FILE)

    # Set input/output files near the top of the process.
    input_file = PROJECT_ROOT / env.get(
        "INPUT_FILE", str(DEFAULT_INPUT_FILE.relative_to(PROJECT_ROOT))
    )
    output_file = PROJECT_ROOT / env.get(
        "OUTPUT_FILE", str(DEFAULT_OUTPUT_FILE.relative_to(PROJECT_ROOT))
    )
    document_title = env.get("DOCUMENT_TITLE", DEFAULT_DOCUMENT_TITLE)

    # Confirm the markdown file exists before trying to read it.
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read and clean the markdown source file.
    md_text = input_file.read_text(encoding="utf-8")
    md_text = sanitize_markdown(md_text)

    # Convert markdown into HTML.
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={
            "toc": {
                "title": "Table of Contents",
                "permalink": False,
            }
        },
    )

    # Add heading IDs and custom styling wrappers.
    html_body = add_ids_to_headings(html_body)
    html_body = enhance_sections(html_body)

    # Build the final standalone HTML document.
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(document_title)}</title>
    <style>
{CSS}
    </style>
</head>
<body>
    <main class="document">
{html_body}
    </main>
</body>
</html>
"""

    # Create the output folder if needed, then write the HTML file.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(full_html, encoding="utf-8")

    print(f"Created: {output_file}")


if __name__ == "__main__":
    main()
