# T&E Policy Chatbot - Improvement TODO

Working list from the 2026-09-22 review of the RAG pipeline. Ordered roughly by
expected impact on answer quality. Check items off as they land.

## Retrieval quality

- [x] **Embed only the atomic rule + Q&A, not the full section markdown.** (done 2026-09-22)
  Every chunk currently carries the whole policy section, so sibling rules from
  the same section embed almost identically. Top-5 often returns 3 near-duplicates
  from one section and crowds out other relevant sections. Keep the section text
  in the record and attach it at answer time instead of embedding it.
  Files: `code/11.build_retrieval_text.py`, `code/14.chat_rag.py`

- [x] **Add keyword (BM25) search alongside FAISS.** (done 2026-09-22)
  Pure vector search is weakest on exact terms: dollar amounts, section numbers,
  jargon like "per diem". Hybrid retrieval is cheap and helps these cases.
  Files: `code/14.chat_rag.py` (BM25 is built in memory from the FAISS docstore; no
  second index on disk). New dependency: `rank-bm25`.
  Lesson: keep the fusion weights equal. With 0.6/0.4 a keyword-only hit at rank 1
  scored below a vector hit at rank 5 and never reached the top 5.

## Evaluation

- [ ] **Build an evaluation set.**
  30 to 50 real employee questions, each with the expected answer and the policy
  section(s) that support it. Without this, quality is anecdotal and there is no
  way to tell whether a retrieval or prompt change helped or hurt.

- [ ] **Script the eval run.**
  Once the set exists, add a script that runs every question through the chain
  and reports retrieval hit rate (expected section in top-k) and answer quality.

## Chat behaviour

- [x] **Lower temperature from 0.4 to 0.1.** (done 2026-09-22)
  Policy answers should be consistent between users asking the same thing.
  File: `code/14.chat_rag.py`

- [x] **Raise or remove the 300 max-token cap.** (done 2026-09-22)
  Multi-part answers (e.g. rules with several exceptions) get truncated.
  Raised to 1024 rather than removed: Bedrock always applies a cap and a ceiling
  guards against runaway output. A 5-part meal question now answers in full.
  File: `code/14.chat_rag.py`

- [x] **Add conversation memory.** (done 2026-09-22)
  Follow-ups like "what about international?" fail today because every question
  is answered in isolation. Users expect chat; this is currently a lookup tool.
  Implemented as a rolling window of the last 5 turns. Follow-ups are rewritten
  into a standalone question (one extra model call) before retrieval, and the
  history is passed to the answering prompt. `new` / `clear` reset it.
  Observation from testing: a single question spanning 5 topics only gets the
  topics that fit in the top-5 chunks. Covered by the Evaluation items.
  File: `code/14.chat_rag.py`

## Data pipeline robustness

- [x] **Harden the tagging format.** (done 2026-09-22)
  A stray `[a]` typo in Excel silently drops a rule. Options: fail the run on
  parse warnings, or emit a summary report that someone actually reviews.
  Did both. The parser now collects ERRORS (untagged text, mistyped tag lines,
  multi-tag or duplicate rule tags, Q&A blocks missing Q:/A:, Q&A tags with no
  rule, duplicate record ids) and WARNINGS (tag gaps, ID vs heading mismatch,
  legacy `||` format), prints a report, and refuses to write the JSON on any
  error (exit 1). `--allow-errors` overrides. Blank trailing rows are skipped.
  File: `code/8.csv_to_json.py`

- [x] **Decide on section 9 numbering: ID column vs document heading.** (done 2026-09-23)
  15 rows had IDs one level deeper than the heading in the policy text. Fixed in
  the Excel by renaming the IDs to match the headings (e.g. `9.2.1.1.2` ->
  `9.2.1.2`). CSV was regenerated from the Excel data sheet (ID, Format-Only
  Markdown - Edited, Atomic Rule, Q&A; cp1252, CRLF) and the 12 drafted Q&A
  blocks were re-applied. Parser reports no issues; 364 records.

- [x] **Add the 12 drafted Q&A blocks to the Excel.** (done 2026-09-23)
  Written into the Q&A column of `T&E_Policy_RAG_Data` with openpyxl in rich-text
  mode. Verified against a backup: only the 10 target cells changed; README sheet
  images, table, freeze panes and column widths intact. Excel and CSV now match.

- [x] **Make the Excel -> CSV export a scripted step.** (done 2026-09-23)
  New `code/6.excel_to_csv.py` exports ID, Format-Only Markdown - Edited, Atomic
  Rule and Q&A from the data sheet to `7.T&E_RAG_V1.4.csv` (UTF-8 BOM, CRLF) and
  prints added / removed / changed rows versus the previous CSV. Do not edit the
  CSV by hand any more; edit the Excel and rerun step 6.
  Files from the CSV onward were renumbered by +1 to make room:
  7 csv, 8 csv_to_json, 9 policyInJson, 10 validate_json, 11 build_retrieval_text,
  12 retrieval_chunks, 13 build_index, 14 chat_rag.

## Delivery and expectations

- [ ] **Move off the terminal.**
  A CLI script will not get adoption. Slack or Teams integration is a separate
  piece of work but decides whether anyone uses this.

- [ ] **Set user expectations in the prompt / UI.**
  Many T&E questions are "can I expense X in my situation", and the policy
  often says "with manager approval". The bot restates policy; it does not make
  decisions. Say so up front.
