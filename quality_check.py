#!/usr/bin/env python3
"""blocks.jsonl + PDFs -> quality flags. Pure rules, zero model cost.

The point is not to prove the parse is correct -- it's to make the set of
pages a human or vision model must look at as small as possible, while
keeping the false-negative rate low enough that "unflagged" means something.

Usage:  python3 quality_check.py [doc_id ...]
Output: quality_report.json  (+ a summary table on stdout)
"""

import argparse
import fitz
import yaml
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

from parse import find_gaps

from corpus import load_corpus

ROOT = Path(__file__).parent

# A page is only worth a (paid, slow) vision pass if a rule fired on it or it
# claims to contain a table -- tables are where a silent column-attribution
# error does the most damage to a numeric claim.
CID_RE = re.compile(r"\(cid:\d+\)")
# A measurement whose *decimal point* is broken by whitespace ("12 .5 ms",
# "12. 5 ms"). Deliberately narrow: two adjacent integers before a unit is
# ordinary prose ("25,600 4 KB pages"), and matching that buries the signal,
# whereas a split decimal reads as valid while being off by 10x.
UNIT = r"(?:ns|µs|us|ms|GB/s|MB/s|TiB|GiB|MiB|KiB|TB|GB|MB|KB|%)"
BROKEN_MEASUREMENT_RE = re.compile(
    rf"\b\d+(?:\s+\.\s*\d+|\.\s+\d+)\s*{UNIT}\b", re.I
)
# A number immediately followed by Mathematical Alphanumeric letters is a unit
# whose font carries a broken ToUnicode table. M5 renders "54 µs" through
# LibertineMathMI and it extracts as "54" + U+1D44D U+1D440 -> "54ZM"; "270 ns"
# becomes "270LM". The text decodes without error and is simply wrong, so no
# encoding check catches it and no query for "µs" will ever match. This is the
# most dangerous corruption class in the corpus: the value is present, looks
# fine, and is invisible.
MISENCODED_UNIT_RE = re.compile(r"\d\s*[\U0001D400-\U0001D7FF]{1,4}")
SENTENCE_END = tuple(".!?:;”\"')]")


def alnum_count(s):
    return sum(c.isalnum() for c in s)


def select_entries(corpus, wanted=None):
    entries = corpus.entries
    return [e for e in entries if e["doc_id"] in wanted] if wanted else entries


def load_blocks(corpus):
    by_doc_page = defaultdict(list)
    with open(corpus.blocks_path) as f:
        for line in f:
            b = json.loads(line)
            by_doc_page[(b["doc_id"], b["page"])].append(b)
    return by_doc_page


def check_page(entry, page, page_no, blocks, flags):
    """Append {check_name, detail, severity} flags for one page."""
    doc_id = entry["doc_id"]

    def flag(check, detail, severity="review"):
        flags.append({
            "doc_id": doc_id, "page": page_no, "check_name": check,
            "detail": detail, "severity": severity,
        })

    raw = page.get_text()
    captured = " ".join(b["text"] for b in blocks)

    # --- 1. CID garble ------------------------------------------------------
    # A broken font CMap yields either literal "(cid:NN)" or a stream that
    # decodes to punctuation noise. Both mean the glyphs are unreadable as
    # text, so nothing downstream should trust this page's numbers.
    n_cid = len(CID_RE.findall(raw))
    if n_cid:
        flag("cid_garble", f"{n_cid} literal (cid:N) sequences in page text", "blocker")
    dense = len(re.sub(r"\s", "", raw))
    if dense >= 200:
        ratio = alnum_count(raw) / dense
        if ratio < 0.55:
            flag("low_alnum_ratio",
                 f"only {ratio:.0%} of non-space page chars are alphanumeric "
                 f"(possible broken CMap)", "blocker")

    # --- 2. coverage gap ---------------------------------------------------
    # Running heads/page numbers are dropped on purpose, so judge on the
    # absolute number of characters lost rather than the ratio -- otherwise
    # every short page trips the check.
    raw_a, got_a = alnum_count(raw), alnum_count(captured)
    lost = raw_a - got_a
    if raw_a and lost > 120 and got_a / raw_a < 0.92:
        flag("coverage_gap",
             f"{lost} alnum chars in the PDF are missing from blocks "
             f"({got_a}/{raw_a} = {got_a/raw_a:.0%} captured)")

    # --- 3. bbox spanning the column gutter --------------------------------
    # On a two-column page, a prose block whose bbox crosses the gutter means
    # the column split failed and two unrelated columns were spliced -- the
    # failure mode that silently fabricates sentences.
    prose = [b for b in blocks if b["block_type"] in ("prose", "heading")]
    if len(prose) >= 4:
        # Find the gutter the same way parse.py does, from the whitespace
        # between block extents, so this is a genuine cross-check of that
        # decision rather than an independently-guessed midline.
        x0s = min(b["bbox"][0] for b in prose)
        x1s = max(b["bbox"][2] for b in prose)
        gaps = find_gaps([(b["bbox"][0], b["bbox"][2]) for b in prose], x0s, x1s, min_gap=15.0)
        center = (x0s + x1s) / 2
        gaps = [g for g in gaps
                if x0s + (x1s - x0s) * 0.25 < g[0] < x1s - (x1s - x0s) * 0.25]
        if gaps:
            g = min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - center))
            for b in prose:
                x0, x1 = b["bbox"][0], b["bbox"][2]
                # a wide heading or a spanning title legitimately crosses the
                # gutter; a long *prose* body block doing so is the splice
                # failure mode worth a human look
                if x0 < g[0] - 2 and x1 > g[1] + 2 and len(b["text"]) > 200:
                    flag("bbox_crosses_gutter",
                         f"prose block spans the column gutter "
                         f"[{g[0]:.0f},{g[1]:.0f}]: {b['text'][:60]!r}")

    # --- 4. table shape ----------------------------------------------------
    for b in blocks:
        if b["block_type"] != "table":
            continue
        rows = b.get("table_rows") or []
        if b.get("quality_flag"):
            flag("table_parse_flag", f"parser flagged this table: {b['quality_flag']}")
        if not rows:
            continue
        filled = [sum(1 for c in r if c.strip()) for r in rows]
        if len(rows) >= 3:
            modal = Counter(filled).most_common(1)[0][0]
            odd = [i for i, n in enumerate(filled) if n < max(1, modal - 1)]
            if len(odd) > len(rows) * 0.34:
                flag("table_ragged_rows",
                     f"{len(odd)}/{len(rows)} rows have fewer filled cells than "
                     f"the modal {modal} (row indices {odd[:6]})")
        empties = [i for i, n in enumerate(filled) if n == 0]
        if empties:
            flag("table_empty_row", f"row indices {empties[:6]} are entirely empty")

    # --- 5. numeric / unit sanity ------------------------------------------
    # Coordinate-order bugs show up in extracted numbers as an intruding
    # space. These are exactly the tokens `check.py` will compare, so a
    # malformed one is worse than a missing one.
    for b in blocks:
        for m in BROKEN_MEASUREMENT_RE.finditer(b["text"]):
            ctx = b["text"][max(0, m.start() - 25):m.end() + 25]
            flag("broken_measurement", f"measurement broken by whitespace: ...{ctx!r}...")

    # check the PDF text layer, not our blocks: parse.py normalizes the math
    # alphanumeric range to ASCII, which turns the corruption into innocuous
    # letters ("54ZM") and hides the very thing we are looking for
    for m in MISENCODED_UNIT_RE.finditer(raw):
        ctx = " ".join(raw[max(0, m.start() - 40):m.end() + 30].split())
        flag("misencoded_unit",
             f"number followed by math-italic letters -- unit lost to a broken "
             f"font ToUnicode table, and unfindable by unit name: ...{ctx}...",
             "blocker")

    # --- 6. mid-sentence paragraph break ----------------------------------
    # Two prose blocks where the first ends without terminal punctuation and
    # the second resumes lowercase = one paragraph wrongly cut in two, which
    # splits a claim across chunks and hurts retrieval.
    seq = [b for b in blocks if b["block_type"] == "prose"]
    for a, b in zip(seq, seq[1:]):
        ta, tb = a["text"].rstrip(), b["text"].lstrip()
        if not ta or not tb:
            continue
        # Only within one column: a sentence continuing across a column or
        # page break is normal typesetting, not a parse error, and counting
        # those drowns the check in false positives.
        same_column = abs(a["bbox"][0] - b["bbox"][0]) < 25
        directly_below = b["bbox"][1] >= a["bbox"][3] - 2
        if (same_column and directly_below
                and not ta.endswith(SENTENCE_END) and tb[:1].islower()):
            flag("paragraph_split_mid_sentence",
                 f"...{ta[-40:]!r} || {tb[:40]!r}...", "info")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--corpus", default=None, help="corpus directory")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    entries = select_entries(corpus, set(args.doc_ids) or None)
    by_doc_page = load_blocks(corpus)

    flags = []
    pages_total = 0
    table_pages = set()

    for entry in entries:
        doc = fitz.open(corpus.root / entry["file"])
        for i, page in enumerate(doc):
            page_no = i + 1
            pages_total += 1
            blocks = by_doc_page[(entry["doc_id"], page_no)]
            if any(b["block_type"] == "table" for b in blocks):
                table_pages.add((entry["doc_id"], page_no))
            check_page(entry, page, page_no, blocks, flags)
        doc.close()

    # `info` flags are logged for the report but must not pull a page into the
    # vision queue -- they mark cosmetic splits, not suspect content
    flagged_pages = {(f["doc_id"], f["page"]) for f in flags if f["severity"] != "info"}
    # B2: the vision pass covers rule-flagged pages *plus* every table page
    review_pages = sorted(flagged_pages | table_pages)

    report = {
        "pages_total": pages_total,
        "pages_flagged": len(flagged_pages),
        "pages_with_tables": len(table_pages),
        "pages_for_vision_review": len(review_pages),
        "flag_rate": round(len(flagged_pages) / pages_total, 4) if pages_total else 0,
        "by_check": dict(Counter(f["check_name"] for f in flags)),
        "by_severity": dict(Counter(f["severity"] for f in flags)),
        "review_pages": [{"doc_id": d, "page": p} for d, p in review_pages],
        "flags": flags,
    }
    corpus.quality_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"pages total            : {pages_total}")
    print(f"pages flagged by rules : {len(flagged_pages)} ({report['flag_rate']:.1%})")
    print(f"pages containing tables: {len(table_pages)}")
    print(f"pages for vision review: {len(review_pages)}")
    print()
    print("flags by check:")
    for name, n in sorted(report["by_check"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {name}")
    print()
    print("flags by severity:")
    for sev, n in sorted(report["by_severity"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {sev}")
    print(f"\nwrote {corpus.quality_report}")


if __name__ == "__main__":
    main()
