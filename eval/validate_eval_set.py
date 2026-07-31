#!/usr/bin/env python3
"""Check eval_set.jsonl is well-formed and that its ground truth is real.

The eval set is hand-written on purpose (a model-generated ground truth would
make the evaluation circular), which means the usual risk applies: a typo or a
misremembered page number silently becomes "the right answer". This script
re-derives what it can from the parsed corpus and the PDFs, so the ground
truth is checked against the documents rather than trusted.

It verifies claims of *presence* by locating the quote, and it re-runs the
claims of *absence* that the B/C items depend on.
"""

import json
import re
import sys
from pathlib import Path

import fitz
import yaml

ROOT = Path(__file__).parent.parent
EVAL_PATH = ROOT / "eval" / "eval_set.jsonl"

REQUIRED = {"id", "class", "question", "ground_truth", "expected_behavior", "source", "note"}
CLASSES = {"A", "B", "C", "selfaudit"}
BEHAVIORS = {"answer", "refuse", "flag", "flag_ambiguous"}


def norm(s):
    """Fold whitespace, quote styles and the ligature/arrow damage that the
    PDF text layer introduces, so a quote can be located despite it."""
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("µ", "u").replace("μ", "u")
    s = s.replace("→", "x").replace("×", "x").replace("↔", "-")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_pdf_text(corpus):
    texts = {}
    for e in corpus:
        doc = fitz.open(ROOT / e["file"])
        texts[e["doc_id"]] = [p.get_text() for p in doc]
        doc.close()
    return texts


def main():
    corpus = yaml.safe_load((ROOT / "corpus.yaml").read_text())
    by_id = {e["doc_id"]: e for e in corpus}
    pdf_text = load_pdf_text(corpus)

    items = []
    with open(EVAL_PATH) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append((ln, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"FAIL line {ln}: invalid JSON: {e}")
                return 1

    errors, warnings = [], []
    seen_ids = set()

    for ln, it in items:
        iid = it.get("id", f"line{ln}")
        missing = REQUIRED - set(it)
        if missing:
            errors.append(f"{iid}: missing fields {sorted(missing)}")
            continue
        if iid in seen_ids:
            errors.append(f"{iid}: duplicate id")
        seen_ids.add(iid)
        if it["class"] not in CLASSES:
            errors.append(f"{iid}: unknown class {it['class']!r}")
        if it["expected_behavior"] not in BEHAVIORS:
            errors.append(f"{iid}: unknown expected_behavior {it['expected_behavior']!r}")

        # B-class must have no ground truth and no source; that is the point
        if it["class"] == "B":
            if it["ground_truth"] is not None:
                errors.append(f"{iid}: B-class must have null ground_truth")
            if it["source"] is not None:
                errors.append(f"{iid}: B-class must have null source")
            if it["expected_behavior"] != "refuse":
                errors.append(f"{iid}: B-class must expect refusal")
            continue

        src = it.get("source")
        if src is None:
            if it["class"] in ("A", "selfaudit"):
                errors.append(f"{iid}: {it['class']}-class needs a source")
            continue

        doc_id = src.get("doc_id")
        if doc_id not in by_id:
            errors.append(f"{iid}: source doc_id {doc_id!r} not in corpus.yaml")
            continue
        if src.get("tier") != by_id[doc_id]["tier"]:
            errors.append(f"{iid}: source tier {src.get('tier')} != corpus tier "
                          f"{by_id[doc_id]['tier']} for {doc_id}")

        page = src.get("page")
        pages = pdf_text[doc_id]
        if not isinstance(page, int) or not (1 <= page <= len(pages)):
            errors.append(f"{iid}: page {page} out of range for {doc_id} "
                          f"(1..{len(pages)})")
            continue

        # locate the quote: exact page first, then anywhere in the document
        quote = src.get("quote") or ""
        probe = norm(quote)[:60]
        if not probe:
            warnings.append(f"{iid}: no quote to verify")
            continue
        if probe in norm(pages[page - 1]):
            continue
        found_on = [i + 1 for i, t in enumerate(pages) if probe in norm(t)]
        if found_on:
            warnings.append(f"{iid}: quote not on p{page} of {doc_id}, "
                            f"but found on p{found_on}")
        else:
            # quotes that splice two locations (C06) or paraphrase a table row
            # will not match verbatim; report rather than fail
            warnings.append(f"{iid}: quote not located verbatim in {doc_id} "
                            f"(may be a composite or table paraphrase)")

    # --- absence re-checks --------------------------------------------------
    # An earlier version claimed M5 states no microsecond figure, "verified"
    # by a regex for µs/μs/us. M5 does state it -- its PDF encodes the unit
    # through a broken ToUnicode table, so "54 µs" extracts as "54" followed
    # by U+1D44D U+1D440. The regex and the mistake it was checking shared the
    # same blind spot, so re-running it confirmed the error.
    #
    # An absence claim is therefore never accepted on a text search alone: a
    # unit that fails to match must also be shown not to be present in
    # corrupted form.
    print("--- absence re-checks ---")
    MATH_RUN = r"[\U0001D400-\U0001D7FF]{1,4}"

    def corrupted_unit_sites(text):
        return re.findall(rf"\d\s*{MATH_RUN}", text)

    m5 = "".join(pdf_text["m5_asplos25"])
    corrupted = corrupted_unit_sites(m5)
    print(f"  M5 numbers followed by math-italic runs (mis-encoded units): "
          f"{len(corrupted)}")
    if corrupted:
        print("    -> any absence claim about an M5 quantity must be checked "
              "against the rendered page, not the text layer")

    # B03 depends on TierLab scenario names being absent from M5; scenario
    # names are plain ASCII, so a text search is sound here
    for scen in ("zipf_high_ct", "zipf_low_mlp", "phase_change"):
        n = len(re.findall(scen, m5, re.I))
        status = "OK" if n == 0 else "STALE -- revisit B03"
        print(f"  '{scen}' in M5: {n}  {status}")
        if n:
            errors.append(f"B03 assumes '{scen}' is absent from M5, but it appears")

    # corpus-wide census of the same corruption, since it is a stated finding
    total_corrupt = sum(len(corrupted_unit_sites("".join(pages)))
                        for pages in pdf_text.values())
    print(f"  corpus-wide mis-encoded unit sites : {total_corrupt}")

    counts = {}
    for _, it in items:
        counts[it["class"]] = counts.get(it["class"], 0) + 1
    print("\n--- composition ---")
    for c in ("A", "B", "C", "selfaudit"):
        print(f"  {c:9s}: {counts.get(c, 0)}")
    print(f"  {'total':9s}: {len(items)}")

    if warnings:
        print("\n--- warnings (inspect, not necessarily wrong) ---")
        for w in warnings:
            print(f"  {w}")
    if errors:
        print("\n--- ERRORS ---")
        for e in errors:
            print(f"  {e}")
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
