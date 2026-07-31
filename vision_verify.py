#!/usr/bin/env python3
"""Visual re-check of parsed pages against the rendered page image.

Runs on the pages quality_check.py routed here (rule-flagged pages plus every
page claiming a table). The model sees the page image and the text we
extracted from it, and reports where they disagree.

Two things this module refuses to do:

1. Assume the image arrived. A vision call can succeed at the HTTP level with
   the image silently dropped, and the model will then happily produce a
   confident-looking comparison report from the text alone. Every request
   stamps a random token onto the image and requires it back; a wrong token
   means the page was never actually seen, and the page goes to a human.
2. Force a verdict. `severity` includes `cannot_verify` precisely so the model
   has somewhere to put "the image is too blurry / the region is cut off"
   other than a guess.

Usage:
  python3 vision_verify.py                 # all queued pages
  python3 vision_verify.py --limit 5       # smoke test
  python3 vision_verify.py --doc m5_asplos25
Output: vision_report.jsonl (one record per page)
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import fitz
import yaml

import llm

ROOT = Path(__file__).parent
CORPUS_YAML = ROOT / "corpus.yaml"
BLOCKS_PATH = ROOT / "blocks.jsonl"
QUALITY_REPORT = ROOT / "quality_report.json"
OUT_PATH = ROOT / "vision_report.jsonl"

RENDER_DPI = 130

SEVERITIES = {"ok", "minor", "major", "cannot_verify"}

PROMPT = """You are auditing a PDF text-extraction pipeline.

You are shown an image of one page from an academic paper, plus the text our
parser extracted from that same page.

FIRST, before anything else, read the short verification code stamped in the
coloured box at the very top-left corner of the image and copy it exactly.
If you cannot see such a box, set "image_token" to "NONE".

THEN compare the extracted text against what the page image actually shows.
Concentrate on: numbers, units, table cell alignment (is each value under the
right column and beside the right row label?), and whether any visible text is
missing from the extraction.

Report ONLY real discrepancies. If the extraction matches the image, return an
empty "discrepancies" list.

Reply with ONLY a JSON object, no prose and no code fences:

{
  "image_token": "<the code you read, or NONE>",
  "page_summary": "<one short sentence describing what is on this page>",
  "discrepancies": [
    {
      "location": "<where on the page, e.g. 'Table 2, row VoltDB, column Mem'>",
      "extracted": "<what the parser text says>",
      "image_shows": "<what the page image actually shows>",
      "severity": "<one of: minor, major, cannot_verify>"
    }
  ]
}

Use severity "cannot_verify" whenever the image is unreadable at that spot or
you are not certain. Do not guess: an honest "cannot_verify" is much more
useful to us than a confident answer that might be wrong."""


def make_token(n=6):
    # avoid characters that render ambiguously at small sizes (0/O, 1/I)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(n))


def render_page_png(page, token):
    """Render the page and stamp `token` in a high-contrast box top-left.

    The stamp is the channel health check: it exists only in the image, never
    in the text we send, so the model can only report it by actually seeing
    the picture."""
    pix = page.get_pixmap(dpi=RENDER_DPI)
    doc = fitz.open()
    newpage = doc.new_page(width=pix.width, height=pix.height)
    newpage.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)

    box = fitz.Rect(8, 8, 8 + 13 * len(token) + 16, 44)
    newpage.draw_rect(box, color=(0, 0, 0), fill=(1, 0.95, 0.2), width=1.5)
    newpage.insert_text(
        (box.x0 + 8, box.y1 - 12), token,
        fontsize=22, fontname="hebo", color=(0, 0, 0),
    )
    out = newpage.get_pixmap(dpi=72).tobytes("png")
    doc.close()
    return out


def call_vision(prompt, png_bytes, timeout=900):
    return llm.complete_vision(prompt, png_bytes, json_mode=True,
                               temperature=0.0, timeout=timeout)


def parse_model_json(text):
    data = llm.parse_json_reply(text)
    return (data, None) if data is not None else (None, "unparseable model output")


def normalize_record(data, token, doc_id, page_no):
    """Validate the model's reply and decide whether we can trust it at all."""
    rec = {
        "doc_id": doc_id, "page": page_no,
        "image_token_expected": token,
        "image_token_reported": (data.get("image_token") or "").strip().upper(),
        "page_summary": data.get("page_summary", ""),
        "discrepancies": [],
    }
    rec["channel_ok"] = rec["image_token_reported"] == token
    if not rec["channel_ok"]:
        # The model never saw the image, so anything else it said about the
        # page is unfounded -- drop its findings rather than record them.
        rec["status"] = "channel_failed_needs_human"
        return rec

    dropped = 0
    for d in data.get("discrepancies") or []:
        sev = str(d.get("severity", "")).strip().lower()
        if sev not in SEVERITIES:
            sev = "cannot_verify"
        extracted = str(d.get("extracted", ""))[:300]
        image_shows = str(d.get("image_shows", ""))[:300]
        # The model sometimes fills the template for a cell it merely checked,
        # reporting the same string in both fields. Identical values are by
        # definition agreement, so drop them rather than send a human to look
        # at a page where nothing differs.
        if extracted.strip() and extracted.strip() == image_shows.strip():
            dropped += 1
            continue
        rec["discrepancies"].append({
            "location": str(d.get("location", ""))[:300],
            "extracted": extracted,
            "image_shows": image_shows,
            "severity": sev,
        })
    if dropped:
        rec["self_agreeing_dropped"] = dropped
    worst = {"major": 3, "minor": 2, "cannot_verify": 1}
    rec["status"] = "discrepancies_found" if rec["discrepancies"] else "clean"
    rec["max_severity"] = max(
        (d["severity"] for d in rec["discrepancies"]),
        key=lambda s: worst.get(s, 0), default="none",
    )
    return rec


#: statuses that mean we never got a usable answer for that page
UNRESOLVED = {"request_failed", "model_output_unparseable", "channel_failed_needs_human"}


def latest_records():
    """Most recent record per page -- the file is append-only, so a retry
    pass adds a newer record rather than rewriting the old one."""
    latest = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                latest[(r["doc_id"], r["page"])] = r
    return latest


def load_queue(doc_filter=None, retry_failed=False, skip_done=False):
    if not QUALITY_REPORT.exists():
        sys.exit("quality_report.json not found -- run quality_check.py first")
    report = json.loads(QUALITY_REPORT.read_text())
    pages = [(p["doc_id"], p["page"]) for p in report["review_pages"]]
    if doc_filter:
        pages = [p for p in pages if p[0] == doc_filter]
    if retry_failed or skip_done:
        latest = latest_records()
        if retry_failed:
            pages = [p for p in pages
                     if p in latest and latest[p].get("status") in UNRESOLVED]
        else:
            pages = [p for p in pages
                     if p not in latest or latest[p].get("status") in UNRESOLVED]
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only N pages (smoke test)")
    ap.add_argument("--doc", default=None, help="restrict to one doc_id")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retry-failed", action="store_true",
                    help="only pages whose latest record never produced a usable answer")
    ap.add_argument("--resume", action="store_true",
                    help="skip pages already answered; useful after an interrupted run")
    args = ap.parse_args()

    corpus = {e["doc_id"]: e for e in yaml.safe_load(CORPUS_YAML.read_text())}
    by_doc_page = defaultdict(list)
    with open(BLOCKS_PATH) as f:
        for line in f:
            b = json.loads(line)
            by_doc_page[(b["doc_id"], b["page"])].append(b)

    queue = load_queue(args.doc, retry_failed=args.retry_failed, skip_done=args.resume)
    if args.limit:
        queue = queue[:args.limit]
    print(f"{len(queue)} page(s) queued for visual re-check\n", file=sys.stderr)

    done = 0
    with open(OUT_PATH, "a") as out:
        for doc_id, page_no in queue:
            entry = corpus[doc_id]
            pdf = fitz.open(ROOT / entry["file"])
            page = pdf[page_no - 1]
            token = make_token()
            png = render_page_png(page, token)

            blocks = by_doc_page[(doc_id, page_no)]
            extracted = "\n\n".join(
                f"[{b['block_type']}] {b['text']}" for b in blocks
            ) or "(the parser produced no blocks for this page)"

            prompt = (
                f"{PROMPT}\n\n--- PARSER OUTPUT FOR THIS PAGE ---\n{extracted}\n"
            )
            t0 = time.time()
            try:
                raw = call_vision(prompt, png, timeout=args.timeout)
                data, err = parse_model_json(raw)
                if data is None:
                    rec = {"doc_id": doc_id, "page": page_no,
                           "status": "model_output_unparseable",
                           "error": err, "raw": raw[:800]}
                else:
                    rec = normalize_record(data, token, doc_id, page_no)
            except (llm.LLMError, TimeoutError, OSError) as e:
                rec = {"doc_id": doc_id, "page": page_no,
                       "status": "request_failed", "error": str(e)}
            rec["elapsed_s"] = round(time.time() - t0, 1)
            pdf.close()

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            print(f"[{done}/{len(queue)}] {doc_id} p{page_no}: {rec['status']} "
                  f"({rec['elapsed_s']}s)", file=sys.stderr)

    print(f"\nappended {done} record(s) to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
