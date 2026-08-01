#!/usr/bin/env python3
"""Grounded question answering over the indexed corpus.

Same model and same index as check.py -- the only difference is how the query
is built and how retrieval is scoped. Nothing here is fine-tuned.

Three properties the prompt has to enforce, because they are what separates
this from asking a chatbot:

  * every number in the answer carries a source marker, and the marker is
    resolved back to doc/page/section by *this code*, not by the model
    (a model that invents a citation label is the failure we are guarding
    against, so it is never trusted to report its own provenance);
  * when the retrieved text does not contain the answer, the answer is
    "not stated" -- a plausible number is worse than no number here;
  * derived values are refused. "54 us / 170 ns = 317 accesses" is arithmetic
    the model must not do silently, because the result looks like a quotation
    from the paper and isn't.

Tier is surfaced on every citation: tier-0 is the survey's own paraphrase of
someone else's work, tier-1 is the original. Conflating them is the specific
error this project exists to catch.

Usage:
  python3 ask.py "how much does a 4 KB page migration cost in M5?"
  python3 ask.py "what is AOL?" --tier 1        # originals only
  python3 ask.py "what does the survey claim about Telescope?" --tier 0
  python3 ask.py "..." --doc m5_asplos25 --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

import llm

from corpus import load_corpus

ROOT = Path(__file__).parent

NOT_STATED = "資料未提及 (not stated in the retrieved sources)"

SYSTEM_PROMPT = """You answer questions about computer-systems papers using ONLY the sources provided below.

Rules, in order of importance:

1. Use ONLY the numbered sources. You have no other knowledge of these papers.
   If the sources do not contain the answer, set "answered" to false. Do not
   fill the gap from memory or from what sounds plausible.

2. Never compute, derive, convert, or estimate a value that is not written in
   the sources. If a source says "54 us" and the question asks for a number of
   accesses, that is a derivation -- refuse it. Report only figures that appear
   verbatim.

3. Every factual statement in your answer must cite the source it came from,
   written as [S1], [S2], and so on. A sentence with a number and no marker is
   not acceptable.

4. Quote numbers exactly as written, including units and any qualifying
   condition attached to them ("under 2 MB THP", "at 15 cores", "for read-only
   workloads"). A number stripped of its condition changes meaning.

5. If sources disagree, say so and cite both rather than picking one.

Reply with ONLY a JSON object:

{
  "answered": true or false,
  "answer": "<your answer with [S#] markers, or an empty string if answered is false>",
  "used_sources": ["S1", "S3"],
  "unsupported_note": "<if answered is false, one sentence on what is missing>"
}"""


def build_context(hits):
    parts = []
    for i, h in enumerate(hits, 1):
        pages = (f"p{h['page_start']}" if h["page_start"] == h["page_end"]
                 else f"p{h['page_start']}-{h['page_end']}")
        tier_note = ("TIER-0, the survey's own paraphrase" if h["tier"] == 0
                     else "TIER-1, original paper")
        section = h.get("section") or "(no section heading)"
        parts.append(
            f"[S{i}] ({tier_note}) {h['doc_id']} {pages}, section: {section}\n"
            f"{h['text']}"
        )
    return "\n\n".join(parts)


def call_model(prompt, timeout=600):
    return llm.complete(prompt, json_mode=True, temperature=0.0, timeout=timeout)


def parse_json_reply(text):
    return llm.parse_json_reply(text)


def ask(question, retriever=None, k=6, doc_id=None, tier=None, exclude_tier=None,
        timeout=600):
    from retrieve import Retriever
    r = retriever or Retriever()
    hits = r.search(question, k=k, doc_id=doc_id, tier=tier, exclude_tier=exclude_tier)

    if not hits:
        return {
            "question": question, "answered": False, "answer": "",
            "not_stated": NOT_STATED,
            "unsupported_note": "retrieval returned no candidate passages",
            "citations": [], "sources": [],
        }

    prompt = (f"{SYSTEM_PROMPT}\n\n--- SOURCES ---\n{build_context(hits)}\n\n"
              f"--- QUESTION ---\n{question}\n")
    raw = call_model(prompt, timeout=timeout)
    data = parse_json_reply(raw) or {}

    # Resolve source markers ourselves. The model names S1/S2; the mapping from
    # marker to doc/page/section comes from the retrieval result, never from
    # anything the model wrote.
    sources = []
    for i, h in enumerate(hits, 1):
        sources.append({
            "marker": f"S{i}", "doc_id": h["doc_id"], "tier": h["tier"],
            "page_start": h["page_start"], "page_end": h["page_end"],
            "section": h.get("section"), "chunk_id": h["chunk_id"],
            "source_block_ids": h.get("source_block_ids", []),
            "text": h["text"],
        })
    by_marker = {s["marker"]: s for s in sources}

    # Take the markers from the answer text as the primary signal: the model
    # reliably writes [S1] inline but often omits the separate used_sources
    # field, and a citation that appears in the prose is the one the reader
    # will actually try to follow.
    used = []
    for m in re.findall(r"\[\s*(S\d+)\s*\]", data.get("answer") or ""):
        if m in by_marker and m not in used:
            used.append(m)
    for m in (data.get("used_sources") or []):
        m = str(m).strip().strip("[]").upper()
        if m in by_marker and m not in used:
            used.append(m)
    used.sort(key=lambda m: int(m[1:]))
    answered = bool(data.get("answered")) and bool((data.get("answer") or "").strip())
    return {
        "question": question,
        "answered": answered,
        "answer": (data.get("answer") or "").strip(),
        "not_stated": None if answered else NOT_STATED,
        "unsupported_note": data.get("unsupported_note") or "",
        "citations": [by_marker[m] for m in used],
        "sources": sources,
        "filters": {"doc_id": doc_id, "tier": tier, "exclude_tier": exclude_tier},
        "model": llm.describe(),
    }


def render(result, show_all_sources=False, retriever_corpus=None):
    corpus = (retriever_corpus or load_corpus()).by_doc_id
    print(f"Q: {result['question']}\n")
    if not result["answered"]:
        print(result["not_stated"])
        if result["unsupported_note"]:
            print(f"  reason: {result['unsupported_note']}")
    else:
        print(result["answer"])

    cites = result["citations"] if result["citations"] else (
        result["sources"] if show_all_sources else [])
    if cites:
        print("\nSources:")
        for s in cites:
            meta = corpus.get(s["doc_id"], {})
            pages = (f"p{s['page_start']}" if s["page_start"] == s["page_end"]
                     else f"p{s['page_start']}-{s['page_end']}")
            tier_label = ("tier-0 SURVEY PARAPHRASE -- not the original"
                          if s["tier"] == 0 else "tier-1 original")
            print(f"  [{s['marker']}] {meta.get('short', s['doc_id'])} "
                  f"({tier_label}) {pages}")
            print(f"        section: {s.get('section') or '(none)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--doc", default=None)
    ap.add_argument("--tier", type=int, default=None,
                    help="0 = survey only, 1 = original papers only")
    ap.add_argument("--exclude-tier", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-sources", action="store_true",
                    help="print retrieved sources even when uncited")
    args = ap.parse_args()

    result = ask(args.question, k=args.k, doc_id=args.doc, tier=args.tier,
                 exclude_tier=args.exclude_tier)
    if args.json:
        slim = dict(result)
        for s in slim["sources"]:
            s["text"] = s["text"][:200]
        for c in slim["citations"]:
            c["text"] = c["text"][:200]
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    else:
        render(result, show_all_sources=args.show_sources)


if __name__ == "__main__":
    main()
