#!/usr/bin/env python3
"""Interactive session for the two things this corpus is actually used for:
looking something up while writing, and proofreading what was written.

Why this exists rather than repeated CLI calls: loading the embedder and the
cross-encoder costs ~17s, and a one-shot command pays that on every question.
Held open, the same query costs ~0.5s. That is the difference between a tool
you consult while writing and one you avoid.

Retrieval needs no API key. Looking a passage up and seeing its page and
section is often the whole answer, so that path works offline; generated
answers and claim-checking light up when a key is set.

    python3 refcheck.py

    > how much does a page migration cost          look it up (no key needed)
    > /ask what migration penalty does TierLab use  generated answer + citations
    > /check <paste a sentence>                     verify claims in that text
    > /doc m5        /tier 1        /k 8            scope the search
    > /tables 2mb                                   set query over tables
    > /help
"""

import argparse
import re
import sys
import time
from pathlib import Path

import yaml

from corpus import load_corpus

ROOT = Path(__file__).parent

BANNER = """paper-refcheck -- lookup & proofreading

  <question>        search the source text (no API key needed)
  /ask <question>    generated answer with citations (needs API key)
  /check <text>      verify each claim in the text against sources (needs API key)
  /tables <pattern>  set query over the table layer, e.g. /tables 2mb
  /doc <id|off>      restrict to one document, e.g. /doc m5
  /tier <0|1|off>    0 = survey's own paraphrase, 1 = original sources
  /k <n>             number of results to return (default 6)
  /docs              list the registered corpus
  /help  /quit
"""


def short_id(corpus, fragment):
    """Accept 'm5' for 'm5_asplos25' -- typing full doc_ids while writing is
    friction the tool does not need to impose."""
    frag = fragment.lower()
    exact = [e["doc_id"] for e in corpus if e["doc_id"].lower() == frag]
    if exact:
        return exact[0]
    hits = [e["doc_id"] for e in corpus
            if frag in e["doc_id"].lower() or frag in (e.get("short") or "").lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"  '{fragment}' matches multiple: {hits}")
        return None
    print(f"  no match for '{fragment}' (see /docs for the list)")
    return None


def fmt_hit(i, h, corpus_by_id, width=320):
    meta = corpus_by_id.get(h["doc_id"], {})
    pages = (f"p{h['page_start']}" if h["page_start"] == h["page_end"]
             else f"p{h['page_start']}-{h['page_end']}")
    tier = "tier-0 paraphrase" if h["tier"] == 0 else "tier-1 original"
    section = h.get("section") or "--"
    text = re.sub(r"\s+", " ", h["text"]).strip()
    if len(text) > width:
        text = text[:width] + "..."
    print(f"\n[{i}] {meta.get('short', h['doc_id'])} . {tier} . {pages}")
    print(f"    section: {section}")
    print(f"    {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=None, help="corpus directory")
    args = ap.parse_args()

    cor = load_corpus(args.corpus)
    corpus = cor.entries
    corpus_by_id = cor.by_doc_id

    print(BANNER)
    print("loading retrieval models...", end="", flush=True)
    t0 = time.time()
    from retrieve import Retriever, query_tables
    r = Retriever(cor)
    r.search("warmup", k=1)          # force the lazy model loads now, not mid-question
    print(f" done ({time.time() - t0:.0f}s; ~0.5s per query after this)\n")

    import llm
    cfg = llm.config()
    has_key = cfg["ready"]
    if not has_key:
        print(f"note: {cfg['key_env']} is not set, so /ask and /check are "
              f"unavailable. Plain search still works.\n")

    state = {"doc": None, "tier": None, "k": 6}

    def scope_label():
        bits = []
        if state["doc"]:
            bits.append(f"doc={state['doc']}")
        if state["tier"] is not None:
            bits.append(f"tier={state['tier']}")
        return f" [{' '.join(bits)}]" if bits else ""

    while True:
        try:
            line = input(f"\n>{scope_label()} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/q", "/exit"):
            break
        if line in ("/help", "/h", "?"):
            print(BANNER)
            continue
        if line == "/docs":
            for e in corpus:
                print(f"  [{e['ref']:>2}] {e['doc_id']:<28} tier {e['tier']}  "
                      f"{e.get('short', '')}")
            continue

        m = re.match(r"^/doc\s+(\S+)$", line)
        if m:
            if m.group(1) in ("off", "none", "-"):
                state["doc"] = None
                print("  doc filter cleared")
            else:
                d = short_id(corpus, m.group(1))
                if d:
                    state["doc"] = d
                    print(f"  restricted to {d}")
            continue

        m = re.match(r"^/tier\s+(\S+)$", line)
        if m:
            v = m.group(1)
            if v in ("off", "none", "-"):
                state["tier"] = None
                print("  tier filter cleared")
            elif v in ("0", "1"):
                state["tier"] = int(v)
                print(f"  restricted to tier {v}" +
                      (" (the survey's own paraphrase)" if v == "0"
                       else " (original sources)"))
            else:
                print("  usage: /tier 0 | /tier 1 | /tier off")
            continue

        m = re.match(r"^/k\s+(\d+)$", line)
        if m:
            state["k"] = max(1, min(20, int(m.group(1))))
            print(f"  k = {state['k']}")
            continue

        m = re.match(r"^/tables\s+(.+)$", line)
        if m:
            rows = query_tables(m.group(1).strip(), doc_id=state["doc"],
                                tier=state["tier"], corpus=cor)
            if not rows:
                print("  no matching rows in the table layer")
            for h in rows:
                flag = f"  [flagged: {h['quality_flag']}]" if h.get("quality_flag") else ""
                print(f"  {h['doc_id']} p{h['page']} | {h['row_label']} | "
                      f"{h['column']} = {h['value']!r}{flag}")
            print(f"\n  {len(rows)} row(s) -- the complete set, not a top-k")
            continue

        m = re.match(r"^/(ask|check)\s+(.+)$", line, re.S)
        if m:
            verb, payload = m.group(1), m.group(2).strip()
            if not has_key:
                print(f"  requires {cfg['key_env']}. Plain search works without it.")
                continue
            try:
                if verb == "ask":
                    from ask import ask, render
                    render(ask(payload, retriever=r, k=state["k"],
                               doc_id=state["doc"], tier=state["tier"]))
                else:
                    from check import check_passage, render as render_check
                    render_check(check_passage(payload, retriever=r, k=state["k"]))
            except Exception as e:                      # noqa: BLE001
                print(f"  failed: {type(e).__name__}: {e}")
            continue

        if line.startswith("/"):
            print("  unknown command, see /help")
            continue

        t = time.time()
        hits = r.search(line, k=state["k"], doc_id=state["doc"], tier=state["tier"])
        if not hits:
            print("  no matches")
            continue
        for i, h in enumerate(hits, 1):
            fmt_hit(i, h, corpus_by_id)
        print(f"\n  {len(hits)} result(s) . {time.time() - t:.1f}s")

    print("bye")


if __name__ == "__main__":
    main()
