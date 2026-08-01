#!/usr/bin/env python3
"""Three-arm comparison over eval_set.jsonl.

  arm "none"    - no retrieval; the model answers from its own weights
  arm "raw"     - retrieval over unparsed PDF text (page.get_text(), naive
                  chunks, no coordinate layout, no filters, no reranker)
  arm "system"  - this pipeline

The middle arm exists to separate two things that are easy to conflate: how
much of the improvement comes from *retrieving* at all, and how much comes
from the parsing and scoping work. Without it, a good result would say only
"RAG helps", which was never in question.

Every arm is measured with the same deterministic scorer. Numeric comparison
goes through units.py, never through a model -- a model judging its own
arithmetic would undermine the measurement.

The baseline arm must actually be run. Modern models often refuse rather than
invent when asked about papers they have not seen, and a report that assumed
confident fabrication would be wrong about its own premise.

Usage:
  python3 eval/run_baseline.py --arms none raw system --repeats 3
  python3 eval/run_baseline.py --arms system --repeats 1 --limit 5   # smoke
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import fitz                      # noqa: E402
import numpy as np               # noqa: E402
import yaml                      # noqa: E402

import llm                       # noqa: E402
from units import quantities_match, parse_quantity   # noqa: E402

EVAL_PATH = ROOT / "eval" / "eval_set.jsonl"
RESULTS_PATH = ROOT / "eval" / "baseline_results.jsonl"

REFUSAL_MARKERS = [
    "not stated", "does not", "do not contain", "no information",
    "not mentioned", "not provided", "cannot determine", "not found",
    "unable to", "not specified", "no such", "not report",
]

ANSWER_PROMPT_NO_RETRIEVAL = """Answer the question about computer-systems research papers.

If you do not know the answer, or the paper is not one you have reliable
knowledge of, say so plainly rather than guessing. A wrong number is worse
than an admission of ignorance.

Reply with ONLY a JSON object:
{"answered": true or false, "answer": "<your answer, or empty>"}"""

ANSWER_PROMPT_RETRIEVAL = """Answer the question using ONLY the sources provided.

If the sources do not contain the answer, set "answered" to false. Do not fill
the gap from memory. Do not compute or derive values that are not written in
the sources.

Reply with ONLY a JSON object:
{"answered": true or false, "answer": "<your answer with [S#] markers, or empty>"}"""


# ---------------------------------------------------------------------------
# arm "raw": what retrieval looks like without the parsing work
# ---------------------------------------------------------------------------

class RawTextIndex:
    """Naive baseline index: whole-page text in reading order as PyMuPDF
    returns it, fixed-size chunks, BM25 + dense, no metadata filters and no
    reranker. Deliberately the obvious thing someone would build first."""

    def __init__(self, chunk_chars=1000):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        corpus = yaml.safe_load((ROOT / "corpus.yaml").read_text())
        self.chunks = []
        for e in corpus:
            doc = fitz.open(ROOT / e["file"])
            for pi, page in enumerate(doc):
                text = page.get_text()          # no columns, no tables, no order fixing
                for i in range(0, len(text), chunk_chars):
                    piece = text[i:i + chunk_chars].strip()
                    if len(piece) > 80:
                        self.chunks.append({
                            "doc_id": e["doc_id"], "tier": e["tier"],
                            "page_start": pi + 1, "page_end": pi + 1,
                            "section": None, "text": piece,
                        })
            doc.close()
        self.bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", c["text"].lower())
                               for c in self.chunks])
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.emb = self.model.encode([c["text"] for c in self.chunks],
                                     batch_size=64, convert_to_numpy=True,
                                     normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")

    def search(self, query, k=6, **_ignored):
        bm = self.bm25.get_scores(re.findall(r"[a-z0-9]+", query.lower()))
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
        dn = (self.emb @ q[0])
        bm_rank = {i: r for r, i in enumerate(np.argsort(bm)[::-1][:30])}
        dn_rank = {i: r for r, i in enumerate(np.argsort(dn)[::-1][:30])}
        fused = defaultdict(float)
        for i, r in bm_rank.items():
            fused[i] += 1.0 / (60 + r + 1)
        for i, r in dn_rank.items():
            fused[i] += 1.0 / (60 + r + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for i, _ in top:
            c = dict(self.chunks[i])
            c["chunk_id"] = f"raw:{i}"
            c["source_block_ids"] = []
            out.append(c)
        return out


# ---------------------------------------------------------------------------
# scoring -- all deterministic
# ---------------------------------------------------------------------------

def looks_like_refusal(result):
    if not result.get("answered"):
        return True
    text = (result.get("answer") or "").lower()
    return any(m in text for m in REFUSAL_MARKERS)


def numeric_match(answer, ground_truth):
    """True when every number in the ground truth appears in the answer.

    Requires all of them, so "90%+ precision and recall at 0.9% CPU on 5 TB"
    is not satisfied by an answer that produces only the 90%."""
    if not ground_truth:
        return None
    gt_nums = [m.group(0) for m in re.finditer(
        r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|×|x|[a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?", ground_truth)]
    gt_nums = [g for g in gt_nums if parse_quantity(g)]
    if not gt_nums:
        return None
    ans_nums = [m.group(0) for m in re.finditer(
        r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|×|x|[a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?", answer or "")]
    for g in gt_nums:
        if not any(quantities_match(g, a) for a in ans_nums):
            return False
    return True


def cites_wrong_source(result, item):
    """C-class misattribution: did the answer name the trap's wrong system?"""
    trap = (item.get("trap_answer") or "").lower()
    ans = (result.get("answer") or "").lower()
    if not trap or not ans:
        return None
    trap_terms = set(re.findall(r"[a-z][a-z0-9/]{2,}", trap))
    truth_terms = set(re.findall(r"[a-z][a-z0-9/]{2,}",
                                 (item.get("ground_truth") or "").lower()))
    distinctive = trap_terms - truth_terms - {
        "the", "and", "for", "with", "that", "this", "answering", "reporting",
        "labelled", "labeled", "attributed", "confidently", "either", "only",
        "omitted", "generally", "achieves", "yes", "unconditional", "as", "if",
        "were", "consistent", "source", "corpus", "default", "most", "other",
        "systems", "use", "which", "them", "than", "not",
    }
    return bool(distinctive) and any(t in ans for t in distinctive)


def context_hit(evidence, item):
    """Context recall: did retrieval surface the page the ground truth is on?"""
    src = item.get("source")
    if not src or not evidence:
        return None
    for e in evidence:
        if e["doc_id"] == src["doc_id"] and \
                e["page_start"] <= src["page"] <= e["page_end"]:
            return True
    return False


# ---------------------------------------------------------------------------

def run_one(item, arm, retriever, k, timeout):
    q = item["question"]
    evidence = []
    if arm == "none":
        prompt = f"{ANSWER_PROMPT_NO_RETRIEVAL}\n\n--- QUESTION ---\n{q}\n"
    else:
        hits = retriever.search(q, k=k)
        evidence = hits
        blocks = []
        for i, h in enumerate(hits, 1):
            pages = (f"p{h['page_start']}" if h["page_start"] == h["page_end"]
                     else f"p{h['page_start']}-{h['page_end']}")
            blocks.append(f"[S{i}] {h['doc_id']} {pages}\n{h['text']}")
        prompt = (f"{ANSWER_PROMPT_RETRIEVAL}\n\n--- SOURCES ---\n"
                  + "\n\n".join(blocks) + f"\n\n--- QUESTION ---\n{q}\n")

    t0 = time.time()
    try:
        raw = llm.complete(prompt, json_mode=True, temperature=0.0, timeout=timeout)
        data = llm.parse_json_reply(raw) or {}
        result = {"answered": bool(data.get("answered")),
                  "answer": (data.get("answer") or "").strip()}
        error = None
    except Exception as e:            # noqa: BLE001 - recorded, not swallowed
        result, error = {"answered": False, "answer": ""}, str(e)

    rec = {
        "id": item["id"], "class": item["class"], "arm": arm,
        "question": q, "answer": result["answer"],
        "answered": result["answered"], "error": error,
        "elapsed_s": round(time.time() - t0, 2),
        "refused": looks_like_refusal(result),
        "evidence": [{"doc_id": e["doc_id"], "page_start": e["page_start"],
                      "page_end": e["page_end"]} for e in evidence],
    }
    if item["class"] in ("A", "C"):
        rec["numeric_match"] = numeric_match(result["answer"], item.get("ground_truth"))
    if item["class"] == "C":
        rec["misattributed"] = cites_wrong_source(result, item)
    if arm != "none":
        rec["context_hit"] = context_hit(evidence, item)
    return rec


def summarize(records):
    """Per-arm metrics. Reported as fractions with the denominator, because
    n is small enough that a bare percentage would overstate precision."""
    out = {}
    by_arm = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)

    for arm, rs in by_arm.items():
        a = [r for r in rs if r["class"] == "A"]
        b = [r for r in rs if r["class"] == "B"]
        c = [r for r in rs if r["class"] == "C"]

        def frac(num, den):
            return {"n": num, "of": den,
                    "rate": round(num / den, 3) if den else None}

        num_ok = [r for r in a if r.get("numeric_match") is not None]
        ctx = [r for r in rs if r.get("context_hit") is not None]
        mis = [r for r in c if r.get("misattributed") is not None]
        out[arm] = {
            "A_numeric_exact_match": frac(sum(1 for r in num_ok
                                              if r["numeric_match"]), len(num_ok)),
            "B_refusal_rate": frac(sum(1 for r in b if r["refused"]), len(b)),
            "C_misattribution_rate": frac(sum(1 for r in mis
                                              if r["misattributed"]), len(mis)),
            "A_refused_wrongly": frac(sum(1 for r in a if r["refused"]), len(a)),
            "context_recall": frac(sum(1 for r in ctx if r["context_hit"]), len(ctx)),
            "errors": sum(1 for r in rs if r.get("error")),
            "mean_latency_s": round(sum(r["elapsed_s"] for r in rs) / len(rs), 2)
            if rs else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["none", "raw", "system"],
                    choices=["none", "raw", "system"])
    ap.add_argument("--repeats", type=int, default=3,
                    help="H2 requires n>=3; lower only for smoke tests")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    items = [json.loads(l) for l in open(EVAL_PATH) if l.strip()]
    items = [i for i in items if i["class"] in args.classes]
    if args.limit:
        items = items[:args.limit]

    if args.repeats < 3:
        print(f"WARNING: repeats={args.repeats} is below the n>=3 the write-up "
              f"requires; treat this run as a smoke test only\n", file=sys.stderr)

    retrievers = {}
    if "system" in args.arms:
        from retrieve import Retriever
        retrievers["system"] = Retriever()
    if "raw" in args.arms:
        print("building the raw-text baseline index ...", file=sys.stderr)
        retrievers["raw"] = RawTextIndex()

    run_meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": llm.describe(),
        "temperature": 0.0,
        "k": args.k,
        "repeats": args.repeats,
        "n_items": len(items),
        "arms": args.arms,
        "prompt_no_retrieval_sha": abs(hash(ANSWER_PROMPT_NO_RETRIEVAL)) % (10 ** 8),
        "prompt_retrieval_sha": abs(hash(ANSWER_PROMPT_RETRIEVAL)) % (10 ** 8),
    }
    print(json.dumps(run_meta, indent=2), file=sys.stderr)

    records = []
    total = len(items) * len(args.arms) * args.repeats
    done = 0
    with open(args.out, "a") as f:
        for rep in range(args.repeats):
            for arm in args.arms:
                for item in items:
                    rec = run_one(item, arm, retrievers.get(arm), args.k, args.timeout)
                    rec["repeat"] = rep
                    rec["run_meta"] = run_meta
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    mark = "!" if rec.get("error") else "."
                    print(f"[{done}/{total}] {arm:6s} {item['id']} {mark}",
                          file=sys.stderr)

    summary = summarize(records)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    (ROOT / "eval" / "baseline_summary.json").write_text(
        json.dumps({"run_meta": run_meta, "summary": summary}, indent=2))
    print(f"\nwrote {args.out} and eval/baseline_summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
