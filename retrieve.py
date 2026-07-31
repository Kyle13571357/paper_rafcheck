#!/usr/bin/env python3
"""Hybrid retrieval over the prose index, plus a filter layer for tables.

BM25 catches the acronyms this corpus runs on (PEBS, DCSC, FMAR, CIT, THP) --
a dense model happily returns a paraphrase that never contains the token.
Dense catches the paraphrase BM25 misses. Neither alone is sufficient here,
so results are fused and then reranked by a cross-encoder.

The `doc_id` / `tier` filters are the reason this module exists in the shape
it does: check.py can only verify a claim against the work it cites if the
search can be *restricted* to that work, and it can only avoid grading the
survey against itself if tier-0 can be excluded. Filters are applied to the
candidate set before ranking, not to the results afterwards, so a filtered
search returns the best k *within* the filter rather than whatever survived.

Usage:
  python3 retrieve.py "how much does page migration cost"
  python3 retrieve.py "migration latency" --doc m5_asplos25
  python3 retrieve.py "2 MB THP" --tables
  python3 retrieve.py --acceptance          # Module D check
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "index"
TABLES_DIR = ROOT / "tables"
CORPUS_YAML = ROOT / "corpus.yaml"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60          # standard reciprocal-rank-fusion damping


class Retriever:
    def __init__(self, load_reranker=False):
        import faiss
        from rank_bm25 import BM25Okapi

        self.chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl")]
        self.manifest = json.loads((INDEX_DIR / "manifest.json").read_text())
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "prose.faiss"))
        self.bm25 = BM25Okapi([c["tokens"] for c in self.chunks])
        self._faiss = faiss
        self._embedder = None
        self._reranker = None
        if load_reranker:
            self._load_reranker()

    # -- lazy model loading; a BM25-only query shouldn't pay for a model ----
    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    # -- filtering ---------------------------------------------------------
    def allowed_indices(self, doc_id=None, tier=None, exclude_tier=None,
                        block_types=None):
        """Indices surviving the metadata filters. Returns None when nothing
        is filtered, so callers can take the fast unfiltered path."""
        if doc_id is None and tier is None and exclude_tier is None and not block_types:
            return None
        docs = {doc_id} if isinstance(doc_id, str) else (set(doc_id) if doc_id else None)
        tiers = {tier} if isinstance(tier, int) else (set(tier) if tier is not None else None)
        ex_tiers = ({exclude_tier} if isinstance(exclude_tier, int)
                    else (set(exclude_tier) if exclude_tier is not None else None))
        keep = []
        for i, c in enumerate(self.chunks):
            if docs is not None and c["doc_id"] not in docs:
                continue
            if tiers is not None and c["tier"] not in tiers:
                continue
            if ex_tiers is not None and c["tier"] in ex_tiers:
                continue
            if block_types and not (set(c["block_types"]) & set(block_types)):
                continue
            keep.append(i)
        return np.array(keep, dtype="int64")

    # -- the two retrieval arms -------------------------------------------
    def bm25_search(self, query, k, allowed=None):
        from build_index import tokenize
        scores = self.bm25.get_scores(tokenize(query))
        if allowed is not None:
            if len(allowed) == 0:
                return []
            mask = np.full(len(scores), -np.inf)
            mask[allowed] = scores[allowed]
            scores = mask
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def dense_search(self, query, k, allowed=None):
        model = self._load_embedder()
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = q.astype("float32")
        if allowed is None:
            D, I = self.faiss_index.search(q, k)
        else:
            if len(allowed) == 0:
                return []
            sel = self._faiss.IDSelectorBatch(allowed)
            params = self._faiss.SearchParameters(sel=sel)
            D, I = self.faiss_index.search(q, min(k, len(allowed)), params=params)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    # -- fusion ------------------------------------------------------------
    @staticmethod
    def rrf(rankings, k=RRF_K):
        """Reciprocal rank fusion. Rank-based, so BM25's unbounded scores and
        cosine's [-1,1] never have to be put on a common scale."""
        fused = {}
        for ranked in rankings:
            for rank, (idx, _score) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query, k=8, doc_id=None, tier=None, exclude_tier=None,
               block_types=None, rerank=True, candidates=30):
        allowed = self.allowed_indices(doc_id, tier, exclude_tier, block_types)
        if allowed is not None and len(allowed) == 0:
            return []
        bm = self.bm25_search(query, candidates, allowed)
        dn = self.dense_search(query, candidates, allowed)
        fused = self.rrf([bm, dn])
        if not fused:
            return []
        idxs = [i for i, _ in fused[:candidates]]

        if rerank and len(idxs) > 1:
            ce = self._load_reranker()
            pairs = [(query, self.chunks[i]["text"]) for i in idxs]
            scores = ce.predict(pairs)
            order = np.argsort(scores)[::-1]
            ranked = [(idxs[o], float(scores[o])) for o in order]
            score_name = "rerank_score"
        else:
            ranked = [(i, s) for i, s in fused[:candidates]]
            score_name = "rrf_score"

        bm_rank = {i: r for r, (i, _) in enumerate(bm)}
        dn_rank = {i: r for r, (i, _) in enumerate(dn)}
        out = []
        for i, s in ranked[:k]:
            c = self.chunks[i]
            out.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "tier": c["tier"],
                "section": c["section"], "page_start": c["page_start"],
                "page_end": c["page_end"], "block_types": c["block_types"],
                "source_block_ids": c["source_block_ids"],
                "text": c["text"], score_name: round(s, 4),
                "bm25_rank": bm_rank.get(i), "dense_rank": dn_rank.get(i),
            })
        return out


# --------------------------------------------------------------------------
# Table layer: set queries go here, never through the vector index.
# --------------------------------------------------------------------------

def load_tables(doc_id=None):
    tables = []
    for path in sorted(TABLES_DIR.glob("*.json")):
        if doc_id and path.stem != doc_id:
            continue
        tables.extend(json.loads(path.read_text()))
    return tables


def query_tables(pattern, column=None, doc_id=None, tier=None, normalize=True):
    """Every row whose cell matches -- a complete set, not a top-k.

    `normalize` folds "2 MB"/"2MB" together, which is what makes a size query
    return every system rather than only the ones that happen to be spaced
    the same way as the query."""
    from build_index import normalize_units
    rx = re.compile(pattern, re.I)

    def prep(s):
        return normalize_units(s).replace(" ", "").lower() if normalize else s

    hits = []
    for t in load_tables(doc_id):
        if tier is not None and t["tier"] != tier:
            continue
        header = t.get("header") or []
        cols = range(len(header)) if column is None else [
            i for i, h in enumerate(header) if re.search(column, h, re.I)
        ]
        for row in t.get("rows", []):
            for ci in cols:
                if ci >= len(row):
                    continue
                if rx.search(prep(row[ci])) or rx.search(row[ci]):
                    hits.append({
                        "doc_id": t["doc_id"], "tier": t["tier"], "page": t["page"],
                        "table_id": t["table_id"], "section": t.get("section"),
                        "row_label": row[0] if row else "",
                        "column": header[ci] if ci < len(header) else f"col{ci}",
                        "value": row[ci], "row": row,
                        "quality_flag": t.get("quality_flag"),
                    })
                    break
    return hits


# --------------------------------------------------------------------------

def acceptance_check():
    """Module D: the same query with and without a doc_id filter must
    converge correctly -- every filtered hit inside the filter, and the
    filter must not merely reorder the unfiltered list."""
    r = Retriever()
    query = "page migration latency cost microseconds"
    target = "m5_asplos25"

    unfiltered = r.search(query, k=8)
    filtered = r.search(query, k=8, doc_id=target)

    print("--- Module D acceptance: doc_id filter convergence ---")
    print(f"query: {query!r}\n")
    print(f"unfiltered top-8 docs: {[h['doc_id'] for h in unfiltered]}")
    print(f"filtered  top-8 docs : {[h['doc_id'] for h in filtered]}")

    all_in_scope = all(h["doc_id"] == target for h in filtered)
    got_results = len(filtered) > 0
    print(f"\n  every filtered hit is from {target}: {all_in_scope}")
    print(f"  filtered search returned results   : {got_results} ({len(filtered)})")

    # tier filter: excluding tier 0 must remove every survey chunk
    no_survey = r.search(query, k=8, exclude_tier=0)
    tiers = {h["tier"] for h in no_survey}
    print(f"  exclude_tier=0 leaves tiers        : {sorted(tiers)}")
    tier_ok = 0 not in tiers

    # the table layer must still return the complete 2 MB set
    rows = query_tables(r"2mb", column="granularity")
    systems = sorted({re.sub(r"\s*\[\d+\]\s*", "", h["row_label"]).strip() for h in rows})
    print(f"  table filter '2mb' -> {systems}")
    set_ok = {"MTM", "NOMAD", "NeoMem"}.issubset(set(systems))

    ok = all_in_scope and got_results and tier_ok and set_ok
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--doc", default=None)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--exclude-tier", type=int, default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--tables", action="store_true", help="query the table layer instead")
    ap.add_argument("--column", default=None, help="restrict table query to a column")
    ap.add_argument("--acceptance", action="store_true")
    args = ap.parse_args()

    if args.acceptance:
        sys.exit(0 if acceptance_check() else 1)
    if not args.query:
        ap.error("a query is required (or use --acceptance)")

    if args.tables:
        for h in query_tables(args.query, column=args.column,
                              doc_id=args.doc, tier=args.tier):
            flag = f"  [flagged: {h['quality_flag']}]" if h["quality_flag"] else ""
            print(f"{h['doc_id']} p{h['page']} | {h['row_label']} | "
                  f"{h['column']} = {h['value']!r}{flag}")
        return

    r = Retriever()
    hits = r.search(args.query, k=args.k, doc_id=args.doc, tier=args.tier,
                    exclude_tier=args.exclude_tier, rerank=not args.no_rerank)
    for i, h in enumerate(hits, 1):
        score = h.get("rerank_score", h.get("rrf_score"))
        pages = (f"p{h['page_start']}" if h["page_start"] == h["page_end"]
                 else f"p{h['page_start']}-{h['page_end']}")
        print(f"\n[{i}] {h['doc_id']} (tier {h['tier']}) {pages} "
              f"score={score} bm25#{h['bm25_rank']} dense#{h['dense_rank']}")
        print(f"    section: {h['section']}")
        print(f"    {h['text'][:300]}...")


if __name__ == "__main__":
    main()
