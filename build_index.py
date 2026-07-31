#!/usr/bin/env python3
"""blocks.jsonl -> three separate retrieval paths.

  prose/heading/caption -> chunks -> dense vectors + BM25
  tables                -> tables/<doc_id>.json, a structured layer
  formulas              -> index/formulas.json

Tables deliberately do NOT enter the vector store. A question like "which
systems support 2 MB pages" needs the *complete set*; nearest-neighbour
similarity returns whichever rows happen to be closest and silently omits
the rest, which is the worst possible failure for an audit tool. Set queries
go through a filter over the structured layer instead.

Usage:
  python3 build_index.py            # build everything
  python3 build_index.py --selftest # build, then run the Module C check
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
BLOCKS_PATH = ROOT / "blocks.jsonl"
CORPUS_YAML = ROOT / "corpus.yaml"
INDEX_DIR = ROOT / "index"
TABLES_DIR = ROOT / "tables"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_BLOCK_TYPES = ("prose", "heading", "caption")
TARGET_CHARS = 900          # aim for chunks around this size
MAX_CHARS = 1400            # hard ceiling before a block is split
MIN_MERGE_CHARS = 400       # below this, keep pulling in the next block

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# "2 MB" / "2MB" / "2 mb" all need to match one query token, and BM25 must be
# able to see the acronyms the survey leans on (PEBS, DCSC, FMAR, CIT, THP)
SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB|KiB|MiB|GiB|TiB|ns|µs|us|ms)\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")


def normalize_units(text):
    """Collapse "2 MB" -> "2mb" so a spaced and unspaced form share a token."""
    return SIZE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}".lower(), text)


def tokenize(text):
    """Tokens for BM25. Keeps the unit-collapsed form alongside the raw one so
    both "2 MB" and "2MB" retrieve the same chunks."""
    base = text.lower()
    toks = TOKEN_RE.findall(base)
    normed = TOKEN_RE.findall(normalize_units(base))
    seen, out = set(), []
    for t in toks + normed:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_blocks():
    blocks = []
    with open(BLOCKS_PATH) as f:
        for i, line in enumerate(f):
            b = json.loads(line)
            b["block_id"] = f"{b['doc_id']}:p{b['page']}:{i}"
            blocks.append(b)
    return blocks


def split_long(text, max_chars=MAX_CHARS):
    """Split an oversized block on sentence boundaries, carrying one sentence
    of overlap so a claim that straddles the cut is still wholly present in
    one of the pieces."""
    sents = SENT_SPLIT_RE.split(text)
    if len(sents) == 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    pieces, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x) + 1 for x in cur) >= max_chars:
            pieces.append(" ".join(cur))
            cur = [cur[-1]] if len(cur) > 1 else []
    if cur:
        tail = " ".join(cur)
        if pieces and len(tail) < MIN_MERGE_CHARS:
            pieces[-1] = pieces[-1] + " " + tail
        else:
            pieces.append(tail)
    return pieces


def build_chunks(blocks):
    """Merge small adjacent text blocks, split oversized ones. Chunks never
    span a section boundary or a document boundary."""
    text_blocks = [b for b in blocks if b["block_type"] in TEXT_BLOCK_TYPES]
    grouped = defaultdict(list)
    for b in text_blocks:
        grouped[b["doc_id"]].append(b)

    chunks = []
    for doc_id, docblocks in grouped.items():
        docblocks.sort(key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))
        buf = []

        def flush():
            nonlocal buf
            if not buf:
                return
            text = " ".join(b["text"] for b in buf).strip()
            if not text:
                buf = []
                return
            pieces = split_long(text) if len(text) > MAX_CHARS else [text]
            for pi, piece in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{doc_id}:c{len(chunks):05d}",
                    "doc_id": doc_id,
                    "tier": buf[0]["tier"],
                    "section": buf[0].get("section"),
                    "page_start": min(b["page"] for b in buf),
                    "page_end": max(b["page"] for b in buf),
                    "block_types": sorted({b["block_type"] for b in buf}),
                    "source_block_ids": [b["block_id"] for b in buf],
                    "part": pi if len(pieces) > 1 else None,
                    "text": piece,
                })
            buf = []

        for b in docblocks:
            if buf:
                same_section = buf[-1].get("section") == b.get("section")
                cur_len = sum(len(x["text"]) for x in buf)
                if not same_section or cur_len >= TARGET_CHARS:
                    flush()
            buf.append(b)
            if sum(len(x["text"]) for x in buf) >= TARGET_CHARS:
                flush()
        flush()
    return chunks


def build_tables(blocks, corpus_by_id):
    """Structured layer. Rows stay as rows; nothing here is embedded."""
    by_doc = defaultdict(list)
    for b in blocks:
        if b["block_type"] != "table":
            continue
        rows = b.get("table_rows") or []
        by_doc[b["doc_id"]].append({
            "table_id": f"{b['doc_id']}:t{b['page']}:{len(by_doc[b['doc_id']])}",
            "doc_id": b["doc_id"],
            "tier": b["tier"],
            "page": b["page"],
            "section": b.get("section"),
            "bbox": b["bbox"],
            "header": rows[0] if rows else [],
            "rows": rows[1:] if len(rows) > 1 else [],
            "all_rows": rows,
            "quality_flag": b.get("quality_flag"),
        })
    TABLES_DIR.mkdir(exist_ok=True)
    for doc_id, tables in by_doc.items():
        (TABLES_DIR / f"{doc_id}.json").write_text(
            json.dumps(tables, indent=2, ensure_ascii=False))
    return by_doc


def build_formulas(blocks):
    formulas = [{
        "formula_id": f"{b['doc_id']}:f{b['page']}",
        "doc_id": b["doc_id"], "tier": b["tier"], "page": b["page"],
        "section": b.get("section"), "text": b["text"], "bbox": b["bbox"],
    } for b in blocks if b["block_type"] == "formula"]
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "formulas.json").write_text(
        json.dumps(formulas, indent=2, ensure_ascii=False))
    return formulas


def embed_chunks(chunks, model_name=EMBED_MODEL, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    return vecs.astype("float32")


def selftest_2mb(tables_by_doc):
    """Module C acceptance: the set query must return all three systems.

    MTM lists "2 MB THP"; NOMAD and NeoMem list "4 KB / 2 MB". A literal
    "2 MB THP" match finds only MTM, which is exactly the silent-omission
    failure this layer exists to prevent -- so match on the normalized size
    token instead."""
    expected = {"MTM", "NOMAD", "NeoMem"}
    found = {}
    for doc_id, tables in tables_by_doc.items():
        for t in tables:
            header = [h.lower() for h in t["header"]]
            if not any("granularity" in h for h in header):
                continue
            gcol = next(i for i, h in enumerate(header) if "granularity" in h)
            for row in t["rows"]:
                if gcol >= len(row):
                    continue
                if "2mb" in normalize_units(row[gcol]).lower().replace(" ", ""):
                    system = re.sub(r"\s*\[\d+\]\s*", "", row[0]).strip()
                    found[system] = {
                        "granularity": row[gcol], "doc_id": doc_id,
                        "page": t["page"], "table_id": t["table_id"],
                    }
    print("\n--- Module C acceptance: which systems support 2 MB pages ---")
    for system, info in sorted(found.items()):
        print(f"  {system:12s} granularity={info['granularity']!r} "
              f"({info['doc_id']} p{info['page']})")
    missing = expected - set(found)
    extra = set(found) - expected
    ok = not missing
    print(f"  expected {sorted(expected)}")
    print(f"  MISSING  {sorted(missing)}" if missing else "  nothing missing")
    if extra:
        print(f"  also matched (not an error, just noted): {sorted(extra)}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-embed", action="store_true",
                    help="rebuild chunks/tables only (fast, no model load)")
    args = ap.parse_args()

    corpus_by_id = {e["doc_id"]: e for e in yaml.safe_load(CORPUS_YAML.read_text())}
    blocks = load_blocks()
    INDEX_DIR.mkdir(exist_ok=True)

    chunks = build_chunks(blocks)
    for c in chunks:
        c["tokens"] = tokenize(c["text"])
    tables_by_doc = build_tables(blocks, corpus_by_id)
    formulas = build_formulas(blocks)

    with open(INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_tables = sum(len(v) for v in tables_by_doc.values())
    print(f"chunks   : {len(chunks)}")
    print(f"tables   : {n_tables} (structured layer, NOT embedded)")
    print(f"formulas : {len(formulas)}")

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embed_model": None if args.skip_embed else EMBED_MODEL,
        "n_chunks": len(chunks), "n_tables": n_tables, "n_formulas": len(formulas),
        "target_chars": TARGET_CHARS, "max_chars": MAX_CHARS,
    }

    if not args.skip_embed:
        import faiss
        t0 = time.time()
        vecs = embed_chunks(chunks)
        index = faiss.IndexFlatIP(vecs.shape[1])   # vectors are normalized -> cosine
        index.add(vecs)
        faiss.write_index(index, str(INDEX_DIR / "prose.faiss"))
        np.save(INDEX_DIR / "embeddings.npy", vecs)
        manifest["embed_dim"] = int(vecs.shape[1])
        manifest["embed_seconds"] = round(time.time() - t0, 1)
        print(f"embedded : {vecs.shape[0]} x {vecs.shape[1]} "
              f"in {manifest['embed_seconds']}s")

    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {INDEX_DIR}/")

    if args.selftest:
        ok = selftest_2mb(tables_by_doc)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
