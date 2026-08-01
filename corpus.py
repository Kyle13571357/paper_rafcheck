#!/usr/bin/env python3
"""A corpus: the registry of documents plus every artifact derived from them.

The pipeline is a general citation-audit tool; a particular set of papers is
only its input. So "which papers" is a constructor argument, not a constant --
every module takes a `Corpus` and reads its paths from there, and nothing in
the pipeline names a specific document.

Bootstrapping a new corpus is the part that would otherwise be hand work:

    python3 corpus.py init ~/papers --out ~/review
    python3 corpus.py init ~/papers --bibliography draft.docx

`init` reads every PDF, recovers title / venue / year, proposes a doc_id, and
writes a corpus.yaml for review. Given a draft, it also parses that draft's
bibliography and matches each numbered entry to a file, so the [9] -> doc_id
mapping that check.py depends on is derived rather than typed in by hand.

Layout of a corpus directory:

    corpus.yaml          the registry (reviewed by a human, then authoritative)
    papers/*.pdf         source documents
    blocks.jsonl         parse.py output
    index/               vectors, BM25 tokens, chunk metadata
    tables/*.json        structured table layer
    quality_report.json  rule-gate output
    vision_report.jsonl  page-image re-check output
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import yaml

CORPUS_FILENAME = "corpus.yaml"

VENUE_PATTERNS = [
    (r"\bOSDI\b|USENIX Symposium on Operating Systems", "OSDI"),
    (r"\bSOSP\b|Symposium on Operating Systems Principles", "SOSP"),
    (r"\bASPLOS\b|Architectural Support for Programming Languages", "ASPLOS"),
    (r"\bEuroSys\b|European Conference on Computer Systems", "EuroSys"),
    (r"USENIX Annual Technical Conference|\bATC\b", "USENIX ATC"),
    (r"\bMICRO\b|Symposium on Microarchitecture", "MICRO"),
    (r"\bNSDI\b", "NSDI"),
    (r"\bISCA\b|International Symposium on Computer Architecture", "ISCA"),
    (r"\bHPCA\b", "HPCA"),
    (r"\bFAST\b", "FAST"),
    (r"\bVLDB\b", "VLDB"),
    (r"\bSIGMOD\b", "SIGMOD"),
    (r"\bNeurIPS\b|Neural Information Processing Systems", "NeurIPS"),
    (r"\bICML\b", "ICML"),
    (r"\bACL\b", "ACL"),
]

STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with",
    "via", "using", "towards", "toward", "is", "are", "at", "by", "from",
}


# ---------------------------------------------------------------------------
# The corpus object
# ---------------------------------------------------------------------------

class CorpusError(RuntimeError):
    pass


class Corpus:
    """Registry + artifact locations for one set of documents."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.registry_path = self.root / CORPUS_FILENAME
        if not self.registry_path.exists():
            raise CorpusError(
                f"no {CORPUS_FILENAME} in {self.root}. Create one with:\n"
                f"    python3 corpus.py init <pdf-dir> --out {self.root}")
        data = yaml.safe_load(self.registry_path.read_text()) or []
        if not isinstance(data, list):
            raise CorpusError(f"{self.registry_path} must contain a list of entries")
        self.entries: list[dict[str, Any]] = data

    # -- lookups ----------------------------------------------------------
    @property
    def doc_ids(self) -> list[str]:
        return [e["doc_id"] for e in self.entries]

    @property
    def by_doc_id(self) -> dict[str, dict]:
        return {e["doc_id"]: e for e in self.entries}

    @property
    def by_ref(self) -> dict[int, dict]:
        return {e["ref"]: e for e in self.entries if e.get("ref") is not None}

    def entry(self, doc_id: str) -> dict:
        try:
            return self.by_doc_id[doc_id]
        except KeyError:
            raise CorpusError(f"unknown doc_id {doc_id!r}") from None

    def path_for(self, doc_id: str) -> Path:
        return self.root / self.entry(doc_id)["file"]

    def tiers(self) -> set[int]:
        return {e.get("tier", 1) for e in self.entries}

    def primary_tier0(self) -> dict | None:
        """The document under audit, if the registry declares one."""
        zeros = [e for e in self.entries if e.get("tier") == 0]
        return zeros[0] if len(zeros) == 1 else (zeros[0] if zeros else None)

    def resolve_short(self, fragment: str) -> str | None:
        """'m5' -> 'm5_asplos25'. Typing full ids while writing is friction."""
        frag = fragment.strip().lower()
        for e in self.entries:
            if e["doc_id"].lower() == frag:
                return e["doc_id"]
        hits = [e["doc_id"] for e in self.entries
                if frag in e["doc_id"].lower()
                or frag in str(e.get("short", "")).lower()]
        return hits[0] if len(hits) == 1 else None

    # -- artifact locations ------------------------------------------------
    @property
    def blocks_path(self) -> Path: return self.root / "blocks.jsonl"
    @property
    def index_dir(self) -> Path: return self.root / "index"
    @property
    def tables_dir(self) -> Path: return self.root / "tables"
    @property
    def quality_report(self) -> Path: return self.root / "quality_report.json"
    @property
    def vision_report(self) -> Path: return self.root / "vision_report.jsonl"
    @property
    def chunks_path(self) -> Path: return self.index_dir / "chunks.jsonl"
    @property
    def faiss_path(self) -> Path: return self.index_dir / "prose.faiss"
    @property
    def embeddings_path(self) -> Path: return self.index_dir / "embeddings.npy"
    @property
    def manifest_path(self) -> Path: return self.index_dir / "manifest.json"
    @property
    def formulas_path(self) -> Path: return self.index_dir / "formulas.json"

    # -- corpus-declared expectations --------------------------------------
    # What a correct answer looks like is a property of a particular corpus;
    # the guarantee being tested (a set query returns every member) is a
    # property of the index. So the expectations live in the registry, and
    # the modules only know how to run them.
    @property
    def checks(self) -> dict[str, Any]:
        path = self.root / "checks.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text()) or {}

    def set_query_checks(self) -> list[dict]:
        """[{name, pattern, column, expect:[row labels]}] -- completeness
        expectations for the structured table layer."""
        return self.checks.get("set_queries") or []

    def retrieval_checks(self) -> dict[str, Any]:
        """{query, scoped_doc} -- used to confirm a doc_id filter actually
        confines results rather than merely reordering them."""
        return self.checks.get("retrieval") or {}

    # -- integrity ---------------------------------------------------------
    def validate(self) -> list[str]:
        problems, seen_ids, seen_refs = [], set(), {}
        for e in self.entries:
            for field in ("doc_id", "file", "tier"):
                if field not in e:
                    problems.append(f"entry {e!r} is missing '{field}'")
            doc_id = e.get("doc_id")
            if doc_id in seen_ids:
                problems.append(f"duplicate doc_id {doc_id!r}")
            seen_ids.add(doc_id)
            ref = e.get("ref")
            if ref is not None:
                if ref in seen_refs:
                    problems.append(
                        f"ref {ref} used by both {seen_refs[ref]!r} and {doc_id!r}")
                seen_refs[ref] = doc_id
            if "file" in e and not (self.root / e["file"]).exists():
                problems.append(f"{doc_id}: file not found: {e['file']}")
        if 0 not in self.tiers():
            problems.append(
                "no tier-0 document registered. check.py needs one to know which "
                "document is under audit and which sources are being cited.")
        return problems

    @classmethod
    def discover(cls, start: Path | str | None = None) -> "Corpus":
        """Nearest corpus.yaml at or above `start`, so commands work from any
        subdirectory the way git does."""
        here = Path(start or Path.cwd()).expanduser().resolve()
        for candidate in [here, *here.parents]:
            if (candidate / CORPUS_FILENAME).exists():
                return cls(candidate)
        raise CorpusError(
            f"no {CORPUS_FILENAME} found in {here} or any parent directory")

    def __repr__(self) -> str:
        return f"<Corpus {self.root} ({len(self.entries)} documents)>"


def load_corpus(path: Path | str | None = None) -> Corpus:
    """Entry point every module uses: an explicit --corpus, else discovery."""
    return Corpus(path) if path else Corpus.discover()


# ---------------------------------------------------------------------------
# Bootstrapping a registry from a folder of PDFs
# ---------------------------------------------------------------------------

def _slug(text: str, maxlen: int = 28) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:maxlen].strip("_")


def _title_words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
            if w not in STOPWORDS and len(w) > 2}


def extract_metadata(pdf_path: Path) -> dict[str, Any]:
    """Recover title / venue / year from a PDF.

    Title comes from the largest type on the opening pages rather than the
    PDF's /Title metadata, which is routinely empty or says "Microsoft Word -
    final_v3.docx"."""
    import fitz

    doc = fitz.open(pdf_path)
    head_text = "\n".join(doc[i].get_text() for i in range(min(3, len(doc))))

    # title: biggest font on the first two pages, excluding boilerplate banners
    best: tuple[float, str] | None = None
    for pno in range(min(2, len(doc))):
        for block in doc[pno].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                text = " ".join(s["text"] for s in line["spans"]).strip()
                if len(text) < 8 or len(text) > 200:
                    continue
                # banners, stamps and legal boilerplate are often set larger
                # than the title itself
                if re.search(r"proceedings|is sponsored by|open access|"
                             r"isbn|https?://|copyright|permission to make|"
                             r"^arxiv:|\[cs\.[A-Z]{2}\]|this paper is included",
                             text, re.I):
                    continue
                size = max(s["size"] for s in line["spans"])
                if best is None or size > best[0] + 0.4:
                    best = (size, text)
        if best:
            break

    title = best[1] if best else pdf_path.stem
    # a wrapped title continues on the next line at the same size
    if best:
        for pno in range(min(2, len(doc))):
            lines = []
            for block in doc[pno].get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    t = " ".join(s["text"] for s in line["spans"]).strip()
                    if t:
                        lines.append((max(s["size"] for s in line["spans"]), t))
            for i, (size, t) in enumerate(lines):
                if t == title and i + 1 < len(lines):
                    nxt_size, nxt = lines[i + 1]
                    if abs(nxt_size - size) < 0.3 and len(nxt) < 120 \
                            and not nxt.endswith("."):
                        title = f"{title} {nxt}"
                    break
            break

    venue = next((label for pat, label in VENUE_PATTERNS
                  if re.search(pat, head_text)), None)
    if venue is None and re.search(r"arXiv:\s*\d{4}\.\d{4,5}", head_text):
        m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", head_text)
        venue = f"arXiv:{m.group(1)}"

    # Publication year from page 1 only. Scanning further reaches the
    # reference list, where the years belong to other people's papers; taking
    # the max there produced a 2017 date for a 2024 paper.
    front = doc[0].get_text() if len(doc) else ""
    years = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-4]\d)\b", front)]
    if venue and venue.startswith("arXiv"):
        m = re.search(r"arXiv:\s*(\d{2})(\d{2})\.", head_text)
        if m:                      # arXiv ids encode YYMM
            years.append(2000 + int(m.group(1)))
    year = max(years, default=None)

    doc.close()
    title = re.sub(r"\s+", " ", title).strip()
    return {"title": title, "venue": venue, "year": year,
            "pages": None, "path": pdf_path}


def propose_doc_id(meta: dict, taken: Iterable[str]) -> str:
    """A stable, readable id: leading title words plus venue and year."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", meta["title"])
             if w.lower() not in STOPWORDS]
    stem = _slug("_".join(words[:3]) or meta["path"].stem, 22)
    venue = _slug(meta.get("venue") or "", 10)
    year = str(meta["year"])[-2:] if meta.get("year") else ""
    base = "_".join(p for p in (stem, venue + year) if p) or _slug(meta["path"].stem)
    candidate, n = base, 2
    taken = set(taken)
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def parse_bibliography(text: str) -> dict[int, str]:
    """Numbered bibliography entries from a draft: {9: 'M5: Mastering ...'}.

    Handles the common shapes: "9. [M5] Author, "Title," in Proc...",
    "[9] Author, Title", "9) Author..."."""
    tail = text
    for marker in ("References", "REFERENCES", "Bibliography", "參考文獻"):
        idx = text.rfind(marker)
        if idx != -1:
            tail = text[idx:]
            break

    entries: dict[int, str] = {}
    pattern = re.compile(r"(?:^|\n)\s*(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+")
    matches = list(pattern.finditer(tail))
    for i, m in enumerate(matches):
        num = int(m.group(1) or m.group(2))
        body = tail[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(tail)]
        entries[num] = re.sub(r"\s+", " ", body).strip()[:600]
    if entries:
        return entries

    # Word's automatic list numbering lives in the paragraph's formatting, not
    # its text, so a .docx bibliography extracts with the numbers missing
    # entirely: "References / [MTM] J. Ren, ... / [NOMAD] L. Xiang, ...".
    # Numbering is positional in that case, so recover it from order.
    def looks_like_citation(line: str) -> bool:
        return (len(line) > 40
                and (re.search(r"\b(19|20)\d{2}\b", line)
                     or re.search(r"in Proceedings|Proc\.|arXiv|Conference|"
                                  r"Symposium|Journal", line, re.I)))

    candidates = [re.sub(r"\s+", " ", ln).strip()
                  for ln in tail.split("\n")[1:] if looks_like_citation(ln)]
    return {i: c[:600] for i, c in enumerate(candidates, start=1)}


def match_bibliography(bib: dict[int, str], metas: list[dict]) -> dict[int, str]:
    """Assign ref numbers to files by comparing titles.

    Both signals are needed: token overlap survives reordering and subtitle
    truncation, sequence ratio catches near-identical strings. A weak best
    match is left unassigned rather than guessed -- a wrong ref number sends
    every later verification to the wrong paper."""
    assigned: dict[int, str] = {}
    used: set[str] = set()
    for ref, citation in sorted(bib.items()):
        cite_words = _title_words(citation)
        if not cite_words:
            continue
        best, best_score = None, 0.0
        for meta in metas:
            if meta["_doc_id"] in used:
                continue
            title_words = _title_words(meta["title"])
            if not title_words:
                continue
            overlap = len(cite_words & title_words) / len(title_words)
            ratio = SequenceMatcher(
                None, meta["title"].lower()[:90],
                citation.lower()[:200]).ratio()
            score = 0.75 * overlap + 0.25 * ratio
            if score > best_score:
                best, best_score = meta, score
        if best is not None and best_score >= 0.45:
            assigned[ref] = best["_doc_id"]
            used.add(best["_doc_id"])
    return assigned


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix == ".pdf":
        import fitz
        d = fitz.open(str(path))
        text = "\n".join(p.get_text() for p in d)
        d.close()
        return text
    return path.read_text()


def init_corpus(pdf_dir: Path, out_dir: Path, bibliography: Path | None = None,
                tier0_pdf: Path | None = None,
                copy_files: bool = True, overwrite: bool = False) -> Path:
    """Write a reviewable corpus.yaml for every PDF in `pdf_dir`.

    `bibliography` and `tier0_pdf` are separate on purpose. The bibliography
    is read for its reference *numbers* -- any text format works, since
    parse_bibliography only needs the citation list, not a coordinate-parsable
    layout. The tier-0 document is what parse.py will actually extract from
    later, and that pipeline is PDF-only. The common real case is a draft
    still in .docx with a separate PDF export sitting next to it (true for
    this project's own survey): passing the .docx as --bibliography and the
    .pdf as --tier0-pdf uses each for what it's good for. Passing only a PDF
    bibliography still works as before, using it for both."""
    import shutil

    pdf_dir = Path(pdf_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise CorpusError(f"no PDFs in {pdf_dir}")

    registry_path = out_dir / CORPUS_FILENAME
    if registry_path.exists() and not overwrite:
        raise CorpusError(f"{registry_path} already exists (use --overwrite)")

    papers_dir = out_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    for pdf in pdfs:
        print(f"  reading {pdf.name} ...", file=sys.stderr)
        meta = extract_metadata(pdf)
        meta["_doc_id"] = propose_doc_id(meta, [m["_doc_id"] for m in metas])
        metas.append(meta)

    ref_map: dict[int, str] = {}
    bib_path = Path(bibliography).expanduser().resolve() if bibliography else None
    if bib_path:
        print(f"  parsing bibliography from {bib_path.name} ...", file=sys.stderr)
        bib = parse_bibliography(read_document_text(bib_path))
        ref_map = match_bibliography(bib, metas)
        print(f"  matched {len(ref_map)}/{len(bib)} bibliography entries",
              file=sys.stderr)

    # Resolve which PDF (if any) becomes the tier-0 entry: an explicit
    # --tier0-pdf wins; otherwise a PDF bibliography can serve double duty;
    # otherwise there isn't one, and that is reported plainly rather than
    # silently producing a registry with no document under audit.
    tier0_path = Path(tier0_pdf).expanduser().resolve() if tier0_pdf else None
    if tier0_path is None and bib_path and bib_path.suffix.lower() == ".pdf":
        tier0_path = bib_path
    tier0_title = Path(bibliography).stem if bibliography else None

    bib_doc_id: str | None = None
    if tier0_path:
        if not tier0_path.exists():
            raise CorpusError(f"tier-0 PDF not found: {tier0_path}")
        bib_doc_id = _slug(tier0_path.stem)
        target = papers_dir / f"{bib_doc_id}.pdf"
        if copy_files and tier0_path != target:
            shutil.copy2(tier0_path, target)

    doc_to_ref = {v: k for k, v in ref_map.items()}
    entries = []
    if bib_doc_id:
        entries.append({
            "ref": 0, "doc_id": bib_doc_id, "tier": 0,
            "file": f"papers/{bib_doc_id}.pdf",
            "title": tier0_title, "short": "document under audit",
            "venue": None, "year": None,
        })
    elif bib_path:
        print(f"  NOTE: {bib_path.name} is not a PDF and no --tier0-pdf was "
              f"given, so no tier-0 document was registered. Its reference "
              f"numbers were still used to assign refs below. Pass "
              f"--tier0-pdf <path> if a PDF export exists.", file=sys.stderr)
    for meta in metas:
        doc_id = meta["_doc_id"]
        target = papers_dir / f"{doc_id}.pdf"
        if copy_files and not target.exists():
            shutil.copy2(meta["path"], target)
        entries.append({
            "ref": doc_to_ref.get(doc_id),
            "doc_id": doc_id,
            "tier": 1,
            "file": f"papers/{doc_id}.pdf",
            "title": meta["title"],
            "short": meta["title"].split(":")[0][:28],
            "venue": meta["venue"],
            "year": meta["year"],
        })

    unmatched = [e["doc_id"] for e in entries if e["ref"] is None]
    header = [
        "# Document registry. REVIEW THIS FILE before running the pipeline.",
        "#",
        "# ref    number used to cite this work in the document under audit.",
        "#        check.py resolves '[9]' through this field, so a wrong number",
        "#        sends every verification of that claim to the wrong paper.",
        "# tier   0 = the document being audited (its own paraphrase of others'",
        "#        work), 1 = an original source. Claims without a citation are",
        "#        searched with tier 0 excluded, so the document cannot",
        "#        corroborate itself.",
        "#",
        f"# Generated from {pdf_dir}",
    ]
    if bibliography:
        header.append(f"# Bibliography parsed from {Path(bibliography).name}")
    if unmatched:
        header.append("#")
        header.append("# NEEDS ATTENTION -- no ref number was matched for:")
        for d in unmatched:
            header.append(f"#     {d}")
        header.append("# Assign ref: by hand, or leave null if the document "
                      "never cites it.")

    out_dir.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, allow_unicode=True, sort_keys=False,
                          default_flow_style=False, width=100)
    registry_path.write_text("\n".join(header) + "\n\n" + body)
    return registry_path


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="build a corpus.yaml from a folder of PDFs")
    p_init.add_argument("pdf_dir")
    p_init.add_argument("--out", default=".", help="corpus directory (default: cwd)")
    p_init.add_argument("--bibliography",
                        help="draft whose reference list assigns ref numbers "
                             "(.docx/.pdf/.txt)")
    p_init.add_argument("--tier0-pdf",
                        help="PDF of the document under audit (parse.py needs a "
                             "PDF; only required when --bibliography isn't "
                             "itself one, e.g. a .docx draft with a separate "
                             "PDF export)")
    p_init.add_argument("--no-copy", action="store_true",
                        help="reference the PDFs in place instead of copying")
    p_init.add_argument("--overwrite", action="store_true")

    p_val = sub.add_parser("validate", help="check a corpus.yaml for problems")
    p_val.add_argument("--corpus", default=None)

    p_show = sub.add_parser("show", help="list registered documents")
    p_show.add_argument("--corpus", default=None)

    args = ap.parse_args()

    if args.cmd == "init":
        path = init_corpus(Path(args.pdf_dir), Path(args.out),
                           bibliography=args.bibliography,
                           tier0_pdf=args.tier0_pdf,
                           copy_files=not args.no_copy,
                           overwrite=args.overwrite)
        print(f"\nwrote {path}")
        print("Review it -- especially the ref numbers and the tier-0 entry -- "
              "then run parse.py.")
        return 0

    corpus = load_corpus(args.corpus)
    if args.cmd == "validate":
        problems = corpus.validate()
        print(f"{corpus.root}: {len(corpus.entries)} documents")
        for p in problems:
            print(f"  PROBLEM: {p}")
        print("OK" if not problems else f"\n{len(problems)} problem(s)")
        return 1 if problems else 0

    print(f"{corpus.root}  ({len(corpus.entries)} documents)")
    for e in sorted(corpus.entries, key=lambda x: (x.get("ref") is None,
                                                   x.get("ref") or 0)):
        ref = "  -" if e.get("ref") is None else f"{e['ref']:>3}"
        print(f"  [{ref}] tier {e.get('tier')}  {e['doc_id']:<30} "
              f"{(e.get('venue') or ''):<12} {e.get('year') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
