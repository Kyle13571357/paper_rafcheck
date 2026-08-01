# paper-refcheck

A retrieval and citation-audit pipeline. Feed it a written passage and it checks each factual claim against the source it cites, surfacing every discrepancy alongside the original text.

The system does not adjudicate right or wrong -- it places the suspect claim and its evidence side by side and lets a human decide. That design choice runs through every layer below.

**A general tool, not a script tied to one paper set.** The corpus is a `Corpus` object (`corpus.py`); every module reads from it instead of a hard-coded path. Which documents are registered, which one is under audit, and what a correct answer looks like for a set query are all declared in `corpus.yaml` / `checks.yaml`, not hard-coded in the pipeline. The 12 memory-tiering systems papers and the survey auditing them, used throughout this repo, are the test corpus this was validated against -- not the design target.

```bash
python3 corpus.py init ~/papers --out ~/my-review \
  --bibliography draft.docx --tier0-pdf draft.pdf
```

Reads every PDF's title, venue, and year off its largest first-page type; parses a draft's reference list and matches each numbered citation back to its file by title overlap; writes a reviewable `corpus.yaml` -- work that otherwise takes an afternoon by hand. Run against this project's own real draft (12 source papers, a `.docx` bibliography): 12/12 citations matched correctly, and the tool correctly declined to force a match for a topically unrelated PDF that happened to sit in the same folder (a cache-partitioning survey, never actually cited), flagging it as unmatched instead of guessing.

Point it at a different paper set and a different draft and the same pipeline runs; `checks.yaml`'s acceptance expectations are declarative, not code.

---

## Motivation: a finding that predates the codebase

The survey this project audits defines `AOL` twice:

- Section 2.2 -- `AOL = Latency / MLP`, attributed to Soar/Alto [4]
- Section 3.1 -- `AOL = L_loaded / (1 + alpha*(MLP-1))`, TierLab's (the author's own simulator) implementation

Two LLM-drafted planning documents both described the second form as "the original paper's formula." The actual source, Soar/Alto (OSDI'25), p4:

> we define AOL = Latency / MLP

The first form is the one the source actually defines. And at alpha=1, `1+1*(MLP-1) = MLP` -- the second form reduces exactly to the first. So the two aren't contradictory; the second is an unstated generalization of the first.

The correct output isn't "A is right, B is wrong." It's: the same symbol is defined twice, the relationship between the two definitions is never stated, and one of them is the author's own implementation rather than the cited paper's definition. That needs a fifth verdict class, `underspecified`, alongside supported / contradicted / not_found / condition_mismatch.

---

## Four things this system cannot get wrong

1. **One model, one index.** `ask` and `check` share the same model and the same index; the only difference is how the query is built and how the search is scoped. No fine-tuning anywhere.
2. **`check`'s retrieval must be scoped.** An uncited claim searched globally is searched with tier-0 explicitly excluded -- otherwise the tool can retrieve the survey's own restatement of a claim and use it to "confirm" that same claim, which looks like it's working while proving nothing.
3. **Tables never enter the vector index.** A query like "which systems support 2 MB pages" needs the complete set; semantic similarity returns the closest few matches and silently drops the rest. Set-shaped queries go through a filter over `tables/*.json` instead.
4. **Eval ground truth is never model-generated.** Every item in `eval/eval_set.jsonl` was checked by hand against the source PDFs; `eval/validate_eval_set.py` re-verifies it independently, in code.

---

## Pipeline

```
papers/*.pdf
   |
   |- parse.py           coordinate-based extraction -> blocks.jsonl
   |                     x splits columns, y splits rows; tables use the
   |                     first column as a row anchor
   |- quality_check.py   rule-based quality gate -> quality_report.json
   |- vision_verify.py   page-image re-check with a channel health check
   |                     -> vision_report.jsonl
   |- build_index.py     three paths: vectors+BM25 / table JSON / formulas
   |- retrieve.py        hybrid search + reranker + doc_id/tier filters
   |
   |- ask.py             question -> cited answer
   `- check.py           passage -> per-claim verdict + source span
```

| File | Role |
|---|---|
| `corpus.py` | The `Corpus` object plus the `init` bootstrapper. Every module's corpus path, doc_id resolution, and tier lookup goes through this; no paper is named in pipeline code. |
| `corpus.yaml` | `ref -> doc_id -> tier -> file`. `check.py`'s reference resolution depends on this table entirely. |
| `checks.yaml` | This corpus's own acceptance expectations (what a set query should return, what a doc_id filter should converge to). Swapping corpora means editing this file, not `build_index.py` / `retrieve.py`. |
| `llm.py` | The single entry point for every model call. An abstract `LLMProvider` plus one subclass per backend; the backend is swappable. |
| `units.py` | Deterministic numeric normalization (`54 us` == `0.054 ms`) -- plain code, never routed through a model. |
| `eval/eval_set.jsonl` | Hand-annotated evaluation set spanning answerable, refusal, and misattribution-trap questions. |
| `eval/run_baseline.py` | Three-arm comparison harness (no retrieval / unfiltered retrieval / this system). |

`tier` 0 = the survey itself (secondhand paraphrase), 1 = an original source.

---

## Two real workflows

### 1. Fast lookup while writing

```bash
python3 refcheck.py
```

An interactive session: the retrieval models load once (~16s), then every query after that is sub-second (~0.5s). A one-shot CLI call would pay that ~16s model-load cost on every single question, which is why lookup always goes through this instead.

```
> how much does a page migration cost        direct search, no API key needed
> /doc m5                                    restrict to one document
> sparse page word level tracking
> /tier 1                                    originals only, survey excluded
> /tables 2mb                                set query over the table layer
> /ask what penalty does TierLab use         generated answer with citations (needs a key)
```

**Search works without an API key.** Most of the time, seeing the passage itself with its page and section is the whole answer.

### 2. Proofreading a finished draft

```bash
python3 check.py --file draft.docx
```

Takes `.docx` / `.pdf` / `.txt` directly, no manual export step. Pipeline: split into paragraphs -> extract claims -> resolve citations -> retrieve within scope -> adjudicate -> attach the original span.

The report shows only what needs attention by default, ranked by severity:

```
contradicted -> not_found -> condition_mismatch -> underspecified
-> possibly_missing_citation -> unresolvable_reference
```

`--all` also lists claims that checked out; `--limit N` restricts to the first N paragraphs.

Single-passage check:

```bash
python3 check.py --text "Colloid achieves 1.01-1.76x speedup [3]."
```

---

## Running it

```bash
python3 parse.py                      # PDFs -> blocks.jsonl
```
```bash
python3 quality_check.py              # rule-based gate; lists pages needing review
```
```bash
python3 build_index.py --selftest     # build the index and run its acceptance check
```
```bash
python3 retrieve.py --acceptance      # retrieval-layer acceptance check
```
```bash
python3 eval/validate_eval_set.py     # verify the eval set itself
```

Everything above runs fully offline, no API key required.

The generation layer needs a provider configured (default: DeepSeek):

```bash
export DEEPSEEK_API_KEY=...
```
```bash
python3 ask.py "What migration penalty does TierLab charge?"
```
```bash
python3 check.py --self-audit         # run the system against its own tier-0 document
```
```bash
python3 eval/run_baseline.py --arms none raw system --repeats 3
```

The retrieval layer (embedding model, BM25, reranker) runs entirely local and offline regardless of provider.

### Visual re-check

`vision_verify.py` needs a vision-capable provider. DeepSeek has no vision model, and `llm.py` raises an explicit error rather than silently dropping the image:

```bash
REFCHECK_PROVIDER=openai python3 vision_verify.py
```

---

## Channel health check, and why it exists

A vision call can succeed at the transport level while the image never actually arrives, and the model will still produce a fluent, confident-looking comparison report from the text alone.

A controlled test (same prompt, `images` field removed) demonstrated exactly that:

```json
{
  "image_token": "NONE",
  "page_summary": "A table with three columns: System, Profiling, and Objective.",
  "discrepancies": [{"location": "Table 2, row MTM [1], column Profiling",
                     "image_shows": "HW sampling (PEBS)", ...}]
}
```

The model asserted what an image "showed" -- an image it never received.

The fix: every call overlays a random token on the rendered page and the model must report it back before anything else. Failing to reproduce the token means the image never arrived; the page is marked `channel_failed_needs_human` and every finding from that call is discarded rather than trusted.

---

## Current state

The extraction, quality-gate, indexing, retrieval, ask, and check layers are implemented and pass their respective acceptance checks locally, without any API key. The three-arm comparison harness (`eval/run_baseline.py`) is implemented and ready to run against a configured provider; see [results.md](results.md) for what's been validated so far and what's still open.
