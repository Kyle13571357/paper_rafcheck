# Proposal: Paper Retrieval and Citation-Audit System

Working name `paper-refcheck`

---

## 1. Reason

### The problem
The 12 papers covered by the CS550 survey are 2024-25 OSDI / SOSP / ASPLOS / EuroSys results, outside any LLM's training data. Asking a model directly about their experimental numbers produces plausible-looking fabricated figures with no traceable source, and no reliable secondhand source exists externally either.

### Why RAG, not fine-tuning
| Requirement | Why fine-tuning doesn't fit |
|---|---|
| Papers need to keep being added | every addition means retraining |
| Answers must carry a page citation | knowledge baked into weights can't point to its source |
| Must refuse when data is insufficient | fine-tuning makes a model bolder, not more honest |

Fine-tuning teaches behavior and format, not facts. This project needs traceable facts, so it's RAG.

### Why not just stuff all 13 papers into context
This question is inevitable, and at n=13 the skepticism is reasonable.

1. **Scoped retrieval cannot be substituted with context stuffing.** Verifying a claim tagged `[9]` requires forcing the model to look only at M5. With all 13 papers in context simultaneously, there's no mechanism to constrain what the model draws on -- it will pull from other papers, including the survey's own paraphrase of M5. This isn't a matter of being slower or more expensive; it's simply not achievable.
2. **Span-level traceability.** Reading the PDF directly and answering gives no mechanical way to verify which line a sentence came from. That's the precondition for an audit tool to exist at all.
3. **Scale.** n=13 fits in context; n=100 doesn't. The design needs to scale up.

Cost and attention dilution are side effects, not the main argument.

### Two use cases
**Same model, same index** -- the only difference is how the query is constructed and how retrieval scope is restricted.

- **`ask`** -- takes a question, returns a cited answer
- **`check`** -- takes an already-written passage, checks each claim against the source, flags numeric errors and dropped conditions

`check` is the project's core. `ask` comes almost free once the retrieval layer exists.

### Why `check` is the core
The most common distortion when writing a survey isn't miscopying a number -- it's compressing "achieves X% under a specific configuration" into "achieves X%." Once the condition drops, the conclusion changes in kind. Checking this by hand, claim by claim, is exhausting; it's exactly the kind of thing machine comparison is suited for.

### Positioning
- The system's role is to **surface suspicious points alongside their evidence**, not to adjudicate right or wrong automatically
- The conclusion should state the **reduction** in hallucination and the remaining error categories, not claim elimination

---

## 2. Steps

Phase 1 = A-H. Phase 2 = I. Each group's **acceptance criterion** gates moving to the next.

### A. Corpus construction
- **A1** collect all 12 original PDFs, plus the survey itself: 13 total
- **A2** build `corpus.yaml`: `ref -> doc_id -> tier -> file`. Fixed before any code is written; every downstream module depends on it
- **A3** parser reads coordinates, not text order: x splits into columns, y splits into rows
- **A4** classify blocks as prose / table / caption / formula, each carrying a bbox and page
- **A5** table reconstruction: x splits columns, first-column token is the row anchor, handles multi-line cells
- **A6** filter out headers / footers / page numbers
- **Acceptance**: spot-check 5 random pages against the original PDF verbatim; column order and table columns both correct

### B. Quality gate
- **B1** rule-based checks: CID garble rate, coverage gap, bbox overlap across columns, table column-count consistency, numeric-unit regex, mid-sentence paragraph breaks
- **B2** visual re-check: for pages flagged by B1 **plus every page containing a table**, render to PNG and send to a vision model for cell-by-cell comparison
- **B3** visual re-check output is forced JSON; `severity` must include `cannot_verify`
- **B4** human adjudication of discrepancies B2 reports
- **B5** formula normalization: map the U+1D400 range (Mathematical Alphanumeric Symbols) back to ASCII
- **Acceptance**: produce a quality report -- total pages, flag rate, re-check count, human-correction count

### C. Indexing
- **C1** chunk prose, injecting metadata `{doc_id, tier, section, page, block_type}`
- **C2** build a vector index + BM25 index over prose
- **C3** convert tables to JSON in a structured query layer, **not into the vector store**
- **C4** store formulas as LaTeX
- **Acceptance**: "which systems support 2 MB THP" via filter returns all three correct hits (MTM, NOMAD, NeoMem), none missing

### D. Retrieval layer
- **D1** hybrid: BM25 catches acronyms (PEBS, DCSC, FMAR, CIT, THP) + dense catches semantics
- **D2** attach a reranker
- **D3** support `doc_id` and `tier` filter parameters -- **this is the precondition for group G**
- **Acceptance**: the same query with and without a doc_id filter converges correctly

### E. `ask` mode
- **E1** prompt forces citations; when context has no data, answer "not stated in the sources"; deriving/estimating a value is forbidden
- **E2** answers must label the source as tier-0 (survey paraphrase) or tier-1 (original)
- **E3** support a tier filter: query originals only, or query what the survey itself says
- **Acceptance**: class-A questions get the correct value with the correct section cited

### F. Eval-set annotation
- **F1** class A (corpus has a clear answer), 10-15 questions, each with recorded ground truth and source
- **F2** class B (not in the corpus, correct answer is refusal), 10 questions
- **F3** class C (misattribution traps), 10 questions
- **F4** self-audit seeds: four known catchable points planted in the survey
- **F5** save as `eval_set.jsonl`: `{id, class, question, ground_truth, source, note}`
- **Acceptance**: every ground truth points to a specific page / section / table cell
- **Cannot be outsourced to a model** -- using model-generated ground truth to evaluate a model is circular

### G. `check` mode
- **G1** claim extraction: `{subject, metric, value, unit, relation, condition, cited_ref, source_span}`
- **G2** reference resolution: look up corpus.yaml, `[9] -> m5_asplos25`
- **G3** scope decision (per claim, not per paragraph)
- **G4** four verdict classes: `supported` / `contradicted` / `not_found` / `condition_mismatch`
- **G5** output must carry the original source span
- **Acceptance**: class-C questions aren't fooled; running it on the survey itself catches the four points planted in F4

### H. Comparison experiment and wrap-up
- **H1** three arms: no retrieval / retrieval over unfiltered text / this system
- **H2** record model version, date, prompt, temperature, n>=3 repeats
- **H3** metrics: numeric exact-match (primary), class-B refusal rate, class-C misattribution rate, reference-resolution accuracy, Context Precision / Recall
- **H4** error analysis: categorize remaining errors and their causes
- **H5** write `results.md` and the README
- **Acceptance**: able to state one conclusion backed by a number

### I. Phase 2
- **I1** qualitative claim verification (non-numeric)
- **I2** cross-tier consistency check: answer the same question using only tier-0 and only tier-1, flag disagreement
- **I3** a visualization panel for human adjudication
- **I4** corpus expansion

### Not doing
Chat UI, multi-turn memory, custom embeddings, a hand-written layout model. Off-the-shelf components for infrastructure; verification logic built in-house.

---

## 3. Modules

```
paper-refcheck/
|-- corpus.yaml
|-- parse.py
|-- quality_check.py
|-- vision_verify.py
|-- build_index.py
|-- retrieve.py
|-- ask.py
|-- check.py
|-- eval/
|   |-- eval_set.jsonl
|   `-- run_baseline.py
|-- tables/*.json
`-- results.md
```

### corpus.yaml
Document registry. `ref -> doc_id -> tier -> file`.

```yaml
- ref: 0
  doc_id: cs550_survey
  tier: 0
  file: papers/survey.pdf
- ref: 9
  doc_id: m5_asplos25
  tier: 1
  file: papers/m5.pdf
```

`tier` 0 = our own survey (secondhand paraphrase), 1 = original source (firsthand). `check.py`'s reference resolution depends entirely on this table.

### parse.py
PDF -> `blocks.jsonl`, each record `{doc_id, tier, page, section, block_type, bbox, text}`.

The core is coordinate-based: x splits into columns, y splits into rows; tables are reconstructed with "x splits columns, first-column token as row anchor" to handle multi-line cells. Already tested on the survey -- all 12 rows of Table 1 correct, and it catches the column-attribution errors that flat extraction commits.

**Formulas go through coordinates too, no OCR needed.** The seemingly garbled AOL formula in the survey is actually CambriaMath using the U+1D400 range (`A` italic = U+1D434 MATHEMATICAL ITALIC CAPITAL A) -- a legal codepoint that most terminals just render as a box. Three steps recover it:

- normalize the U+1D400 range back to ASCII
- classify the smaller-size span as sub/superscript
- classify a thin horizontal line inside a drawing rect as a fraction bar, with content above as numerator and below as denominator

Tested result: `AOL = L_loaded / (1 + alpha*(MLP - 1))`, reconstructed deterministically, no vision model needed.

### quality_check.py
Takes `blocks.jsonl` + PDF pages, outputs a flag list `{page, check_name, detail}`. Pure rules, zero cost, goal is to minimize the page count that needs visual re-checking.

### vision_verify.py
Takes flagged pages + every page containing a table, renders to PNG and sends it with the parse result to a vision model. Outputs `{location, extracted, image_shows, severity}`. `severity` must include `cannot_verify` -- without a refusal option the model will guess.

**Also needs a channel-level health check.** The vision channel can fail silently (the image never arrives but the call still returns successfully), in which case the model produces a seemingly normal comparison report without ever having seen the image. Approach: overlay a random token on the image on every call, require the model to report that token first; failing to do so means the image never arrived, and that page is routed back to a human.

### build_index.py
Builds three paths:
- prose -> vector index + BM25
- table -> structured JSON layer (**not into the vector store**)
- formula -> LaTeX

Routing tables through semantic search is wrong. "Which systems support 2 MB THP" needs the complete set; semantic similarity can't answer with a set.

### retrieve.py
Hybrid (BM25 + dense) + reranker, supporting both `doc_id` and `tier` filters. These two parameters are the precondition for `check.py` to work at all.

### ask.py
Question -> retrieval -> generation. Global retrieval by default, tier can be specified. The answer must label its source tier, since tier-0 is our own paraphrase and may be inaccurate.

### check.py
The project's core. Four steps:

1. **Claim extraction** -- the `condition` field is required; without it, a dropped condition can't be detected
2. **Reference resolution** -- look up corpus.yaml
3. **Scope decision**
4. **Verdict + output span**

**Scope decision logic** (per claim, not per paragraph -- if one sentence has three numbers each citing a different paper, it runs three times):

```
has cited_ref
   -> lock retrieval to that doc_id -> four-class verdict

no cited_ref
   |-- is the author's own experimental result -> skip, nothing to verify
   `-- looks like a restatement of someone else's work -> global search excluding tier-0
        `-- if a matching value is found in some original
             -> report "possibly missing a citation"
```

Excluding tier-0 is the key part: allowing retrieval to hit the survey's own narrative turns this into verifying a secondhand description with a secondhand description -- the system looks like it's working but the result is meaningless.

**Four verdict classes**:
- `supported` value and condition both match
- `contradicted` the original clearly states a different value
- `not_found` the original has no such number (wrong paper cited, or self-derived while paraphrasing)
- `condition_mismatch` the value is right but a condition was dropped or altered -- the single most valuable class
- `underspecified` the source itself is ambiguous (the same symbol defined multiple times, a relationship left unstated) -> defer to a human

A binary verdict would miss the last two entirely.

### eval/eval_set.jsonl
Three question classes:

- **Class A** the corpus has a clear answer. Example: TierLab's migration penalty = 54 us (section 3.1)
- **Class B** not in the corpus but a reasonable question; correct answer is refusal. Example: Colloid's P95 on zipf_high_mlp (the simulation runs *policies*, not *systems*)
- **Class C** misattribution traps

Class-C seeds come from real LLM errors that actually happened:

| Question | Correct answer | Trap answer |
|---|---|---|
| Soar/Alto's AOL formula | see below -- the correct answer is **to flag the source ambiguity** | `L/(1+alpha*(MLP-1))` labeled "the original paper's formula" -- both LLM-produced planning documents wrote it this way |
| NOMAD's multi-tenant improvement, in % | no such number | 72% -- actually Adaptive Migration's improvement **relative to** NOMAD, direction reversed |
| Which systems support 2 MB THP | MTM, NOMAD, NeoMem | NeoMem omitted |

**The full AOL case** (the single most important question in this project):

AOL appears with two definitions in the survey:

- Section 2.2 C -- `AOL = Latency / MLP`, attributed to Soar/Alto [4]
- Section 3.1 -- `AOL = L_loaded / (1 + alpha*(MLP-1))`, TierLab's implementation

At alpha=1, `1+1*(MLP-1) = MLP` -- the second form reduces exactly to the first. So the two don't contradict each other; it's an **unstated generalization** -- the original never writes out this relationship.

So the correct answer isn't "A is right, B is wrong." It's to report: the same symbol is defined differently in two places, the relationship is never stated, and one of the two is the author's own implementation rather than the cited paper's definition. This needs the fifth verdict class, `underspecified`, or should be routed to human adjudication.

The value of this case: it's a **real finding** that surfaced before the project had even started being built, and it demonstrates the correct behavior for an auditor -- not judging right or wrong, but surfacing the ambiguity alongside its evidence. The README can open with this case to show the motivation isn't hypothetical.

Self-audit seeds (using the system to audit its own survey):
1. Section 3.3 writes "Figures 1-4," but the figures actually cited are Figure 2-5
2. Conclusion #2's "~300 accesses" is a derived value (54 us / 170 ns = 317.6), not a quoted one
3. Telescope in section 2.1 is "90%+ precision/recall on 5 TB at ~0.9% of a single CPU"; section 3.4.1 compresses this to "High at TB scale / <1% CPU" -> `condition_mismatch`, and it happens within the same document
4. Section 3.1's 54 us figure is labeled as calibrated to M5 [9] -- needs checking against the original to confirm the number exists and under what conditions

### eval/run_baseline.py
Three-arm comparison, metrics primarily deterministic. Values need normalization before comparison (`54 us` = `0.054 ms`).

The baseline arm has to actually run, not be filled in from imagination -- current models faced with an unfamiliar paper are quite likely to refuse outright rather than fabricate; if the report is written on the assumption of "confident fabrication" but the real behavior is refusal, the whole comparison collapses.

### results.md
Numbers + error analysis. Remaining errors are expected to concentrate in: cross-chunk synthesis, reference resolution on compound citations like `[3, 9]`, low recall on condition extraction, and merged table cells.

---

## 4. Four things that cannot be gotten wrong

1. **One model, one index.** The only difference between `ask` and `check` is how the query is constructed and how retrieval scope is restricted -- no model fine-tuning is involved anywhere. This needs to be stated up front in the README, to avoid being misread as two specialized trained models.
2. **`check`'s retrieval must be scoped.** Global retrieval will pull in the survey's own narrative, forming a self-referential loop.
3. **Tables never enter the vector store.** Set-shaped queries need a filter, not semantic similarity.
4. **Eval ground truth cannot be model-generated.** Otherwise the evaluation itself is circular.

## 5. Practical trade-offs
- Key tables can be structured by hand (20-30 of them, one afternoon, 100% accuracy)
- The retrieval layer runs fully local (embedding, BM25, and reranker can all run offline); the generation layer's choice of local model vs. API is decided later
- All parsing scripts stay in the repo; the pipeline must be reproducible
