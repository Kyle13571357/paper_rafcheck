# Results

State as of 2026-07-30.

**Important distinction:** this document has two parts. Part 1 is **already measured** -- every number here comes from deterministic code, no model involved. Part 2 is the **not-yet-run** model comparison experiment -- code complete, waiting on an API key. No number below is invented.

---

## Part 1 -- Measured (deterministic, no model)

### Corpus

| Item | Count |
|---|---|
| Documents | 13 (1 tier-0 survey, 12 tier-1 originals) |
| Pages | 216 |
| Blocks | 8,529 |
| Chunks | 1,238 |
| Tables (structured layer, not embedded) | 88 |
| Formulas | 11 |

Block breakdown: prose 7,974 / heading 271 / caption 185 / table 88 / formula 11.

### Module A acceptance -- parsing correctness

- **All 13 rows of the survey's Table 1 correct**, including multi-line cells and a `(policy-agnostic)` value wrapped across a row boundary.
- **AOL formula reconstructed deterministically**, no vision model needed:
  `AOL = L_loaded / (1+alpha*(MLP-1))`
  Three steps: normalize the U+1D400 range back to ASCII, classify sub/superscript by span `size`, classify a thin horizontal drawing rect as a fraction bar.
- Text coverage (extracted characters / raw PDF characters): median **0.988**, mean **0.985**. The one page below 0.80 is a 154-character page whose entire shortfall is the running header, deliberately filtered.

### Module B acceptance -- quality gate

| Metric | Value |
|---|---|
| Total pages | 216 |
| Pages flagged by rules | 44 (20.4%) |
| Pages routed to visual review | 74 (34.3%) |

Flag breakdown: `misencoded_unit` 97 (blocker), `paragraph_split_mid_sentence` 37 (info), `table_parse_flag` 28, `table_ragged_rows` 7, `table_empty_row` 1.

`bbox_crosses_gutter`: **0 hits** -- this check is independent of `parse.py`'s column-splitting logic, so zero hits means the column decisions are self-consistent.

`misencoded_unit` was added later (see "Mis-encoded units" below); adding it raised the review queue from 60 to 74 pages.

### Module C acceptance -- set-query completeness

> "Which systems support 2 MB THP?"

```
MTM          granularity='2 MB (THP)'   (cs550_survey p5)
NOMAD        granularity='4 KB / 2 MB'  (cs550_survey p5)
NeoMem       granularity='4 KB / 2 MB'  (cs550_survey p3)
RESULT: PASS -- nothing missing
```

The point: NeoMem's cell reads `4 KB / 2 MB`. A literal search for `"2 MB THP"` finds **only MTM**. Normalizing to a `2mb` token recovers all three. This is the actual reason tables never enter the vector index.

### Module D acceptance -- retrieval scope convergence

Same query, `"page migration latency cost microseconds"`:

| Condition | Result |
|---|---|
| No filter | hits spread across 6 different docs |
| `doc_id=m5_asplos25` | 8/8 all from M5 |
| `exclude_tier=0` | remaining tiers are `[1]` only, survey fully excluded |
| Table filter `2mb` | `['MTM','NOMAD','NeoMem']` |

### Module G -- scope decisions (the deterministic part)

| Scenario | Decision |
|---|---|
| `[9]` | locked to `m5_asplos25` |
| `[3, 9]` compound citation | locked to `['colloid_sosp24','m5_asplos25']` |
| `[3, 99]` partially invalid | locked to `colloid_sosp24`, flagged `partially_resolved:[99]` |
| `[99]` invalid | `unresolvable_reference` |
| No citation + author's own result | `skip` |
| No citation + apparent paraphrase of others | global search, **`exclude_tier=0`** |

### The single most important item: mis-encoded units, and how it fooled me too

`check.py`'s `numeric_prescreen` never touches a model -- it's unit normalization followed by a numeric comparison, in code:

```
M5          value='54 us'      -> scope ['m5_asplos25']  -> NOT FOUND   <- false negative
Colloid     value='1.01-1.76x' -> scope ['colloid_sosp24'] -> FOUND '1.76' p10
Telescope   value='90%'        -> scope ['telescope_atc24'] -> FOUND p2
Adaptive    value='72.0%'      -> scope ['adaptive_migration_arxiv25'] -> FOUND p12
```

The first row is a **false negative** -- and at one point I treated it as a citation error in the survey and wrote that conclusion into this document and the eval set. M5 p12 states, in plain text:

> This is not enough of the number of accesses to amortize the cost of page migration
> (~54us in our setup), which requires more than 318 accesses (= 54us/(270ns - 100ns)) on average.

**The survey's citation is correct.** What was wrong was the extraction, and my own verification method.

**Root cause:** M5 sets units in `LibertineMathMI`, whose ToUnicode table maps to the wrong codepoints:

| Source character | Actual codepoint in the text layer |
|---|---|
| `u` (micro) | U+1D44D MATHEMATICAL ITALIC CAPITAL Z |
| `n` | U+1D43F (L) |
| `s` | U+1D440 (M) |

So `54 us` extracts as `54ZM` (mathematical-alphabet codepoints); after `parse.py`'s U+1D400 normalization it becomes the literal, clean-looking string **`54ZM`**; `270 ns` becomes `270LM`.

**Why this is the most dangerous failure class:** the text layer decodes **without any error** -- no replacement character, no CID garble, and the normalized output is valid English letters. No encoding check catches it, and any query using "us"/"microsecond" as a keyword will never match. The value is genuinely present and looks completely normal, yet is invisible to search.

**Corpus-wide census: 97 sites, 8 of 12 papers** affected (MTM's `40kU`, M5's `140-170LM`, Chrono's `5.5LMNOP`, NeoMem's `2.3x2.3mm`, and others).

**A methodological flaw this exposed:** my original `validate_eval_set.py` "re-verified" this absence claim using **the exact same regex**, so the re-check passed. Verifying a method with the same method proves nothing. It now detects "a digit immediately followed by U+1D400-range characters" and, on a hit, warns that any absence claim about that quantity must be checked against the rendered page, not the text layer.

**How it was actually found:** the author looked directly at the rendered PDF page and saw the number was right there. This is the clearest possible case for the B2 visual re-check design -- **the image channel can recover what the text channel silently lost** -- and `misencoded_unit` now routes all 97 sites into the visual review queue at blocker severity.

Correction log: eval-set items `C05` / `C08` / `S02` / `S04` were all reversed; `S06` was added to record the actual defect (in the corpus, not the survey); `B03` was replaced with a genuinely absent fact.

### Real defects found in the source papers during audit-set construction

**Adaptive Migration (ref 5) contradicts itself.** The friendly/unfriendly labels are swapped between two passages:

- p2: migration-**unfriendly** -> 14.8%; migration-**friendly** -> 36.0%
- p12: 14.8% "with migration-**friendly** workloads"; 36.0% "with migration-**unfriendly** workloads"

p2 is the internally consistent version (turning migration off should help workloads that migration hurts). The survey follows p2, correctly. This was found while building the eval set by checking the original paper directly; it's recorded as `C06`, and the correct output is to **report the ambiguity**, not pick a side.

**Colloid's condition was dropped.** The original, p11:

> even at the maximum alternate tier unloaded latency (2.7x of default tier), Colloid still achieves 1.01-1.76x, 1.03-1.76x and 1.01-1.63x performance improvement for HeMem, TPP and MEMTIS, respectively

The survey's section 2.2 compresses this to "Achieves 1.01-1.76x speedup" -- the number is correct, and every qualifying condition (2.7x alternate-tier latency, HeMem only) is gone. A textbook `condition_mismatch`.

### Vision channel health check -- verified by test, not assumed

A negative control: same prompt, `images` field removed, everything else identical.

The model returned a **complete, confident** report -- a `page_summary` sentence ("A table with three columns: System, Profiling, and Objective") and a discrepancy whose `image_shows` field asserted what the page displayed. **It never received any image.**

The token check caught it: reported `NONE` != expected token -> `channel_ok: false` -> `channel_failed_needs_human`, and every finding from that call was discarded.

This isn't a hypothetical risk -- it's observed behavior.

### Visual re-check pilot run (19/60 pages, stopped)

The local 7B vision model ran 19 pages before the run was stopped (200-2200s per page, the machine was overheating). The remaining 41 pages are pending a cloud provider.

| Item | Value |
|---|---|
| Records | 19 |
| `clean` | 9 |
| `discrepancies_found` | 4 |
| `request_failed` | 6 |
| Channel-check pass rate | 13/13 (all successful calls) |
| Discrepancies retained | 6 (4 major / 1 minor / 1 cannot_verify) |
| False positives filtered (self-agreeing) | 32 |

The filtered count (32) far exceeds what was kept (6): the model frequently fills the discrepancy template even when it checked a value and found it matched; those get dropped whenever `extracted == image_shows`.

Real problems it caught:

- `adaptive_migration_arxiv25` p9 **[major]**: Table 2's CXL Memory bandwidth column parsed as empty (`Read : Write :`); the actual figure is `Read : 17.8GB/s, Write : 15.8GB/s` -- genuine data loss.
- `cs550_survey` p12 **[major]**: `Alternate tier` vs. the actual `Alternate tier saturated` -- a multi-line cell cut short.
- `cs550_survey` p8 **[major]**: column order scrambled in the `Migration budget` row.

Two of these match problems found independently during manual review.

---

## Part 2 -- Not yet run

The following need `DEEPSEEK_API_KEY`; the code is complete:

- **H1** three arms: `none` (no retrieval) / `raw` (retrieval over unfiltered text) / `system` (this pipeline)
- **H2** records model version, date, prompt, temperature, n>=3
- **H3** metrics: numeric exact-match (primary), class-B refusal rate, class-C misattribution rate, reference-resolution accuracy, context recall
- **H4** error analysis

Run it:

```bash
export DEEPSEEK_API_KEY=... && python3 eval/run_baseline.py --arms none raw system --repeats 3
```

The `raw` arm is deliberately built as "the most obvious first attempt": `page.get_text()`, fixed-length chunking, no coordinate-based column splitting, no filters, no reranker. Without this arm there's no way to tell whether an improvement comes from retrieval existing at all, or from the parsing and scoping specifically.

**The baseline arm has to actually run.** Current models faced with an unfamiliar paper are quite likely to refuse rather than fabricate; if the report were written on the assumption of "confident fabrication" and the real behavior is refusal, the whole comparison collapses.

### Known limitations (destined for error analysis)

1. **Soar/Alto's own AOL formula extraction is incomplete.** The original renders it as a stacked fraction (`AOL = Latency` over `MLP`) in a regular (non-math) font; `parse.py`'s formula detector requires a math font or a U+1D400-range codepoint, so it misses this one, leaving the text layer with the fragment `we define AOL = Latency`. The survey's own AOL formula (set in CambriaMath) reconstructs completely.
2. **114 -> 98 pages are still somewhat fragmented.** Switching column splitting to an x0 histogram fixed most of it; what remains is mostly chart axis-label text and reference-list pages, which is expected, not a splicing error.
3. **`near_figure_caption_possibly_not_a_table` has middling precision.** `find_tables()` sometimes misreads grid-arranged figure labels as a table. Currently caught only by this flag plus visual review, not automatically excluded.
4. **Visual re-check is only 19/74 pages done**, and used the local 7B model. If the rest is completed on a cloud model, the two batches used different methods and that needs to be noted.
5. **Mis-encoded units are detected but not repaired.** All 97 sites are flagged at blocker severity and routed to visual review, but `blocks.jsonl` still stores `54ZM` -- so a query for "54 us" still misses in M5. A fix needs either recovering the true value from the rendered page (the vision channel) or a per-font substitution table; the former is more robust but waits on a configured vision provider. This is the known defect with the largest impact on retrieval correctness right now.
