#!/usr/bin/env python3
"""PDF -> blocks.jsonl via coordinate-based extraction (not raw text order).

x splits columns, y splits rows within a column. Tables are reconstructed with
the first (leftmost) column as a row anchor so multi-line cells don't get
attributed to the wrong row. Formulas are read off span coordinates too: a
thin filled rect is a fraction bar, and a span noticeably smaller *and*
vertically offset from its neighbor is a sub/superscript.

Usage: python3 parse.py [doc_id ...]   (default: every doc in corpus.yaml)
Output: blocks.jsonl in the repo root, one JSON object per block.
"""

import argparse
import fitz  # PyMuPDF
import yaml
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

from corpus import load_corpus

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF) -> plain Unicode.
# CambriaMath/CMMI-style fonts render formula variables through this block
# (e.g. 𝐴 = U+1D434 MATHEMATICAL ITALIC CAPITAL A). The block is organized as
# repeating CAPITAL/SMALL/DIGIT alphabets under different styles (bold,
# italic, script, fraktur, double-struck, sans-serif, monospace), so instead
# of a lookup table we parse unicodedata's own name for the base letter.
# ---------------------------------------------------------------------------

_GREEK_LOWER = {
    "ALPHA": "α", "BETA": "β", "GAMMA": "γ", "DELTA": "δ", "EPSILON": "ε",
    "ZETA": "ζ", "ETA": "η", "THETA": "θ", "IOTA": "ι", "KAPPA": "κ",
    "LAMBDA": "λ", "MU": "μ", "NU": "ν", "XI": "ξ", "OMICRON": "ο",
    "PI": "π", "RHO": "ρ", "SIGMA": "σ", "TAU": "τ", "UPSILON": "υ",
    "PHI": "φ", "CHI": "χ", "PSI": "ψ", "OMEGA": "ω",
}
_GREEK_UPPER = {
    "ALPHA": "Α", "BETA": "Β", "GAMMA": "Γ", "DELTA": "Δ", "EPSILON": "Ε",
    "ZETA": "Ζ", "ETA": "Η", "THETA": "Θ", "IOTA": "Ι", "KAPPA": "Κ",
    "LAMBDA": "Λ", "MU": "Μ", "NU": "Ν", "XI": "Ξ", "OMICRON": "Ο",
    "PI": "Π", "RHO": "Ρ", "SIGMA": "Σ", "TAU": "Τ", "UPSILON": "Υ",
    "PHI": "Φ", "CHI": "Χ", "PSI": "Ψ", "OMEGA": "Ω",
}
_DIGIT_WORDS = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
                "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}
# Unicode leaves a handful of "holes" in the math-alphanumeric block where a
# pre-existing Letterlike Symbol is reused instead (e.g. italic h -> Planck
# constant). These fall outside U+1D400-1D7FF so the range guard below won't
# catch them.
_HOLES = {
    0x210E: "h", 0x2102: "C", 0x210D: "H", 0x2115: "N", 0x2119: "P",
    0x211A: "Q", 0x211D: "R", 0x2124: "Z", 0x212C: "B", 0x2130: "E",
    0x2131: "F", 0x210B: "H", 0x2110: "I", 0x2112: "L", 0x2133: "M",
    0x211B: "R",
}


def normalize_math_char(ch: str) -> str:
    cp = ord(ch)
    if cp in _HOLES:
        return _HOLES[cp]
    if not (0x1D400 <= cp <= 0x1D7FF):
        return ch
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ch
    if not name.startswith("MATHEMATICAL"):
        return ch
    words = name.split()
    last = words[-1]
    if "DIGIT" in words:
        return _DIGIT_WORDS.get(last, ch)
    if "SMALL" in words:
        if last in _GREEK_LOWER:
            return _GREEK_LOWER[last]
        return last.lower() if len(last) == 1 and last.isalpha() else ch
    if "CAPITAL" in words:
        if last in _GREEK_UPPER:
            return _GREEK_UPPER[last]
        return last.upper() if len(last) == 1 and last.isalpha() else ch
    return ch


def normalize_math_text(text: str) -> str:
    return "".join(normalize_math_char(c) for c in text)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

PAGE_NUM_RE = re.compile(
    r"^\s*(\d+\s*/\s*\d+|\d+|Page\s+\d+(\s+of\s+\d+)?)\s*$", re.I
)
# NOTE: deliberately NOT re.I -- the uppercase requirement after the number is
# what separates "3.1 Simulation Framework" from body text like "5.0 can offer
# the same bandwidth" or "1 while (true) {". Case-insensitivity is scoped to
# the keyword alternation only.
SECTION_RE = re.compile(
    r"^(\d+(\.\d+){0,3}\.?)\s+[A-Z]"      # 1, 2.1, 3.4.1 Heading
    r"|^([IVXLC]+)\.\s+[A-Z]"             # I. Heading
    r"|^([A-Z])\.\s+[A-Z]"                # C. Heading (IEEE subsections)
    r"|^§\s*\d"
    r"|^(?i:Abstract|References|Acknowledg(?:e)?ments?|Conclusions?)\s*$"
)
NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.|§\s*\d+(?:\.\d+)*)\s*")
CAPTION_RE = re.compile(r"^(Table|Figure|Fig\.)\s*\d+\s*[:.]?\s+[A-Z]")
FRAGMENT_RE = re.compile(r"^[\[\(]?\d+[\]\)]?[.:]?$")
MATH_FONT_RE = re.compile(
    r"(?i)math|cmmi|cmsy|cmex|cmbsy|msam|msbm|stix|asana|lm[a-z]{2}\d|xits"
)


def is_heading(text):
    """True for a real section heading, False for a numbered *list item*.

    Both start with "N." so the regex alone can't separate them, and font
    weight isn't usable either -- some venues set headings in plain body
    type (NeoMem's "I. INTRODUCTION" is regular weight at body size). What
    does separate them is that a heading is a short label while a list item
    is running prose: "5. Composability works. Alto [4] + Colloid [3]
    outperforms Colloid by 10%; ..." is a sentence, not a title."""
    if not SECTION_RE.match(text) or len(text) > 80:
        return False
    # "2020. Sketching Algorithms for ..." is a bibliography entry, not §2020
    if re.match(r"^(19|20)\d{2}[.\s]", text):
        return False
    rest = NUMBER_PREFIX_RE.sub("", text).strip()
    if re.search(r"[.:]\s+\S", rest):   # sentence/clause break inside -> prose
        return False
    if "," in rest:                     # "3 CXL transaction layer, and 4 ..."
        return False
    # a title is a label, not a clause: "20 GB for GAPBS graph data sets and
    # 50 GB for SPEC 2017." runs long even though it starts like a heading
    if len(rest.split()) > 9:
        return False
    return not rest.endswith((",", ";"))


def detect_column_split(line_bboxes, page_width):
    """x-coordinate separating a two-column layout, or None if single-column.

    Uses the histogram of line start positions rather than a fully-clear
    whitespace gap. A gap-based test needs the gutter to be untouched by
    *every* line, so a single figure caption or wide table spanning both
    columns erases it -- which happened on more than half the pages in this
    corpus and silently interleaved the two columns into spliced nonsense."""
    if len(line_bboxes) < 8:
        return None
    counts = Counter(round(b.x0 / 2) * 2 for b in line_bboxes)
    peaks = [x for x, n in counts.most_common() if n >= 3]
    if len(peaks) < 2:
        return None
    left = min(peaks)
    right_candidates = [x for x in peaks if x - left > page_width * 0.25]
    if not right_candidates:
        return None
    right = min(right_candidates)

    def near(target):
        return sum(n for x, n in counts.items() if abs(x - target) <= 6)

    # both columns must actually carry text; a stray indented block is not a column
    if near(left) < 3 or near(right) < 3:
        return None
    return right - 4


def overlap_frac(inner, outer):
    """Fraction of `inner`'s area that falls inside `outer`.

    Used instead of a bare .intersects() when deciding whether a text line
    belongs to an already-extracted table/formula region: a detected region
    around a figure often reaches into the neighbouring text column, and a
    mere edge touch would drop real prose."""
    area = abs(inner.get_area())
    if area <= 0:
        return 1.0 if inner in outer else 0.0
    return abs((inner & outer).get_area()) / area


def rects_close(a, b, tol=2.0):
    return (
        a.x0 - tol <= b.x1 and b.x0 - tol <= a.x1 and
        a.y0 - tol <= b.y1 and b.y0 - tol <= a.y1
    )


def dehyphen_join(words):
    """Join word tuples (x0,y0,x1,y1,text,...) sorted reading-order, collapsing
    a trailing '-' into the next token instead of inserting a space."""
    out = ""
    for w in words:
        t = w[4]
        if out.endswith("-"):
            out += t
        elif out:
            out += " " + t
        else:
            out = t
    return out


def find_gaps(intervals, x0, x1, min_gap):
    """Given a list of (a,b) covered x-intervals within [x0,x1], return the
    uncovered gaps of width >= min_gap. Used to find whitespace column/section
    boundaries without assuming a fixed layout.

    Uses exact interval merging rather than a discretized bitmap: a narrow
    but real column gap (e.g. ~7pt between two short numeric columns) sits
    close enough to typical thresholds that bitmap-bin rounding can eat
    enough of it to fall under min_gap."""
    merged = []
    for a, b in sorted(intervals):
        a, b = max(a, x0), min(b, x1)
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    gaps = []
    prev_end = x0
    for a, b in merged:
        if a - prev_end >= min_gap:
            gaps.append((prev_end, a))
        prev_end = max(prev_end, b)
    if x1 - prev_end >= min_gap:
        gaps.append((prev_end, x1))
    return gaps


# ---------------------------------------------------------------------------
# Table reconstruction: x-column split + first-column token as row anchor.
# Validated against the survey's Table 1 (13 rows incl. header, all correct,
# including multi-line cells and a row whose cell wraps a parenthetical
# across what would otherwise look like a row boundary).
# ---------------------------------------------------------------------------

def reconstruct_table(page, bbox):
    # find_tables()'s bbox is sometimes clipped a few points above the actual
    # header row (its column-grid detection anchors on the data rows), which
    # silently drops the header entirely -- pad before pulling words
    tab = fitz.Rect(bbox) + (-2, -10, 2, 2)
    tab = tab & page.rect
    words = [w for w in page.get_text("words") if tab.contains(fitz.Rect(w[:4]))]
    if not words:
        return None
    words.sort(key=lambda w: (w[1], w[0]))

    # --- column boundaries via whitespace-gap detection ---
    # gap-search over the tight content extent, not the (padded) table bbox --
    # otherwise leading/trailing bbox padding shows up as a spurious "gap"
    # that splits off an empty first/last column and starves the row-anchor
    # column of any content at all
    content_x0 = min(w[0] for w in words)
    content_x1 = max(w[2] for w in words)
    gaps = find_gaps([(w[0], w[2]) for w in words], content_x0, content_x1, min_gap=7.0)
    col_bounds = [tab.x0]
    for a, b in gaps:
        col_bounds.append((a + b) / 2)
    col_bounds.append(tab.x1)
    ncols = len(col_bounds) - 1
    if ncols < 2:
        return None

    def col_of(w):
        xc = (w[0] + w[2]) / 2
        for i in range(ncols):
            if col_bounds[i] <= xc < col_bounds[i + 1]:
                return i
        return ncols - 1

    # --- visual line-groups (words sharing a y0 within tolerance) ---
    line_groups = []
    cur = [words[0]]
    for w in words[1:]:
        if abs(w[1] - cur[-1][1]) <= 1.5:
            cur.append(w)
        else:
            line_groups.append(cur)
            cur = [w]
    line_groups.append(cur)

    # --- row triggers: column-0 lines that aren't just a wrapped fragment
    # (e.g. a citation bracket "[2]" that wrapped onto its own line) ---
    triggers = []
    for lg in line_groups:
        c0 = [w for w in lg if col_of(w) == 0]
        if not c0:
            continue
        text = dehyphen_join(sorted(c0, key=lambda w: w[0])).strip()
        if FRAGMENT_RE.match(text):
            continue
        triggers.append(min(w[1] for w in c0))
    if len(triggers) < 2:
        # degenerate case (e.g. col 0 itself is a numeric index column) --
        # fall back to treating every col-0 line as its own row
        triggers = sorted({min(w[1] for w in lg if col_of(w) == 0)
                            for lg in line_groups if any(col_of(w) == 0 for w in lg)})
    if len(triggers) < 2:
        return None

    # row boundary = midpoint between consecutive triggers, not the trigger
    # position itself -- a short row-anchor cell often sits vertically
    # centered below the true top of a taller row, so anchor-to-anchor
    # slicing misattributes that row's leading lines to the row above it.
    bounds = [-1e9] + [(triggers[i - 1] + triggers[i]) / 2 for i in range(1, len(triggers))] + [1e9]

    rows = [[[] for _ in range(ncols)] for _ in range(len(triggers))]
    for lg in line_groups:
        y = lg[0][1]
        ridx = next((i for i in range(len(triggers)) if bounds[i] <= y < bounds[i + 1]), len(triggers) - 1)
        for w in lg:
            rows[ridx][col_of(w)].append(w)

    table = [[dehyphen_join(sorted(cell, key=lambda w: (w[1], w[0]))) for cell in row] for row in rows]

    # repair: a cell whose true bottom line landed just past the row's
    # midpoint boundary and got attributed to the next row instead. Two
    # tells: an unmatched '(' whose ')' shows up at the start of the next
    # row's same column, or a trailing hyphenation break ("mem-") whose
    # continuation ("ory") is the next row's leading word.
    for c in range(ncols):
        for r in range(len(table) - 1):
            cell, nxt = table[r][c], table[r + 1][c]
            if cell.count("(") > cell.count(")") and ")" in nxt:
                m = re.match(r"^(\S*\))\s*(.*)$", nxt)
                if m:
                    table[r][c] = (cell + " " + m.group(1)).strip()
                    table[r + 1][c] = m.group(2)
                    continue
            if cell.endswith("-"):
                m = re.match(r"^(\S+)\s*(.*)$", nxt)
                if m:
                    table[r][c] = cell[:-1] + m.group(1)
                    table[r + 1][c] = m.group(2)

    return table


# ---------------------------------------------------------------------------
# Formula reconstruction: math-font/math-codepoint spans, clustered spatially,
# with fraction bars read off thin filled-rect drawings and sub/superscripts
# read off a span's size + baseline shift relative to its neighbor.
# ---------------------------------------------------------------------------

def is_math_span(span):
    # font-name match or an actual math-alphanumeric codepoint only -- plain
    # operator symbols like x or greater-or-equal also show up inline in
    # ordinary prose ("1.76x speedup"), so they're not a safe signal alone
    if MATH_FONT_RE.search(span["font"]):
        return True
    return any(0x1D400 <= ord(c) <= 0x1D7FF for c in span["text"])


def cluster_formula_spans(page, exclude_bboxes):
    d = page.get_text("dict")
    math_spans = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if not s["text"].strip() or not is_math_span(s):
                    continue
                r = fitz.Rect(s["bbox"])
                if any(overlap_frac(r, ex) > 0.5 for ex in exclude_bboxes):
                    continue
                math_spans.append(s)
    if not math_spans:
        return []

    # connected components by spatial proximity
    clusters, used = [], [False] * len(math_spans)
    for i in range(len(math_spans)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(math_spans)):
                if used[j]:
                    continue
                rj = fitz.Rect(math_spans[j]["bbox"])
                if any(rects_close(rj, fitz.Rect(math_spans[k]["bbox"]), tol=15) for k in group):
                    group.append(j)
                    used[j] = True
                    changed = True
        clusters.append([math_spans[k] for k in group])
    return clusters


def _tidy_formula_text(text):
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(?<!/)\s+\(", "(", text)
    text = re.sub(r"\s*=\s*", " = ", text)
    return re.sub(r"\s+", " ", text).strip()


def render_group(spans):
    # formula tokens read strictly left-to-right; sub/superscript status
    # comes from size+shift below, not from y-ordering (stretchy parens and
    # sub/superscript glyphs jitter in y even within one visual token run)
    spans = sorted(spans, key=lambda s: s["bbox"][0])
    sizes = [s["size"] for s in spans]
    if not sizes:
        return ""
    base_size = Counter(round(s, 1) for s in sizes).most_common(1)[0][0]
    base_centers = [(s["bbox"][1] + s["bbox"][3]) / 2 for s in spans if abs(s["size"] - base_size) < 0.5]
    base_center = sum(base_centers) / len(base_centers) if base_centers else None
    out = ""
    for s in spans:
        text = normalize_math_text(s["text"])
        center = (s["bbox"][1] + s["bbox"][3]) / 2
        marker = ""
        if base_center is not None and s["size"] < base_size * 0.9:
            shift = center - base_center
            if shift > 1.5:
                marker = "_"
            elif shift < -1.5:
                marker = "^"
        sep = "" if (out.endswith("-") or marker) else (" " if out else "")
        out += sep + marker + text
    return out.strip()


def reconstruct_formula(cluster, drawings):
    cbbox = fitz.Rect()
    for s in cluster:
        cbbox |= fitz.Rect(s["bbox"])

    bar = None
    for dr in drawings:
        r = fitz.Rect(dr["rect"])
        if dr.get("type") == "f" and r.width > 8 and r.height < 2.0 and cbbox.intersects(r):
            bar = r
            break

    if bar is not None:
        bar_cy = (bar.y0 + bar.y1) / 2

        def near_bar_x(s):
            return s["bbox"][2] > bar.x0 - 3 and s["bbox"][0] < bar.x1 + 3

        num = [s for s in cluster if near_bar_x(s) and (s["bbox"][1] + s["bbox"][3]) / 2 < bar_cy]
        den = [s for s in cluster if near_bar_x(s) and (s["bbox"][1] + s["bbox"][3]) / 2 >= bar_cy]
        rest = [s for s in cluster if not near_bar_x(s)]
        num_txt, den_txt, rest_txt = render_group(num), render_group(den), render_group(rest)
        frac = f"{num_txt} / ({den_txt})" if num_txt or den_txt else ""
        return _tidy_formula_text((rest_txt + " " + frac).strip() if rest_txt else frac)
    return _tidy_formula_text(render_group(cluster))


# ---------------------------------------------------------------------------
# Header/footer/page-number filtering
# ---------------------------------------------------------------------------

def normalize_band_text(text):
    return re.sub(r"\d+", "#", text.strip())


def find_running_bands(pages_lines, page_height, band_frac=0.10):
    """Lines whose (normalized text) recurs across many pages within the same
    top/bottom band are running headers/footers, not content."""
    band = max(50.0, page_height * band_frac)
    counts = Counter()
    for lines in pages_lines:
        seen_this_page = set()
        for y0, y1, text in lines:
            if y0 <= band or y1 >= page_height - band:
                key = normalize_band_text(text)
                if key and key not in seen_this_page:
                    counts[key] += 1
                    seen_this_page.add(key)
    n_pages = len(pages_lines)
    threshold = max(2, int(n_pages * 0.4))
    return {k for k, c in counts.items() if c >= threshold}


NUMBER_ONLY_RE = re.compile(r"^(\d+(?:\.\d+)*\.?|[IVXLC]+\.?)$")


def body_font_size(pages_dict):
    """Modal character-weighted font size = the document's body text size."""
    sizes = Counter()
    for d in pages_dict:
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def looks_like_heading(text, is_big):
    """Regex path catches venues that set headings at body size (NeoMem's
    "I. INTRODUCTION"); the font-size path catches venues whose headings
    carry no number on the same line (ACM/LaTeX splits "2" and "Background"
    onto separate lines, so neither half matches the regex)."""
    if is_heading(text):
        return True
    return (is_big and 0 < len(text) <= 80 and len(text.split()) <= 9
            and not text.endswith((",", ";", ".")))


# ---------------------------------------------------------------------------
# Main per-document parse
# ---------------------------------------------------------------------------

def parse_document(entry, corpus):
    path = corpus.root / entry["file"]
    doc = fitz.open(path)
    blocks = []

    # pass 1: collect line bboxes/text per page (for header/footer detection)
    pages_dict = [p.get_text("dict") for p in doc]
    pages_lines_raw = []
    for d in pages_dict:
        lines = []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                text = normalize_math_text("".join(s["text"] for s in l["spans"])).strip()
                if text:
                    lines.append((l["bbox"][1], l["bbox"][3], text))
        pages_lines_raw.append(lines)

    page_h = doc[0].rect.height
    running_bands = find_running_bands(pages_lines_raw, page_h)
    body_size = body_font_size(pages_dict)

    current_section = None

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1
        d = pages_dict[page_idx]

        # caption lines, gathered up front so table candidates can check
        # whether they're actually sitting next to a "Figure N" caption --
        # find_tables() also fires on grid-arranged figure labels (e.g. a
        # page-table-walk diagram), which isn't a CID-garble problem (the
        # text reads fine) so the alnum-ratio check below won't catch it
        page_captions = []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                text = normalize_math_text("".join(s["text"] for s in l["spans"])).strip()
                if CAPTION_RE.match(text):
                    page_captions.append((fitz.Rect(l["bbox"]), text))

        def nearest_caption_is_figure(bbox):
            best, best_d = None, 1e9
            for cb, ctext in page_captions:
                d_ = min(abs(cb.y0 - bbox.y1), abs(bbox.y0 - cb.y1))
                if d_ < best_d:
                    best, best_d = ctext, d_
            return best is not None and best_d < 15 and not best.lower().startswith("table")

        # tables
        try:
            found_tables = page.find_tables()
        except Exception:
            found_tables = None
        table_bboxes = []
        for t in (found_tables.tables if found_tables else []):
            rows = reconstruct_table(page, t.bbox)
            if not rows:
                continue
            table_bboxes.append(fitz.Rect(t.bbox))
            text = "\n".join(" | ".join(cell for cell in row) for row in rows)
            # some figures/charts embed a broken-CMap font for axis/legend
            # text (real content, but every codepoint decodes to punctuation
            # noise) and still get flagged as a table by find_tables(); flag
            # rather than silently emit or drop, so a human/vision pass (the
            # quality gate) reviews it instead of it looking like real data
            alnum = sum(c.isalnum() for c in text)
            dense = max(1, len(re.sub(r"\s", "", text)))
            block = {
                "doc_id": entry["doc_id"], "tier": entry["tier"], "page": page_no,
                "section": current_section, "block_type": "table",
                "bbox": list(t.bbox), "text": text, "table_rows": rows,
            }
            if alnum / dense < 0.4:
                block["quality_flag"] = "low_alnum_ratio_possible_cid_garble"
            elif nearest_caption_is_figure(fitz.Rect(t.bbox)):
                block["quality_flag"] = "near_figure_caption_possibly_not_a_table"
            blocks.append(block)

        # formulas
        drawings = page.get_drawings()
        formula_clusters = cluster_formula_spans(page, table_bboxes)
        formula_bboxes = []
        OPERATOR_CHARS = set("=+−×÷∙≤≥≈→±/")
        for cl in formula_clusters:
            bbox = fitz.Rect()
            for s in cl:
                bbox |= fitz.Rect(s["bbox"])
            if bbox.height > 45 or len(cl) > 15:
                # algorithm/pseudocode listings are dense with math-italic
                # variable names line after line, so proximity clustering
                # can chain a whole multi-line listing into one "cluster";
                # a real equation is compact -- bail and let it fall through
                # to prose rather than emit an unreadable blob
                continue
            has_bar = any(
                dr.get("type") == "f" and fitz.Rect(dr["rect"]).width > 8
                and fitz.Rect(dr["rect"]).height < 2.0 and bbox.intersects(fitz.Rect(dr["rect"]))
                for dr in drawings
            )
            has_operator = any(c in OPERATOR_CHARS for s in cl for c in s["text"])
            # a single inline math-italic variable (e.g. an "L" in "the
            # latency L increases...") is not a formula worth its own block --
            # pulling it out would leave a hole in the middle of that sentence.
            # Only carve out a real equation: multiple tokens plus an
            # operator, or an actual fraction bar.
            if not (has_bar or (len(cl) >= 2 and has_operator)):
                continue
            text = reconstruct_formula(cl, drawings)
            if not any(c.isalnum() for c in text):
                continue
            formula_bboxes.append(bbox)
            blocks.append({
                "doc_id": entry["doc_id"], "tier": entry["tier"], "page": page_no,
                "section": current_section, "block_type": "formula",
                "bbox": list(bbox), "text": text,
            })

        exclude = table_bboxes + formula_bboxes

        # reading order over remaining lines: x splits columns, y splits rows
        lines = []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                lb = fitz.Rect(l["bbox"])
                text = normalize_math_text("".join(s["text"] for s in l["spans"])).strip()
                if not text or any(overlap_frac(lb, ex) > 0.5 for ex in exclude):
                    continue
                key = normalize_band_text(text)
                near_edge = lb.y0 <= max(50.0, page_h * 0.10) or lb.y1 >= page_h - max(50.0, page_h * 0.10)
                if near_edge and (key in running_bands or PAGE_NUM_RE.match(text)):
                    continue
                lines.append({"bbox": lb, "text": text, "spans": l["spans"]})
        lines.sort(key=lambda x: (x["bbox"].y0, x["bbox"].x0))

        mid = detect_column_split([ln["bbox"] for ln in lines], page.rect.width)

        def col_side(ln):
            if mid is None:
                return "single"
            if ln["bbox"].x1 <= mid + 3:
                return "left"
            if ln["bbox"].x0 >= mid - 3:
                return "right"
            return "full"

        ordered = []
        left_buf, right_buf = [], []

        def flush():
            ordered.extend(left_buf)
            ordered.extend(right_buf)
            left_buf.clear()
            right_buf.clear()

        for ln in lines:
            side = col_side(ln)
            if side in ("single", "full"):
                flush()
                ordered.append(ln)
            elif side == "left":
                left_buf.append(ln)
            else:
                right_buf.append(ln)
        flush()

        # merge consecutive non-heading lines in the same run into paragraphs,
        # tracking section headings as we go
        para_lines, para_bbox = [], None
        caption_lines, caption_bbox = [], None

        def flush_para():
            nonlocal para_lines, para_bbox
            if para_lines:
                blocks.append({
                    "doc_id": entry["doc_id"], "tier": entry["tier"], "page": page_no,
                    "section": current_section, "block_type": "prose",
                    "bbox": list(para_bbox), "text": " ".join(para_lines),
                })
            para_lines, para_bbox = [], None

        def flush_caption():
            nonlocal caption_lines, caption_bbox
            if caption_lines:
                blocks.append({
                    "doc_id": entry["doc_id"], "tier": entry["tier"], "page": page_no,
                    "section": current_section, "block_type": "caption",
                    "bbox": list(caption_bbox), "text": " ".join(caption_lines),
                })
            caption_lines, caption_bbox = [], None

        prev_bbox = None
        pending_number = None      # a lone "2" awaiting its title line
        last_heading = None        # for merging a heading wrapped over 2 lines
        for ln in ordered:
            text = ln["text"]
            ln_size = max((s["size"] for s in ln["spans"]), default=body_size)
            size_big = ln_size > body_size + 0.8
            # Page 1 is front matter: the title, every author name and every
            # affiliation is set larger than body text, so size alone must not
            # promote a line to a heading there. It is still needed to spot the
            # split "1" / "Introduction" pair, which the regex then confirms.
            is_big = size_big and page_no > 1

            # ACM/LaTeX puts the section number on its own line above the
            # title; hold the number and attach it to the title that follows
            if size_big and NUMBER_ONLY_RE.match(text):
                flush_para()
                flush_caption()
                pending_number = text
                prev_bbox = None
                continue
            if pending_number:
                # Rejoin either way: if the pair really is a heading the regex
                # below now sees "2 Background", and if it isn't, the number
                # goes back into the prose it came from instead of vanishing.
                text = f"{pending_number} {text}"
                pending_number = None

            if looks_like_heading(text, is_big):
                flush_para()
                flush_caption()
                # a long heading wraps onto a second line ("3 CXL-driven Page
                # and Word Access" / "Counting"); fold it back into one
                if (last_heading is not None and is_big
                        and not NUMBER_PREFIX_RE.match(text)
                        and ln["bbox"].y0 - last_heading["_y1"] < 6):
                    last_heading["text"] += " " + text
                    current_section = last_heading["text"]
                    last_heading["section"] = current_section
                    last_heading["_y1"] = ln["bbox"].y1
                    prev_bbox = None
                    continue
                current_section = text.strip()
                # emit the heading as its own block rather than consuming it:
                # if the heading test ever misfires on a real content line,
                # the worst outcome is a mislabeled block_type, never a
                # silently dropped sentence
                hb = {
                    "doc_id": entry["doc_id"], "tier": entry["tier"], "page": page_no,
                    "section": current_section, "block_type": "heading",
                    "bbox": list(ln["bbox"]), "text": text,
                    "_y1": ln["bbox"].y1,
                }
                blocks.append(hb)
                last_heading = hb
                prev_bbox = None
                continue
            last_heading = None
            if CAPTION_RE.match(text):
                flush_para()
                caption_lines.append(text)
                caption_bbox = ln["bbox"] if caption_bbox is None else caption_bbox | ln["bbox"]
                prev_bbox = ln["bbox"]
                continue
            # a "same paragraph" continuation must be a small *positive* y-step
            # in the *same column* -- a plain small-gap check also matches the
            # large negative gap produced when reading order wraps from the
            # bottom of the left column back to the top of the right column,
            # which would otherwise splice unrelated columns into one block
            def _continues(ln, prev):
                gap = ln["bbox"].y0 - prev.y1
                return -4 <= gap < 8 and abs(ln["bbox"].x0 - prev.x0) < 25

            same_run = (
                prev_bbox is not None
                and _continues(ln, prev_bbox)
                and not caption_lines
            )
            if caption_lines and prev_bbox is not None and _continues(ln, prev_bbox):
                caption_lines.append(text)
                caption_bbox |= ln["bbox"]
                prev_bbox = ln["bbox"]
                continue
            flush_caption()
            if not same_run:
                flush_para()
            para_lines.append(text)
            para_bbox = ln["bbox"] if para_bbox is None else para_bbox | ln["bbox"]
            prev_bbox = ln["bbox"]
        flush_para()
        flush_caption()

    doc.close()
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc_ids", nargs="*",
                    help="only these documents (default: every one registered)")
    ap.add_argument("--corpus", default=None,
                    help="corpus directory (default: nearest corpus.yaml)")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    wanted = set(args.doc_ids)
    entries = [e for e in corpus.entries if e["doc_id"] in wanted] if wanted \
        else corpus.entries

    all_blocks = []
    for entry in entries:
        print(f"parsing {entry['doc_id']} ({entry['file']}) ...", file=sys.stderr)
        blocks = parse_document(entry, corpus)
        print(f"  -> {len(blocks)} blocks", file=sys.stderr)
        all_blocks.extend(blocks)

    with open(corpus.blocks_path, "w") as f:
        for b in all_blocks:
            b.pop("_y1", None)   # internal bookkeeping for heading merging
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_blocks)} blocks to {corpus.blocks_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
