#!/usr/bin/env python3
"""Numeric normalization, so "54 µs" and "0.054 ms" compare equal.

Kept deterministic and separate from any model. Verdicts turn on whether two
numbers match, and that comparison must not depend on a model's arithmetic --
asking a model whether 54 µs equals 0.054 ms reintroduces exactly the kind of
silent error the project is trying to detect.
"""

import re

# everything reduces to a base unit within its dimension
SCALES = {
    "time": {
        "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
        "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
        "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    },
    # Binary, not decimal: in a memory-tiering corpus "4 KB page" and
    # "2 MB THP" are page sizes, so KB/MB mean KiB/MiB. Treating them as
    # decimal makes "2 MB" and "2048 KB" differ by 2.4% and compare unequal.
    "bytes": {
        "b": 1, "byte": 1, "bytes": 1,
        "kb": 1024, "kib": 1024,
        "mb": 1024 ** 2, "mib": 1024 ** 2,
        "gb": 1024 ** 3, "gib": 1024 ** 3,
        "tb": 1024 ** 4, "tib": 1024 ** 4,
        "pb": 1024 ** 5,
    },
    "bandwidth": {
        "b/s": 1, "kb/s": 1e3, "mb/s": 1e6, "gb/s": 1e9, "tb/s": 1e12,
    },
    "percent": {"%": 1.0, "percent": 1.0},
    "ratio": {"x": 1.0, "×": 1.0, "times": 1.0},
}

UNIT_TO_DIM = {}
for dim, table in SCALES.items():
    for unit in table:
        UNIT_TO_DIM[unit] = dim

NUM_RE = re.compile(
    r"(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|×|[a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?"
)
RANGE_RE = re.compile(
    r"(?P<lo>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?:-|–|—|to|~|↔)\s*"
    r"(?P<hi>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|×|[a-zA-Zµμ]+(?:/[a-zA-Z]+)?)?"
)


def _canon_unit(unit):
    if unit is None:
        return None
    u = unit.strip().lower()
    return u if u in UNIT_TO_DIM else None


def parse_quantity(text):
    """First quantity in `text` as (value_in_base_unit, dimension, raw).

    Returns None when there is no number, or when the number carries a unit
    this module does not understand -- an unknown unit must not be silently
    treated as dimensionless."""
    if not text:
        return None
    s = str(text).strip()

    m = RANGE_RE.search(s)
    if m:
        unit = _canon_unit(m.group("unit"))
        dim = UNIT_TO_DIM.get(unit) if unit else None
        scale = SCALES[dim][unit] if dim else 1.0
        lo = float(m.group("lo").replace(",", "")) * scale
        hi = float(m.group("hi").replace(",", "")) * scale
        return {"kind": "range", "lo": lo, "hi": hi, "dim": dim, "raw": m.group(0)}

    m = NUM_RE.search(s)
    if not m:
        return None
    unit = _canon_unit(m.group("unit"))
    dim = UNIT_TO_DIM.get(unit) if unit else None
    scale = SCALES[dim][unit] if dim else 1.0
    val = float(m.group("num").replace(",", "")) * scale
    return {"kind": "point", "value": val, "dim": dim, "raw": m.group(0)}


def quantities_match(a, b, rel_tol=0.02):
    """True when two quantity strings denote the same magnitude.

    A point value inside a stated range counts as a match: a paper reporting
    "1.01-1.76x" does support a claim of "1.5x", though the *condition* may
    still differ -- that is a separate judgement, deliberately not made here."""
    qa, qb = parse_quantity(a), parse_quantity(b)
    if not qa or not qb:
        return None                       # nothing numeric to compare
    if qa.get("dim") and qb.get("dim") and qa["dim"] != qb["dim"]:
        return False                      # e.g. a time vs a percentage

    def within(x, y):
        if x == 0 or y == 0:
            return abs(x - y) < 1e-12
        return abs(x - y) / max(abs(x), abs(y)) <= rel_tol

    if qa["kind"] == "point" and qb["kind"] == "point":
        return within(qa["value"], qb["value"])
    if qa["kind"] == "range" and qb["kind"] == "range":
        return within(qa["lo"], qb["lo"]) and within(qa["hi"], qb["hi"])
    pt, rng = (qa, qb) if qa["kind"] == "point" else (qb, qa)
    lo, hi = min(rng["lo"], rng["hi"]), max(rng["lo"], rng["hi"])
    return lo * (1 - rel_tol) <= pt["value"] <= hi * (1 + rel_tol)


def find_quantities(text):
    """Every quantity in a passage, for locating candidate evidence."""
    out = []
    for m in RANGE_RE.finditer(text or ""):
        q = parse_quantity(m.group(0))
        if q:
            out.append(q)
    seen_spans = {(m.start(), m.end()) for m in RANGE_RE.finditer(text or "")}
    for m in NUM_RE.finditer(text or ""):
        if any(s <= m.start() < e for s, e in seen_spans):
            continue
        q = parse_quantity(m.group(0))
        if q:
            out.append(q)
    return out


if __name__ == "__main__":
    cases = [
        ("54 µs", "0.054 ms", True),
        ("54 µs", "54 ms", False),
        ("90%", "90 percent", True),
        ("1.01-1.76x", "1.5x", True),
        ("1.01-1.76x", "2.4x", False),
        ("2 MB", "2048 KB", True),
        ("47%", "47 %", True),
        ("54 µs", "90%", False),
        ("170 ns", "0.17 µs", True),
    ]
    bad = 0
    for a, b, want in cases:
        got = quantities_match(a, b)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {a!r} vs {b!r} -> {got} (want {want})")
    print("all pass" if not bad else f"{bad} failures")
