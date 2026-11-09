"""The fixed review fields (Decision 2026-11-09, docs/tool_substrate.md).

Every review of a tool carries the same four fields in every jurisdiction,
all required (a review missing one is invalid):

  - ``correctness`` -1..1: the tool works and is good at what it claims.
  - ``simplicity``   0..1: a nice-to-have; its factor in the tool's score
    runs from half (least simple) to full (simplest).
  - ``novelty``      0..1: how much of the tool's value is actually new
    beyond its dependencies and everything that already exists, in the
    library or outside it, as the reviewer finds it (not whether the
    author's novelty claim is right).
  - ``malware``      yes or no, by NIST SP 800-53 Rev. 5 "malicious code".

The charter axes are separate: the DAO's values, each optional per review.
An axis a review leaves unscored counts as ``UNSCORED_AXIS_DEFAULT`` (a
Registry value the DAO can change) with the review's weight.

On the wire (ToolBond) and in the close's review snapshot every value is in
hundredths: correctness and charter axes -100..100, simplicity and novelty
0..100, malware 0 or 100. A tool's or round's ``{"sum", "mass"}`` per field
is REP weight x value and REP weight, so ``sum / mass / 100`` is the
REP-weighted mean, and for malware the share of the reviewing REP that said
yes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

REVIEW_FIELDS: Tuple[str, ...] = ("correctness", "simplicity", "novelty", "malware")
SCORED_FIELDS: Tuple[str, ...] = ("correctness", "simplicity", "novelty")
FIELD_RANGES: Dict[str, Tuple[float, float]] = {
    "correctness": (-1.0, 1.0),
    "simplicity": (0.0, 1.0),
    "novelty": (0.0, 1.0),
}
UNSCORED_AXIS_DEFAULT = 0.5
MALWARE_VERDICT_SHARE = 0.5   # more than this share of the reviewing REP

MALWARE_DEFINITION = (
    "Software or firmware intended to perform an unauthorized process that "
    "will have adverse impacts on the confidentiality, integrity, or "
    "availability of a system (NIST SP 800-53 Rev. 5, \"malicious code\")")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def parse_fields(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``({correctness, simplicity, novelty, malware}, "")`` from a review's
    raw input, scores clamped to their ranges, or ``(None, why)`` when a
    field is missing or not a number (malware must be true or false)."""
    out: Dict[str, Any] = {}
    missing = []
    for name in SCORED_FIELDS:
        v = _number(raw.get(name))
        if v is None:
            missing.append(name)
            continue
        lo, hi = FIELD_RANGES[name]
        out[name] = max(lo, min(hi, v))
    malware = raw.get("malware")
    if not isinstance(malware, bool):
        missing.append("malware")
    else:
        out["malware"] = malware
    if missing:
        return None, ("every review needs correctness (-1..1), simplicity "
                      "(0..1), novelty (0..1) and malware (true or false); "
                      "missing or invalid: " + ", ".join(missing))
    return out, ""
