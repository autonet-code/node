"""Registry parameters the tool mint reads (Decisions 2026-11-07, 2026-11-09).

The close reads them from the DAO Registry's last writes pinned to the
snapshot block, so every daemon applies the same values. Unset or malformed
keys keep the defaults.
"""

from __future__ import annotations

from typing import Dict

# Breakthrough gate defaults (bps of REP supply, bps of REP-holding
# households): DAO-set as jurisdiction.<id>.mint_gate_rep_bps / _houses_bps.
# A tool concludes, and its malware verdict forms, once its reviewers clear
# both.
MINT_GATE_DEFAULTS = (100, 50)


def mint_gate_from_writes(writes: Dict[str, str], jurisdiction_id: str) -> tuple:
    out = list(MINT_GATE_DEFAULTS)
    for i, key in enumerate(("mint_gate_rep_bps", "mint_gate_houses_bps")):
        raw = writes.get(f"jurisdiction.{jurisdiction_id}.{key}")
        if raw in (None, ""):
            continue
        try:
            out[i] = max(0, int(str(raw).strip()))
        except ValueError:
            continue
    return tuple(out)


def review_params_from_writes(writes: Dict[str, str], jurisdiction_id: str) -> tuple:
    """(review_decay, reviewer_min_rep_bps) from the Registry
    (``jurisdiction.<id>.review_decay``, a factor in [0, 1] applied per
    close to a tool's carried review rounds; ``.reviewer_min_rep_bps``, the
    household REP a reviewer needs to be paid, as bps of REP supply).
    Unset or malformed keys keep the defaults (1.0, 0)."""
    decay, min_bps = 1.0, 0
    raw = writes.get(f"jurisdiction.{jurisdiction_id}.review_decay")
    if raw not in (None, ""):
        try:
            decay = max(0.0, min(1.0, float(str(raw).strip())))
        except ValueError:
            pass
    raw = writes.get(f"jurisdiction.{jurisdiction_id}.reviewer_min_rep_bps")
    if raw not in (None, ""):
        try:
            min_bps = max(0, min(10_000, int(str(raw).strip())))
        except ValueError:
            pass
    return decay, min_bps
