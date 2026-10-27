"""Consensus-tiered containment: the pure tier rule (Decision 2026-10-27).

One copy shared by the daemon (atn/tool_security.py, which tool runs may do
what) and the federated close (a tool the network denied mints nothing).
The close reads the parameters from the DAO Registry pinned to the snapshot
block, so every daemon applies the same thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

CONTAINMENT_DENIED = "denied"
CONTAINMENT_STRICT = "strict"
CONTAINMENT_DECLARED = "declared"
# trust_mass_bps 10 (0.1% of REP supply), not 200: at 2% the GPU sim denied
# 0 of 933 harmful tools at 100k households (experiments/econ_gpu REPORT 3.4).
CONTAINMENT_DEFAULTS = {"deny_harm": 50.0, "trust_harm": 10.0,
                        "trust_mass_bps": 10.0}
# Breakthrough gate defaults (bps of REP supply, bps of REP-holding
# households): DAO-set as jurisdiction.<id>.mint_gate_rep_bps / _houses_bps.
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


def tier_for_standing(standing: Optional[Dict[str, Any]],
                      params: Dict[str, float]) -> tuple[str, str]:
    """Pure tier rule over ``{harm_sum, harm_mass, supply}`` (REP units)."""
    if not standing or not int(standing.get("harm_mass") or 0):
        return CONTAINMENT_STRICT, "not yet reviewed for harm"
    mass = int(standing["harm_mass"])
    harm = int(standing.get("harm_sum") or 0) / mass
    supply = int(standing.get("supply") or 0)
    share_bps = mass * 10_000 / supply if supply > 0 else 0.0
    quorate = share_bps >= params["trust_mass_bps"]
    if harm >= params["deny_harm"] and quorate:
        return CONTAINMENT_DENIED, f"settled harm {harm:.0f}/100"
    if harm < params["trust_harm"] and quorate:
        return CONTAINMENT_DECLARED, f"reviewed safe (harm {harm:.0f}/100)"
    if not quorate:
        return CONTAINMENT_STRICT, (f"harm {harm:.0f}/100 from too little REP "
                                    f"({share_bps:.0f} bps of supply)")
    return CONTAINMENT_STRICT, f"contested (harm {harm:.0f}/100)"


def params_from_writes(writes: Dict[str, str], jurisdiction_id: str) -> Dict[str, float]:
    """The thresholds in force, from the Registry's last writes at a block
    (``jurisdiction.<id>.containment_*``); unset or malformed keys keep the
    defaults."""
    out = dict(CONTAINMENT_DEFAULTS)
    for key in out:
        raw = writes.get(f"jurisdiction.{jurisdiction_id}.containment_{key}")
        if raw in (None, ""):
            continue
        try:
            out[key] = float(str(raw).strip())
        except ValueError:
            continue
    return out


def denied_digests(review_snapshot: Optional[Dict[str, Any]], supply: int,
                   params: Dict[str, float]) -> list[str]:
    """Tools the snapshot denies: settled harm, finalized or still in its
    appeal window, over enough REP (the same standing the daemon contains
    by)."""
    snap = review_snapshot or {}
    tools = snap.get("tools") or {}
    cooling = snap.get("cooling_harm") or {}
    out = []
    for digest in sorted(set(tools) | set(cooling)):
        harm = (tools.get(digest) or {}).get("harm") or {}
        cool = cooling.get(digest) or {}
        standing = {"harm_sum": int(harm.get("sum") or 0) + int(cool.get("sum") or 0),
                    "harm_mass": int(harm.get("mass") or 0) + int(cool.get("mass") or 0),
                    "supply": int(supply or 0)}
        if tier_for_standing(standing, params)[0] == CONTAINMENT_DENIED:
            out.append(str(digest).lower().removeprefix("0x"))
    return out
