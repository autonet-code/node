"""Tool usage aggregation — deterministic per-manifest counts from events.

Design: ``docs/tool_substrate.md``. ``tool_used`` receipts ride the
canonical event rail; at epoch close every honest daemon aggregates them
with this module and gets bit-identical results (sorted iteration,
plain integer/float accumulation, rounded fee sums).

This is the consensus input for:
  - mint (v4.1 gradient trust, ratified 2026-07-09 —
    memory/tool_economy_v4_gradient_trust.md): author attribution =
    usage_term ONLY (the v2 standing multiplier and the v3 greenlight
    gate are both retired). ``compute_tool_mint`` collapses callers to
    households, damps once per household (log1p), and scales each
    household's credit by its MINT weight (raw reputation share for
    rep-holders, ε for zero-rep, the aggregate zero-rep ε-weight capped
    at a supply-pegged β). ATN and reputation mint at DECOUPLED amounts:
    zero-rep-weighted usage mints ATN but grants no reputation.
  - position drift: per-axis review scores (``axis_reviews_by_caller``)
    PLUS vet inspection reviews (``vet_axis_reviews_by_caller`` — a vet
    is now a review that moves position and mints nothing) feed the
    tool's drifted charter head, weighted by rep_share × credibility
    (no ε floor: zero-rep reviews move nothing).
"""

from __future__ import annotations

from typing import Any, Dict, List

_FEE_DECIMALS = 10


def tool_usage_from_events(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate ``tool_used`` events into per-manifest usage stats.

    Returns a canonically-ordered dict:

      {manifest_digest: {
          "count": int,             # all receipts (both tiers)
          "ok_count": int,          # successful invocations (both tiers)
          "fee_total": float,       # Σ fee_atn across receipts
          "tool_author": str,       # from the receipts (first sorted wins
                                    # on disagreement — consensus tie-break)
          "callers": {caller_id: count},
          # Cognitive-attestation tier (the only mint input — spec:
          # Attestation section). Keyed by attesting AGENT; the gossip
          # batch's signing key rides along per attestation ("senders")
          # solely for the wire-level self-attestation dedup in
          # compute_tool_mint — it is transport plumbing, never an
          # economic entity.
          "attested_ok_by_caller": {caller_id: count},
          "attester_senders": {caller_id: [sender_hex, ...]},
          # v3 per-axis review scores (spec Decision 2026-07-08).
          # Per attesting agent, per charter axis: sum + count of the
          # signed scores that agent submitted. compute_tool_mint damps
          # per caller (log1p) and applies the same exclusions as the
          # usage term, then folds the result into the tool's drifted
          # charter head. Axes an agent never scored are simply absent.
          "axis_reviews_by_caller": {caller_id: {axis_id: {"sum": float,
                                                           "n": int}}},
          # Vets are NOT usage — excluded from every count above. v4.1
          # (2026-07-09) retired the greenlight GATE, so these no longer
          # accumulate toward mint eligibility; the carried vetting state
          # is kept tolerantly (dead knobs) and a vet's surviving value is
          # its per-axis inspection review below. Only ok=True vets are
          # recorded (ok=False is verdict-layer material).
          "vets_by_caller": {caller_id: count},
          "vet_senders": {caller_id: [sender_hex, ...]},
          # v4.1 [R2]: an inspection review — a vet event that carries
          # per-axis scores. Same [-1,1]-clamped sum/count shape as
          # ``axis_reviews_by_caller``, but sourced from vet events, so it
          # moves position (drift) WITHOUT contributing to usage/mint. The
          # vet gate is gone (v4.1); these scores are the surviving value
          # of a vet. compute_tool_mint merges this map with
          # ``axis_reviews_by_caller`` at drift time.
          "vet_axis_reviews_by_caller": {caller_id: {axis_id: {"sum": float,
                                                              "n": int}}},
      }}

    Deterministic: events are processed in (author_agent, seq,
    receipt_digest) order regardless of input order, all maps are
    key-sorted, fee sums rounded to swamp IEEE jitter.
    """
    receipts = [e for e in events if e.get("kind") == "tool_used"
                and e.get("manifest_digest")]
    receipts.sort(key=lambda e: (e.get("author_agent", ""),
                                 e.get("seq", 0),
                                 e.get("receipt_digest", "")))

    usage: Dict[str, Dict[str, Any]] = {}
    for ev in receipts:
        digest = ev["manifest_digest"]
        entry = usage.get(digest)
        if entry is None:
            entry = usage[digest] = {
                "count": 0,
                "ok_count": 0,
                "fee_total": 0.0,
                "tool_author": str(ev.get("tool_author") or ""),
                "callers": {},
                "attested_ok_by_caller": {},
                "attester_senders": {},
                "axis_reviews_by_caller": {},
                "vets_by_caller": {},
                "vet_senders": {},
                "vet_axis_reviews_by_caller": {},
            }
        if ev.get("vet"):
            # Third flavor: a vet is a judgment about the CODE, not a
            # use of it — it must never inflate usage counts. Only
            # affirmative vets accumulate (greenlight input — DORMANT in
            # v4.1, kept as a rebuildable-cache field).
            if bool(ev.get("ok", True)):
                by = entry["vets_by_caller"]
                caller = str(ev.get("author_agent") or "")
                by[caller] = by.get(caller, 0) + 1
                sender = str(ev.get("sender") or "")
                senders = entry["vet_senders"].setdefault(caller, [])
                if sender and sender not in senders:
                    senders.append(sender)
                # v4.1 [R2]: a vet's per-axis scores are an INSPECTION
                # REVIEW — they drift position without minting. Same
                # clamp/accumulate shape as attested-usage reviews, kept
                # in a parallel map so mint accounting is untouched.
                raw_axes = ev.get("axes")
                if isinstance(raw_axes, dict) and raw_axes:
                    per_caller = entry["vet_axis_reviews_by_caller"].setdefault(
                        caller, {})
                    for axis_id in sorted(raw_axes):
                        try:
                            value = max(-1.0, min(1.0, float(raw_axes[axis_id])))
                        except (TypeError, ValueError):
                            continue
                        cell = per_caller.setdefault(
                            str(axis_id), {"sum": 0.0, "n": 0})
                        cell["sum"] += value
                        cell["n"] += 1
            continue
        entry["count"] += 1
        ok = bool(ev.get("ok", True))
        if ok:
            entry["ok_count"] += 1
        entry["fee_total"] += float(ev.get("fee_atn") or 0.0)
        caller = str(ev.get("author_agent") or "")
        entry["callers"][caller] = entry["callers"].get(caller, 0) + 1
        if ok and ev.get("attested"):
            by = entry["attested_ok_by_caller"]
            by[caller] = by.get(caller, 0) + 1
            sender = str(ev.get("sender") or "")
            senders = entry["attester_senders"].setdefault(caller, [])
            if sender and sender not in senders:
                senders.append(sender)
            # v3 per-axis review scores ride only attested-ok receipts.
            raw_axes = ev.get("axes")
            if isinstance(raw_axes, dict) and raw_axes:
                per_caller = entry["axis_reviews_by_caller"].setdefault(
                    caller, {})
                for axis_id in sorted(raw_axes):
                    try:
                        value = max(-1.0, min(1.0, float(raw_axes[axis_id])))
                    except (TypeError, ValueError):
                        continue
                    cell = per_caller.setdefault(
                        str(axis_id), {"sum": 0.0, "n": 0})
                    cell["sum"] += value
                    cell["n"] += 1

    for entry in usage.values():
        entry["fee_total"] = round(entry["fee_total"], _FEE_DECIMALS)
        entry["callers"] = dict(sorted(entry["callers"].items()))
        entry["attested_ok_by_caller"] = dict(
            sorted(entry["attested_ok_by_caller"].items()))
        entry["attester_senders"] = {
            k: sorted(v) for k, v in sorted(entry["attester_senders"].items())
        }
        entry["axis_reviews_by_caller"] = {
            caller: {
                axis: {"sum": round(cell["sum"], _FEE_DECIMALS),
                       "n": cell["n"]}
                for axis, cell in sorted(axes.items())
            }
            for caller, axes in sorted(
                entry["axis_reviews_by_caller"].items())
        }
        entry["vets_by_caller"] = dict(sorted(entry["vets_by_caller"].items()))
        entry["vet_senders"] = {
            k: sorted(v) for k, v in sorted(entry["vet_senders"].items())
        }
        entry["vet_axis_reviews_by_caller"] = {
            caller: {
                axis: {"sum": round(cell["sum"], _FEE_DECIMALS),
                       "n": cell["n"]}
                for axis, cell in sorted(axes.items())
            }
            for caller, axes in sorted(
                entry["vet_axis_reviews_by_caller"].items())
        }
    return dict(sorted(usage.items()))


def loadout_adoption_from_events(
    events: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate the ``loadout`` stamps on attested receipts.

    Returns (canonically ordered):
      {loadout_digest: {"by_caller": {caller: count},
                        "senders": {caller: [sender_hex, ...]}}}

    Adoption is derived at close as DISTINCT FLEETS (callers collapsed
    by the chain owner map — that happens in compute_tool_mint where
    the map lives); counts here are kept only so the evidence trail is
    inspectable. Same determinism rules as tool_usage_from_events.
    """
    receipts = [e for e in events if e.get("kind") == "tool_used"
                and e.get("attested") and e.get("ok", True)
                and e.get("loadout")]
    receipts.sort(key=lambda e: (e.get("author_agent", ""),
                                 e.get("seq", 0),
                                 e.get("receipt_digest", "")))
    out: Dict[str, Dict[str, Any]] = {}
    for ev in receipts:
        loadout = str(ev["loadout"])
        entry = out.setdefault(loadout, {"by_caller": {}, "senders": {}})
        caller = str(ev.get("author_agent") or "")
        entry["by_caller"][caller] = entry["by_caller"].get(caller, 0) + 1
        sender = str(ev.get("sender") or "")
        senders = entry["senders"].setdefault(caller, [])
        if sender and sender not in senders:
            senders.append(sender)
    for entry in out.values():
        entry["by_caller"] = dict(sorted(entry["by_caller"].items()))
        entry["senders"] = {k: sorted(v)
                            for k, v in sorted(entry["senders"].items())}
    return dict(sorted(out.items()))
