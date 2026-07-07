"""Tool usage aggregation — deterministic per-manifest counts from events.

Design: ``docs/tool_substrate.md``. ``tool_used`` receipts ride the
canonical event rail; at epoch close every honest daemon aggregates them
with this module and gets bit-identical results (sorted iteration,
plain integer/float accumulation, rounded fee sums).

This is the consensus input for:
  - mint: author attribution ∝ standing × usage (reconcile side), and
  - attested-class standing decay: ``epochs_since_last_receipt`` is
    derived from which epochs contained receipts for a manifest.
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
          # Vetting tier (spec: Vetting section). Vets are NOT usage —
          # they are excluded from every count above and only accumulate
          # toward the manifest's greenlight in compute_tool_mint.
          # Only ok=True vets are recorded (ok=False is debate material
          # for the verdict layer, not a greenlight input).
          "vets_by_caller": {caller_id: count},
          "vet_senders": {caller_id: [sender_hex, ...]},
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
            }
        if ev.get("vet"):
            # Third flavor: a vet is a judgment about the CODE, not a
            # use of it — it must never inflate usage counts. Only
            # affirmative vets accumulate (greenlight input).
            if bool(ev.get("ok", True)):
                by = entry["vets_by_caller"]
                caller = str(ev.get("author_agent") or "")
                by[caller] = by.get(caller, 0) + 1
                sender = str(ev.get("sender") or "")
                senders = entry["vet_senders"].setdefault(caller, [])
                if sender and sender not in senders:
                    senders.append(sender)
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
