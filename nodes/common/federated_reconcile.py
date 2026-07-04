"""Federated reconciliation: bit-identical agent_mint maps across daemons.

Phase 5.4 of native world-model integration.

Wraps ``world_model_substrate.reconcile.reconcile_epoch`` with two
guarantees that the underlying function does NOT make on its own:

  1. **Deterministic iteration order**. The wrapped function iterates
     a Python ``set`` of node ids and ``dict.items()`` of attribution
     maps. Float addition is non-associative, so different iteration
     orders produce different sums. We override the iteration to be
     sorted (lexicographic on node_id, agent_id) so every honest
     daemon performs the *same sequence of float operations*.

  2. **Output normalization**. Even with sorted iteration, IEEE 754
     edge cases (subnormals, ulp-level jitter from cross-platform
     math libraries) can produce trivially-different float results.
     We round all output mint/novelty values to ``OUTPUT_DECIMALS``
     decimal places before returning. This is a normalization step,
     not a precision loss for the chain side: the ``ChainSubmitter``
     scales by 1e6 anyway.

Why a wrapper rather than editing reconcile.py
----------------------------------------------

``reconcile_epoch`` is used by the existing per-task substrate path
(``aggregator/main.py`` etc.) where bit-identicality is not required.
A wrapper preserves backward compat while letting the federated
codepath get the stricter guarantee.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World, equilibrate

from .canonical_ordering import CanonicalOrder
from .world_model_substrate.adapter import (
    _all_node_ids,
    _EventRecorder,
    build_charter_world,
)
from .world_model_substrate.aggregate import apply_events
from .world_model_substrate.mint_gate import (
    DEFAULT_CHARTER_IDS,
    charter_violation_score,
)
from .world_model_substrate.reconcile import (
    EpochSnapshots,
    _attribute_node_mint,
    _node_mint,
    _node_novelty,
    scale_node_agent_mint_by_violation,
    snapshot_node_scores_ordered,
)


logger = logging.getLogger(__name__)


# Number of decimal places we round mint/novelty outputs to.
# Empirically: 10 is far below any meaningful precision (substrate
# scores are typically O(10) to O(1000) and chain scaling truncates
# at 6 decimals), but well above IEEE 754 jitter (~15 sig figs).
OUTPUT_DECIMALS = 10


def compute_tool_mint(
    world: World,
    events: List[Dict[str, Any]],
    *,
    receipt_history: Optional[Dict[str, int]] = None,
    registrations: Optional[Dict[str, Dict[str, str]]] = None,
    decay_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Tool-author mint from canonical events (docs/tool_substrate.md).

    ``tool_mint(m) = max(0, effective_standing(m)) × log1p(ok_count(m))``
    per manifest with usage this epoch, attributed to the author carried
    on the registration sprout's ``manifest_meta`` — consensus data, so
    a blob-store miss can never fork the close.

    Cross-epoch state comes in as explicit params, both derived purely
    from prior epochs' canonical events (so bit-identical everywhere):
    ``receipt_history`` (digest → epochs since last receipt) and
    ``registrations`` (digest → manifest_meta, accumulated across
    epochs — a registration in epoch 1 still attributes mint in epoch
    5). A carried registration wins over a same-digest re-registration
    event: mint can never be re-attributed. Manifests known to neither
    mint nothing. Cross-epoch STANDING additionally requires seeding the
    close from the prior epoch's world (``seed_world``) so the manifest
    claim node persists; on a fresh-charter close only manifests
    (re-)sprouted this epoch have standing.

    Returns (all maps key-sorted, pure function of canonical inputs):
      {
        "per_digest": {digest: {node_id, author, trust_class, standing,
                                effective_standing, ok_count, mint}},
        "node_agent": {node_id: {author: mint}},   # gate-mergeable
        "receipt_history_next": {digest: epochs_since_last_receipt},
        "registrations_next": {digest: manifest_meta},
      }
    """
    import math

    from .world_model_substrate.infer import _artifact_standing, _standing_of
    from .world_model_substrate.tool_standing import (
        DEFAULT_ATTESTED_DECAY,
        effective_standing,
        update_receipt_history,
    )
    from .world_model_substrate.tool_usage import tool_usage_from_events

    history = dict(receipt_history or {})
    rate = DEFAULT_ATTESTED_DECAY if decay_rate is None else float(decay_rate)

    # Registrations: carried-over map first (immutable attribution),
    # then this epoch's sprouts in canonical author/seq order — a later
    # re-registration cannot re-attribute an existing digest.
    known_regs: Dict[str, Dict[str, str]] = dict(registrations or {})
    reg_events = [
        e for e in events
        if e.get("kind") == "sub_claim_sprouted"
        and e.get("artifact_digest") and e.get("manifest_meta")
    ]
    reg_events.sort(key=lambda e: (e.get("author_agent", ""), e.get("seq", 0)))
    for ev in reg_events:
        known_regs.setdefault(ev["artifact_digest"], dict(ev["manifest_meta"]))

    usage = tool_usage_from_events(events)
    by_digest = _artifact_standing(world)

    per_digest: Dict[str, Dict[str, Any]] = {}
    node_agent: Dict[str, Dict[str, float]] = {}
    for digest in sorted(set(usage) & set(known_regs)):
        meta = known_regs[digest]
        author = str(meta.get("author") or "")
        if not author:
            continue
        nodes = by_digest.get(digest, [])
        standing = _standing_of(nodes)
        eff = effective_standing(
            standing, str(meta.get("trust_class") or ""),
            history.get(digest, 0), decay_rate=rate,
        )
        ok_count = int(usage[digest]["ok_count"])
        mint = max(0.0, eff) * math.log1p(ok_count)
        if mint <= 0.0:
            continue
        # Anchor the mint on the manifest's claim node so the
        # violator-pays gate prices it: a won charter CON against the
        # tool suppresses exactly this share. Deterministic node pick:
        # lexicographically-first id among the digest's claim nodes.
        node_id = (sorted({n.id for n in nodes})[0] if nodes
                   else f"tool:{digest}")
        per_digest[digest] = {
            "node_id": node_id,
            "author": author,
            "trust_class": str(meta.get("trust_class") or ""),
            "standing": standing,
            "effective_standing": eff,
            "ok_count": ok_count,
            "mint": mint,
        }
        bucket = node_agent.setdefault(node_id, {})
        bucket[author] = bucket.get(author, 0.0) + mint

    history_next = update_receipt_history(
        history,
        used_digests=set(usage),
        known_digests=set(history) | set(known_regs),
    )
    return {
        "per_digest": dict(sorted(per_digest.items())),
        "node_agent": {k: dict(sorted(v.items()))
                       for k, v in sorted(node_agent.items())},
        "receipt_history_next": history_next,
        "registrations_next": dict(sorted(known_regs.items())),
    }


def federated_reconcile_epoch(
    world: World,
    snapshots: EpochSnapshots,
    events: List[Dict[str, Any]],
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    apply_gate: bool = True,
    gate_strength: float = 1.0,
    output_decimals: int = OUTPUT_DECIMALS,
    emission_pool: Optional[float] = None,
    extra_node_agent_mint: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Run reconcile_epoch deterministically.

    Same inputs (world topology, snapshots, events list, optional
    agent_weights) at every honest daemon produce **bit-identical**
    output dicts.

    ``emission_pool``: when set, post-gate mints are normalized to
    shares of this fixed pool (see reconcile.apply_emission_pool).
    MUST be derived from canonical data (e.g. emission_rate x the
    epoch duration measured between anchored timestamps), never from
    a daemon-local clock, or the bit-identical guarantee breaks.

    Returns the same shape as ``reconcile_epoch`` plus:
      - ``output_decimals``: the rounding precision used.
      - ``gate_applied``: whether the mint gate ran.
    """
    agent_weights = dict(agent_weights or {})

    # Discover all node ids that appear in any snapshot. Sort them so
    # the iteration order — and therefore the sequence of float adds
    # into agent_mint — is deterministic across daemons.
    all_node_ids: set[str] = set()
    all_node_ids.update(snapshots.start.keys())
    for cp in snapshots.checkpoints:
        all_node_ids.update(cp.keys())
    all_node_ids.update(snapshots.close.keys())
    sorted_node_ids = sorted(all_node_ids)

    node_mint: Dict[str, float] = {}
    node_novelty: Dict[str, float] = {}
    agent_mint: Dict[str, float] = {}
    agent_novelty: Dict[str, float] = {}
    # Per-(node, agent) mint breakdown, retained so the violator-pays
    # gate can scale a flagged node's mint before it aggregates into
    # per-agent totals. reconcile.py's uniform-ratio fallback discarded
    # this; keeping it is what lets a CON win redistribute correctly.
    node_agent_mint: Dict[str, Dict[str, float]] = {}

    for node_id in sorted_node_ids:
        nov = _node_novelty(node_id, snapshots)
        mint = _node_mint(node_id, snapshots, world=world)
        if nov > 0:
            node_novelty[node_id] = nov
        if mint > 0:
            node_mint[node_id] = mint

        # Attribute novelty proportionally over agents whose events
        # landed under this node. Sort the attribution dict so the
        # sum of float shares is associative-stable across daemons.
        attribution = _attribute_node_mint(node_id, 1.0, events, world)
        for agent in sorted(attribution.keys()):
            share = attribution[agent]
            agent_novelty[agent] = agent_novelty.get(agent, 0.0) + nov * share

        if mint > 0:
            mint_attribution = _attribute_node_mint(node_id, mint, events, world)
            per_agent: Dict[str, float] = {}
            for agent in sorted(mint_attribution.keys()):
                amount = mint_attribution[agent]
                weight = agent_weights.get(agent, 1.0)
                weighted = amount * weight
                agent_mint[agent] = agent_mint.get(agent, 0.0) + weighted
                per_agent[agent] = per_agent.get(agent, 0.0) + weighted
            node_agent_mint[node_id] = per_agent

    # Extra per-(node, agent) mint from non-score sources — currently
    # tool-author mint (compute_tool_mint). Merged BEFORE the gate so a
    # won charter CON against the anchoring node suppresses it like any
    # other mint, and before the pool so it competes for the same
    # emission. Sorted iteration keeps the float-add order canonical.
    for node_id in sorted((extra_node_agent_mint or {}).keys()):
        extra = extra_node_agent_mint[node_id]
        per_agent = node_agent_mint.setdefault(node_id, {})
        node_total = 0.0
        for agent in sorted(extra.keys()):
            amount = float(extra[agent])
            if amount <= 0.0:
                continue
            agent_mint[agent] = agent_mint.get(agent, 0.0) + amount
            per_agent[agent] = per_agent.get(agent, 0.0) + amount
            node_total += amount
        if node_total > 0.0:
            node_mint[node_id] = node_mint.get(node_id, 0.0) + node_total

    # Output normalization: round all per-agent and per-node values
    # to `output_decimals` places. This swamps any IEEE 754 jitter
    # without losing meaningful precision.
    agent_mint = _round_dict(agent_mint, output_decimals)
    agent_novelty = _round_dict(agent_novelty, output_decimals)
    node_mint = _round_dict(node_mint, output_decimals)
    node_novelty = _round_dict(node_novelty, output_decimals)

    # Re-sort dicts so JSON serialization (and dict equality with
    # other daemons') is order-stable. dict() in CPython preserves
    # insertion order, so building from a sorted iterator gives a
    # canonical layout.
    agent_mint = dict(sorted(agent_mint.items()))
    agent_novelty = dict(sorted(agent_novelty.items()))
    node_mint = dict(sorted(node_mint.items()))
    node_novelty = dict(sorted(node_novelty.items()))

    total_mint = round(sum(agent_mint.values()), output_decimals)
    total_novelty = round(sum(agent_novelty.values()), output_decimals)

    result = {
        "agent_mint": agent_mint,
        "agent_novelty": agent_novelty,
        "node_mint": node_mint,
        "node_novelty": node_novelty,
        "total_mint": total_mint,
        "total_novelty": total_novelty,
        "output_decimals": output_decimals,
        "gate_applied": False,
    }

    if apply_gate:
        # Violator-pays gate: scale each flagged node's per-agent mint
        # BEFORE aggregating, so suppression falls only on the flagged
        # node's authors (not uniformly across all agents like
        # apply_mint_gate's fallback). Combined with the emission pool
        # below, a winning CON redistributes the violator's mint to
        # everyone else.
        node_violation: Dict[str, float] = {}
        for node_id in sorted(node_agent_mint.keys()):
            node_violation[node_id] = charter_violation_score(
                world, node_id, DEFAULT_CHARTER_IDS,
            )
        gated_agent_mint = scale_node_agent_mint_by_violation(
            node_agent_mint, node_violation, gate_strength=gate_strength,
        )
        # Recompute the gated per-node mint (for diagnostics + pool
        # scaling of node_mint), scaling each node by its violation.
        gated_node_mint: Dict[str, float] = {}
        for node_id in sorted(node_mint.keys()):
            v = node_violation.get(node_id, 0.0)
            scale = max(0.0, 1.0 - gate_strength * v)
            gm = node_mint[node_id] * scale
            if gm > 0:
                gated_node_mint[node_id] = gm

        result["agent_mint"] = dict(sorted(
            _round_dict(gated_agent_mint, output_decimals).items()
        ))
        result["node_mint"] = dict(sorted(
            _round_dict(gated_node_mint, output_decimals).items()
        ))
        result["node_violation"] = dict(sorted(node_violation.items()))
        result["total_mint"] = round(
            sum(result["agent_mint"].values()), output_decimals,
        )
        result["gate_applied"] = True

    # Fixed-emission normalization — AFTER the gate, so mint suppressed
    # by debate is redistributed to the surviving contributors. Scaling
    # runs over sorted keys and is re-rounded, so determinism holds.
    if emission_pool is not None:
        from .world_model_substrate.reconcile import apply_emission_pool
        result = apply_emission_pool(result, emission_pool)
        result["agent_mint"] = dict(sorted(
            _round_dict(result.get("agent_mint", {}), output_decimals).items()
        ))
        result["node_mint"] = dict(sorted(
            _round_dict(result.get("node_mint", {}), output_decimals).items()
        ))
        result["total_mint"] = round(
            sum(result["agent_mint"].values()), output_decimals,
        )

    logger.info(
        "federated reconciliation: %d nodes, %d agents, "
        "total mint %.6f, total novelty %.6f",
        len(result["node_mint"]),
        len(result["agent_mint"]),
        result["total_mint"],
        result["total_novelty"],
    )
    return result


def federated_epoch_close(
    canonical: CanonicalOrder,
    *,
    seed_world: Optional[World] = None,
    bandwidth: float = 1.5,
    embedding_dim: int = 1024,
    apply_gate: bool = True,
    gate_strength: float = 1.0,
    output_decimals: int = OUTPUT_DECIMALS,
    equilibrate_rounds: int = 8,
    equilibrate_tolerance: float = 1e-3,
    emission_pool: Optional[float] = None,
    pricing: str = "ledger",
    tool_receipt_history: Optional[Dict[str, int]] = None,
    tool_registrations: Optional[Dict[str, Dict[str, str]]] = None,
    tool_decay_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a full federated epoch close given a canonical sequence.

    Builds a throwaway replay world from a fresh charter (or from the
    supplied ``seed_world`` if continuing from a prior epoch's
    authoritative state), replays the canonical batch sequence, and
    runs federated_reconcile_epoch.

    ``pricing`` selects how per-node scores are derived:

      - ``"ledger"`` (default, post-phase8): replay applies causal
        events only — NO equilibrate rounds, NO derived-sprout capture.
        A node's score is the ``net_score`` tree recursion (posts +
        signed pro/con children). Score moves only when causal events
        land on/under the node during the epoch. The O(N^2) equilibrate
        leaves the hot path. Cycle memoization is made deterministic by
        evaluating net_score in sorted node-id order (see
        ``snapshot_node_scores_ordered``).

      - ``"equilibrated"``: the pre-phase8 experimental kernel. Replays
        each batch with per-batch equilibration and captures the
        derived sprouts it produces, attributing them to the batch's
        author. Preserved BIT-FOR-BIT for the experimental kernel.

    The returned ``result`` is bit-identical across daemons that
    received the same canonical sequence — which by 5.3's contract
    they will, given the same set of batches.

    Notes
    -----

    Phase 5.4 builds the replay world from a **fresh charter**, which
    is correct as long as epochs don't carry any persistent state
    forward. Phase 5.5+ may want to seed from the prior authoritative
    epoch's snapshot (so cross-epoch lineage works); that's why
    ``seed_world`` is exposed.
    """
    if pricing not in ("ledger", "equilibrated"):
        raise ValueError(f"unknown pricing mode: {pricing!r}")

    if seed_world is None:
        world = build_charter_world(
            bandwidth=bandwidth, embedding_dim=embedding_dim,
        )
    else:
        world = seed_world

    snapshots = EpochSnapshots()

    if pricing == "ledger":
        # Ledger replay: causal events only, no equilibration, no
        # derived-sprout capture. Snapshots use sorted-order net_score
        # so co-parented-cycle memoization is deterministic.
        #
        # Attribution needs each causal sprout event's node_id to be the
        # LIVE (content-addressed) id, but the wire carries solver-side
        # labels. We capture the replay remap and rewrite the events so
        # ``_events_under_node`` (which matches raw event node_ids
        # against live descendant ids) attributes to the real author.
        snapshots.start = snapshot_node_scores_ordered(world)
        all_events: List[Dict[str, Any]] = []
        for batch in canonical.ordered_batches:
            if not batch.events:
                continue
            remap: Dict[str, str] = {}
            apply_events(
                world, batch.events,
                equilibrate_after=False, remap_out=remap,
            )
            # Build attribution copies with node_id rewritten to the
            # live id. We do NOT mutate batch.events (the canonical log
            # is fed downstream to the state-sync tracker verbatim).
            for ev in batch.events:
                if ev.get("kind") == "sub_claim_sprouted":
                    live = remap.get(ev.get("node_id", ""))
                    if live and live != ev.get("node_id"):
                        ev = dict(ev)
                        ev["solver_node_id"] = ev.get("node_id")
                        ev["node_id"] = live
                all_events.append(ev)
        snapshots.close = snapshot_node_scores_ordered(world)
    else:
        # Equilibrated (experimental) kernel — unchanged behavior.
        snapshots.record_start(world)
        # Replay each batch as its own apply_events + equilibrate, so
        # equilibration history matches what would have happened with
        # in-order live submission.
        #
        # Attribution gotcha (Phase 5.4): the gossip wire format carries
        # only causal events. Reconcile, however, attributes mint by
        # finding sub_claim_sprouted events under each minting node —
        # which means it needs the *derived* sprouts that equilibration
        # produced too. We capture them here and attribute them to the
        # batch's author, so attribution reproduces deterministically.
        #
        # Determinism: cross-tendency discovery and equilibration are
        # deterministic given world state, so two daemons replaying the
        # same canonical sequence produce the same derived events.
        all_events = []
        for batch in canonical.ordered_batches:
            # Author of this batch's derived events = whoever signed the
            # batch. We use sender_pubkey hex as a stable agent identity.
            # If batch.events carry an explicit author_agent, prefer that
            # (the daemon may run on behalf of multiple agents).
            author = _author_for_batch(batch)
            before_ids = _all_node_ids(world)
            if batch.events:
                apply_events(world, batch.events)
            equilibrate(
                world,
                max_rounds=equilibrate_rounds,
                tolerance=equilibrate_tolerance,
            )
            # Capture derived sprouts post-equilibrate, attributed to
            # this batch's author. The event recorder uses its own
            # sequence numbering; we offset to keep events globally
            # unique-ish in the all_events list (reconcile only matches
            # by node_id and position, not seq).
            recorder = _EventRecorder(agent_id=author)
            recorder.sub_claims_after_equilibrate(world, before_ids)
            derived = [e.to_dict() for e in recorder.events]

            all_events.extend(batch.events)
            all_events.extend(derived)

        snapshots.record_close(world)

    # Tool substrate: author mint ∝ standing × usage, anchored on the
    # manifest's claim node so the violator-pays gate prices it
    # (docs/tool_substrate.md). Pure function of the replayed world +
    # canonical events + explicitly-passed receipt history, so the
    # bit-identical guarantee holds.
    tool_result = compute_tool_mint(
        world, all_events,
        receipt_history=tool_receipt_history,
        registrations=tool_registrations,
        decay_rate=tool_decay_rate,
    )

    result = federated_reconcile_epoch(
        world, snapshots, all_events,
        apply_gate=apply_gate,
        gate_strength=gate_strength,
        output_decimals=output_decimals,
        emission_pool=emission_pool,
        extra_node_agent_mint=tool_result["node_agent"],
    )
    result["tool_mint"] = tool_result["per_digest"]
    result["tool_receipt_history"] = tool_result["receipt_history_next"]
    result["tool_registrations"] = tool_result["registrations_next"]
    result["epoch_root"] = canonical.epoch_root().hex()
    result["n_batches"] = len(canonical.ordered_batches)
    result["n_events"] = len(all_events)
    result["scope"] = "federated"
    result["authoritative"] = True
    result["pricing"] = pricing

    # The on-chain anchor (Phase 5.5) commits the merkle of an
    # ``authoritative_payload`` dict. Only fields whose keys are
    # deterministic across daemons can live here. Node ids are now
    # fully deterministic (content-addressed children; roots are
    # ``root_<tendency_id>``), but per-node score maps are unbounded
    # in size, so the chain payload still carries only ``agent_mint``
    # + ``agent_novelty`` (keyed by user-supplied agent_id) plus
    # ``epoch_root``. Per-node fields stay in ``result`` for local
    # diagnostics only.
    result["authoritative_payload"] = {
        "schema": 1,
        "epoch_root": result["epoch_root"],
        "agent_mint": result["agent_mint"],
        "agent_novelty": result["agent_novelty"],
        "total_mint": result["total_mint"],
        "total_novelty": result["total_novelty"],
        "output_decimals": result.get("output_decimals", output_decimals),
        "gate_applied": result.get("gate_applied", False),
        "pricing": pricing,
        "n_batches": result["n_batches"],
        "n_events": result["n_events"],
    }
    return result


def _author_for_batch(batch) -> str:
    """Determine which agent_id derived events should be attributed to.

    Preference order:
      1. The author_agent of the first observation_added event in the
         batch (the batch was made on behalf of that user-supplied id).
      2. The sender_pubkey hex (a stable cryptographic identity, used
         when the batch carries no causal observation events).

    Determinism: every honest daemon picks the same author given the
    same canonical batch, so derived-event attribution reproduces.
    """
    for ev in batch.events:
        if ev.get("kind") == "observation_added":
            author = ev.get("author_agent")
            if author:
                return str(author)
    # Fall back: derive a stable identity from the signer.
    return f"pk:{batch.sender_pubkey.hex()[:16]}"


def _round_dict(d: Dict[str, float], decimals: int) -> Dict[str, float]:
    """Round every value in a {str: float} dict to ``decimals`` places.
    Drops zero values so empty contributions don't pollute the map."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        rv = round(float(v), decimals)
        if rv != 0.0:
            out[k] = rv
    return out
