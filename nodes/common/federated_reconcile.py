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
from typing import Any, Dict, List, Optional, Tuple

from world_model.generalized import World, equilibrate

from .canonical_ordering import CanonicalOrder
from .world_model_substrate.adapter import (
    _all_node_ids,
    _EventRecorder,
    build_charter_world,
)
from .world_model_substrate.aggregate import apply_events
from .world_model_substrate.charter import active_charter
from .world_model_substrate.mint_gate import (
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

# Composition fan-out (docs/tool_substrate.md, Composition section):
# one attestation of a composite carries total weight 1 — the root
# keeps ROOT_SHARE, the remainder splits equally among declared deps,
# recursively to MAX_DEPTH. Cycles and unregistered deps FORFEIT their
# share (total may be < 1, never > 1): conservation of attestation.
COMPOSITE_ROOT_SHARE = 0.7
COMPOSITE_MAX_DEPTH = 4
# Reviewed composition split (Decision 2026-10-13 §9): COMPOSITE_ROOT_SHARE
# is now only the DEFAULT. Attestations may carry ``own_share`` in [0,1]
# (how much of a composite's value is its own logic vs its declared deps); a
# composite's root share is the rep-share-weighted running mean of those
# estimates, carried in its ``tool_positions`` entry as ``own_share: {sum,
# mass}``. No own_share estimates => the default.

# Reviewer share of a tool's mint (docs/multi_jurisdiction.md decision 10):
# the reviewers of the tool's rounds finalized since the previous anchor
# split this share by stake x accuracy x novelty; the author keeps the rest
# (100% when no such round has a weighted reviewer). The live value is
# ToolBond ``reviewerShareBps``, read pinned into the review snapshot.
REVIEWER_SHARE_BPS_DEFAULT = 4000
BPS = 10_000

# Vetting pipeline (docs/tool_substrate.md, Vetting section). A published
# manifest enters the CANDIDATE pool: visible, debatable, NOT mint-
# eligible. GREENLIGHT = affirmative vets from distinct fleets whose
# summed weights reach VET_QUORUM; validators then earn a conserved
# royalty slice of the tool's mint for VET_ROYALTY_EPOCHS closes — the
# royalty IS the stake: a post-greenlight charter bust (violation ≥
# VET_BUST_THRESHOLD on the manifest's claim node) zeroes the remaining
# royalty and permanently discounts each validator's future vet weight
# (weight = 1/(1+busts) — slashing without a staking contract).
# PROVISIONAL VALUES: quorum/share/epochs/threshold are economic
# parameters pending sim sweep + user blessing (sims/tool_economy).
VET_QUORUM = 2.0
VET_ROYALTY_SHARE = 0.1
VET_ROYALTY_EPOCHS = 8
VET_BUST_THRESHOLD = 0.5

# Household voice (docs/tool_substrate.md, Decision 2026-07-08 addendum,
# voice SOURCE = reputation not balance per the same-day "ATN = money,
# reputation = voice" decision). A "household" is the economic identity
# behind a caller: its registered owner wallet (proven on-chain by the
# EIP-712 owner binding), falling back to the agent id when unbound.
# Usage and review mass collapse per household BEFORE damping (log1p once
# per household — N agents under one owner are one voice, closing the
# per-agent log1p amplification), then multiply by the household's voice
# weight: epsilon + household_REPUTATION / rep_supply, LINEAR in
# reputation so splitting earnings across wallets never gains weight.
# VOICE_EPSILON is the floor for households the weight map doesn't know —
# it bounds what a zero-reputation identity can contribute on the MINT
# rail and lets a cold-start network (supply = 0) bootstrap.
#
# fees-only + REP-from-earnings (Decision 2026-07-10, docs/tool_substrate.md
# §5): the supply-pegged β cap is DELETED (tool users are spenders, spenders
# never earn REP, so honest usage is all zero-rep by construction and any β
# throttle on zero-rep weight zeroes honest author income). The ε MINT-weight
# floor survives — REP dilution, not β, is the youth defense now, and the sim
# validated the model with ε present. v4.1 (2026-07-09) drops this ε floor
# from DRIFT weight (drift = raw rep_share × credibility); it survived only
# here, on the mint rail.
#
# ε = 0 (2026-10-17, docs/multi_jurisdiction.md decision 8): a household's
# mint weight is its REP share, no floor. A fake owner wallet is free, so a
# floor paid every fake household for farming its own tools. Newcomers
# still earn as authors when REP holders use their tools, and earn REP by
# staking accurate reviews (decision 10). A DAO starts with genesis REP, so
# the weights are never all zero.
VOICE_EPSILON = 0.0

# ---------------------------------------------------------------------------
# Tool economy v4.1 "gradient trust" (ratified 2026-07-09) as amended by
# fees-only + REP-from-earnings (Decision 2026-07-10). The v3 vet GATE is
# removed (tools mint from first attested use); the surviving trust surface
# is (a) zero-rep reviews carry ZERO drift weight and (b) continuous
# reversal-aware credibility multiplies drift weight. The supply-pegged β
# cap and the rep/ATN mint split are DELETED (2026-07-10): the close
# computes MONEY ONLY — REP is a pure function of ratified earnings,
# claimed DAO-side (RepToken), never minted on the ATN close path.
# ---------------------------------------------------------------------------

# Staked reviews (docs/multi_jurisdiction.md decision 10, 2026-10-17)
# replaced the gossip review drift and the off-chain reversal-aware
# credibility: ToolBond is the single source of ratings. A wrong review
# costs REP on chain in proportion to how far off it was, so REP is the
# reputation and there is no separate credibility number.


# Fees-only emission (docs/tool_substrate.md, Decision 2026-07-10 §1):
#   pool(N) = recycled(N)
# where recycled(N) = Σ ServiceFee.burned between the previous two anchors
# (read on-chain by voice_state.read_voice_state, same pinned rail as voice
# weights). No base emission: ATN mints each epoch EXACTLY what fees burned
# that epoch — money is always demand-backed. Zero service volume => zero
# pool => zero mint, and that is correct behavior (the economy does not
# pretend to exist before demand does). Conservation by construction
# (Σ minted == Σ burned); wash-pumping the pool is a strict ATN loss. The
# fee/treasury split lives on-chain (Substrate.sol serviceFeeBps).


def _normalize_vetting(vetting: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonicalize carried-over vetting state (sorted, typed, copied).

    Shape: {"manifests": {digest: {vets: {agent: [sender,...]},
    greenlit, royalty_left, validators, busted}}, "busts": {agent: n}}.
    Derived purely from prior epochs' canonical events + world state,
    so — like ``registrations`` — the persisted copy is a rebuildable
    cache and identical on every honest daemon.
    """
    state: Dict[str, Any] = {"manifests": {}, "busts": {}}
    if not vetting:
        return state
    for digest, m in sorted((vetting.get("manifests") or {}).items()):
        state["manifests"][str(digest)] = {
            "vets": {str(a): sorted(str(s) for s in senders)
                     for a, senders in sorted((m.get("vets") or {}).items())},
            "greenlit": bool(m.get("greenlit")),
            "royalty_left": int(m.get("royalty_left") or 0),
            "validators": sorted(str(v) for v in (m.get("validators") or [])),
            "busted": bool(m.get("busted")),
        }
    state["busts"] = {str(k): int(v)
                      for k, v in sorted((vetting.get("busts") or {}).items())}
    return state


def _composition_shares(
    root: str,
    deps_of: Dict[str, List[str]],
    *,
    root_share: float = COMPOSITE_ROOT_SHARE,
    max_depth: int = COMPOSITE_MAX_DEPTH,
    root_shares: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Split one unit of attestation weight over a declared-dep DAG.

    ``root_shares`` (digest -> reviewed own share in [0,1]) overrides
    ``root_share`` per composite at every level of the recursion: a
    composite dep keeps ITS reviewed share of whatever reaches it.

    Pure + deterministic (sorted recursion over consensus-carried
    declarations). Returns {digest: weight} with sum <= 1.0.
    """
    shares: Dict[str, float] = {}
    reviewed = root_shares or {}

    def _spread(digest: str, weight: float, depth: int, path: frozenset) -> None:
        declared = [d for d in deps_of.get(digest, []) if d]
        if not declared or depth >= max_depth:
            shares[digest] = shares.get(digest, 0.0) + weight
            return
        own = max(0.0, min(1.0, float(reviewed.get(digest, root_share))))
        shares[digest] = shares.get(digest, 0.0) + weight * own
        # Split by the FULL declared count; shares of cyclic or
        # unregistered deps are forfeited, never redistributed to
        # siblings (conservation: total <= 1, gaming by padding with
        # dead deps only burns the padder's own credit).
        child_w = weight * (1.0 - own) / len(declared)
        for dep in sorted(declared):
            if dep in path or dep not in deps_of:
                continue
            _spread(dep, child_w, depth + 1, path | {digest})

    _spread(root, 1.0, 0, frozenset({root}))
    return shares


def _normalize_review_snapshot(
    snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Canonicalize the ToolBond review snapshot (docs/multi_jurisdiction.md
    decision 10; read by voice_state.read_tool_review_snapshot pinned to the
    previous anchor, the close only consumes it).

    Shape: {"reviewer_share_bps": int, "supply": int (REP at the snapshot),
    "tools": {digest: {"axes": {axis_id: {"sum", "mass"}}, "harm": {"sum",
    "mass"}}}, "rounds": {digest: [{"epoch", "supply", "prior_mass",
    "reviews": [{"reviewer", "stake", "accuracy", "axes", "harm"}]}]}}.
    Integers stay integers (REP and ATN base units; sums are REP x score in
    -100..100 units, so a mean is sum / mass / 100). Digests lose any 0x,
    addresses are lowercased, bps clamps to [0, BPS], garbage is dropped.
    """
    out: Dict[str, Any] = {"reviewer_share_bps": REVIEWER_SHARE_BPS_DEFAULT,
                           "supply": 0, "tools": {}, "rounds": {}}
    if not snapshot:
        return out

    def _int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    out["reviewer_share_bps"] = max(0, min(BPS, _int(
        snapshot.get("reviewer_share_bps"), REVIEWER_SHARE_BPS_DEFAULT)))
    out["supply"] = max(0, _int(snapshot.get("supply")))
    tools: Dict[str, Any] = {}
    for digest, entry in sorted((snapshot.get("tools") or {}).items()):
        if not isinstance(entry, dict):
            continue
        axes = {}
        for axis_id, cell in sorted((entry.get("axes") or {}).items()):
            if isinstance(cell, dict) and _int(cell.get("mass")) > 0:
                axes[str(axis_id)] = {"sum": _int(cell.get("sum")),
                                      "mass": _int(cell.get("mass"))}
        harm = entry.get("harm") or {}
        tools[str(digest).lower().removeprefix("0x")] = {
            "axes": axes,
            "harm": {"sum": max(0, _int(harm.get("sum"))),
                     "mass": max(0, _int(harm.get("mass")))},
        }
    out["tools"] = dict(sorted(tools.items()))
    rounds: Dict[str, List[Dict[str, Any]]] = {}
    for digest, rs in sorted((snapshot.get("rounds") or {}).items()):
        norm_rs = []
        for rnd in rs or []:
            if not isinstance(rnd, dict):
                continue
            revs = []
            for v in rnd.get("reviews") or []:
                if not isinstance(v, dict) or not v.get("reviewer"):
                    continue
                acc = v.get("accuracy")
                revs.append({
                    "reviewer": str(v["reviewer"]).lower(),
                    "stake": max(0, _int(v.get("stake"))),
                    "accuracy": None if acc is None else max(0, min(10 ** 18, _int(acc))),
                    "axes": sorted(str(a) for a in (v.get("axes") or [])),
                    "harm": bool(v.get("harm")),
                })
            norm_rs.append({
                "epoch": _int(rnd.get("epoch")),
                "supply": max(0, _int(rnd.get("supply"))),
                "prior_mass": {str(k): max(0, _int(m)) for k, m in
                               sorted((rnd.get("prior_mass") or {}).items())},
                "reviews": revs,
            })
        if norm_rs:
            rounds[str(digest).lower().removeprefix("0x")] = sorted(
                norm_rs, key=lambda r: r["epoch"])
    out["rounds"] = rounds
    if isinstance(snapshot.get("bonds"), dict):
        bonds: Dict[str, Dict[str, Any]] = {}
        for digest, b in sorted(snapshot["bonds"].items()):
            if not isinstance(b, dict) or not b.get("author"):
                continue
            bonds[str(digest).lower().removeprefix("0x")] = {
                "author": str(b["author"]).lower(),
                "stake": max(0, _int(b.get("stake"))),
                "unbonding": bool(b.get("unbonding")),
            }
        out["bonds"] = bonds
        out["min_stake"] = max(0, _int(snapshot.get("min_stake")))
    return out


def reviewer_weights(rounds: List[Dict[str, Any]],
                     fallback_supply: int = 0) -> Dict[str, float]:
    """Each reviewer's claim on a tool's reviewer share, over the rounds of
    it finalized in this close's window (decision 10):

        weight = accuracy x novelty   (stake > 0 required, not weighed:
                                       the stake is the bet, 2026-10-28),
        novelty = mean over the axes the review scored (harm counting as
                  one) of 1 / (1 + M / supply),

    M = the tool's REP mass on that axis BEFORE the round (the rounds
    finalized earlier), supply = the REP supply at the round's epoch start
    (the close's snapshot supply when the epoch reports none). Accuracy is
    ToolBond's settled 1e18-scale value; an uninformative review (None) and
    a zero stake weigh nothing. Pure and deterministic (sorted keys)."""
    out: Dict[str, float] = {}
    for rnd in rounds:
        supply = int(rnd.get("supply") or 0) or int(fallback_supply or 0)
        prior = rnd.get("prior_mass") or {}
        for v in rnd.get("reviews") or []:
            acc = v.get("accuracy")
            stake = int(v.get("stake") or 0)
            if acc is None or stake <= 0 or int(acc) <= 0:
                continue
            scored = list(v.get("axes") or []) + (["harm"] if v.get("harm") else [])
            if not scored:
                continue
            nov = 0.0
            for axis in sorted(scored):
                m = int(prior.get(axis, 0))
                nov += 1.0 / (1.0 + (m / supply if supply > 0 else 0.0))
            nov /= len(scored)
            # Accuracy alone, not stake: weighting by ATN staked let money buy
            # REP through reviews (GPU sim, experiments/econ_gpu REPORT 3).
            w = (int(acc) / 10 ** 18) * nov
            if w > 0:
                key = str(v["reviewer"]).lower()
                out[key] = out.get(key, 0.0) + w
    return dict(sorted(out.items()))


def tool_rating_position(entry: Optional[Dict[str, Any]], axis_ids: List[str],
                         supply: int) -> Dict[str, Any]:
    """A ToolBond rating as the close's tool position: per charter axis
    head = sum / mass / 100 (the REP-weighted mean, -1..1) and mass = REP
    mass / REP supply; harm the same way (0..1). Unrated axes sit at 0 with
    0 mass."""
    head: List[float] = []
    mass: List[float] = []
    axes = (entry or {}).get("axes") or {}
    for axis_id in axis_ids:
        cell = axes.get(axis_id) or {}
        m = int(cell.get("mass") or 0)
        head.append(round(int(cell.get("sum") or 0) / m / 100.0, 9) if m > 0 else 0.0)
        mass.append(round(m / supply, 9) if supply > 0 else 0.0)
    harm = (entry or {}).get("harm") or {}
    hm = int(harm.get("mass") or 0)
    return {
        "head": head,
        "mass": mass,
        "harm": round(int(harm.get("sum") or 0) / hm / 100.0, 9) if hm > 0 else 0.0,
        "harm_mass": round(hm / supply, 9) if supply > 0 else 0.0,
    }


def compute_tool_mint(
    world: World,
    events: List[Dict[str, Any]],
    *,
    registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    voice_weights: Optional[Dict[str, float]] = None,
    vetting: Optional[Dict[str, Any]] = None,
    positions: Optional[Dict[str, Dict[str, Any]]] = None,
    # Raw REP share per household: weighs the own_share estimates only.
    rep_shares: Optional[Dict[str, float]] = None,
    # Staked reviews (docs/multi_jurisdiction.md decision 10): the pinned
    # ToolBond review snapshot (ratings + the reviewer-share rounds).
    review_snapshot: Optional[Dict[str, Any]] = None,
    # The jurisdiction's gossip domain: attested receipts must be signed by
    # the agent they name for it (nodes/common/tool_receipts.py). None (no
    # chain, tests) = receipts unchecked.
    receipt_domain: Optional[str] = None,
    # Tools the network denied at the snapshot: they mint nothing.
    denied_tools: Optional[List[str]] = None,
    # Sanctioned households: their usage credits no tool.
    sanctioned_households: Optional[List[str]] = None,
    # Breakthrough gate (Decision 2026-10-27, sized 2026-10-28 by the GPU
    # sim): (rep_bps, houses_bps). A tool mints only when this epoch's users
    # hold >= rep_bps of REP supply and are >= houses_bps of the REP-holding
    # households. None, or no rep_shares (no chain), = no gate.
    mint_gate: Optional[Tuple[int, int]] = None,
    vet_quorum: float = VET_QUORUM,
    vet_royalty_share: float = VET_ROYALTY_SHARE,
    vet_royalty_epochs: int = VET_ROYALTY_EPOCHS,
    vet_bust_threshold: float = VET_BUST_THRESHOLD,
) -> Dict[str, Any]:
    """Tool-author mint from canonical events (docs/tool_substrate.md v3).

    ``tool_mint(m) = usage_term(m)`` per PINNED manifest (v3, Decision
    2026-07-08: usage alone mints — the v2 standing multiplier is
    retired; reviews rank discovery and drift position instead),
    attributed to the author carried on the registration sprout's
    ``manifest_meta`` — consensus data, so a blob-store miss can never
    fork the close. Mint is pinned-only: attested (connector-backed)
    tools are publishable and reviewable but draw no emission, and
    remote services never enter the substrate at all
    (docs/services_market.md).

    ``usage_term`` is the combo damper collapsed to HOUSEHOLDS
    (Decision 2026-07-08 addendum, voice = REPUTATION not balance):
    callers group by their proven owner wallet (``agent_owner_map``;
    unbound callers are their own household), attested-ok counts sum per
    household, log1p damps ONCE per household, and the damped value
    multiplies the household's MINT weight (``voice_weights``: the
    ε-floored reputation share; ``voice_weights=None`` in the local/no-
    chain regime weighs every household 1.0). Decision 2026-07-10 deleted
    the supply-pegged β cap and the rep/ATN mint split — mint is MONEY
    ONLY now; ``rep_shares`` survives to drive DRIFT weight (below).
    ``voice_weights=None`` (no chain access) means every household weighs
    1.0 — the pre-voice behavior; a non-None map defaults unknown
    households to ``VOICE_EPSILON``. Only COGNITIVE attestations count;
    mechanical receipts are exhaust. Two exclusions run first:

      1. Owner-rooted (final form, spec "Owner-rooted registration"):
         the author's own household is excluded — this subsumes the
         self-attestation check and the same-owner check in one
         comparison. ``agent_owner_map`` (agent id → owner 0x) is an
         explicit carry-over input like ``registrations`` — sourced
         from chain owner-binding data by the driver, identical at
         every daemon, absent entries = unknown owner = the caller is
         its own household.
      2. Wire-level (interim floor until bound registrations are
         ubiquitous): a household ANY of whose attestation batches
         were signed with the same key as the registration batch is
         excluded (co-hosted sybil agents collapse to nothing). The
         sender key is transport plumbing — the agent is the only
         economic entity; the key appears in no output.

    Cross-epoch state comes in as one explicit param, derived purely
    from prior epochs' canonical events (so bit-identical everywhere):
    ``registrations`` (digest → manifest_meta, accumulated across
    epochs — a registration in epoch 1 still attributes mint in epoch
    5). A carried registration wins over a same-digest re-registration
    event: mint can never be re-attributed. Cross-epoch STANDING
    additionally requires seeding the close from the prior epoch's
    world (``seed_world``) so the manifest claim node persists; on a
    fresh-charter close only manifests (re-)sprouted this epoch have
    standing.

    Vetting (v4.1 [R1], ratified 2026-07-09): the vet GATE is REMOVED —
    tools mint from first attested use, author keeps 100% (no greenlight
    requirement, no validator royalty). ``vetting`` is still accepted and
    carried forward tolerantly (rebuildable-cache contract), and vets are
    still aggregated (``vet_quorum``/``vet_royalty_*`` retained as dead
    knobs), but they no longer skip mint or split it. A vet's surviving
    value is its per-axis scores, which enter position drift as an
    INSPECTION REVIEW (v4.1 [R2]) — moving the head, minting nothing.

    Ratings (docs/multi_jurisdiction.md decision 10, 2026-10-17): the
    tool's position is no longer drifted from gossip reviews and there is
    no off-chain credibility. ``review_snapshot`` (ToolBond, pinned to the
    previous anchor, see ``_normalize_review_snapshot``) supplies each
    tool's settled rating: ``positions_next[digest]`` = head per charter
    axis sum / mass / 100, mass = REP mass / REP supply, plus ``harm`` and
    ``harm_mass``. ``rep_shares`` (raw REP share per household, None in the
    local regime = weight 1.0) weighs only the ``own_share`` estimates.

    MINT weight per household: the ε-floored reputation share
    (``voice_weights``), no β cap. Mint is MONEY ONLY (Decision
    2026-07-10): REP is claimed DAO-side on ratified ATN earnings, never
    minted here.

    Three more close rules:

      - Reviewer share (decision 10): ``reviewer_share_bps`` of a tool's
        mint goes to the reviewers of its rounds finalized since the
        previous anchor, split by stake x accuracy x novelty
        (``reviewer_weights``); the author keeps the rest, 100% when no
        such round has a weighted reviewer. Redistribution only: the
        tool's total mint is unchanged (reviewer parts first, the author
        gets the exact remainder), and the pool normalization + base-unit
        floor run downstream as before. Reviewer addresses map onto an
        agent id already present in the close (case-insensitive,
        sorted-first spelling), else the lowercased address.
      - Reviewed composition split: a composite's root share is the
        rep-share-weighted mean of its attestations' ``own_share``
        estimates (carried as ``positions[digest]["own_share"] = {sum,
        mass}``), default ``COMPOSITE_ROOT_SHARE``. Updated BEFORE the
        fan-out so this close's estimates count; damp-then-split is
        unchanged.
      - Successor merit carry: a manifest first appearing with
        ``version_of`` naming a registered predecessor by the SAME author
        inherits the predecessor's carried own_share and, until ToolBond
        has rated the successor itself, its rating. A different author
        inherits nothing.

    Returns (all maps key-sorted, pure function of canonical inputs):
      {
        "per_digest": {digest: {node_id, author, trust_class,
                                ok_count, attesters, usage_term, mint,
                                greenlit, validators, royalty_left,
                                reviewers, reviewer_mint}},
        "node_agent": {node_id: {author: mint}},   # gate-mergeable
        "registrations_next": {digest: manifest_meta},
        "vetting_next": {"manifests": {...}, "busts": {agent: n}},
        "positions_next": {digest: {"head": [n_dims], "mass": [n_dims],
                                    "harm", "harm_mass",
                                    "own_share"?: {"sum", "mass"}}},
      }
    """
    import math

    # Platform-independent log1p: the mint must not depend on the C library.
    from world_model.generalized import detmath as _detmath
    from .world_model_substrate.infer import _artifact_standing
    from .world_model_substrate.tool_manifest import TRUST_PINNED
    from .world_model_substrate.tool_usage import tool_usage_from_events

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
        meta = dict(ev["manifest_meta"])
        # Registration batch's signing key, for the wire-level self-
        # attestation dedup below. Carried registrations keep the key
        # from their original epoch (persisted in the carry-over map).
        if ev.get("sender"):
            meta.setdefault("sender", str(ev["sender"]))
        known_regs.setdefault(ev["artifact_digest"], meta)

    from .tool_receipts import verified_receipts
    usage = tool_usage_from_events(verified_receipts(events, receipt_domain))
    by_digest = _artifact_standing(world)

    # ------------------------------------------------------------------
    # Vetting pipeline (spec: Vetting section) — three deterministic
    # passes over sorted keys BEFORE any mint math:
    #   1. merge this epoch's affirmative vets into the carried state
    #      (same exclusions as the damper: self-vet, same registered
    #      owner, vet batch signed with the registration batch's key);
    #   2. bust detection: a greenlit manifest whose claim node carries
    #      a winning charter violation zeroes its remaining royalty and
    #      increments every validator's bust count (their future vet
    #      weight decays as 1/(1+busts));
    #   3. greenlight: Σ over distinct fleets (owner-map collapse,
    #      fallback per-agent) of max member weight ≥ vet_quorum →
    #      greenlit; validators frozen at that instant (late vets earn
    #      nothing — the risk window is the incentive).
    # ------------------------------------------------------------------
    vet_state = _normalize_vetting(vetting)
    # Addresses are case-insensitive: the chain-read maps key by lowercase
    # 0x while events carry whatever spelling the agent used (often
    # checksummed). Every household lookup lowercases both sides, so a
    # checksummed caller still finds its owner, weight and rep share.
    owner_map_all = _lower_map(agent_owner_map or {}, values=True)
    if voice_weights is not None:
        voice_weights = _lower_map(voice_weights)
    if rep_shares is not None:
        rep_shares = _lower_map(rep_shares)

    def _owner(agent_id: str) -> str:
        return owner_map_all.get(str(agent_id).lower(), "")

    for digest in sorted(set(usage) & set(known_regs)):
        vets_by_caller = usage[digest].get("vets_by_caller") or {}
        if not vets_by_caller:
            continue
        meta = known_regs[digest]
        author = str(meta.get("author") or "")
        reg_sender = str(meta.get("sender") or "")
        author_owner = _owner(author)
        m = vet_state["manifests"].setdefault(digest, {
            "vets": {}, "greenlit": False, "royalty_left": 0,
            "validators": [], "busted": False,
        })
        for vetter in sorted(vets_by_caller.keys()):
            if vetter == author:
                continue
            if author_owner and _owner(vetter) == author_owner:
                continue
            senders = usage[digest].get("vet_senders", {}).get(vetter, [])
            if reg_sender and reg_sender in senders:
                continue
            pooled = m["vets"].setdefault(vetter, [])
            for s in senders:
                if s not in pooled:
                    pooled.append(s)
            m["vets"][vetter] = sorted(pooled)

    # v3 (spec Decision 2026-07-08): the CON-triggered bust of a greenlit
    # manifest is DORMANT — debate machinery left the live path, so there
    # is no charter-violation signal to trigger it. The bust fields
    # (busted, busts) remain in the carried state so validator weights
    # keep honoring any historical busts and the rail can be reactivated
    # (the named mitigation if review-only defenses prove insufficient).

    for digest in sorted(vet_state["manifests"].keys()):
        m = vet_state["manifests"][digest]
        if m["greenlit"]:
            continue
        fleets: Dict[str, float] = {}
        for vetter in sorted(m["vets"].keys()):
            fleet_key = _owner(vetter) or str(vetter).lower()
            weight = 1.0 / (1.0 + vet_state["busts"].get(vetter, 0))
            fleets[fleet_key] = max(fleets.get(fleet_key, 0.0), weight)
        total_weight = round(
            sum(fleets[k] for k in sorted(fleets.keys())), 10)
        if total_weight >= vet_quorum:
            m["greenlit"] = True
            m["royalty_left"] = int(vet_royalty_epochs)
            m["validators"] = sorted(m["vets"].keys())

    # Composition fan-out (docs/tool_substrate.md — Composition rule 3):
    # each receipt digest's attested counts distribute over its declared
    # dependency DAG with conserved total weight 1 (root keeps its
    # reviewed root share, default COMPOSITE_ROOT_SHARE, deps split the
    # rest, recursively; see the reviewed split below).
    #
    # ORDER OF OPERATIONS IS THE ANTI-AMPLIFICATION GUARANTEE: the
    # per-caller count is DAMPED FIRST (log1p once, at the composite the
    # caller actually attested), then the damped value splits linearly
    # over the DAG. Damping per-node after splitting would let concavity
    # mint free credit (log1p(0.7)+log1p(0.3) > log1p(1)) — self-padding
    # a composite with your own deps would amplify. Linear distribution
    # of the damped quantity makes padding exactly neutral at equal
    # standing. Attester wire identities are pooled per caller so the
    # co-hosting dedup applies wherever the credit lands.
    deps_of: Dict[str, List[str]] = {
        d: [x for x in str(m.get("deps") or "").split(",") if x]
        for d, m in known_regs.items()
    }

    # Household collapse (voice addendum): the economic unit behind a
    # caller is its proven owner wallet; unbound callers stand alone.
    def _household(agent_id: str) -> str:
        return _owner(agent_id) or str(agent_id).lower()

    rep_share_map = rep_shares or {}

    def _rep_share(house: str) -> float:
        return float(rep_share_map.get(house, 0.0))

    # ------------------------------------------------------------------
    # SUCCESSOR MERIT CARRY (Decision 2026-10-13 §6): a manifest that
    # first appears this close (no carried position) and declares
    # ``version_of`` = a registered predecessor by the SAME author starts
    # from the predecessor's carried own_share estimate, and (below) takes
    # the predecessor's ToolBond rating until it has one of its own. A
    # different author's ``version_of`` inherits nothing (no forks: they
    # compete on their own merit). Lookups read the ORIGINAL carried map
    # only, so a chain of same-close successors cannot cascade and the
    # result is independent of iteration order.
    # ------------------------------------------------------------------
    import copy as _copy

    orig_pos = positions or {}
    carried_pos: Dict[str, Dict[str, Any]] = dict(orig_pos)
    successor_of: Dict[str, str] = {}
    for digest in sorted(known_regs.keys()):
        meta = known_regs[digest]
        pred = str(meta.get("version_of") or "").lower().removeprefix("0x")
        author = str(meta.get("author") or "")
        if not pred or pred == digest or not author:
            continue
        pred_meta = known_regs.get(pred)
        if pred_meta is None or str(pred_meta.get("author") or "") != author:
            continue
        successor_of[digest] = pred
        if digest not in orig_pos and pred in orig_pos:
            carried_pos[digest] = _copy.deepcopy(orig_pos[pred])

    # ------------------------------------------------------------------
    # Reviewed composition split (Decision 2026-10-13 §9). Per composite
    # (declared deps), the root share is the running weighted mean of the
    # ``own_share`` estimates its attestations carried: log1p(n) x
    # rep_share per household, no ε floor (zero-rep estimates move
    # nothing), author household and co-hosted wire identities excluded;
    # local regime (rep_shares None) weighs 1.0. Computed BEFORE the
    # fan-out so this close's estimates count. No estimates yet =>
    # COMPOSITE_ROOT_SHARE.
    # ------------------------------------------------------------------
    root_share_of: Dict[str, float] = {}
    own_share_next: Dict[str, Dict[str, float]] = {}
    for digest in sorted(known_regs.keys()):
        # A carried estimate survives on any tool (e.g. one inherited by a
        # successor); new estimates only accumulate on composites.
        prior_own = (carried_pos.get(digest) or {}).get("own_share") or {}
        try:
            o_sum = float(prior_own.get("sum") or 0.0)
            o_mass = float(prior_own.get("mass") or 0.0)
        except (TypeError, ValueError, AttributeError):
            o_sum, o_mass = 0.0, 0.0
        if not o_mass > 0.0:
            o_sum, o_mass = 0.0, 0.0
        entry = usage.get(digest) or {}
        by_caller = (entry.get("own_share_by_caller") or {}
                     if deps_of.get(digest) else {})
        if by_caller:
            meta = known_regs[digest]
            author_house = _household(str(meta.get("author") or ""))
            reg_sender = str(meta.get("sender") or "")
            cells: Dict[str, Dict[str, float]] = {}
            cell_senders: Dict[str, set] = {}
            for caller in sorted(by_caller.keys()):
                house = _household(caller)
                agg = cells.setdefault(house, {"sum": 0.0, "n": 0})
                agg["sum"] += float(by_caller[caller].get("sum") or 0.0)
                agg["n"] += int(by_caller[caller].get("n") or 0)
                cell_senders.setdefault(house, set()).update(
                    entry.get("attester_senders", {}).get(caller, []))
            for house in sorted(cells.keys()):
                if house == author_house:
                    continue
                if reg_sender and reg_sender in cell_senders[house]:
                    continue
                n = int(cells[house]["n"])
                if n <= 0:
                    continue
                if rep_shares is None:
                    base_w = 1.0
                else:
                    base_w = _rep_share(house)
                    if base_w <= 0.0:
                        continue
                w = _detmath.log1p(n) * base_w
                o_sum += w * (cells[house]["sum"] / n)
                o_mass += w
        if o_mass > 0.0:
            own = {"sum": round(o_sum, 10), "mass": round(o_mass, 10)}
            if own["mass"] > 0.0:
                own_share_next[digest] = own
                root_share_of[digest] = round(
                    max(0.0, min(1.0, own["sum"] / own["mass"])), 9)

    # eff buckets are keyed by HOUSEHOLD: counts sum across a
    # household's agents at each attested composite, then log1p damps
    # once — N co-owned agents are one voice, not N log1p terms.
    eff: Dict[str, Dict[str, float]] = {}
    house_senders: Dict[str, set] = {}
    sanctioned_set = {str(h).lower() for h in (sanctioned_households or ())}
    denied_set = {str(d).lower().removeprefix("0x") for d in (denied_tools or ())}
    for digest in sorted(set(usage) & set(known_regs)):
        shares = _composition_shares(digest, deps_of,
                                     root_shares=root_share_of)
        entry = usage[digest]
        counts: Dict[str, int] = {}
        for caller in sorted(entry["attested_ok_by_caller"].keys()):
            house = _household(caller)
            if house in sanctioned_set:
                continue   # a sanctioned household's usage credits nobody
            counts[house] = (counts.get(house, 0)
                             + int(entry["attested_ok_by_caller"][caller]))
            house_senders.setdefault(house, set()).update(
                entry["attester_senders"].get(caller, []))
        for house in sorted(counts.keys()):
            damped = _detmath.log1p(counts[house])
            for target in sorted(shares.keys()):
                bucket = eff.setdefault(target, {})
                bucket[house] = bucket.get(house, 0.0) + damped * shares[target]

    # Resident grammar — ADOPTION (spec: Resident tools, loadouts,
    # distros): loadout stamps on attestations credit the distro by
    # DISTINCT FLEETS, volume-blind: callers collapse by the chain
    # owner map (fallback: each caller its own fleet — documented
    # degradation when registrations are sparse), each fleet
    # contributes log1p(1) once per epoch regardless of how many
    # attestations it emitted. The damped value injects at the distro
    # root and fans over its dep DAG like any composite. Exclusions
    # (author, same-owner, wire dedup) run in the mint loop below via
    # the fleet's sorted representative caller, whose sender pool is
    # the union of the fleet's senders.
    from .world_model_substrate.tool_usage import loadout_adoption_from_events
    adoption = loadout_adoption_from_events(events)
    for loadout in sorted(set(adoption) & set(known_regs)):
        shares = _composition_shares(loadout, deps_of,
                                     root_shares=root_share_of)
        entry = adoption[loadout]
        fleets: Dict[str, List[str]] = {}
        for caller in sorted(entry["by_caller"].keys()):
            fleets.setdefault(_household(caller), []).append(caller)
        for fleet_key in sorted(fleets.keys()):
            # Fleet == household: credit lands on the household key
            # directly, senders pool there for the wire dedup below.
            pooled = house_senders.setdefault(fleet_key, set())
            for member in fleets[fleet_key]:
                pooled.update(entry["senders"].get(member, []))
            damped = _detmath.log1p(1)
            for target in sorted(shares.keys()):
                bucket = eff.setdefault(target, {})
                bucket[fleet_key] = (bucket.get(fleet_key, 0.0)
                                     + damped * shares[target])

    # ------------------------------------------------------------------
    # MINT weight per household (Decision 2026-07-10: the β cap and the
    # rep/ATN split are DELETED — mint is MONEY ONLY, distributed pro-rata
    # over the ε-floored reputation-share weights). ``rep_shares`` survives
    # only for DRIFT weight (position section below). ``voice_weights``
    # carries the ε-floored share; None => local/no-chain regime, weight
    # 1.0. ``rep_share_map`` (defined above) is kept for the drift math.
    # ------------------------------------------------------------------
    def _mint_weight(house: str) -> float:
        # ε-floored reputation share (voice_weights); local regime weighs
        # every household 1.0 (the pre-voice behavior).
        if voice_weights is not None:
            return round(float(voice_weights.get(house, VOICE_EPSILON)), 9)
        return 1.0

    # Reviewer share (decision 10). Reviewer addresses resolve onto an
    # agent id the close already knows (registered authors, owner-map
    # agents, receipt callers), matched case-insensitively with the
    # sorted-first spelling winning, so one address never becomes two mint
    # leaves; unknown reviewers key by the lowercased address (claimable
    # once that agent is registered).
    rev_snap = _normalize_review_snapshot(review_snapshot)
    reviewer_bps = int(rev_snap["reviewer_share_bps"])
    id_index: Dict[str, str] = {}
    if rev_snap["rounds"]:
        known_ids: set = set((agent_owner_map or {}).keys())
        for m in known_regs.values():
            if m.get("author"):
                known_ids.add(str(m["author"]))
        for u in usage.values():
            known_ids.update(str(c) for c in (u.get("callers") or {}))
        for ident in sorted(known_ids):
            if ident:
                id_index.setdefault(ident.lower(), ident)

    per_digest: Dict[str, Dict[str, Any]] = {}
    node_agent: Dict[str, Dict[str, float]] = {}
    bonds = rev_snap.get("bonds")
    gate_on = mint_gate is not None and rep_shares is not None
    gate_rep_bps, gate_houses_bps = (int(mint_gate[0]), int(mint_gate[1])) if gate_on else (0, 0)
    n_holders = sum(1 for v in (rep_shares or {}).values() if float(v) > 0)
    for digest in sorted(eff.keys()):
        if digest in denied_set:
            continue   # denied by the network: mints nothing
        meta = known_regs[digest]
        author = str(meta.get("author") or "")
        if bonds is not None:
            # Only a tool bonded on ToolBond at the pinned snapshot mints,
            # and its mint goes to the bond's author (Decision 2026-10-27):
            # a gossiped registration alone attributes nothing.
            b = bonds.get(digest)
            if not b or b["unbonding"] or b["stake"] < int(rev_snap.get("min_stake") or 0) \
                    or b["stake"] <= 0:
                continue
            if author.lower() != b["author"]:
                author = b["author"]
        if not author:
            continue
        # Pinned-only emission (spec v2): the network mints only for
        # capability it can actually KNOW (hash-locked code).
        if str(meta.get("trust_class") or "") != TRUST_PINNED:
            continue
        # v4.1 [R1]: the vet GATE is REMOVED — tools mint from first
        # attested use. The carried vet state is still surfaced (greenlit
        # flag is informational only) but never skips mint here.
        vm = vet_state["manifests"].get(digest)
        greenlit = bool(vm and vm.get("greenlit"))
        nodes = by_digest.get(digest, [])
        # Combo damper over HOUSEHOLDS: counts collapsed + log1p'd at
        # fan-out time, author's household excluded (subsumes self and
        # same-owner), co-hosted households (any attestation batch signed
        # with the registration batch's key) excluded, then each
        # household's damped credit scales by its ε-floored MINT weight.
        reg_sender = str(meta.get("sender") or "")
        author_house = _household(author)
        attested_by = eff[digest]
        usage_term = 0.0
        attesters = 0
        gate_rep = 0.0
        gate_holders = 0
        for house in sorted(attested_by.keys()):
            if house == author_house:
                continue
            if reg_sender and reg_sender in house_senders.get(house, set()):
                continue
            usage_term += attested_by[house] * round(_mint_weight(house), 12)
            attesters += 1
            share = _rep_share(house)
            if share > 0:
                gate_rep += share
                gate_holders += 1
        if gate_on and (gate_rep * 10_000 < gate_rep_bps
                        or gate_holders * 10_000 < gate_houses_bps * n_holders):
            continue   # not broken through yet: earns nothing this epoch
        # v3/v4.1: USAGE ALONE mints. No standing multiplier, no vet gate.
        mint = usage_term
        if mint <= 0.0:
            continue
        # Anchor the mint on the manifest's claim node: lexicographically-
        # first id among the digest's claim nodes (the manifest anchor;
        # also where position drift writes the reviewed charter head).
        node_id = (sorted({n.id for n in nodes})[0] if nodes
                   else f"tool:{digest}")
        validators = list(vm["validators"]) if vm else []
        # Reviewer share: reviewers' parts first, in sorted order, then the
        # author takes the exact remainder, so the tool's total is
        # redistributed, never changed. The author's own reviews never
        # count on ToolBond, but an author address is skipped here too.
        reviewer_parts: Dict[str, float] = {}
        weights = {a: w for a, w in reviewer_weights(
            rev_snap["rounds"].get(digest, []), rev_snap["supply"]).items()
            if a != author.lower()}
        total_w = sum(weights[a] for a in sorted(weights))
        if weights and reviewer_bps > 0 and total_w > 0:
            share_pool = mint * reviewer_bps / BPS
            for addr in sorted(weights):
                agent = id_index.get(addr, addr)
                reviewer_parts[agent] = (reviewer_parts.get(agent, 0.0)
                                         + share_pool * weights[addr] / total_w)
        reviewer_total = sum(reviewer_parts[a] for a in sorted(reviewer_parts))
        author_part = max(0.0, mint - reviewer_total)
        per_digest[digest] = {
            "node_id": node_id,
            "author": author,
            "trust_class": TRUST_PINNED,
            "ok_count": int(usage.get(digest, {}).get("ok_count", 0)),
            "attesters": attesters,
            "usage_term": round(usage_term, 10),
            "mint": mint,
            "greenlit": greenlit,
            "validators": len(validators),
            "royalty_left": int(vm.get("royalty_left") or 0) if vm else 0,
            "reviewers": len(reviewer_parts),
            "reviewer_mint": round(reviewer_total, 10),
        }
        bucket = node_agent.setdefault(node_id, {})
        bucket[author] = bucket.get(author, 0.0) + author_part
        for agent in sorted(reviewer_parts.keys()):
            bucket[agent] = bucket.get(agent, 0.0) + reviewer_parts[agent]

    # Royalty window ticks once per close for every greenlit manifest
    # (kept for the carried-state contract; no mint effect under v4.1).
    for digest in sorted(vet_state["manifests"].keys()):
        m = vet_state["manifests"][digest]
        if m["greenlit"] and m["royalty_left"] > 0:
            m["royalty_left"] -= 1

    # ------------------------------------------------------------------
    # Positions = ToolBond ratings (decision 10). Each registered tool's
    # charter head is its settled REP-weighted mean per active-charter axis,
    # read pinned to the previous anchor; mass is the REP behind it as a
    # share of REP supply; harm likewise. No gossip review moves it and no
    # credibility weighs it. A same-author successor with no rating of its
    # own shows its predecessor's. Unrated tools sit at the zero head.
    # ------------------------------------------------------------------
    _charter = active_charter()
    axis_ids = list(_charter.ids)
    supply = int(rev_snap["supply"])
    positions_next: Dict[str, Dict[str, Any]] = {}
    for digest in sorted(known_regs.keys()):
        entry = rev_snap["tools"].get(digest)
        pred = successor_of.get(digest)
        if entry is None and pred is not None:
            entry = rev_snap["tools"].get(pred)
        positions_next[digest] = tool_rating_position(entry, axis_ids, supply)
        if digest in own_share_next:
            positions_next[digest]["own_share"] = own_share_next[digest]

    return {
        "per_digest": dict(sorted(per_digest.items())),
        "node_agent": {k: dict(sorted(v.items()))
                       for k, v in sorted(node_agent.items())},
        "registrations_next": dict(sorted(known_regs.items())),
        "positions_next": positions_next,
        "vetting_next": {
            "manifests": {d: {
                "vets": {a: sorted(s)
                         for a, s in sorted(m["vets"].items())},
                "greenlit": m["greenlit"],
                "royalty_left": m["royalty_left"],
                "validators": sorted(m["validators"]),
                "busted": m["busted"],
            } for d, m in sorted(vet_state["manifests"].items())},
            "busts": dict(sorted(vet_state["busts"].items())),
        },
    }


def federated_reconcile_epoch(
    world: World,
    snapshots: EpochSnapshots,
    events: List[Dict[str, Any]],
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    # v4.1 [R7] / divergence D2: default OFF, aligned with the outer
    # federated_epoch_close (the violator-pays gate is DORMANT). A True
    # default here silently reactivated the gate for direct callers.
    apply_gate: bool = False,
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
    shares of this fixed pool (see reconcile.apply_emission_pool). Under
    fees-only emission (Decision 2026-07-10) this is the burned-fee total
    for the epoch window (Σ ServiceFee.burned between anchors), read on
    the same pinned rail as the voice weights; a zero pool yields zero
    mint. MUST be derived from canonical data (never a daemon-local
    clock) or the bit-identical guarantee breaks.

    Returns the same shape as ``reconcile_epoch`` plus:
      - ``output_decimals``: the rounding precision used.
      - ``gate_applied``: whether the mint gate ran.

    ONE EARNING RAIL (ratified 2026-07-04, late): score-movement mint
    is RETIRED. Usage of a tool is a choice — attested usage is the
    usefulness signal the economics run on, and text-claim score
    movement double-counted the same "did it land" question with
    weaker ground truth (phase8). ``agent_mint`` therefore consists
    solely of ``extra_node_agent_mint`` (tool-author mint from
    compute_tool_mint), gate-scaled and pool-normalized. The score/
    standing machinery is fully retained — standing PRICES tool mint
    and the gate reads charter debates — it just no longer pays for
    prose. Work units remain data-plane retrieval artifacts (that
    value was validated); they earn nothing directly. ``node_mint`` /
    ``agent_novelty`` remain computed as diagnostics.
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
        score_mint = _node_mint(node_id, snapshots, world=world)
        if nov > 0:
            node_novelty[node_id] = nov
        if score_mint > 0:
            # DIAGNOSTIC ONLY (one earning rail): per-node score
            # movement is reported but never enters agent_mint.
            node_mint[node_id] = score_mint

        # Attribute novelty proportionally over agents whose events
        # landed under this node. Sort the attribution dict so the
        # sum of float shares is associative-stable across daemons.
        attribution = _attribute_node_mint(node_id, 1.0, events, world)
        for agent in sorted(attribution.keys()):
            share = attribution[agent]
            agent_novelty[agent] = agent_novelty.get(agent, 0.0) + nov * share

    # The ONLY earning rail: tool-author mint (compute_tool_mint),
    # anchored per manifest claim node. Merged BEFORE the gate so a won
    # charter CON against the anchoring node suppresses it like any
    # other mint, and before the pool so pool normalization applies.
    # ``agent_weights`` (stake-style weighting) applies here as it did
    # to the retired score rail. Sorted iteration keeps the float-add
    # order canonical. Money only (Decision 2026-07-10): no rep basis is
    # accumulated — REP is claimed DAO-side on ratified ATN earnings.
    for node_id in sorted((extra_node_agent_mint or {}).keys()):
        extra = extra_node_agent_mint[node_id]
        per_agent = node_agent_mint.setdefault(node_id, {})
        node_total = 0.0
        for agent in sorted(extra.keys()):
            amount = float(extra[agent])
            if amount <= 0.0:
                continue
            weighted = amount * agent_weights.get(agent, 1.0)
            agent_mint[agent] = agent_mint.get(agent, 0.0) + weighted
            per_agent[agent] = per_agent.get(agent, 0.0) + weighted
            node_total += weighted
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
                world, node_id, active_charter().ids,
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
        # On-chain conservation (2026-10-12): Substrate caps each epoch's
        # cumulative claimFor at the integer fee pool, so the
        # ratified mints are FLOORED to integer base units whose sum never
        # exceeds it (float rounding could otherwise over-allocate by an
        # ulp and the last claimer would revert).
        result["agent_mint"] = floor_mint_to_pool(
            result.get("agent_mint", {}) or {}, emission_pool)
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


def apply_tool_positions(
    world: World, positions: Dict[str, Dict[str, Any]],
) -> int:
    """Write drifted charter heads onto manifest claim nodes (v3 drift).

    For every node whose ``artifact_digest`` appears in ``positions``,
    the head (first ``len(pos["head"])`` coords) of BOTH coordinate
    stores is replaced: the per-tendency ``CoordinateClaim.anchor``
    (what serialize_world / checkpoints read) and the backing
    ``world.observations`` entry (what the visualization surfaces
    read). Embedding tails are untouched. Deterministic: purely a
    function of (world, positions).

    Returns the number of (tendency, node) coordinate writes.
    """
    from dataclasses import replace as _dc_replace

    if not positions:
        return 0
    updated = 0
    seen_obs: set = set()
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            digest = getattr(node, "artifact_digest", "") or ""
            pos = positions.get(digest) if digest else None
            if not pos:
                continue
            head = tuple(float(x) for x in (pos.get("head") or ()))
            if not head:
                continue
            claim = tendency._node_to_claim.get(node.id)
            if claim is not None and claim.anchor and len(claim.anchor) >= len(head):
                claim.anchor = head + tuple(claim.anchor)[len(head):]
                updated += 1
            obs_id = getattr(node, "observation_id", None)
            if obs_id and obs_id not in seen_obs:
                obs = world.observations.get(obs_id)
                if obs is not None and obs.coords and len(obs.coords) >= len(head):
                    world.observations[obs_id] = _dc_replace(
                        obs, coords=head + tuple(obs.coords)[len(head):])
                    seen_obs.add(obs_id)
    return updated


def federated_epoch_close(
    canonical: CanonicalOrder,
    *,
    seed_world: Optional[World] = None,
    bandwidth: float = 1.5,
    embedding_dim: int = 1024,
    # v3 (spec Decision 2026-07-08): the violator-pays gate is DORMANT —
    # debate machinery left the live path, so there are no charter CONs
    # to price. The machinery stays intact behind the flag for the named
    # future mitigation; default OFF.
    apply_gate: bool = False,
    gate_strength: float = 1.0,
    output_decimals: int = OUTPUT_DECIMALS,
    equilibrate_rounds: int = 8,
    equilibrate_tolerance: float = 1e-3,
    emission_pool: Optional[float] = None,
    pricing: str = "ledger",
    tool_registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    voice_weights: Optional[Dict[str, float]] = None,
    tool_vetting: Optional[Dict[str, Any]] = None,
    tool_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    # Raw REP share per household (weighs own_share estimates).
    rep_shares: Optional[Dict[str, float]] = None,
    # Staked reviews (docs/multi_jurisdiction.md decision 10): the ToolBond
    # review snapshot, read by the caller pinned to the previous anchor
    # (same snapshot as voice_weights / rep_shares / emission_pool). None =
    # no ratings (zero heads) and no reviewer shares.
    review_snapshot: Optional[Dict[str, Any]] = None,
    # Households under a DAO sanction at the same snapshot
    # (docs/accountability.md): their mint is withheld (left unminted, not
    # passed to others) and their estimates carry no drift weight. Their
    # reviews are already dropped from the ratings in ``review_snapshot``.
    sanctioned_households: Optional[List[str]] = None,
    # Gossip domain for signed receipts (compute_tool_mint).
    receipt_domain: Optional[str] = None,
    # Tools denied at the snapshot (compute_tool_mint).
    denied_tools: Optional[List[str]] = None,
    # Breakthrough gate (compute_tool_mint).
    mint_gate: Optional[Tuple[int, int]] = None,
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
            sender_hex = batch.sender_pubkey.hex()
            for ev in batch.events:
                if ev.get("kind") == "sub_claim_sprouted":
                    live = remap.get(ev.get("node_id", ""))
                    if live and live != ev.get("node_id"):
                        ev = dict(ev)
                        ev["solver_node_id"] = ev.get("node_id")
                        ev["node_id"] = live
                ev = _annotate_sender(ev, sender_hex)
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

            sender_hex = batch.sender_pubkey.hex()
            all_events.extend(_annotate_sender(ev, sender_hex)
                              for ev in batch.events)
            all_events.extend(derived)

        snapshots.record_close(world)

    # Tool substrate: author mint = usage_term (pinned only, v3 —
    # docs/tool_substrate.md Decision 2026-07-08), anchored on the
    # manifest's claim node. Pure function of the replayed world +
    # canonical events + explicitly-passed carry-over registrations,
    # so the bit-identical guarantee holds.
    sanctioned = sorted({str(h).lower() for h in (sanctioned_households or ())})
    if sanctioned and rep_shares is not None:
        # No drift weight: a sanctioned household's own_share estimates
        # move nothing (zero rep share, like any zero-rep household).
        rep_shares = {h: w for h, w in rep_shares.items()
                      if str(h).lower() not in set(sanctioned)}
    tool_result = compute_tool_mint(
        world, all_events,
        registrations=tool_registrations,
        agent_owner_map=agent_owner_map,
        voice_weights=voice_weights,
        vetting=tool_vetting,
        positions=tool_positions,
        rep_shares=rep_shares,
        review_snapshot=review_snapshot,
        receipt_domain=receipt_domain,
        denied_tools=denied_tools,
        sanctioned_households=sanctioned,
        mint_gate=mint_gate,
    )
    # Write the ToolBond-rated charter heads onto the replay world so the
    # checkpoint (world_cid) and any world reader carry them forward. Coords
    # never feed scores, so the score snapshots above are unaffected; the
    # write is a pure function of (pinned snapshot, carry, canonical
    # events), bit-identical everywhere.
    apply_tool_positions(world, tool_result["positions_next"])

    result = federated_reconcile_epoch(
        world, snapshots, all_events,
        apply_gate=apply_gate,
        gate_strength=gate_strength,
        output_decimals=output_decimals,
        emission_pool=emission_pool,
        extra_node_agent_mint=tool_result["node_agent"],
    )
    # Keep tool_mint in the SAME UNITS as agent_mint: when the emission
    # pool normalized agent mints, scale the per-digest display values
    # by the identical factor (usage_term stays raw — it's the input
    # signal, not a payout). Sorted iteration + rounding for the
    # bit-identical guarantee.
    per_digest = tool_result["per_digest"]
    _pool = result.get("emission_pool")
    _raw_total = float(result.get("raw_mint_total") or 0.0)
    if _pool is not None and _raw_total > 0.0:
        _scale = float(_pool) / _raw_total
        per_digest = {
            d: {**e, "mint": round(e["mint"] * _scale, output_decimals),
                "reviewer_mint": round(e.get("reviewer_mint", 0.0) * _scale,
                                       output_decimals)}
            for d, e in sorted(per_digest.items())
        }
    # Sanctions: withhold the mint of every agent whose household is
    # sanctioned. Applied AFTER the pool split, so the withheld amount stays
    # unminted instead of raising everyone else's share: a sanction never
    # pays the households that voted for it.
    withheld: Dict[str, float] = {}
    if sanctioned:
        house_set = set(sanctioned)
        owners = {str(k).lower(): str(v).lower()
                  for k, v in (agent_owner_map or {}).items()}
        kept: Dict[str, float] = {}
        for agent in sorted(result.get("agent_mint", {}) or {}):
            amount = result["agent_mint"][agent]
            a = str(agent).lower()
            if owners.get(a, a) in house_set or a in house_set:
                if amount:
                    withheld[agent] = amount
            else:
                kept[agent] = amount
        result["agent_mint"] = kept
        result["total_mint"] = round(sum(kept.values()), output_decimals)
    result["sanctioned_households"] = sanctioned
    result["withheld_mint"] = dict(sorted(withheld.items()))
    # The owner wallet claims (Decision 2026-10-27): key each mint by the
    # earning agent's pinned owner, summed per household. An agent with no
    # owner entry (bound after the snapshot, or genesis) stays its own
    # claimant. The per-agent split is kept for diagnostics only.
    result["agent_mint_by_agent"] = result.get("agent_mint", {}) or {}
    result["agent_mint"] = household_mint(
        result["agent_mint_by_agent"], agent_owner_map, output_decimals)
    # Pool-floored mints are whole base units, so the household sums are
    # exact and never exceed the pool.
    result["total_mint"] = round(sum(result["agent_mint"].values()), output_decimals)

    result["tool_mint"] = per_digest
    result["tool_registrations"] = tool_result["registrations_next"]
    result["tool_vetting"] = tool_result["vetting_next"]
    result["tool_positions"] = tool_result["positions_next"]
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
    # Charter version hash: the anchored charter this close ran against. A
    # charter change is a forward-only fork boundary, so the hash belongs in
    # the authoritative payload (imported lazily to keep the import block small
    # and avoid a circular import via adapter).
    from .world_model_substrate.charter_version import charter_hash as _charter_hash

    result["authoritative_payload"] = {
        # schema 3 (fees-only, Decision 2026-07-10): MONEY ONLY — the
        # anchor's mint merkle commits (agent, agent_mint) on a 2-field
        # leaf. ``agent_rep`` is gone: REP is claimed DAO-side on ratified
        # ATN earnings, not federation-minted here. (Schema 2 was the v4.1
        # 3-field agent_rep shape; not reused for this different shape.)
        # schema 4: + carry_cid (set by FederatedCloseDriver, like
        # world_cid) committing the next close's carry-over bundle.
        # schema 5 (Decision 2026-10-26): the encoding also commits
        # charter_hash and sanctioned_households.
        # schema 6 (Decision 2026-10-27): agent_mint is keyed by household
        # claimant (lowercase owner wallet), the leaf claimFor verifies.
        "schema": 6,
        "epoch_root": result["epoch_root"],
        "agent_mint": result["agent_mint"],
        "agent_novelty": result["agent_novelty"],
        "total_mint": result["total_mint"],
        "total_novelty": result["total_novelty"],
        "output_decimals": result.get("output_decimals", output_decimals),
        "gate_applied": result.get("gate_applied", False),
        "pricing": pricing,
        "charter_hash": _charter_hash(),
        # The sanction list this close ran against (docs/accountability.md).
        "sanctioned_households": result["sanctioned_households"],
        "n_batches": result["n_batches"],
        "n_events": result["n_events"],
    }
    return result


def household_mint(
    agent_mint: Dict[str, float],
    agent_owner_map: Optional[Dict[str, str]],
    output_decimals: int,
) -> Dict[str, float]:
    """Re-key a per-agent mint map by household claimant.

    Each agent's amount goes to its owner in the pinned ``agent_owner_map``
    (lowercased), or to the agent itself when it has none. Keys are
    lowercase so the merkle leaf and every submitter agree on spelling.
    Sorted accumulation, then rounding, for the bit-identical guarantee.
    """
    owners = {str(k).lower(): str(v).lower()
              for k, v in (agent_owner_map or {}).items()}
    out: Dict[str, float] = {}
    for agent in sorted(agent_mint):
        a = str(agent).lower()
        claimant = owners.get(a, a)
        out[claimant] = out.get(claimant, 0.0) + float(agent_mint[agent])
    return {k: round(v, output_decimals) for k, v in sorted(out.items())}


def _annotate_sender(ev: Dict[str, Any], sender_hex: str) -> Dict[str, Any]:
    """Stamp the gossip batch's signing key onto a COPY of the events
    the tool-mint pass reads (tool_used receipts and manifest_meta
    registration sprouts).

    The AGENT is the only authoritative economic entity; the sender key
    is transport plumbing used solely for the wire-level dedup in
    compute_tool_mint (self-attestation via co-hosted sybil agents
    collapses to nothing). It never enters formula outputs, attribution
    maps, or any chain surface. Deterministic: canonical batches carry
    the key, so every daemon stamps identically. batch.events is never
    mutated (the canonical log flows downstream verbatim).
    """
    if ev.get("kind") == "tool_used" or (
        ev.get("kind") == "sub_claim_sprouted" and ev.get("manifest_meta")
    ):
        ev = dict(ev)
        ev["sender"] = sender_hex
    return ev


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


def floor_mint_to_pool(
    agent_mint: Dict[str, float],
    emission_pool: float,
    mint_scale: int = 1_000_000,
) -> Dict[str, float]:
    """Floor pool-normalized mints to integer base units (x mint_scale)
    whose sum is <= the integer pool, then return them as floats k/scale.

    ``emission_pool`` is the ATN-denominated pool (chain base units /
    mint_scale, exact for pools below ~9e9 ATN), so the integer pool is
    recovered by rounding. Any float excess left after flooring is taken
    one unit at a time from the largest allocations (ties by key), which
    is deterministic. The dust (pool - sum) stays unminted. ``scale_mint``
    rounds, so each leaf scales back to exactly k.
    """
    import math

    pool_raw = int(round(float(emission_pool) * mint_scale))
    keys = sorted(agent_mint)
    ks: Dict[str, int] = {}
    for a in keys:
        v = float(agent_mint[a])
        ks[a] = max(0, int(math.floor(v * mint_scale))) if v > 0 else 0
    excess = sum(ks[a] for a in keys) - max(pool_raw, 0)
    if excess > 0:
        order = sorted(keys, key=lambda a: (-ks[a], a))
        i = 0
        while excess > 0 and order:
            a = order[i % len(order)]
            if ks[a] > 0:
                ks[a] -= 1
                excess -= 1
            i += 1
            if i > 10 * len(order) + excess * len(order):  # pragma: no cover
                break
    return {a: ks[a] / mint_scale for a in keys}


def _lower_map(d: Dict[str, Any], *, values: bool = False) -> Dict[str, Any]:
    """Lowercase a map's keys (and string values when ``values``), in
    sorted key order so a case collision resolves the same everywhere."""
    out: Dict[str, Any] = {}
    for k in sorted(d, key=str):
        v = d[k]
        out[str(k).lower()] = str(v).lower() if values else v
    return out


def _round_dict(d: Dict[str, float], decimals: int) -> Dict[str, float]:
    """Round every value in a {str: float} dict to ``decimals`` places.
    Drops zero values so empty contributions don't pollute the map."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        rv = round(float(v), decimals)
        if rv != 0.0:
            out[k] = rv
    return out
