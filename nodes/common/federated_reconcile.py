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
ATN_BASE_UNITS = 1_000_000   # 6 decimals: 1 ATN = 1e6 base units
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


UNMINTED = "~unminted"
"""Sentinel recipient for the part of a concluding tool's share that pays
nobody (the author's unpaid rating remainder, forfeited dependency shares,
a reviewer share with no eligible reviewer). It is normalized with the rest
so every tool's share of the pool is fixed by its REP alone, then dropped
before the claim tree: that ATN stays unminted and carries over."""

REVIEW_DECAY_DEFAULT = 1.0
"""Per-epoch factor applied to a tool's carried review rounds until it
concludes (Registry ``review_decay``). 1.0 = no decay."""

REVIEWER_MIN_REP_BPS_DEFAULT = 0
"""Minimum household REP, as bps of REP supply, for a reviewer to be paid
(Registry ``reviewer_min_rep_bps``). 0 = any REP above zero."""


def _normalize_tool_rounds(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonicalize the carried review-round state (sorted, typed, copied).

    Shape: {"pool_carry": int (ATN base units left unminted so far),
    "counts": {epoch: {household: n reviews}},
    "tools": {digest: {"concluded": bool, "rounds": [{"epoch", "decay",
    "reviews": [{"reviewer", "house", "rep", "n", "accuracy"}]}]}}}.
    Derived purely from prior closes' pinned snapshots, so it is identical
    on every honest daemon (the carry-over commitment covers it)."""
    out: Dict[str, Any] = {"pool_carry": 0, "counts": {}, "tools": {}}
    if not state:
        return out
    try:
        out["pool_carry"] = max(0, int(state.get("pool_carry") or 0))
    except (TypeError, ValueError):
        out["pool_carry"] = 0
    for epoch, houses in sorted((state.get("counts") or {}).items(), key=lambda kv: str(kv[0])):
        if isinstance(houses, dict):
            out["counts"][str(epoch)] = {str(h).lower(): int(n) for h, n in sorted(houses.items())
                                         if int(n) > 0}
    for digest, t in sorted((state.get("tools") or {}).items()):
        if not isinstance(t, dict):
            continue
        rounds = []
        for rnd in t.get("rounds") or []:
            revs = []
            for v in rnd.get("reviews") or []:
                acc = v.get("accuracy")
                revs.append({
                    "reviewer": str(v.get("reviewer") or "").lower(),
                    "house": str(v.get("house") or "").lower(),
                    "rep": round(float(v.get("rep") or 0.0), 12),
                    "n": max(1, int(v.get("n") or 1)),
                    "accuracy": None if acc is None else round(float(acc), 12),
                })
            rounds.append({"epoch": int(rnd.get("epoch") or 0),
                           "decay": round(float(rnd.get("decay", 1.0)), 12),
                           "reviews": revs})
        out["tools"][str(digest).lower().removeprefix("0x")] = {
            "concluded": bool(t.get("concluded")),
            "rounds": sorted(rounds, key=lambda r: r["epoch"]),
        }
    return out


def _tool_rating(entry: Optional[Dict[str, Any]], axis_ids: List[str]) -> float:
    """A tool's rating in [0, 1] for the author's part: the mean over the
    rated charter axes of the REP-weighted mean score (-1..1), negative
    clipped to 0. Unrated = 0."""
    axes = (entry or {}).get("axes") or {}
    means = []
    for axis_id in axis_ids:
        cell = axes.get(axis_id) or {}
        m = int(cell.get("mass") or 0)
        if m > 0:
            means.append(int(cell.get("sum") or 0) / m / 100.0)
    if not means:
        return 0.0
    return round(max(0.0, min(1.0, sum(means) / len(means))), 12)


def compute_tool_mint(
    world: World,
    events: List[Dict[str, Any]],
    *,
    registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    positions: Optional[Dict[str, Dict[str, Any]]] = None,
    rep_shares: Optional[Dict[str, float]] = None,
    review_snapshot: Optional[Dict[str, Any]] = None,
    tool_rounds: Optional[Dict[str, Any]] = None,
    denied_tools: Optional[List[str]] = None,
    sanctioned_households: Optional[List[str]] = None,
    mint_gate: Optional[Tuple[int, int]] = None,
    review_decay: float = REVIEW_DECAY_DEFAULT,
    reviewer_min_rep_bps: int = REVIEWER_MIN_REP_BPS_DEFAULT,
) -> Dict[str, Any]:
    """Tool mint from staked reviews (Decision 2026-11-07).

    A tool is paid once, in the close where its review rounds CONCLUDE.
    Each close adds the ToolBond rounds finalized in its window to the
    tool's carried rounds (older rounds scaled by ``review_decay`` per
    close). A tool concludes when its reviewers clear the breakthrough gate
    (``mint_gate`` = (rep_bps, houses_bps): distinct reviewing households
    holding >= rep_bps of REP and >= houses_bps of the REP-holding
    households; no gate in the local regime = the first round concludes).
    Rounds of a concluded tool pay nothing more (they still rate it).

    At conclusion the tool's share of the pool is its REP weight W: the sum
    over its reviews of the reviewing household's REP share / n (n = the
    reviews that household made in that epoch, so reviewing many tools
    never multiplies a household's REP) x the round's decay.
      - reviewers: reviewer_share_bps of W, whatever the verdict, split by
        accuracy x decay among reviewers whose household holds at least
        ``reviewer_min_rep_bps`` of REP (any REP when 0);
      - author side: the rest of W x the tool's rating (0..1), fanned over
        its declared dependencies (composite keeps COMPOSITE_ROOT_SHARE);
      - everything else goes to UNMINTED and stays unminted.
    Excluded: denied tools and every tool depending on one (transitively),
    unbonded tools, unpinned tools, the author's own household as a
    reviewer, sanctioned households in the gate and in W.

    Attestations (tool_used receipts) no longer count anywhere: nothing
    can observe a household's private use.
    """
    from .world_model_substrate.tool_manifest import TRUST_PINNED
    from .world_model_substrate.infer import _artifact_standing

    known_regs: Dict[str, Dict[str, str]] = dict(registrations or {})
    reg_events = [
        e for e in events
        if e.get("kind") == "sub_claim_sprouted"
        and e.get("artifact_digest") and e.get("manifest_meta")
    ]
    reg_events.sort(key=lambda e: (e.get("author_agent", ""), e.get("seq", 0)))
    for ev in reg_events:
        meta = dict(ev["manifest_meta"])
        if ev.get("sender"):
            meta.setdefault("sender", str(ev["sender"]))
        known_regs.setdefault(ev["artifact_digest"], meta)
    by_digest = _artifact_standing(world)

    owner_map_all = _lower_map(agent_owner_map or {}, values=True)
    rep_map = _lower_map(rep_shares) if rep_shares is not None else None

    def _household(agent_id: str) -> str:
        a = str(agent_id).lower()
        return owner_map_all.get(a, a)

    def _rep(house: str) -> float:
        if rep_map is None:
            return 1.0      # local regime: every household weighs 1
        return float(rep_map.get(house, 0.0))

    deps_of: Dict[str, List[str]] = {
        d: [x for x in str(m.get("deps") or "").split(",") if x]
        for d, m in known_regs.items()
    }

    # Denial passes up the dependency chain: a composite of a denied tool
    # can't be fetched whole, so it is denied too (its author is not
    # slashed for it).
    denied = {str(d).lower().removeprefix("0x") for d in (denied_tools or ())}
    changed = True
    while changed:
        changed = False
        for digest in sorted(deps_of):
            if digest not in denied and any(d in denied for d in deps_of[digest]):
                denied.add(digest)
                changed = True

    sanctioned = {str(h).lower() for h in (sanctioned_households or ())}
    rev_snap = _normalize_review_snapshot(review_snapshot)
    reviewer_bps = int(rev_snap["reviewer_share_bps"])
    bonds = rev_snap.get("bonds")
    min_stake = int(rev_snap.get("min_stake") or 0)
    state = _normalize_tool_rounds(tool_rounds)
    decay = max(0.0, min(1.0, float(review_decay)))

    def _bonded_author(digest: str) -> str:
        """The tool's payable author, or '' when it can't be paid."""
        meta = known_regs.get(digest)
        if meta is None or digest in denied:
            return ""
        if str(meta.get("trust_class") or "") != TRUST_PINNED:
            return ""
        author = str(meta.get("author") or "")
        if bonds is not None:
            b = bonds.get(digest)
            if (not b or b["unbonding"] or b["stake"] <= 0
                    or b["stake"] < min_stake):
                return ""
            author = b["author"]
        return author

    # 1. Older rounds of every open tool fade by one step.
    for digest in sorted(state["tools"]):
        t = state["tools"][digest]
        if t["concluded"] or decay == 1.0:
            continue
        for rnd in t["rounds"]:
            rnd["decay"] = round(rnd["decay"] * decay, 12)

    # 2. This window's finalized rounds: count each household's reviews per
    # epoch (cumulative across closes, for the REP / n split), then append
    # them to their open tools.
    window = rev_snap["rounds"]
    for digest in sorted(window):
        for rnd in window[digest]:
            ep = str(rnd["epoch"])
            counts = state["counts"].setdefault(ep, {})
            for v in rnd["reviews"]:
                house = _household(v["reviewer"])
                counts[house] = counts.get(house, 0) + 1
    for digest in sorted(window):
        t = state["tools"].setdefault(digest, {"concluded": False, "rounds": []})
        if t["concluded"]:
            continue
        for rnd in window[digest]:
            ep = str(rnd["epoch"])
            revs = []
            for v in rnd["reviews"]:
                house = _household(v["reviewer"])
                acc = v.get("accuracy")
                revs.append({
                    "reviewer": v["reviewer"],
                    "house": house,
                    "rep": round(_rep(house), 12),
                    "n": max(1, int(state["counts"].get(ep, {}).get(house, 1))),
                    "accuracy": None if acc is None else round(int(acc) / 10 ** 18, 12),
                })
            revs.sort(key=lambda v: v["reviewer"])
            t["rounds"].append({"epoch": int(rnd["epoch"]), "decay": 1.0,
                                "reviews": revs})
        t["rounds"].sort(key=lambda r: r["epoch"])

    gate_on = mint_gate is not None and rep_map is not None
    gate_rep_bps, gate_houses_bps = (int(mint_gate[0]), int(mint_gate[1])) if gate_on else (0, 0)
    n_holders = sum(1 for v in (rep_map or {}).values() if float(v) > 0)
    min_share = reviewer_min_rep_bps / BPS

    # 3. Conclusions.
    per_digest: Dict[str, Dict[str, Any]] = {}
    node_agent: Dict[str, Dict[str, float]] = {}
    _charter = active_charter()
    axis_ids = list(_charter.ids)
    for digest in sorted(state["tools"]):
        t = state["tools"][digest]
        if t["concluded"] or not t["rounds"]:
            continue
        author = _bonded_author(digest)
        if not author:
            continue
        author_house = _household(author)
        house_rep: Dict[str, float] = {}
        weight = 0.0
        for rnd in t["rounds"]:
            for v in rnd["reviews"]:
                h = v["house"]
                if h == author_house or h in sanctioned:
                    continue
                r = v["rep"] * rnd["decay"]
                if r <= 0:
                    continue
                house_rep[h] = max(house_rep.get(h, 0.0), r)
                weight += r / v["n"]
        if gate_on:
            gate_rep = sum(house_rep[h] for h in sorted(house_rep))
            if (gate_rep * BPS < gate_rep_bps
                    or len(house_rep) * BPS < gate_houses_bps * n_holders):
                continue   # not conclusive yet: carried to the next close
        elif not house_rep:
            continue
        weight = round(weight, 12)
        t["concluded"] = True
        if weight <= 0.0:
            continue
        nodes = by_digest.get(digest, [])
        node_id = (sorted({n.id for n in nodes})[0] if nodes
                   else f"tool:{digest}")
        bucket = node_agent.setdefault(node_id, {})

        # Reviewers: their share whatever the verdict, by accuracy x decay.
        rev_w: Dict[str, float] = {}
        for rnd in t["rounds"]:
            for v in rnd["reviews"]:
                if v["house"] == author_house or v["accuracy"] is None:
                    continue
                w = v["accuracy"] * rnd["decay"]
                if v["house"] in sanctioned:
                    # A sanction withholds, never redistributes: the
                    # sanctioned reviewer's cut stays unminted.
                    if w > 0:
                        rev_w[UNMINTED] = rev_w.get(UNMINTED, 0.0) + w
                    continue
                if rep_map is not None and (v["rep"] <= 0 or v["rep"] < min_share):
                    continue
                if w > 0:
                    rev_w[v["reviewer"]] = rev_w.get(v["reviewer"], 0.0) + w
        reviewer_pool = weight * reviewer_bps / BPS
        total_w = sum(rev_w[a] for a in sorted(rev_w))
        reviewer_paid = 0.0
        if total_w > 0:
            for a in sorted(rev_w):
                part = reviewer_pool * rev_w[a] / total_w
                bucket[a] = bucket.get(a, 0.0) + part
                if a != UNMINTED:
                    reviewer_paid += part

        # Author side, scaled by the rating, fanned over the dependencies.
        rating = _tool_rating(rev_snap["tools"].get(digest), axis_ids)
        author_side = (weight - reviewer_pool) * rating
        author_paid = 0.0
        shares = _composition_shares(digest, deps_of)
        for target in sorted(shares):
            recipient = author if target == digest else _bonded_author(target)
            part = author_side * shares[target]
            if not recipient or part <= 0:
                continue
            bucket[recipient] = bucket.get(recipient, 0.0) + part
            author_paid += part
        rest = weight - reviewer_paid - author_paid - bucket.get(UNMINTED, 0.0)
        if rest > 0:
            bucket[UNMINTED] = bucket.get(UNMINTED, 0.0) + rest
        per_digest[digest] = {
            "node_id": node_id,
            "author": author,
            "trust_class": TRUST_PINNED,
            "weight": weight,
            "rating": rating,
            "reviewers": len([a for a in rev_w if a != UNMINTED]),
            "reviewer_mint": round(reviewer_paid, 12),
            "author_mint": round(author_paid, 12),
            "mint": round(reviewer_paid + author_paid, 12),
        }

    # Ratings as positions (decision 10), unchanged: the ToolBond-settled
    # REP-weighted mean per charter axis; a same-author successor with no
    # rating of its own shows its predecessor's.
    successor_of: Dict[str, str] = {}
    for digest in sorted(known_regs):
        meta = known_regs[digest]
        pred = str(meta.get("version_of") or "").lower().removeprefix("0x")
        pred_meta = known_regs.get(pred) if pred and pred != digest else None
        if pred_meta is not None and str(pred_meta.get("author") or "") == str(meta.get("author") or ""):
            successor_of[digest] = pred
    supply = int(rev_snap["supply"])
    positions_next: Dict[str, Dict[str, Any]] = {}
    for digest in sorted(known_regs):
        entry = rev_snap["tools"].get(digest)
        pred = successor_of.get(digest)
        if entry is None and pred is not None:
            entry = rev_snap["tools"].get(pred)
        positions_next[digest] = tool_rating_position(entry, axis_ids, supply)

    # Concluded tools need only their flag from here on.
    for digest in sorted(state["tools"]):
        if state["tools"][digest]["concluded"]:
            state["tools"][digest]["rounds"] = []

    # Review counts are needed only for epochs an open tool still carries.
    live_epochs = {str(rnd["epoch"]) for t in state["tools"].values()
                   for rnd in t["rounds"]}
    state["counts"] = {e: c for e, c in sorted(state["counts"].items())
                       if e in live_epochs}

    return {
        "per_digest": dict(sorted(per_digest.items())),
        "node_agent": {k: dict(sorted(v.items()))
                       for k, v in sorted(node_agent.items())},
        "registrations_next": dict(sorted(known_regs.items())),
        "positions_next": positions_next,
        "tool_rounds_next": state,
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
    tool_rounds: Optional[Dict[str, Any]] = None,
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
    # Tools denied at the snapshot (compute_tool_mint).
    denied_tools: Optional[List[str]] = None,
    # Breakthrough gate (compute_tool_mint).
    mint_gate: Optional[Tuple[int, int]] = None,
    # Carried rounds fade per close; reviewer pay needs this much REP.
    review_decay: float = REVIEW_DECAY_DEFAULT,
    reviewer_min_rep_bps: int = REVIEWER_MIN_REP_BPS_DEFAULT,
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
        positions=tool_positions,
        rep_shares=rep_shares,
        review_snapshot=review_snapshot,
        tool_rounds=tool_rounds,
        denied_tools=denied_tools,
        sanctioned_households=sanctioned,
        mint_gate=mint_gate,
        review_decay=review_decay,
        reviewer_min_rep_bps=reviewer_min_rep_bps,
    )
    rounds_next = tool_result["tool_rounds_next"]
    # The pool carries over (Decision 2026-11-07): what earlier closes left
    # unminted joins this epoch's burned fees. Substrate caps claims on the
    # cumulative pool, so the carry can be spent in any later epoch.
    if emission_pool is not None:
        emission_pool = float(emission_pool) + rounds_next["pool_carry"] / ATN_BASE_UNITS
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
    # The unpaid part of each concluding tool's share stays unminted.
    for key in ("agent_mint", "node_mint"):
        result[key] = {k: v for k, v in (result.get(key) or {}).items()
                       if k != UNMINTED}
    if UNMINTED in (result.get("agent_mint") or {}):  # pragma: no cover
        raise AssertionError("unminted sentinel leaked")
    result["total_mint"] = round(sum(result["agent_mint"].values()), output_decimals)
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
    if emission_pool is not None:
        pool_raw = int(round(float(emission_pool) * ATN_BASE_UNITS))
        minted_raw = sum(int(round(float(v) * ATN_BASE_UNITS))
                         for v in result["agent_mint"].values())
        rounds_next["pool_carry"] = max(0, pool_raw - minted_raw)
    result["tool_rounds"] = rounds_next
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
