"""The federated epoch close: bit-identical mint maps across daemons.

Every daemon that closes an epoch runs ``federated_epoch_close`` over the
same canonical batch sequence (``canonical_ordering``), the carry-over the
previous anchor committed (registrations, review rounds, positions) and the
chain inputs pinned to that anchor (REP shares, the ToolBond review
snapshot, the fee pool, sanctions). The only mint rail is the review mint
(``compute_tool_mint``, Decision 2026-11-07). Two guarantees make the
output bit-identical:

  1. **Deterministic iteration order**. Float addition is
     non-associative, so every sum runs over sorted keys (digest, then
     agent) and every honest daemon performs the same sequence of float
     operations.

  2. **Output normalization**. Mints are rounded to ``OUTPUT_DECIMALS``
     places and then floored to whole ATN base units within the pool,
     which swamps any ulp-level jitter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .canonical_ordering import CanonicalOrder
from .library.charter import active_charter


logger = logging.getLogger(__name__)


# Number of decimal places mint outputs are rounded to: far below any
# meaningful precision (chain scaling truncates at 6 decimals), well above
# IEEE 754 jitter (~15 significant figures).
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
    events: List[Dict[str, Any]],
    *,
    registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
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
    from .library.tool_manifest import TRUST_PINNED

    known_regs: Dict[str, Dict[str, str]] = dict(registrations or {})
    reg_events = [
        e for e in events
        if e.get("kind") == "tool_registered"
        and e.get("artifact_digest") and e.get("manifest_meta")
    ]
    reg_events.sort(key=lambda e: (e.get("author_agent", ""), e.get("seq", 0)))
    for ev in reg_events:
        meta = dict(ev["manifest_meta"])
        if ev.get("sender"):
            meta.setdefault("sender", str(ev["sender"]))
        known_regs.setdefault(ev["artifact_digest"], meta)

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
        bucket = node_agent.setdefault(digest, {})

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


def federated_epoch_close(
    canonical: CanonicalOrder,
    *,
    output_decimals: int = OUTPUT_DECIMALS,
    emission_pool: Optional[float] = None,
    tool_registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    tool_rounds: Optional[Dict[str, Any]] = None,
    # Raw REP share per household (weighs reviews).
    rep_shares: Optional[Dict[str, float]] = None,
    # Staked reviews (docs/multi_jurisdiction.md decision 10): the ToolBond
    # review snapshot, read by the caller pinned to the previous anchor
    # (same snapshot as rep_shares / emission_pool). None = no ratings
    # (zero heads) and no review rounds.
    review_snapshot: Optional[Dict[str, Any]] = None,
    # Households under a DAO sanction at the same snapshot
    # (docs/accountability.md): their mint is withheld (left unminted, not
    # passed to others) and their reviews weigh nothing.
    sanctioned_households: Optional[List[str]] = None,
    # Tools denied at the snapshot (compute_tool_mint).
    denied_tools: Optional[List[str]] = None,
    # Breakthrough gate (compute_tool_mint).
    mint_gate: Optional[Tuple[int, int]] = None,
    # Carried rounds fade per close; reviewer pay needs this much REP.
    review_decay: float = REVIEW_DECAY_DEFAULT,
    reviewer_min_rep_bps: int = REVIEWER_MIN_REP_BPS_DEFAULT,
) -> Dict[str, Any]:
    """Run a full federated epoch close over a canonical batch sequence.

    A pure function of (canonical events, carry-over, pinned chain
    inputs): daemons that received the same batches return bit-identical
    results, and the ``authoritative_payload`` is what the anchor commits.
    The canonical events matter for their tool registrations
    (``tool_registered``); everything a mint needs beyond them is carried
    or read from the chain.
    """
    all_events: List[Dict[str, Any]] = []
    for batch in canonical.ordered_batches:
        sender_hex = batch.sender_pubkey.hex()
        for ev in batch.events or []:
            all_events.append(_annotate_sender(ev, sender_hex))

    sanctioned = sorted({str(h).lower() for h in (sanctioned_households or ())})
    if sanctioned and rep_shares is not None:
        # A sanctioned household's reviews weigh nothing (zero REP share).
        rep_shares = {h: w for h, w in rep_shares.items()
                      if str(h).lower() not in set(sanctioned)}
    tool_result = compute_tool_mint(
        all_events,
        registrations=tool_registrations,
        agent_owner_map=agent_owner_map,
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

    # Per-agent mint: every concluding tool's bucket, digest then agent
    # order. The unpaid part of each tool's share rides as UNMINTED so the
    # pool split below leaves it unminted.
    agent_mint: Dict[str, float] = {}
    node_agent = tool_result["node_agent"]
    for digest in sorted(node_agent):
        for agent in sorted(node_agent[digest]):
            amount = float(node_agent[digest][agent])
            if amount > 0.0:
                agent_mint[agent] = agent_mint.get(agent, 0.0) + amount
    result: Dict[str, Any] = {
        "agent_mint": dict(sorted(_round_dict(agent_mint, output_decimals).items())),
        "output_decimals": output_decimals,
    }
    if emission_pool is not None:
        result = apply_emission_pool(result, emission_pool)
        # On-chain conservation (2026-10-12): Substrate caps each epoch's
        # cumulative claimFor at the integer fee pool, so the ratified
        # mints are FLOORED to integer base units whose sum never exceeds
        # it (float rounding could otherwise over-allocate by an ulp and
        # the last claimer would revert).
        result["agent_mint"] = floor_mint_to_pool(
            result.get("agent_mint", {}) or {}, emission_pool)
        result["agent_mint"] = dict(sorted(
            _round_dict(result.get("agent_mint", {}), output_decimals).items()
        ))
    result["agent_mint"] = {k: v for k, v in result["agent_mint"].items()
                            if k != UNMINTED}
    # Keep tool_mint in the SAME UNITS as agent_mint: when the pool
    # normalized agent mints, scale the per-digest display values by the
    # identical factor. Sorted iteration + rounding.
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
        for agent in sorted(result["agent_mint"]):
            amount = result["agent_mint"][agent]
            a = str(agent).lower()
            if owners.get(a, a) in house_set or a in house_set:
                if amount:
                    withheld[agent] = amount
            else:
                kept[agent] = amount
        result["agent_mint"] = kept
    result["sanctioned_households"] = sanctioned
    result["withheld_mint"] = dict(sorted(withheld.items()))
    # The owner wallet claims (Decision 2026-10-27): key each mint by the
    # earning agent's pinned owner, summed per household. An agent with no
    # owner entry (bound after the snapshot, or genesis) stays its own
    # claimant. The per-agent split is kept for diagnostics only.
    result["agent_mint_by_agent"] = result["agent_mint"]
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
        logger.info(
            "tool mint: %d tool(s) concluded, minted %.6f of a %.6f pool, "
            "%.6f carried over", len(tool_result["per_digest"]),
            minted_raw / ATN_BASE_UNITS, pool_raw / ATN_BASE_UNITS,
            rounds_next["pool_carry"] / ATN_BASE_UNITS)
    result["tool_rounds"] = rounds_next
    result["tool_positions"] = tool_result["positions_next"]
    result["epoch_root"] = canonical.epoch_root().hex()
    result["n_batches"] = len(canonical.ordered_batches)
    result["n_events"] = len(all_events)
    result["scope"] = "federated"
    result["authoritative"] = True

    # The on-chain anchor commits the merkle of an ``authoritative_payload``
    # dict: only fields every daemon computes identically live here.
    from .library.charter_version import charter_hash as _charter_hash

    result["authoritative_payload"] = {
        # schema 7 (Decision 2026-11-08, the world leaves the system): the
        # payload drops agent_novelty, total_novelty, gate_applied and
        # world_cid. agent_mint is keyed by household claimant (lowercase
        # owner wallet, schema 6), the 2-field leaf claimFor verifies;
        # carry_cid (set by FederatedCloseDriver) commits the next close's
        # carry-over bundle; charter_hash and sanctioned_households decide
        # the mint, so they are committed too.
        "schema": 7,
        "epoch_root": result["epoch_root"],
        "agent_mint": result["agent_mint"],
        "total_mint": result["total_mint"],
        "output_decimals": output_decimals,
        "charter_hash": _charter_hash(),
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
    """Stamp the gossip batch's signing key onto a COPY of a tool
    registration event (it rides the registration as ``sender``).

    The AGENT is the only authoritative economic entity; the sender key is
    transport plumbing. It never enters formula outputs or any chain
    surface. Deterministic: canonical batches carry the key, so every
    daemon stamps identically. batch.events is never mutated (the
    canonical log flows downstream verbatim).
    """
    if ev.get("kind") == "tool_registered" and ev.get("manifest_meta"):
        ev = dict(ev)
        ev["sender"] = sender_hex
    return ev


def apply_emission_pool(result: Dict[str, Any], emission_pool: float) -> Dict[str, Any]:
    """Normalize a close's mints to shares of the epoch's pool.

    Deterministic given (result, pool): values are scaled in sorted-key
    order, so every daemon computes the same map. ``emission_pool`` must
    come from canonical data (the burned fees read pinned to the previous
    anchor, plus the carried pool), never from a daemon-local clock.
    """
    out = dict(result)
    agent_mint = result.get("agent_mint", {}) or {}
    raw_total = sum(agent_mint[a] for a in sorted(agent_mint))
    out["raw_mint_total"] = raw_total
    out["emission_pool"] = float(emission_pool)
    if raw_total <= 0:
        return out
    scale = float(emission_pool) / raw_total
    out["agent_mint"] = {a: agent_mint[a] * scale for a in sorted(agent_mint)}
    out["total_mint"] = sum(out["agent_mint"][a] for a in sorted(out["agent_mint"]))
    return out


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
