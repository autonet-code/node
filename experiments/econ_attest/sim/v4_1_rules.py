"""v4.1 "gradient trust" ruleset — SIM-ONLY reimplementation.

=====================================================================
SAME GROUND RULES AS v4_rules.py: this is NOT production code and does
NOT import the real mint/drift functions. The v4.1 rules were ratified
after the v4 sim findings; they are sim-validated here BEFORE any
production build. Deviations from the v3 real close are tagged
[v4-A], [v4-B], [v4-C] (unchanged from v4) and [v4.1-D'], [v4.1-E']
(the two revisions).
=====================================================================

Carried over from v4 (unchanged):
  A. No vet gate — mint from first attested use.               [v4-A]
  B. Vetting merged into reviews: inspection reviews drift
     position, mint nothing.                                   [v4-B]
  C. Drift weight = credibility * household_rep/rep_supply with
     NO epsilon floor (zero-rep -> zero drift mass). Author
     prior mass stays 1.0.                                     [v4-C]

Revised in v4.1:
  D' (replaces D). DROP the beta aggregate cap. Zero-rep callers'
     usage still mints ATN (they carry an epsilon weight on the ATN
     side), but that ATN is DECOUPLED from reputation: an author's
     reputation increment per epoch = ONLY the portion of their mint
     attributable to REP-HOLDING callers' weight. Consequence: a dust
     ring skims a bounded ATN trickle forever but NEVER gains voice —
     the rule-D leak dies.                                     [v4.1-D']

  E' (replaces E). Continuous, reversal-aware sanctions, NO
     stabilization moment. Every epoch, each household's credibility
     is recomputed against the CURRENT voice-weighted score of each
     tool it reviewed: deviation > delta -> dock proportionally;
     deviation shrank since it reviewed (score moved toward it) ->
     restore proportionally. Symmetric. No Q threshold; sanctions only
     weigh reviews on tools whose total review mass >= MASS_FLOOR so a
     single-review tool sanctions no one. A captured score that later
     reverses retroactively docks the capturers (the review record is
     carried and re-evaluated against the moving score).       [v4.1-E']
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of

N_DIMS = 6
CORRECTNESS_AXIS = 4
SIMPLICITY_AXIS = 5

CRED_FLOOR = 0.1
CRED_RECOVERY = 0.10          # passive drift back toward 1.0 (E' still symmetric)
MASS_FLOOR = 3.0              # [v4.1-E'] don't sanction on thin tools
EPS_ATN = 0.05               # zero-rep ATN-side weight (mint only, D')


@dataclass
class OpenReview:
    """A review record carried across epochs so E' can re-evaluate it
    against the CURRENT (moving) score — this is what lets a later
    reversal retroactively dock the original capturers."""
    household: str
    digest: str
    axis_id: str
    score: float          # the score the household asserted
    filed_epoch: int
    # the tool's head on this axis AT THE TIME the review was filed —
    # used to detect whether the score later moved toward or away.
    head_when_filed: float


@dataclass
class V41State:
    positions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    credibility: Dict[str, float] = field(default_factory=dict)
    review_mass: Dict[str, float] = field(default_factory=dict)   # digest -> total axis mass
    registrations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # [v4.1-E'] open review ledger: (household,digest,axis) -> OpenReview,
    # kept so continuous reversal-aware sanctions can re-score it each epoch.
    open_reviews: Dict[Tuple[str, str, str], OpenReview] = field(default_factory=dict)
    epoch: int = 0


@dataclass
class Review:
    household: str
    axes: Dict[str, float]
    inspection: bool = False


def _voice_mass(agents: Dict[str, Agent]) -> Tuple[Dict[str, float], float]:
    house_rep: Dict[str, float] = {}
    for a in agents.values():
        house_rep[household_of(a)] = house_rep.get(household_of(a), 0.0) + a.reputation
    return house_rep, sum(house_rep.values())


def _axis_idx(axis_id: str) -> Optional[int]:
    return {"correctness": CORRECTNESS_AXIS, "simplicity": SIMPLICITY_AXIS,
            "life_precious": 0, "self_preservation": 1,
            "promotion_of_intelligence": 2, "evolution": 3}.get(axis_id)


def compute_v41_epoch(
    *,
    agents: Dict[str, Agent],
    state: V41State,
    tools: Dict[str, Tool],
    usage_counts: Dict[str, Dict[str, float]],
    reviews: Dict[str, List[Review]],
    author_house: Dict[str, str],
    emission_pool: float,
    delta: float,
    epsilon: float = EPS_ATN,
    beta: Optional[float] = None,
) -> Dict[str, Any]:
    """One v4.1 close. Returns agent_mint (pool-normalized), rep_increment
    (the decoupled reputation gain per household, D'), positions,
    credibility, plus diagnostics for the metric reframe. Mutates state.

    ``beta`` (v4.1 + accepted β cap): a REP-INDEPENDENT aggregate cap on
    the total zero-rep ATN weight this epoch, expressed as a fraction of
    total ATN weight. ``None`` = uncapped (the pure-v4.1 behavior). When
    the observed zero-rep share exceeds β, ALL zero-rep households are
    scaled down uniformly (pro-rata) so their aggregate is exactly β of
    the total. This throttles honest newcomers too — the trade-off the
    sweep quantifies. Note it is on the ATN side ONLY; D' already ensures
    zero-rep usage grants no reputation, so this never touches voice.
    """
    house_rep, supply = _voice_mass(agents)

    def rep_share(house: str) -> float:
        return (house_rep.get(house, 0.0) / supply) if supply > 0 else 0.0

    def is_zero_rep(house: str) -> bool:
        return house_rep.get(house, 0.0) <= 0.0

    # ---- β cap (rep-independent, ATN-side): compute a uniform scale on
    # zero-rep ATN weight so its aggregate share of total ATN weight <= β.
    # Uses this epoch's OBSERVED weights, so it is pegged to real demand,
    # not a static guess. We first tally the damped usage weight each
    # household contributes (rep-holders at rep_share, zero-rep at epsilon)
    # to derive the cap scale, then apply it in the mint loop below.
    rep_weight_total = 0.0
    zero_weight_total = 0.0
    for digest in usage_counts:
        if digest not in state.registrations and digest not in tools:
            continue
        ah = author_house.get(digest, "")
        for house in usage_counts[digest]:
            if house == ah:
                continue
            damped = math.log1p(usage_counts[digest][house])
            if is_zero_rep(house):
                zero_weight_total += damped * epsilon
            else:
                zero_weight_total += 0.0
                rep_weight_total += damped * rep_share(house)
    total_weight = rep_weight_total + zero_weight_total
    observed_zero_share = (zero_weight_total / total_weight) if total_weight > 0 else 0.0
    zero_scale = 1.0
    if beta is not None and zero_weight_total > 0:
        # allowed zero-rep weight so that allowed/(rep + allowed) = β
        allowed = (beta / (1.0 - beta)) * rep_weight_total if beta < 1.0 else float("inf")
        if allowed < zero_weight_total:
            zero_scale = allowed / zero_weight_total

    # [v4.1-D'] MINT weight: rep-holders carry rep/supply; zero-rep callers
    # carry a flat epsilon (β-scaled if the cap binds this epoch).
    def atn_weight(house: str) -> float:
        r = house_rep.get(house, 0.0)
        if r > 0:
            return rep_share(house)
        return epsilon * zero_scale

    # We split each author's mint into an ATN component (all callers) and a
    # REP component (rep-holding callers only). Reputation increments come
    # ONLY from the rep component -> dust rings never gain voice.
    raw_atn: Dict[str, float] = {}         # digest -> usage_term (ATN basis)
    raw_rep: Dict[str, float] = {}         # digest -> usage_term (rep basis)
    node_atn: Dict[str, float] = {}        # author household -> ATN basis
    node_rep_basis: Dict[str, float] = {}  # author household -> REP basis
    for digest in sorted(usage_counts.keys()):
        if digest not in state.registrations and digest not in tools:
            continue
        ah = author_house.get(digest, "")
        term_atn = 0.0
        term_rep = 0.0
        for house in sorted(usage_counts[digest].keys()):
            if house == ah:
                continue
            damped = math.log1p(usage_counts[digest][house])
            term_atn += damped * atn_weight(house)
            if house_rep.get(house, 0.0) > 0:
                term_rep += damped * rep_share(house)   # rep-holders only
        if term_atn <= 0:
            continue
        raw_atn[digest] = term_atn
        raw_rep[digest] = term_rep
        node_atn[ah] = node_atn.get(ah, 0.0) + term_atn
        node_rep_basis[ah] = node_rep_basis.get(ah, 0.0) + term_rep

    # ---- POSITION DRIFT (B, C) + open-review bookkeeping (E') ---------
    positions_next: Dict[str, Dict[str, List[float]]] = {}
    for digest in sorted(set(reviews) | set(state.positions) |
                         set(state.registrations)):
        prior = state.positions.get(digest, {})
        head = list(prior.get("head") or [0.0] * N_DIMS)
        mass = list(prior.get("mass") or [1.0] * N_DIMS)
        if len(head) != N_DIMS:
            head = [0.0] * N_DIMS
        if len(mass) != N_DIMS:
            mass = [1.0] * N_DIMS

        revs = reviews.get(digest, [])
        ah = author_house.get(digest, "")
        add_mass = [0.0] * N_DIMS
        add_val = [0.0] * N_DIMS
        cells: Dict[str, Dict[str, Dict[str, float]]] = {}
        for r in revs:
            if r.household == ah:
                continue
            for axis_id, score in r.axes.items():
                agg = cells.setdefault(r.household, {}).setdefault(
                    axis_id, {"sum": 0.0, "n": 0})
                agg["sum"] += max(-1.0, min(1.0, score))
                agg["n"] += 1

        for house in sorted(cells.keys()):
            r = house_rep.get(house, 0.0)
            base_w = (r / supply) if (supply > 0 and r > 0) else 0.0
            if base_w <= 0.0:
                continue
            cred = state.credibility.get(house, 1.0)
            for axis_id in sorted(cells[house].keys()):
                idx = _axis_idx(axis_id)
                if idx is None:
                    continue
                cell = cells[house][axis_id]
                n = int(cell["n"])
                if n <= 0:
                    continue
                mean_score = cell["sum"] / n
                w = math.log1p(n) * base_w * cred
                add_mass[idx] += w
                add_val[idx] += w * mean_score
                # [v4.1-E'] record/refresh the open review against the
                # head AS IT WAS BEFORE this epoch's drift (so we can tell
                # if the score later moves toward or away from the review).
                key = (house, digest, axis_id)
                state.open_reviews[key] = OpenReview(
                    household=house, digest=digest, axis_id=axis_id,
                    score=mean_score, filed_epoch=state.epoch,
                    head_when_filed=head[idx])

        for i in range(N_DIMS):
            if add_mass[i] > 0.0:
                head[i] = (mass[i] * head[i] + add_val[i]) / (mass[i] + add_mass[i])
                mass[i] = mass[i] + add_mass[i]
        positions_next[digest] = {"head": [round(h, 9) for h in head],
                                  "mass": [round(m, 9) for m in mass]}
        state.review_mass[digest] = sum(mass) - N_DIMS  # subtract author prior

    state.positions = positions_next

    # ---- [v4.1-E'] CONTINUOUS REVERSAL-AWARE SANCTIONS ---------------
    # Re-evaluate EVERY open review against the CURRENT score. Symmetric:
    #   deviation > delta  -> dock proportional to (dev - delta)
    #   deviation <= delta -> restore proportional to how far inside delta
    # Only on tools whose review mass >= MASS_FLOOR (thin tools sanction
    # no one). This is what makes a later reversal retroactively dock the
    # early capturers: their open review's score is now far from the
    # reversed head, so they get docked in the epoch the reversal lands.
    cred_delta: Dict[str, float] = {}    # net signed adjustment this epoch
    for key, orv in list(state.open_reviews.items()):
        pos = positions_next.get(orv.digest)
        if not pos:
            continue
        total_mass = state.review_mass.get(orv.digest, 0.0)
        if total_mass < MASS_FLOOR:
            continue
        idx = _axis_idx(orv.axis_id)
        if idx is None:
            continue
        cur = pos["head"][idx]
        dev = abs(orv.score - cur)
        if dev > delta:
            penalty = (dev - delta) / (2.0 - delta)      # in [0,1]
            # accumulate the WORST dock across this household's reviews
            cred_delta[orv.household] = min(
                cred_delta.get(orv.household, 0.0), -penalty)
        else:
            # inside delta -> reward restoration proportional to closeness
            restore = (delta - dev) / max(1e-9, delta)   # in [0,1]
            cred_delta[orv.household] = max(
                cred_delta.get(orv.household, 0.0),
                +CRED_RECOVERY * restore)

    # apply: passive recovery toward 1.0, then the signed E' adjustment,
    # floored. A household with no open review on a mass-floored tool just
    # drifts back toward 1.0 (self-restore for honest reviewers).
    docked_this_epoch: Dict[str, float] = {}
    all_houses = set(list(state.credibility.keys()) + list(cred_delta.keys()))
    for house in all_houses:
        c = state.credibility.get(house, 1.0)
        adj = cred_delta.get(house)
        if adj is None:
            c = min(1.0, c + CRED_RECOVERY * (1.0 - c))   # passive restore
        elif adj < 0:
            c = c * (1.0 + adj)                            # proportional dock
            docked_this_epoch[house] = -adj
        else:
            c = min(1.0, c + adj * (1.0 - c))             # proportional restore
        state.credibility[house] = max(CRED_FLOOR, c)

    # ---- EMISSION POOL NORMALIZATION (ATN side) ----------------------
    raw_total = sum(node_atn.values())
    agent_mint: Dict[str, float] = {}
    scale = 0.0
    if raw_total > 0:
        scale = emission_pool / raw_total
        agent_mint = {h: node_atn[h] * scale for h in sorted(node_atn)}

    # [v4.1-D'] reputation increment = ONLY the rep-holder-attributable
    # portion, in the SAME pool-normalized ATN units. Zero-rep-only mint
    # grants ZERO reputation.
    rep_increment: Dict[str, float] = {}
    if scale > 0:
        for h in sorted(node_rep_basis):
            inc = node_rep_basis[h] * scale
            if inc > 0:
                rep_increment[h] = inc

    state.epoch += 1

    return {
        "agent_mint": agent_mint,                 # ATN (pool-normalized)
        "rep_increment": rep_increment,           # decoupled voice gain (D')
        "raw_atn": raw_atn,
        "raw_rep": raw_rep,
        "positions": positions_next,
        "credibility": dict(state.credibility),
        "docked_this_epoch": docked_this_epoch,
        "review_mass": dict(state.review_mass),
        "total_mint": sum(agent_mint.values()),
        # β-cap diagnostics: the observed zero-rep weight share BEFORE the
        # cap (the peg signal for the adaptive rule) and the scale applied.
        "observed_zero_share": observed_zero_share,
        "zero_cap_scale": zero_scale,
    }


def v41_rank_score(tool: Tool, positions: Dict[str, Dict[str, List[float]]]) -> float:
    """Discovery lift, unchanged: multiplicative, no filter."""
    pos = positions.get(tool.digest)
    head = list(pos.get("head")) if pos else [0.0] * N_DIMS
    if len(head) != N_DIMS:
        head = [0.0] * N_DIMS
    rating = (head[CORRECTNESS_AXIS] + head[SIMPLICITY_AXIS]) / 2.0
    return tool.topic_match * (1.0 + math.tanh(rating))
