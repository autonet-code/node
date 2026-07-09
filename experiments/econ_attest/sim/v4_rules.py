"""v4 "gradient trust" ruleset — SIM-ONLY reimplementation.

=====================================================================
THIS IS NOT PRODUCTION CODE AND DOES NOT IMPORT THE REAL MINT/DRIFT
FUNCTIONS. The v4 rules do not exist in the codebase yet; the user
ratified the redesign and wants it sim-validated BEFORE any production
build. So — unlike harness.py, which imports compute_tool_mint verbatim
— this module RE-IMPLEMENTS the close-time math with the v4 changes.
It is a faithful port of the v3 algorithm in
nodes/common/federated_reconcile.py (household collapse, log1p usage
damper, per-axis mint-weighted position drift, emission-pool
normalization) with six deliberate deviations, each tagged [v4-X].
=====================================================================

The v4 ruleset (coordinator brief):

  A. No vet gate. Tools mint from first attested use; no greenlight
     quorum, no vet royalty.                                    [v4-A]
  B. Vetting merged into reviews. The old vet event survives as an
     INSPECTION review: per-axis review with NO usage receipt (read
     the code, didn't run it). Inspection reviews move POSITION like
     usage reviews but mint nothing.                            [v4-B]
  C. Zero-rep reviews carry no DRIFT weight. drift weight =
     household_rep/rep_supply with NO epsilon floor (zero-rep
     household -> zero drift mass). Author prior mass stays 1.0. [v4-C]
  D. epsilon capped on MINT weight. Zero-rep households' usage still
     mints, but the AGGREGATE mint weight of all zero-rep households
     per epoch is capped at fraction beta of total voice mass, split
     pro-rata among them.                                       [v4-D]
  E. Cross-epoch review sanctions (credibility). Per-household
     credibility c in (0,1], carried across epochs, multiplies that
     household's DRIFT weight (not mint). When a tool's score
     stabilizes (accumulated independent review mass >= Q), reviewers
     whose recorded review deviates > delta from the stabilized score
     get c docked proportional to deviation. Sanction-only, floor
     c=0.1, slow recovery +10%/epoch toward 1.                  [v4-E]

Everything else (household collapse, owner/wire exclusions, log1p
damper, emission pool = fixed pie, reputation = cumulative ATN) matches
v3 so the comparison is apples-to-apples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of

N_DIMS = 6
CORRECTNESS_AXIS = 4
SIMPLICITY_AXIS = 5

CRED_FLOOR = 0.1          # rule E: credibility never sanctioned below this
CRED_RECOVERY = 0.10      # rule E: +10%/epoch toward 1.0


@dataclass
class V4State:
    """Cross-epoch carry state the v4 close threads between epochs."""
    # digest -> {"head":[N], "mass":[N]}  (per-axis mint-weighted centroid)
    positions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # household -> credibility multiplier c in [floor, 1]
    credibility: Dict[str, float] = field(default_factory=dict)
    # digest -> accumulated independent review mass (for the Q gate)
    review_mass: Dict[str, float] = field(default_factory=dict)
    # digest -> whether its score has "stabilized" (mass crossed Q)
    stabilized: Dict[str, bool] = field(default_factory=dict)
    # registrations carry (digest -> {"author":..}) like v3
    registrations: Dict[str, Dict[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Review record: a review is (household, axis_scores, is_inspection)
# ---------------------------------------------------------------------------

@dataclass
class Review:
    household: str
    axes: Dict[str, float]        # axis_id -> score in [-1,1]
    inspection: bool = False      # True = inspection (no usage), rule B


def _voice_mass(agents: Dict[str, Agent]) -> Tuple[Dict[str, float], float]:
    """household -> reputation, and total supply. (No epsilon here; the
    epsilon logic differs per rule below.)"""
    house_rep: Dict[str, float] = {}
    for a in agents.values():
        house_rep[household_of(a)] = house_rep.get(household_of(a), 0.0) + a.reputation
    return house_rep, sum(house_rep.values())


def compute_v4_epoch(
    *,
    agents: Dict[str, Agent],
    state: V4State,
    # digest -> Tool (must include this epoch's registrations, merged into
    # state.registrations by the caller before/after)
    tools: Dict[str, Tool],
    # digest -> {household -> ok_attested_count}  (usage receipts, collapsed
    # per household already; author/self excluded by caller)
    usage_counts: Dict[str, Dict[str, float]],
    # digest -> list[Review]  (both usage-linked and inspection reviews)
    reviews: Dict[str, List[Review]],
    # author household per digest (excluded from its own tool)
    author_house: Dict[str, str],
    emission_pool: float,
    beta: float,
    delta: float,
    Q: float,
    epsilon: float = 0.05,
) -> Dict[str, Any]:
    """Run one v4 close. Returns:
      {"agent_mint": {household: atn},   # pool-normalized
       "raw_mint": {digest: raw_usage_term},
       "positions": {digest: {head, mass}},
       "credibility": {household: c},
       "stabilized": {digest: bool}}

    Mutates ``state`` in place (positions, credibility, review_mass,
    stabilized) so the caller threads it forward.
    """
    house_rep, supply = _voice_mass(agents)

    # ---- MINT WEIGHTS (rules A, D) -----------------------------------
    # Rule A: no vet gate — every registered pinned tool is mint-eligible
    # from its first attested use. (No greenlight filter.)
    #
    # Rule D: a household's mint weight is its rep share; zero-rep
    # households get epsilon, BUT the aggregate epsilon mass across ALL
    # zero-rep households is capped at beta * (total voice mass) and
    # split pro-rata. Non-zero-rep households keep rep/supply.
    def rep_share(house: str) -> float:
        return (house_rep.get(house, 0.0) / supply) if supply > 0 else 0.0

    # total voice mass of rep-holding households (the "real economy")
    rep_mass_total = sum(rep_share(h) for h in house_rep if house_rep[h] > 0)
    # identify zero-rep households that appear as callers this epoch
    zero_rep_callers = set()
    for d, counts in usage_counts.items():
        for h in counts:
            if house_rep.get(h, 0.0) <= 0.0:
                zero_rep_callers.add(h)
    n_zero = len(zero_rep_callers)
    # [v4-D] aggregate zero-rep mint budget = beta * total-mass, pro-rata
    if n_zero > 0:
        # total mass = rep mass + capped zero-rep budget; budget is beta of
        # the WHOLE (so zero-rep can never exceed beta share of the pie).
        # Solve budget b s.t. b/(rep_mass_total + b) = beta  =>
        #   b = beta/(1-beta) * rep_mass_total, capped, and if rep_mass_total
        #   is ~0 (cold start) fall back to beta absolute so bootstrap works.
        if rep_mass_total > 0:
            zero_budget = (beta / (1.0 - beta)) * rep_mass_total
        else:
            zero_budget = beta   # cold start: zero-rep carry beta absolute
        per_zero = zero_budget / n_zero
    else:
        per_zero = 0.0

    def mint_weight(house: str) -> float:
        r = house_rep.get(house, 0.0)
        if r > 0:
            return rep_share(house)
        return per_zero   # capped, pro-rata zero-rep weight

    raw_mint: Dict[str, float] = {}       # digest -> usage_term
    node_mint: Dict[str, float] = {}      # household -> raw atn (author)
    for digest in sorted(usage_counts.keys()):
        if digest not in state.registrations and digest not in tools:
            continue
        ah = author_house.get(digest, "")
        term = 0.0
        for house in sorted(usage_counts[digest].keys()):
            if house == ah:
                continue          # author's own household excluded (v3 rule)
            damped = math.log1p(usage_counts[digest][house])
            term += damped * mint_weight(house)
        if term <= 0:
            continue
        raw_mint[digest] = term
        author = tools[digest].author if digest in tools else \
            state.registrations.get(digest, {}).get("author", "")
        # attribute to author household
        node_mint[ah or author] = node_mint.get(ah or author, 0.0) + term

    # ---- POSITION DRIFT (rules B, C, E) ------------------------------
    # [v4-C] drift weight = credibility * (household_rep/supply) with NO
    #        epsilon floor. Zero-rep households move nothing.
    # [v4-B] inspection reviews (no usage) drift position, mint nothing.
    #        They are already in the `reviews` list flagged inspection=True.
    # [v4-E] credibility multiplier applied to drift weight.
    positions_next: Dict[str, Dict[str, List[float]]] = {}
    # collect, per household, the (digest, axis, score, weight) it filed —
    # for the post-close sanction pass.
    filed: List[Tuple[str, str, str, float, float]] = []  # house,digest,axis,score,w

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
        # pool per-household-per-axis (a household may file once; keep it
        # simple — sum then average like v3's cell aggregation)
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
            # [v4-C] zero-rep -> zero drift weight (NO epsilon)
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
                w = math.log1p(n) * base_w * cred   # [v4-E] credibility
                add_mass[idx] += w
                add_val[idx] += w * mean_score
                filed.append((house, digest, axis_id, mean_score, w))

        for i in range(N_DIMS):
            if add_mass[i] > 0.0:
                head[i] = (mass[i] * head[i] + add_val[i]) / (mass[i] + add_mass[i])
                mass[i] = mass[i] + add_mass[i]
        positions_next[digest] = {"head": [round(h, 9) for h in head],
                                  "mass": [round(m, 9) for m in mass]}

        # accumulate independent review mass for the Q/stabilize gate
        ep_mass = sum(add_mass)
        state.review_mass[digest] = state.review_mass.get(digest, 0.0) + ep_mass
        if not state.stabilized.get(digest) and state.review_mass[digest] >= Q:
            state.stabilized[digest] = True

    state.positions = positions_next

    # ---- CREDIBILITY SANCTION PASS (rule E) --------------------------
    # For every filed review on a STABILIZED tool, dock the reviewer's
    # credibility proportional to how far its score deviated from the
    # tool's (now stabilized) head on that axis. Sanction-only.
    docked: Dict[str, float] = {}
    for (house, digest, axis_id, score, w) in filed:
        if not state.stabilized.get(digest):
            continue
        idx = _axis_idx(axis_id)
        if idx is None:
            continue
        stable_score = positions_next[digest]["head"][idx]
        dev = abs(score - stable_score)
        if dev > delta:
            # dock proportional to deviation beyond delta, scaled to [0,2]
            # range of possible deviation.
            penalty = (dev - delta) / (2.0 - delta)   # in [0,1]
            docked[house] = max(docked.get(house, 0.0), penalty)

    # apply recovery first (+10%/epoch toward 1), then sanctions
    for house in set(list(state.credibility.keys()) + list(docked.keys())):
        c = state.credibility.get(house, 1.0)
        c = min(1.0, c + CRED_RECOVERY * (1.0 - c))     # slow recovery
        if house in docked:
            c = c * (1.0 - docked[house])               # proportional dock
        state.credibility[house] = max(CRED_FLOOR, c)

    # ---- EMISSION POOL NORMALIZATION (matches v3 apply_emission_pool) -
    raw_total = sum(node_mint.values())
    agent_mint: Dict[str, float] = {}
    if raw_total > 0:
        scale = emission_pool / raw_total
        agent_mint = {h: node_mint[h] * scale for h in sorted(node_mint)}

    return {
        "agent_mint": agent_mint,
        "raw_mint": raw_mint,
        "positions": positions_next,
        "credibility": dict(state.credibility),
        "stabilized": dict(state.stabilized),
        "review_mass": dict(state.review_mass),
        "docked_this_epoch": docked,
    }


def _axis_idx(axis_id: str) -> Optional[int]:
    return {"correctness": CORRECTNESS_AXIS, "simplicity": SIMPLICITY_AXIS,
            "life_precious": 0, "self_preservation": 1,
            "promotion_of_intelligence": 2, "evolution": 3}.get(axis_id)


def v4_rank_score(tool: Tool, positions: Dict[str, Dict[str, List[float]]]) -> float:
    """Same discovery lift as v3: base_cosine * (1 + tanh(mean(corr,simp)))
    on the drifted head. (Ranking rule is unchanged in v4.)"""
    pos = positions.get(tool.digest)
    head = list(pos.get("head")) if pos else [0.0] * N_DIMS
    if len(head) != N_DIMS:
        head = [0.0] * N_DIMS
    rating = (head[CORRECTNESS_AXIS] + head[SIMPLICITY_AXIS]) / 2.0
    return tool.topic_match * (1.0 + math.tanh(rating))
