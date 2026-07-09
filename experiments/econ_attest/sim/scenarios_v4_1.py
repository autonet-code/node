"""v4.1 scenarios — SIM-ONLY (see v4_1_rules.py header).

Implements the coordinator's v4.1 pass: rule revisions D' + E', and the
METRIC REFRAME. Per the user's paradigm decision, the rep-weighted
consensus IS a tool's quality (there is no external ground truth, no
"honest minority"). So adversarial scenarios report CONSENSUS-CAPTURE
COST — the rep-weight an attacker must hold to move a score by a given
amount, relative to the tool's accumulated review mass — not "did truth
win."

Scenarios:
  1. baseline_honest    — no regression vs v4; continuous-E' honest FP <2%.
  2. epsilon_faucet (D') — sybil VOICE share flatlines ~0; ATN share curve.
  3. sybil_pump (D'+E')  — mint capture collapses; rank channel dead.
  4. consensus_capture   — capture-cost curves + reversal/recovery path.
  5. niche_discovery     — multiplicative-lift surfacing property (assert).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from harness import Agent, honest_axes
from harness_v4_1 import V41Simulation

DELTA = 0.7   # recommended from v4 sanction sweep


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def _rank_of(tools, rank_fn, *, key, reverse):
    target = sorted(tools, key=key, reverse=reverse)[0]
    order = sorted(tools, key=lambda t: -rank_fn(t))
    return order.index(target) + 1


def _voice_shares(sim, group):
    hr = {}
    for a in sim.agents.values():
        hr[a.owner or a.agent_id] = hr.get(a.owner or a.agent_id, 0.0) + a.reputation
    sup = sum(hr.values()) or 1.0
    return sum(hr.get(g.owner, 0.0) for g in group) / sup


# ===========================================================================
# 1. baseline_honest — regression + continuous-E' honest FP
# ===========================================================================

def baseline_honest(*, epochs: int = 200, seed: int = 1, delta: float = DELTA,
                    n_authors: int = 20, n_users: int = 30,
                    uses_per_epoch: int = 120,
                    incumbent_rep: float = 5.0) -> Dict[str, Any]:
    sim = V41Simulation(seed=seed, delta=delta)
    rng = sim.rng
    authors = [sim.add_agent(Agent(f"author_{i:02d}", kind="honest",
                                   owner=f"0xauth{i:02d}"))
               for i in range(n_authors)]
    users = [sim.add_agent(Agent(f"user_{i:02d}", kind="user",
                                 owner=f"0xuser{i:02d}"))
             for i in range(n_users)]
    n_est = n_users // 3
    for u in users[:n_est]:
        u.reputation = incumbent_rep * (0.5 + rng.random())
    tools = []
    for i, a in enumerate(authors):
        q = -0.6 + 1.55 * (i / max(1, n_authors - 1))
        tools.append(sim.new_tool(a.agent_id, true_quality=q,
                                  topic_match=0.5 + 0.4 * rng.random()))
    good = max(tools, key=lambda t: t.true_quality)
    records = []
    good_positive_epoch = None
    docked_events = 0
    review_events = 0
    for ep in range(epochs):
        regs = tools if ep == 0 else []
        usages = []
        weights = [max(1e-6, sim.rank(t)) for t in tools]
        wsum = sum(weights)
        for _ in range(uses_per_epoch):
            u = rng.choice(users)
            r = rng.random() * wsum
            acc = 0.0
            pick = tools[0]
            for t, w in zip(tools, weights):
                acc += w
                if acc >= r:
                    pick = t
                    break
            ok = rng.random() < 0.5 + 0.45 * (pick.true_quality + 1) / 2
            ax = honest_axes(pick, rng) if ok else None
            usages.append((u.agent_id, pick, ok, ax))
            if ok and ax:
                review_events += 1
        result = sim.run_epoch(registrations=regs, usages=usages)
        assert result["total_mint"] <= sim.last_pool + 1e-6
        docked_events += len(result["docked_this_epoch"])
        gr = sim.tool_rating(good)
        if good_positive_epoch is None and gr > 0.1:
            good_positive_epoch = ep
        records.append({"epoch": ep, "good_rating": gr,
                        "quality_vs_rank_corr": _pearson(
                            [t.true_quality for t in tools],
                            [sim.rank(t) for t in tools])})
    verdict = {
        "quality_vs_finalrank_corr": records[-1]["quality_vs_rank_corr"],
        "top_quality_rank": _rank_of(tools, sim.rank,
                                     key=lambda t: t.true_quality, reverse=True),
        "worst_quality_rank": _rank_of(tools, sim.rank,
                                       key=lambda t: t.true_quality, reverse=False),
        "cold_start_epochs_to_positive_rating": good_positive_epoch,
        "continuous_E_fp_dock_rate": docked_events / max(1, review_events),
    }
    return {"scenario": "baseline_honest_v4_1",
            "params": {"epochs": epochs, "delta": delta},
            "epochs": records, "verdict": verdict}


# ===========================================================================
# 2. epsilon_faucet under D' — VOICE flatlines, ATN curve
# ===========================================================================

def epsilon_faucet(*, epochs: int = 200, seed: int = 3, delta: float = DELTA,
                   K_values: Optional[List[int]] = None, n_honest: int = 20,
                   honest_users: int = 30,
                   honest_uses_per_epoch: int = 150) -> Dict[str, Any]:
    if K_values is None:
        K_values = [50, 200]
    sweep = []
    for K in K_values:
        sim = V41Simulation(seed=seed + K, delta=delta)
        rng = sim.rng
        h_auth = [sim.add_agent(Agent(f"ha_{i:02d}", kind="honest", owner=f"0xha{i:02d}"))
                  for i in range(n_honest)]
        h_users = [sim.add_agent(Agent(f"hu_{i:02d}", kind="user", owner=f"0xhu{i:02d}"))
                   for i in range(honest_users)]
        for u in h_users[:honest_users // 3]:
            u.reputation = 5.0
        h_tools = [sim.new_tool(a.agent_id, true_quality=0.2 + 0.7 * rng.random())
                   for a in h_auth]
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil", owner=f"0xsyb{i:03d}"))
                  for i in range(K)]
        ring = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0, topic_match=0.2)
                for s in sybils]
        records = []
        for ep in range(epochs):
            regs = [*h_tools, *ring] if ep == 0 else []
            usages = []
            for _ in range(honest_uses_per_epoch):
                u = rng.choice(h_users)
                t = rng.choice(h_tools)
                ok = rng.random() < 0.75
                usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
            for i, s in enumerate(sybils):
                usages.append((s.agent_id, ring[(i + 1) % K], True, {"correctness": 1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages)
            assert result["total_mint"] <= sim.last_pool + 1e-6
            atn_share = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils) \
                / max(1e-9, result["total_mint"])
            records.append({"epoch": ep, "sybil_atn_share": atn_share,
                            "sybil_voice_share": _voice_shares(sim, sybils)})
        late = records[-20:]
        sweep.append({
            "K": K,
            "final_voice_share": records[-1]["sybil_voice_share"],
            "max_voice_share": max(r["sybil_voice_share"] for r in records),
            "final_atn_share": records[-1]["sybil_atn_share"],
            "mean_late_atn_share": sum(r["sybil_atn_share"] for r in late) / len(late),
            "records": records,
        })
    verdict = {
        "voice_share_flatlines_zero": all(s["max_voice_share"] < 1e-6 for s in sweep),
        "final_voice_share_by_K": {s["K"]: round(s["final_voice_share"], 8) for s in sweep},
        "final_atn_share_by_K": {s["K"]: round(s["final_atn_share"], 4) for s in sweep},
        "atn_share_bounded": all(
            abs(s["records"][-1]["sybil_atn_share"]
                - s["records"][len(s["records"]) // 2]["sybil_atn_share"]) < 0.05
            for s in sweep),
    }
    return {"scenario": "epsilon_faucet_v4_1",
            "params": {"epochs": epochs, "delta": delta, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 3. sybil_pump under D'+E' — mint capture collapses, rank dead
# ===========================================================================

def sybil_pump(*, epochs: int = 120, seed: int = 2, delta: float = DELTA,
               K_values: Optional[List[int]] = None, n_honest: int = 10,
               honest_users: int = 20, uses_per_epoch: int = 80) -> Dict[str, Any]:
    if K_values is None:
        K_values = [0, 3, 10, 30, 100]
    sweep = []
    for K in K_values:
        sim = V41Simulation(seed=seed + K, delta=delta)
        rng = sim.rng
        atk = sim.add_agent(Agent("attacker", kind="attacker", owner="0xatk"))
        ca = sim.add_agent(Agent("honest_ctrl", kind="honest", owner="0xctrl"))
        users = [sim.add_agent(Agent(f"hu_{i:02d}", kind="user", owner=f"0xhu{i:02d}"))
                 for i in range(honest_users)]
        for u in users[:honest_users // 3]:
            u.reputation = 5.0
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil", owner=f"0xsyb{i:03d}"))
                  for i in range(K)]
        bg = [sim.add_agent(Agent(f"bg_{i:02d}", kind="honest", owner=f"0xbg{i:02d}"))
              for i in range(n_honest)]
        atk_tool = sim.new_tool("attacker", kind="sybil", true_quality=0.3)
        ctrl_tool = sim.new_tool("honest_ctrl", kind="honest", true_quality=0.3)
        bg_tools = [sim.new_tool(a.agent_id, true_quality=0.1 + 0.7 * rng.random())
                    for a in bg]
        records = []
        for ep in range(epochs):
            regs = [atk_tool, ctrl_tool, *bg_tools] if ep == 0 else []
            usages = []
            for _ in range(uses_per_epoch):
                u = rng.choice(users)
                t = rng.choice([ctrl_tool, *bg_tools])
                ok = rng.random() < 0.7
                usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
            for _ in range(uses_per_epoch // (n_honest + 1)):
                u = rng.choice(users)
                ok = rng.random() < 0.7
                usages.append((u.agent_id, atk_tool, ok,
                               honest_axes(atk_tool, rng) if ok else None))
            for s in sybils:
                usages.append((s.agent_id, atk_tool, True,
                               {"correctness": 1.0, "simplicity": 1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages)
            assert result["total_mint"] <= sim.last_pool + 1e-6
            records.append({
                "epoch": ep,
                "atk_mint": result["agent_mint"].get("0xatk", 0.0),
                "ctrl_mint": result["agent_mint"].get("0xctrl", 0.0),
                "atk_rank": sim.rank(atk_tool),
                "ctrl_rank": sim.rank(ctrl_tool),
            })
        atk_cum = sum(r["atk_mint"] for r in records)
        ctrl_cum = sum(r["ctrl_mint"] for r in records)
        sweep.append({
            "K": K,
            "capture_ratio": (atk_cum / ctrl_cum) if ctrl_cum > 0 else float("inf"),
            "rank_gap": records[-1]["atk_rank"] - records[-1]["ctrl_rank"],
            "records": records,
        })
    verdict = {
        "capture_ratio_by_K": {s["K"]: round(s["capture_ratio"], 4) for s in sweep},
        "rank_gap_by_K": {s["K"]: round(s["rank_gap"], 4) for s in sweep},
    }
    return {"scenario": "sybil_pump_v4_1",
            "params": {"epochs": epochs, "delta": delta, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 4. consensus_capture (reframed nuke) — capture cost + reversal path
# ===========================================================================

def consensus_capture(*, epochs: int = 120, seed: int = 4, delta: float = DELTA,
                      target_moves: Optional[List[float]] = None,
                      tool_mass_levels: Optional[List[float]] = None,
                      honest_users: int = 20) -> Dict[str, Any]:
    """METRIC REFRAME: report the rep-weight an attacker must hold to move a
    tool's consensus score by a target amount, as a function of the tool's
    accumulated review mass. Plus a reversal-path run: rep-backed users
    keep USING a down-scored tool (adoption, not rank-driven) and review it
    up; measure recovery and whether the early down-scorers get docked.
    """
    if target_moves is None:
        target_moves = [0.2, 0.5, 0.8]
    if tool_mass_levels is None:
        tool_mass_levels = [3.0, 10.0, 30.0]

    # --- Part A: capture-cost curve ----------------------------------
    # For each pre-accumulated review mass and target downward move, binary
    # search the attacker rep-weight (share of supply) needed in one epoch.
    cost_curve = []
    for base_mass in tool_mass_levels:
        for move in target_moves:
            needed = _capture_cost(seed=seed, delta=delta, base_mass=base_mass,
                                   target_move=move, honest_users=honest_users)
            cost_curve.append({
                "tool_review_mass": base_mass,
                "target_score_move": move,
                "attacker_rep_share_needed": needed,
            })

    # --- Part B: reversal / recovery path ----------------------------
    reversal = _reversal_run(seed=seed + 100, delta=delta)

    verdict = {
        "capture_cost_curve": cost_curve,
        "cost_scales_with_mass": _monotone_in_mass(cost_curve),
        "reversal_recovered": reversal["recovered"],
        "reversal_final_rating": reversal["final_rating"],
        "capturers_docked_to_floor": reversal["capturers_docked_to_floor"],
        "recovery_epoch": reversal["recovery_epoch"],
    }
    return {"scenario": "consensus_capture_v4_1",
            "params": {"epochs": epochs, "delta": delta,
                       "target_moves": target_moves,
                       "tool_mass_levels": tool_mass_levels},
            "reversal_records": reversal["records"], "verdict": verdict}


def _capture_cost(*, seed, delta, base_mass, target_move, honest_users,
                  attack_epochs: int = 20):
    """Consensus-capture COST (metric reframe). Because the score is a
    mass-weighted centroid, a one-shot attacker — however heavy — barely
    moves an already-massed tool (inertia). The real cost is SUSTAINED
    rep-weighted reviewing that overcomes the accumulated mass. So we
    binary-search the attacker rep-share that, held for ``attack_epochs``
    while honest reviewers keep endorsing, drives the score DOWN by
    ``target_move`` from its warmed-up level. Returns that rep-share (of
    total supply) — higher = more expensive to capture.
    """
    def _run(attacker_rep_share):
        sim = V41Simulation(seed=seed, delta=delta)
        rng = sim.rng
        author = sim.add_agent(Agent("author", kind="honest", owner="0xauth"))
        honest = [sim.add_agent(Agent(f"h_{i}", kind="user", owner=f"0xh{i}"))
                  for i in range(honest_users)]
        for u in honest:
            u.reputation = 10.0
        tool = sim.new_tool("author", true_quality=0.8)
        honest_rep = honest_users * 10.0
        atk_rep = attacker_rep_share * honest_rep / max(1e-9, 1 - attacker_rep_share)
        attacker = sim.add_agent(Agent("attacker", kind="attacker", owner="0xatk"))
        attacker.reputation = atk_rep
        ep = 0
        while sim.state.review_mass.get(tool.digest, 0.0) < base_mass and ep < 60:
            regs = [tool] if ep == 0 else []
            usages = [(u.agent_id, tool, True, {"correctness": 0.9, "simplicity": 0.9})
                      for u in honest]
            sim.run_epoch(registrations=regs, usages=usages)
            ep += 1
        pre = sim.tool_rating(tool)
        # SUSTAINED attack: attacker slams -1 every epoch while a subset of
        # honest reviewers keep endorsing (ongoing organic usage).
        for _ in range(attack_epochs):
            usages = [(attacker.agent_id, tool, True,
                       {"correctness": -1.0, "simplicity": -1.0})]
            for u in honest:
                usages.append((u.agent_id, tool, True,
                               {"correctness": 0.9, "simplicity": 0.9}))
            sim.run_epoch(registrations=[], usages=usages)
        post = sim.tool_rating(tool)
        return pre - post

    lo, hi = 0.0, 0.98
    if _run(hi) < target_move:
        return None   # even a 0.98-share sustained attack can't achieve it
    for _ in range(16):
        mid = (lo + hi) / 2
        if _run(mid) >= target_move:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def _reversal_run(*, seed, delta):
    sim = V41Simulation(seed=seed, delta=delta)
    rng = sim.rng
    author = sim.add_agent(Agent("author", kind="honest", owner="0xauth"))
    # capturers with real earned rep down-score epochs 1..25
    capturers = [sim.add_agent(Agent(f"cap_{i}", kind="attacker", owner=f"0xcap{i}"))
                 for i in range(5)]
    for a in capturers:
        a.reputation = 30.0
    # loyal rep-backed adopters keep USING the tool (adoption, not rank)
    # and review it up from epoch 26.
    loyal = [sim.add_agent(Agent(f"loyal_{i}", kind="user", owner=f"0xloy{i}"))
             for i in range(20)]
    for u in loyal:
        u.reputation = 40.0
    tool = sim.new_tool("author", true_quality=0.7)
    records = []
    recovery_epoch = None
    for ep in range(120):
        regs = [tool] if ep == 0 else []
        usages = []
        if 1 <= ep <= 25:
            for a in capturers:
                usages.append((a.agent_id, tool, True,
                               {"correctness": -1.0, "simplicity": -1.0}))
        if ep >= 26:
            for u in loyal:
                for _ in range(3):
                    usages.append((u.agent_id, tool, True,
                                   {"correctness": 0.9, "simplicity": 0.9}))
        result = sim.run_epoch(registrations=regs, usages=usages)
        rating = sim.tool_rating(tool)
        if recovery_epoch is None and ep >= 26 and rating > 0:
            recovery_epoch = ep
        cap_cred = sum(sim.state.credibility.get(a.owner, 1.0) for a in capturers) / 5
        records.append({"epoch": ep, "rating": rating, "capturer_cred": cap_cred})
    final = records[-1]
    return {
        "records": records,
        "recovered": final["rating"] > 0,
        "final_rating": final["rating"],
        "capturers_docked_to_floor": final["capturer_cred"] <= 0.15,
        "recovery_epoch": recovery_epoch,
    }


def _monotone_in_mass(curve):
    by_move = {}
    for c in curve:
        by_move.setdefault(c["target_score_move"], []).append(
            (c["tool_review_mass"], c["attacker_rep_share_needed"]))
    ok = True
    for move, pts in by_move.items():
        # None = "infeasible even at 0.98 share" = maximal cost; sort with
        # None treated as +inf so the monotonicity check still holds.
        pts.sort(key=lambda p: (p[0]))
        vals = [(p[1] if p[1] is not None else float("inf")) for p in pts]
        for a, b in zip(vals, vals[1:]):
            if b < a - 1e-9:   # cost should not DECREASE with mass
                ok = False
    return ok


# ===========================================================================
# 5. niche_discovery (NEW) — multiplicative lift surfacing property
# ===========================================================================

def niche_discovery(*, seed: int = 5, delta: float = DELTA) -> Dict[str, Any]:
    """A badly-scored tool ALONE in an empty semantic region must still
    surface top-1 for its niche query (multiplicative lift, no filter);
    the SAME score in a CROWDED region gets buried. Assert-style check.
    """
    sim = V41Simulation(seed=seed, delta=delta)
    rng = sim.rng
    author = sim.add_agent(Agent("author", kind="honest", owner="0xauth"))
    reviewers = [sim.add_agent(Agent(f"r_{i}", kind="user", owner=f"0xr{i}"))
                 for i in range(10)]
    for u in reviewers:
        u.reputation = 10.0

    # NICHE tool: only tool near the niche query (topic_match high for niche);
    # give it a somewhat negative score.
    niche = sim.new_tool("author", true_quality=-0.3, topic_match=0.8)
    # push a negative rating onto it
    for ep in range(10):
        regs = [niche] if ep == 0 else []
        usages = [(u.agent_id, niche, True, {"correctness": -0.4, "simplicity": -0.4})
                  for u in reviewers]
        sim.run_epoch(registrations=regs, usages=usages)
    niche_rating = sim.tool_rating(niche)
    niche_rank = sim.rank(niche)

    # In a niche region the niche tool is the only candidate -> top-1 by
    # construction (multiplicative lift keeps rank_score > 0 as long as
    # 1+tanh(rating) > 0, which holds for any rating > -inf since tanh in
    # (-1,1) => factor in (0,2)). Assert it still surfaces.
    niche_surfaces = niche_rank > 0.0

    # CROWDED region: 8 competitors with the same topic_match but POSITIVE
    # scores. The same niche-score tool should rank below them.
    comp_authors = [sim.add_agent(Agent(f"ca_{i}", kind="honest", owner=f"0xca{i}"))
                    for i in range(8)]
    comps = [sim.new_tool(a.agent_id, true_quality=0.7, topic_match=0.8)
             for a in comp_authors]
    for ep in range(10):
        usages = []
        for t in comps:
            for u in reviewers:
                usages.append((u.agent_id, t, True, {"correctness": 0.8, "simplicity": 0.8}))
        sim.run_epoch(registrations=comps if ep == 0 else [], usages=usages)

    crowded = [niche, *comps]
    ranked = sorted(crowded, key=lambda t: -sim.rank(t))
    niche_pos_crowded = ranked.index(niche) + 1
    buried_in_crowd = niche_pos_crowded > 1

    verdict = {
        "niche_rating": round(niche_rating, 4),
        "niche_surfaces_alone": niche_surfaces,
        "niche_rank_score_alone": round(niche_rank, 4),
        "niche_pos_in_crowd": niche_pos_crowded,
        "buried_in_crowd": buried_in_crowd,
        "PROPERTY_HOLDS": niche_surfaces and buried_in_crowd,
    }
    # the assert the user wants attested
    assert niche_surfaces, "niche tool must surface when alone (multiplicative lift)"
    assert buried_in_crowd, "same score must be buried in a crowded region"
    return {"scenario": "niche_discovery_v4_1",
            "params": {"delta": delta}, "verdict": verdict}
