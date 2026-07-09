"""v4 "gradient trust" scenarios — SIM-ONLY (see v4_rules.py header).

Mirrors scenarios.py but drives the v4 ruleset (harness_v4.V4Simulation /
v4_rules.compute_v4_epoch). Same populations, seeds, and metrics so the
v3-vs-v4 comparison is apples-to-apples. Adds two NEW scenarios the v4
rules enable: spam_burial (rule B inspection reviews) and
sanction_false_positives (rule E chilling-effect).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from harness import Agent, honest_axes
from harness_v4 import V4Simulation


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


# ===========================================================================
# 1. baseline_honest — cold-start check (rule C: no epsilon on drift)
# ===========================================================================

def baseline_honest(*, epochs: int = 200, seed: int = 1, beta: float = 0.1,
                    delta: float = 0.5, Q: float = 5.0,
                    n_authors: int = 20, n_users: int = 30,
                    uses_per_epoch: int = 120,
                    incumbent_rep: float = 5.0) -> Dict[str, Any]:
    """Same honest population as v3 baseline, BUT under rule C only
    rep-holding households move positions. We seed a modest incumbent rep
    distribution (a fraction of users are established reviewers with some
    rep) so the drift channel isn't dead at genesis, then measure how long
    a fresh good tool takes to earn a positive rating. Cold-start cost of
    rule C is reported explicitly.
    """
    sim = V4Simulation(seed=seed, beta=beta, delta=delta, Q=Q)
    rng = sim.rng

    authors = [sim.add_agent(Agent(f"author_{i:02d}", kind="honest",
                                   owner=f"0xauth{i:02d}"))
               for i in range(n_authors)]
    users = [sim.add_agent(Agent(f"user_{i:02d}", kind="user",
                                 owner=f"0xuser{i:02d}"))
             for i in range(n_users)]
    # seed a modest incumbent-reviewer rep distribution: 1/3 of users are
    # "established" with some prior reputation so drift can move at all.
    n_established = n_users // 3
    for u in users[:n_established]:
        u.reputation = incumbent_rep * (0.5 + rng.random())

    tools = []
    for i, a in enumerate(authors):
        q = -0.6 + 1.55 * (i / max(1, n_authors - 1))
        tools.append(sim.new_tool(a.agent_id, true_quality=q,
                                  topic_match=0.5 + 0.4 * rng.random()))

    # designate ONE fresh good tool published at epoch 0 to time its rating
    good = max(tools, key=lambda t: t.true_quality)

    records = []
    good_positive_epoch = None
    for ep in range(epochs):
        regs = tools if ep == 0 else []
        usages = []
        # discovery-weighted usage
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
            usages.append((u.agent_id, pick, ok,
                           honest_axes(pick, rng) if ok else None))
        result = sim.run_epoch(registrations=regs, usages=usages)
        assert result["total_mint"] <= sim.last_pool + 1e-6
        gh = result["positions"].get(good.digest, {}).get("head", [0] * 6)
        good_rating = (gh[4] + gh[5]) / 2
        if good_positive_epoch is None and good_rating > 0.1:
            good_positive_epoch = ep
        records.append({
            "epoch": ep,
            "total_mint": result["total_mint"],
            "good_rating": good_rating,
            "quality_vs_rank_corr": _pearson(
                [t.true_quality for t in tools], [sim.rank(t) for t in tools]),
        })

    cum = {t.digest: 0.0 for t in tools}
    # cumulative mint via raw shares is complex; approximate with rating-rank
    verdict = {
        "quality_vs_finalrank_corr": records[-1]["quality_vs_rank_corr"],
        "top_quality_rank": _rank_of(tools, sim.rank,
                                     key=lambda t: t.true_quality, reverse=True),
        "worst_quality_rank": _rank_of(tools, sim.rank,
                                       key=lambda t: t.true_quality, reverse=False),
        "cold_start_epochs_to_positive_rating": good_positive_epoch,
        "final_good_rating": records[-1]["good_rating"],
    }
    return {"scenario": "baseline_honest_v4",
            "params": {"epochs": epochs, "beta": beta, "delta": delta, "Q": Q,
                       "incumbent_rep": incumbent_rep},
            "epochs": records, "verdict": verdict}


# ===========================================================================
# 2. sybil_pump — expect capture ~1.0 and rank channel dead
# ===========================================================================

def sybil_pump(*, epochs: int = 120, seed: int = 2, beta: float = 0.1,
               delta: float = 0.5, Q: float = 5.0,
               K_values: Optional[List[int]] = None, n_honest: int = 10,
               honest_users: int = 20, uses_per_epoch: int = 80) -> Dict[str, Any]:
    if K_values is None:
        K_values = [0, 3, 10, 30, 100]
    sweep = []
    for K in K_values:
        sim = V4Simulation(seed=seed + K, beta=beta, delta=delta, Q=Q)
        rng = sim.rng
        attacker = sim.add_agent(Agent("attacker", kind="attacker", owner="0xatk"))
        ctrl = sim.add_agent(Agent("honest_ctrl", kind="honest", owner="0xctrl"))
        users = [sim.add_agent(Agent(f"hu_{i:02d}", kind="user", owner=f"0xhu{i:02d}"))
                 for i in range(honest_users)]
        # seed some user rep so drift works (established reviewers)
        for u in users[:honest_users // 3]:
            u.reputation = 5.0
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil",
                                      owner=f"0xsyb{i:03d}"))
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
                "atk_mint": result["raw_mint"].get(atk_tool.digest, 0.0),
                "ctrl_mint": result["raw_mint"].get(ctrl_tool.digest, 0.0),
                "atk_rank": sim.rank(atk_tool),
                "ctrl_rank": sim.rank(ctrl_tool),
            })
        atk_cum = sum(r["atk_mint"] for r in records)
        ctrl_cum = sum(r["ctrl_mint"] for r in records)
        sweep.append({
            "K": K,
            "capture_ratio": (atk_cum / ctrl_cum) if ctrl_cum > 0 else float("inf"),
            "final_atk_rank": records[-1]["atk_rank"],
            "final_ctrl_rank": records[-1]["ctrl_rank"],
            "rank_gap": records[-1]["atk_rank"] - records[-1]["ctrl_rank"],
            "records": records,
        })
    verdict = {
        "capture_ratio_by_K": {s["K"]: round(s["capture_ratio"], 4) for s in sweep},
        "rank_gap_by_K": {s["K"]: round(s["rank_gap"], 4) for s in sweep},
    }
    return {"scenario": "sybil_pump_v4",
            "params": {"epochs": epochs, "beta": beta, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 3. epsilon_faucet — expect sybil pool share plateau at ~beta regardless of K
# ===========================================================================

def epsilon_faucet(*, epochs: int = 120, seed: int = 3, beta: float = 0.1,
                   delta: float = 0.5, Q: float = 5.0,
                   K_values: Optional[List[int]] = None, n_honest: int = 20,
                   honest_users: int = 30,
                   honest_uses_per_epoch: int = 150) -> Dict[str, Any]:
    if K_values is None:
        K_values = [0, 5, 20, 50, 100, 200]
    sweep = []
    for K in K_values:
        sim = V4Simulation(seed=seed + K, beta=beta, delta=delta, Q=Q)
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
                if K > 0:
                    target = ring[(i + 1) % K]
                    usages.append((s.agent_id, target, True, {"correctness": 1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages)
            assert result["total_mint"] <= sim.last_pool + 1e-6
            sybil_mint = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
            records.append({
                "epoch": ep,
                "sybil_pool_share": (sybil_mint / result["total_mint"])
                if result["total_mint"] > 0 else 0.0,
            })
        sweep.append({
            "K": K,
            "final_pool_share": records[-1]["sybil_pool_share"],
            "mean_late_share": sum(r["sybil_pool_share"] for r in records[-20:]) / 20,
            "records": records,
        })
    verdict = {
        "final_pool_share_by_K": {s["K"]: round(s["final_pool_share"], 5) for s in sweep},
        "beta": beta,
        "plateau_at_beta": all(s["mean_late_share"] <= beta + 0.03
                               for s in sweep if s["K"] > 0),
    }
    return {"scenario": "epsilon_faucet_v4",
            "params": {"epochs": epochs, "beta": beta, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 4. review_nuke — nukers hold rep, get credibility-docked post-stabilization
# ===========================================================================

def review_nuke(*, epochs: int = 120, seed: int = 4, beta: float = 0.1,
                delta: float = 0.5, Q: float = 5.0,
                J_values: Optional[List[int]] = None, honest_users: int = 20,
                uses_per_epoch: int = 60, nuke_start: int = 1,
                nuker_rep: float = 10.0) -> Dict[str, Any]:
    """Nukers must now hold rep to move the drift (rule C), so we give
    them nuker_rep. Once the victim's score stabilizes (mass >= Q), a
    nuker whose -1 reviews deviate > delta from the stabilized (positive)
    head gets credibility docked (rule E). Measure (a) young-tool
    nukeability pre-stabilization, (b) the nukers' credibility cost.
    """
    if J_values is None:
        J_values = [0, 1, 3, 10, 30]
    sweep = []
    for J in J_values:
        sim = V4Simulation(seed=seed + J, beta=beta, delta=delta, Q=Q)
        rng = sim.rng
        va = sim.add_agent(Agent("victim_author", kind="honest", owner="0xvic"))
        ca = sim.add_agent(Agent("ctrl_author", kind="honest", owner="0xctrl"))
        users = [sim.add_agent(Agent(f"u_{i:02d}", kind="user", owner=f"0xu{i:02d}"))
                 for i in range(honest_users)]
        for u in users[:honest_users // 3]:
            u.reputation = 8.0
        nukers = [sim.add_agent(Agent(f"nk_{i:02d}", kind="attacker",
                                      owner=f"0xnk{i:02d}"))
                  for i in range(J)]
        for nk in nukers:
            nk.reputation = nuker_rep   # rule C: nukers must hold rep to bite
        victim = sim.new_tool("victim_author", kind="victim", true_quality=0.6)
        ctrl = sim.new_tool("ctrl_author", kind="honest", true_quality=0.6)
        records = []
        stabilized_epoch = None
        for ep in range(epochs):
            regs = [victim, ctrl] if ep == 0 else []
            usages = []
            for _ in range(uses_per_epoch):
                u = rng.choice(users)
                t = rng.choice([victim, ctrl])
                ok = rng.random() < 0.85
                usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
            if ep >= nuke_start:
                for nk in nukers:
                    usages.append((nk.agent_id, victim, True,
                                   {"correctness": -1.0, "simplicity": -1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages)
            assert result["total_mint"] <= sim.last_pool + 1e-6
            if stabilized_epoch is None and result["stabilized"].get(victim.digest):
                stabilized_epoch = ep
            nuker_cred = (sum(sim.state.credibility.get(nk.owner, 1.0)
                              for nk in nukers) / J) if J else 1.0
            records.append({
                "epoch": ep,
                "victim_rank": sim.rank(victim),
                "ctrl_rank": sim.rank(ctrl),
                "victim_stabilized": bool(result["stabilized"].get(victim.digest)),
                "mean_nuker_cred": nuker_cred,
            })
        final = records[-1]
        # pre-stabilization nukeability: min rank ratio before stabilize
        pre = [r for r in records if not r["victim_stabilized"]]
        min_pre_ratio = min((r["victim_rank"] / r["ctrl_rank"]
                             for r in pre if r["ctrl_rank"] > 0), default=1.0)
        sweep.append({
            "J": J,
            "final_rank_ratio": (final["victim_rank"] / final["ctrl_rank"])
            if final["ctrl_rank"] > 0 else float("inf"),
            "min_pre_stabilize_ratio": min_pre_ratio,
            "stabilized_epoch": stabilized_epoch,
            "final_nuker_cred": final["mean_nuker_cred"],
            "records": records,
        })
    verdict = {
        "final_rank_ratio_by_J": {s["J"]: round(s["final_rank_ratio"], 4) for s in sweep},
        "pre_stabilize_ratio_by_J": {s["J"]: round(s["min_pre_stabilize_ratio"], 4)
                                     for s in sweep},
        "final_nuker_cred_by_J": {s["J"]: round(s["final_nuker_cred"], 4) for s in sweep},
    }
    return {"scenario": "review_nuke_v4",
            "params": {"epochs": epochs, "beta": beta, "delta": delta, "Q": Q,
                       "J_values": J_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 5. spam_burial (NEW) — SEO flood + honest inspection reviews bury it
# ===========================================================================

def spam_burial(*, epochs: int = 120, seed: int = 5, beta: float = 0.1,
                delta: float = 0.5, Q: float = 5.0,
                M_values: Optional[List[int]] = None,
                inspectors: int = 5, honest_users: int = 20) -> Dict[str, Any]:
    """M SEO manifests flood a topic (high claimed topic_match, ~no real
    usage). Honest agents INSPECT some spam and file negative inspection
    reviews (rule B: drift without usage). Measure whether spam gets
    buried out of the top-k vs v3 (where inspection reviews didn't exist,
    so burial had nothing to grip — spam ranked on raw cosine).
    """
    if M_values is None:
        M_values = [5, 20, 50]
    K_TOP = 5
    sweep = []
    for M in M_values:
        sim = V4Simulation(seed=seed + M, beta=beta, delta=delta, Q=Q)
        rng = sim.rng
        honest_auth = sim.add_agent(Agent("honest", kind="honest", owner="0xh"))
        insp = [sim.add_agent(Agent(f"insp_{i}", kind="user", owner=f"0xin{i}"))
                for i in range(inspectors)]
        for a in insp:
            a.reputation = 10.0     # inspectors are established (rule C)
        users = [sim.add_agent(Agent(f"u_{i}", kind="user", owner=f"0xu{i}"))
                 for i in range(honest_users)]
        for u in users[:honest_users // 3]:
            u.reputation = 5.0
        # one honest tool, decent topic_match; M spam manifests with HIGHER
        # claimed topic_match (SEO keyword stuffing) but true_quality low
        honest_tool = sim.new_tool("honest", true_quality=0.7, topic_match=0.7)
        spam_authors = [sim.add_agent(Agent(f"spam_{i}", kind="sybil",
                                            owner=f"0xsp{i}"))
                        for i in range(M)]
        spam_tools = [sim.new_tool(sa.agent_id, kind="sybil", true_quality=-0.5,
                                   topic_match=0.85)
                      for sa in spam_authors]
        all_tools = [honest_tool, *spam_tools]
        records = []
        for ep in range(epochs):
            regs = all_tools if ep == 0 else []
            usages, inspections = [], []
            # honest users use the honest tool (spam has no real usage)
            for _ in range(honest_users * 3):
                u = rng.choice(users)
                ok = rng.random() < 0.8
                usages.append((u.agent_id, honest_tool, ok,
                               honest_axes(honest_tool, rng) if ok else None))
            # inspectors inspect a sample of spam each epoch, file negative
            # inspection reviews (read the code, it's junk).
            sample = rng.sample(spam_tools, min(len(spam_tools), inspectors * 2))
            for st in sample:
                insp_agent = rng.choice(insp)
                inspections.append((insp_agent.agent_id, st,
                                    {"correctness": -0.9, "simplicity": -0.8}))
            result = sim.run_epoch(registrations=regs, usages=usages,
                                   inspections=inspections)
            assert result["total_mint"] <= sim.last_pool + 1e-6
            # top-k by rank
            ranked = sorted(all_tools, key=lambda t: -sim.rank(t))
            honest_rank_pos = ranked.index(honest_tool) + 1
            spam_in_topk = sum(1 for t in ranked[:K_TOP] if t.kind == "sybil")
            records.append({
                "epoch": ep,
                "honest_rank_pos": honest_rank_pos,
                "spam_in_topk": spam_in_topk,
                "honest_in_topk": honest_rank_pos <= K_TOP,
            })
        final = records[-1]
        sweep.append({
            "M": M,
            "final_honest_rank_pos": final["honest_rank_pos"],
            "final_spam_in_topk": final["spam_in_topk"],
            "final_honest_in_topk": final["honest_in_topk"],
            # epoch honest first re-enters top-k after inspection burial
            "records": records,
        })
    verdict = {
        "honest_rank_pos_by_M": {s["M"]: s["final_honest_rank_pos"] for s in sweep},
        "spam_in_topk_by_M": {s["M"]: s["final_spam_in_topk"] for s in sweep},
        "honest_in_topk_by_M": {s["M"]: s["final_honest_in_topk"] for s in sweep},
    }
    return {"scenario": "spam_burial_v4",
            "params": {"epochs": epochs, "beta": beta, "M_values": M_values,
                       "K_top": K_TOP},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# 6. sanction_false_positives (NEW) — honest noisy reviewers, chilling price
# ===========================================================================

def sanction_false_positives(*, epochs: int = 120, seed: int = 6,
                             beta: float = 0.1,
                             delta_values: Optional[List[float]] = None,
                             Q_values: Optional[List[float]] = None,
                             n_reviewers: int = 30, n_tools: int = 8,
                             review_noise: float = 0.25) -> Dict[str, Any]:
    """ONLY honest noisy reviewers (no attackers). Measure the rate at
    which honest reviewers get credibility-docked (false positives) as a
    function of delta and Q. Recommend the (delta, Q) region where honest
    FP rate stays < ~2%.
    """
    if delta_values is None:
        delta_values = [0.3, 0.5, 0.7]
    if Q_values is None:
        Q_values = [3.0, 5.0, 10.0]
    grid = []
    for delta in delta_values:
        for Q in Q_values:
            sim = V4Simulation(seed=seed, beta=beta, delta=delta, Q=Q)
            rng = sim.rng
            authors = [sim.add_agent(Agent(f"a_{i}", kind="honest", owner=f"0xa{i}"))
                       for i in range(n_tools)]
            reviewers = [sim.add_agent(Agent(f"r_{i}", kind="user", owner=f"0xr{i}"))
                         for i in range(n_reviewers)]
            for rv in reviewers:
                rv.reputation = 5.0 + 5.0 * rng.random()   # all established
            tools = [sim.new_tool(a.agent_id, true_quality=-0.3 + 0.9 * rng.random())
                     for a in authors]
            docked_events = 0
            review_events = 0
            for ep in range(epochs):
                regs = tools if ep == 0 else []
                usages = []
                for _ in range(n_reviewers * 4):
                    rv = rng.choice(reviewers)
                    t = rng.choice(tools)
                    usages.append((rv.agent_id, t, True,
                                   honest_axes(t, rng, noise=review_noise)))
                    review_events += 1
                result = sim.run_epoch(registrations=regs, usages=usages)
                assert result["total_mint"] <= sim.last_pool + 1e-6
                docked_events += len(result["docked_this_epoch"])
            fp_rate = docked_events / max(1, review_events)
            # also: fraction of reviewers with credibility below 0.95 at end
            low_cred = sum(1 for rv in reviewers
                           if sim.state.credibility.get(rv.owner, 1.0) < 0.95)
            grid.append({
                "delta": delta, "Q": Q,
                "fp_dock_rate": fp_rate,
                "frac_reviewers_dinged": low_cred / n_reviewers,
                "mean_final_cred": sum(sim.state.credibility.get(rv.owner, 1.0)
                                       for rv in reviewers) / n_reviewers,
            })
    # recommend region: FP dock rate < 2% AND most reviewers keep full cred
    ok_cells = [g for g in grid
                if g["fp_dock_rate"] < 0.02 and g["frac_reviewers_dinged"] < 0.15]
    verdict = {
        "grid": [{k: round(v, 5) if isinstance(v, float) else v
                  for k, v in g.items()} for g in grid],
        "safe_region": [{"delta": g["delta"], "Q": g["Q"]} for g in ok_cells],
    }
    return {"scenario": "sanction_false_positives_v4",
            "params": {"epochs": epochs, "delta_values": delta_values,
                       "Q_values": Q_values, "review_noise": review_noise},
            "verdict": verdict}


# ===========================================================================
# 7. service_clone — quick sanity re-run under v4
# ===========================================================================

def service_clone(*, epochs: int = 200, seed: int = 7, beta: float = 0.1,
                  delta: float = 0.5, Q: float = 5.0,
                  phi: float = 0.7, demand: float = 200.0,
                  service_price: float = 0.5, c_local: float = 0.2,
                  clone_epoch: int = 40, n_honest: int = 15,
                  honest_users: int = 30, honest_uses_per_epoch: int = 120,
                  ) -> Dict[str, Any]:
    """Same structure as v3 service_clone but under v4 (no vet gate, no
    royalty drag). Expect unchanged or slightly better clone payback."""
    sim = V4Simulation(seed=seed, beta=beta, delta=delta, Q=Q)
    rng = sim.rng
    h_auth = [sim.add_agent(Agent(f"ha_{i:02d}", kind="honest", owner=f"0xha{i:02d}"))
              for i in range(n_honest)]
    h_users = [sim.add_agent(Agent(f"hu_{i:02d}", kind="user", owner=f"0xhu{i:02d}"))
               for i in range(honest_users)]
    for u in h_users[:honest_users // 3]:
        u.reputation = 5.0
    h_tools = [sim.new_tool(a.agent_id, true_quality=0.2 + 0.7 * rng.random())
               for a in h_auth]
    clone_auth = sim.add_agent(Agent("clone_author", kind="clone", owner="0xclone"))
    d_users = [sim.add_agent(Agent(f"du_{i:02d}", kind="user", owner=f"0xdu{i:02d}"))
               for i in range(honest_users)]
    for u in d_users[:honest_users // 3]:
        u.reputation = 5.0

    clone_tool = None
    users_switch = c_local < service_price
    switch_frac = phi if users_switch else 0.0
    RECYCLE = 0.0125
    clone_cum = 0.0
    records = []
    for ep in range(epochs):
        regs = list(h_tools) if ep == 0 else []
        served_clone = 0.0
        if clone_tool is not None:
            surfaced = sim.rank(clone_tool) > 0.1
            served_clone = demand * (switch_frac if surfaced else 0.0)
        served_service = demand - served_clone
        service_rev = served_service * service_price
        sim.recycled_fees = service_rev * RECYCLE
        usages = []
        for _ in range(honest_uses_per_epoch):
            u = rng.choice(h_users)
            t = rng.choice(h_tools)
            ok = rng.random() < 0.75
            usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
        if ep == clone_epoch:
            clone_tool = sim.new_tool("clone_author", kind="clone",
                                      true_quality=0.7, topic_match=0.75)
            regs.append(clone_tool)
        if clone_tool is not None and served_clone > 0:
            n_uses = max(1, int(round(served_clone / 5.0)))
            for k in range(n_uses):
                u = d_users[k % len(d_users)]
                ok = rng.random() < 0.9
                usages.append((u.agent_id, clone_tool, ok,
                               honest_axes(clone_tool, rng) if ok else None))
        result = sim.run_epoch(registrations=regs, usages=usages)
        assert result["total_mint"] <= sim.last_pool + 1e-6
        cm = result["agent_mint"].get("0xclone", 0.0)
        clone_cum += cm
        records.append({"epoch": ep, "service_rev": service_rev,
                        "clone_mint": cm, "clone_cum": clone_cum})
    verdict = {
        "clone_cum_mint": round(clone_cum, 3),
        "final_service_rev_frac": round(
            records[-1]["service_rev"] / (demand * service_price), 4),
        "expected_moat_rent_frac": round(1 - phi, 4),
    }
    return {"scenario": "service_clone_v4",
            "params": {"epochs": epochs, "phi": phi, "beta": beta},
            "epochs": records, "verdict": verdict}
