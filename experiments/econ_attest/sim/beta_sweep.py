"""β-cap sweep — SIM-ONLY, on top of v4.1 (D'/E').

The user accepted a rep-INDEPENDENT aggregate cap on zero-rep ATN mint
weight but wants the VALUE to emerge from sims. This sweep quantifies the
core trade-off: β throttles ALL zero-rep usage weight, including honest
NEWCOMERS (who in a young network are most of the demand signal). Too low
-> honest authors' earnings stop tracking real newcomer demand; too high
-> dust rings skim.

Sweep: β ∈ {0.02,0.05,0.1,0.2,0.3,0.5} × network maturity (honest
zero-rep share of usage: 0.6 young / 0.3 growing / 0.1 mature) × dust ring
K ∈ {0,50,200}. Per cell:
  1. honest distortion = 1 - corr(honest true demand, realized mint) drop
     vs the UNCAPPED (β=None) baseline.
  2. sybil skim = ring's ATN pool share.
  3. combined verdict = smallest β with corr-drop < 0.05 at each maturity.

Then: does a simple ADAPTIVE rule (β pegged to last-epoch observed
zero-rep weight share, hard ceiling) beat any fixed value, and can a ring
game the peg (inflate observed share) beyond the ceiling?

Short run, reuses harness_v4_1.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from harness import Agent, honest_axes
from harness_v4_1 import V41Simulation

DELTA = 0.7
EPOCHS = 60
BETA_VALUES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
MATURITIES = {"young": 0.6, "growing": 0.3, "mature": 0.1}
K_VALUES = [0, 50, 200]


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def _build(sim: V41Simulation, *, n_honest: int, n_rep_users: int,
           n_newcomers: int, K: int):
    """Populate: honest authors, a pool of REP-holding users (established)
    and a pool of ZERO-rep newcomers (the young-network demand), plus a
    dust ring of K zero-rep sybils authoring junk tools."""
    rng = sim.rng
    authors = [sim.add_agent(Agent(f"auth_{i:02d}", kind="honest",
                                   owner=f"0xauth{i:02d}"))
               for i in range(n_honest)]
    rep_users = [sim.add_agent(Agent(f"rep_{i:02d}", kind="user",
                                     owner=f"0xrep{i:02d}"))
                 for i in range(n_rep_users)]
    for u in rep_users:
        u.reputation = 10.0
    newcomers = [sim.add_agent(Agent(f"new_{i:03d}", kind="user",
                                     owner=f"0xnew{i:03d}"))
                 for i in range(n_newcomers)]   # zero rep
    tools = [sim.new_tool(a.agent_id, kind="honest",
                          true_quality=0.2 + 0.7 * rng.random(),
                          topic_match=0.4 + 0.4 * rng.random())
             for a in authors]
    sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil",
                                  owner=f"0xsyb{i:03d}"))
              for i in range(K)]
    ring = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                         topic_match=0.2)
            for s in sybils]
    return authors, rep_users, newcomers, tools, sybils, ring


def _run_cell(*, seed, beta, zero_share, K, adaptive=False, beta_ceiling=0.2,
              ring_inflates=False) -> Dict[str, Any]:
    """One (β, maturity, K) run. ``zero_share`` = target honest zero-rep
    (newcomer) fraction of honest usage. Returns distortion + skim metrics.

    True demand per honest author = total honest receipts on its tool
    (rep-user + newcomer), the real signal the mint should track.
    """
    sim = V41Simulation(seed=seed, delta=DELTA, beta=beta, adaptive_beta=adaptive,
                        beta_ceiling=beta_ceiling)
    rng = sim.rng
    n_honest, n_rep_users, n_newcomers = 12, 12, 24
    authors, rep_users, newcomers, tools, sybils, ring = _build(
        sim, n_honest=n_honest, n_rep_users=n_rep_users,
        n_newcomers=n_newcomers, K=K)

    HONEST_USES = 150
    true_demand: Dict[str, float] = {t.digest: 0.0 for t in tools}
    realized_mint: Dict[str, float] = {t.digest: 0.0 for t in tools}
    sybil_atn_shares: List[float] = []
    observed_shares: List[float] = []

    for ep in range(EPOCHS):
        regs = [*tools, *ring] if ep == 0 else []
        usages = []
        # honest demand: split between rep-users and newcomers per zero_share
        for _ in range(HONEST_USES):
            t = rng.choice(tools)
            if rng.random() < zero_share:
                u = rng.choice(newcomers)     # zero-rep newcomer
            else:
                u = rng.choice(rep_users)     # established rep user
            ok = rng.random() < 0.8
            usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
            if ok:
                true_demand[t.digest] += 1.0
        # dust ring cross-attests its own tools (zero-rep)
        for i, s in enumerate(sybils):
            if K > 0:
                usages.append((s.agent_id, ring[(i + 1) % K], True,
                               {"correctness": 1.0}))
        # ring peg-gaming: also spray dust usage on HONEST tools to inflate
        # the observed zero-rep share (attack the adaptive peg).
        if ring_inflates and K > 0:
            for s in sybils:
                t = rng.choice(tools)
                usages.append((s.agent_id, t, True, None))

        result = sim.run_epoch(registrations=regs, usages=usages)
        assert result["total_mint"] <= sim.last_pool + 1e-6
        for t in tools:
            realized_mint[t.digest] += _mint_for_author(result, sim, t)
        sybil_atn = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
        sybil_atn_shares.append(sybil_atn / max(1e-9, result["total_mint"]))
        observed_shares.append(result.get("observed_zero_share", 0.0))

    corr = _pearson([true_demand[t.digest] for t in tools],
                    [realized_mint[t.digest] for t in tools])
    late = slice(-20, None)
    return {
        "corr_demand_mint": corr,
        "sybil_atn_share": sum(sybil_atn_shares[late]) / len(sybil_atn_shares[-20:]),
        "mean_observed_zero_share": sum(observed_shares[late]) / len(observed_shares[-20:]),
    }


def _mint_for_author(result, sim, tool) -> float:
    """ATN credited to a tool's author household this epoch (author == its
    own household in these populations)."""
    author = sim.agents[tool.author]
    return result["agent_mint"].get(author.owner or author.agent_id, 0.0)


def run_sweep(seed: int = 20) -> Dict[str, Any]:
    # baseline (uncapped) corr per (maturity, K) for the distortion delta
    uncapped: Dict[tuple, float] = {}
    for mat, zs in MATURITIES.items():
        for K in K_VALUES:
            r = _run_cell(seed=seed, beta=None, zero_share=zs, K=K)
            uncapped[(mat, K)] = r["corr_demand_mint"]

    cells: List[Dict[str, Any]] = []
    for mat, zs in MATURITIES.items():
        for K in K_VALUES:
            for beta in BETA_VALUES:
                r = _run_cell(seed=seed, beta=beta, zero_share=zs, K=K)
                base_corr = uncapped[(mat, K)]
                cells.append({
                    "maturity": mat, "zero_share": zs, "K": K, "beta": beta,
                    "corr_demand_mint": round(r["corr_demand_mint"], 4),
                    "corr_drop_vs_uncapped": round(base_corr - r["corr_demand_mint"], 4),
                    "sybil_atn_share": round(r["sybil_atn_share"], 4),
                    "observed_zero_share": round(r["mean_observed_zero_share"], 4),
                })

    # combined verdict: smallest β with corr-drop < 0.05 at each maturity
    # (worst case over K), and the sybil skim at that β (worst over K>0).
    beta_star: Dict[str, Any] = {}
    for mat in MATURITIES:
        chosen = None
        for beta in BETA_VALUES:
            drops = [c["corr_drop_vs_uncapped"] for c in cells
                     if c["maturity"] == mat and c["beta"] == beta]
            if drops and max(drops) < 0.05:
                chosen = beta
                break
        skim = None
        if chosen is not None:
            skims = [c["sybil_atn_share"] for c in cells
                     if c["maturity"] == mat and c["beta"] == chosen and c["K"] > 0]
            skim = round(max(skims), 4) if skims else None
        beta_star[mat] = {"beta_star": chosen, "sybil_skim_at_beta_star": skim,
                          "honest_zero_share": MATURITIES[mat]}

    # adaptive rule: peg β to observed zero-rep share, ceiling 0.2.
    adaptive: Dict[str, Any] = {}
    for mat, zs in MATURITIES.items():
        r = _run_cell(seed=seed, beta=None, zero_share=zs, K=200,
                      adaptive=True, beta_ceiling=0.2)
        base_corr = uncapped[(mat, 200)]
        adaptive[mat] = {
            "corr_demand_mint": round(r["corr_demand_mint"], 4),
            "corr_drop_vs_uncapped": round(base_corr - r["corr_demand_mint"], 4),
            "sybil_atn_share": round(r["sybil_atn_share"], 4),
            "mean_eff_zero_share": round(r["mean_observed_zero_share"], 4),
        }
    # peg-gaming: ring sprays honest tools to inflate observed share, young net
    gamed = _run_cell(seed=seed, beta=None, zero_share=0.6, K=200,
                      adaptive=True, beta_ceiling=0.2, ring_inflates=True)
    gamed_ceiling_only = _run_cell(seed=seed, beta=0.2, zero_share=0.6, K=200,
                                   ring_inflates=True)

    verdict = {
        "beta_star_by_maturity": beta_star,
        "adaptive_rule": adaptive,
        "peg_gaming": {
            "adaptive_ring_inflates_sybil_share": round(gamed["sybil_atn_share"], 4),
            "adaptive_ring_inflates_observed_share": round(
                gamed["mean_observed_zero_share"], 4),
            "fixed_ceiling_0.2_ring_inflates_sybil_share": round(
                gamed_ceiling_only["sybil_atn_share"], 4),
        },
    }
    return {"scenario": "beta_sweep_v4_1",
            "params": {"seed": seed, "epochs": EPOCHS, "beta_values": BETA_VALUES,
                       "maturities": MATURITIES, "K_values": K_VALUES,
                       "delta": DELTA},
            "uncapped_corr": {f"{m}|K{k}": round(v, 4) for (m, k), v in uncapped.items()},
            "cells": cells, "verdict": verdict}
