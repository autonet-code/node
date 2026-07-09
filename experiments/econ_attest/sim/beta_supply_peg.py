"""Supply-pegged β schedule — end-to-end life-cycle validation. SIM-ONLY,
on top of v4.1 (D'/E' + the accepted rep-independent β cap).

The β-cap sweep (beta_sweep.py) showed a single fixed β can't be right:
cheap when mature, prohibitive when young. The proposed fix is to make β a
FUNCTION OF TOTAL REPUTATION SUPPLY (a maturity proxy that — under D' —
dust rings cannot inflate, since zero-rep mint grants no rep). This module
validates that schedule end to end.

Two functional forms (β_min = 0.05):
  hyperbolic:  β(S) = β_min + (1-β_min) * S0 / (S0 + S)
  exponential: β(S) = max(β_min, exp(-S / S0))

S0 is calibrated in "epochs-worth of pool" units: 1 epoch = 100 ATN of
rep supply, so S0 ∈ {10,50,200} epochs -> {1000, 5000, 20000} rep units.

ONE continuous life-cycle sim (not static cells): genesis (near-zero
supply -> β≈1, uncapped) -> supply grows endogenously from honest
rep-weighted activity -> dust rings injected in three windows (early /
middle / late). Measured over the whole trajectory:
  1. honest distortion per life stage,
  2. ring ATN skim per window,
  3. ring VOICE share (D' regression: must stay 0),
  4. adversarial peg check: supply trajectory with vs without rings
     (must be ~identical -> ring can't move β),
  5. the transition: worst-case corr-drop across the trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from harness import Agent, honest_axes
from harness_v4_1 import V41Simulation

BETA_MIN = 0.05
POOL = 100.0
DELTA = 0.7

# S0 in rep units = epochs * pool.
S0_EPOCHS = [10, 50, 200]
S0_VALUES = [e * POOL for e in S0_EPOCHS]

EPOCHS = 300
# life-stage windows (epoch ranges) + a dust ring active in each.
WINDOWS = {
    "early":  (30, 60),
    "middle": (130, 160),
    "late":   (240, 270),
}
RING_K = {"early": 200, "middle": 200, "late": 200}


def hyperbolic(S0: float) -> Callable[[float], float]:
    return lambda S: BETA_MIN + (1.0 - BETA_MIN) * S0 / (S0 + S)


def exponential(S0: float) -> Callable[[float], float]:
    return lambda S: max(BETA_MIN, math.exp(-S / S0))


FORMS = {"hyperbolic": hyperbolic, "exponential": exponential}


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def _run_lifecycle(*, seed: int, beta_fn: Optional[Callable[[float], float]],
                   inject_rings: bool = True,
                   newcomer_share_schedule: Optional[Callable[[int], float]] = None
                   ) -> Dict[str, Any]:
    """One continuous life-cycle. Returns per-epoch trajectory + per-window
    aggregates. ``beta_fn=None`` => uncapped (the honest-distortion
    baseline). ``inject_rings=False`` => the peg-adversary control (supply
    must match the ring case)."""
    import random as _random
    sim = V41Simulation(seed=seed, delta=DELTA, beta_fn=beta_fn)
    rng = sim.rng
    # DEDICATED promotion RNG, independent of ring activity, so the
    # adversarial peg check (supply with vs without rings) is not corrupted
    # by RNG-stream desync — the ring's usage draws must not shift which
    # newcomers get promoted. Honest maturation is thus identical in both
    # runs, isolating the ONLY channel by which a ring could move supply:
    # earning reputation (which D' forbids).
    promo_rng = _random.Random(seed ^ 0x9E3779B9)
    # DEDICATED honest-usage RNG too: ring tool creation / usage draws from
    # sim.rng, which would desync the honest demand stream between the with-
    # and without-rings runs. Isolating honest draws makes honest demand
    # bit-identical in both, so any supply difference is REAL rep leakage.
    honest_rng = _random.Random(seed ^ 0x51EDC6F1)

    # honest population: 12 authors, a growing rep-user base, and a pool of
    # zero-rep newcomers. Newcomer SHARE of honest demand falls as the
    # network matures (default schedule) unless overridden.
    n_authors = 12
    authors = [sim.add_agent(Agent(f"auth_{i:02d}", kind="honest",
                                   owner=f"0xauth{i:02d}"))
               for i in range(n_authors)]
    tools = [sim.new_tool(a.agent_id, kind="honest",
                          true_quality=0.2 + 0.7 * rng.random(),
                          topic_match=0.4 + 0.4 * rng.random())
             for a in authors]
    rep_users = [sim.add_agent(Agent(f"rep_{i:02d}", kind="user",
                                     owner=f"0xrep{i:02d}"))
                 for i in range(20)]
    # seed a few FOUNDER rep-holders so supply bootstraps from epoch 1
    # (genesis has near-zero supply -> β≈1 uncapped, which is correct, but
    # we want it to start GROWING immediately, not deadlock at zero). This
    # is a modest seed representing prior earned standing.
    for u in rep_users[:3]:
        u.reputation = 5.0
    # rep_users EARN their rep endogenously by authoring nothing but being
    # rep-holders is the maturity signal; we seed a slow trickle so supply
    # grows from honest mint (authors) — reviewers get rep only if they also
    # author. To keep it simple + honest: authors accrue rep from mint (D'),
    # and we promote a few reviewers to rep-holders over time to model a
    # maturing reviewer base.
    newcomers = [sim.add_agent(Agent(f"new_{i:03d}", kind="user",
                                     owner=f"0xnew{i:03d}"))
                 for i in range(40)]

    def newcomer_share(ep: int) -> float:
        if newcomer_share_schedule is not None:
            return newcomer_share_schedule(ep)
        # young: mostly newcomers; matures toward mostly rep-users.
        frac = 1.0 - ep / EPOCHS
        return max(0.1, 0.15 + 0.7 * frac)   # ~0.85 -> ~0.15

    HONEST_USES = 150
    records: List[Dict[str, Any]] = []
    # rolling per-window accumulators for demand-vs-mint correlation
    stage_demand: Dict[str, Dict[str, float]] = {w: {t.digest: 0.0 for t in tools}
                                                 for w in WINDOWS}
    stage_mint: Dict[str, Dict[str, float]] = {w: {t.digest: 0.0 for t in tools}
                                               for w in WINDOWS}
    # also a continuous rolling correlation window (every epoch)
    roll_demand = {t.digest: 0.0 for t in tools}
    roll_mint = {t.digest: 0.0 for t in tools}

    ring_agents: Dict[str, List[Agent]] = {}
    ring_tools: Dict[str, List] = {}

    for ep in range(EPOCHS):
        # which window (if any) is this epoch in?
        cur_window = None
        for w, (a, b) in WINDOWS.items():
            if a <= ep < b:
                cur_window = w
                break

        # spin up a ring at each window start (distinct sybils per window)
        if inject_rings and cur_window and cur_window not in ring_agents:
            K = RING_K[cur_window]
            ring_agents[cur_window] = [
                sim.add_agent(Agent(f"syb_{cur_window}_{i:03d}", kind="sybil",
                                    owner=f"0xsyb{cur_window}{i:03d}"))
                for i in range(K)]
            ring_tools[cur_window] = [
                sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                             topic_match=0.2)
                for s in ring_agents[cur_window]]

        regs = [t for t in tools if t.born_epoch == 0] if ep == 0 else []
        if cur_window and cur_window in ring_tools and \
                all(rt.born_epoch == ep for rt in ring_tools[cur_window]):
            regs = regs + ring_tools[cur_window]

        ncs = newcomer_share(ep)
        usages = []
        demand_this = {t.digest: 0.0 for t in tools}
        for _ in range(HONEST_USES):
            t = honest_rng.choice(tools)
            if honest_rng.random() < ncs:
                u = honest_rng.choice(newcomers)      # zero-rep
            else:
                u = honest_rng.choice(rep_users)      # rep-holder (see promotion)
            ok = honest_rng.random() < 0.8
            usages.append((u.agent_id, t, ok,
                           honest_axes(t, honest_rng) if ok else None))
            if ok:
                demand_this[t.digest] += 1.0

        # dust ring cross-attests its own ring tools (zero-rep), only while
        # its window is active.
        if inject_rings and cur_window and cur_window in ring_agents:
            agentsK = ring_agents[cur_window]
            toolsK = ring_tools[cur_window]
            K = len(agentsK)
            for i, s in enumerate(agentsK):
                usages.append((s.agent_id, toolsK[(i + 1) % K], True,
                               {"correctness": 1.0}))

        result = sim.run_epoch(registrations=regs, usages=usages)
        assert result["total_mint"] <= sim.last_pool + 1e-6

        # promote a reviewer to rep-holder occasionally so the rep-user base
        # (and thus supply) grows with maturity — models newcomers earning
        # standing. Promotion rate is independent of rings (honest-only).
        if ep > 0 and ep % 3 == 0:
            for u in newcomers:
                if u.reputation == 0.0 and promo_rng.random() < 0.02:
                    # give them a starter rep by having them "author" — but
                    # to keep D' clean we just move them into rep_users via a
                    # small rep grant representing prior earned mint.
                    u.reputation = 5.0
                    rep_users.append(u)
                    break

        supply = sum(a.reputation for a in sim.agents.values())
        eff_beta = sim._last_eff_beta if sim._last_eff_beta is not None else 1.0
        # ring metrics
        ring_atn = 0.0
        ring_voice = 0.0
        if cur_window and cur_window in ring_agents:
            ring_atn = sum(result["agent_mint"].get(s.owner, 0.0)
                           for s in ring_agents[cur_window])
            ring_voice = sum(s.reputation for s in ring_agents[cur_window])
        ring_atn_share = ring_atn / max(1e-9, result["total_mint"])

        # accumulate demand/mint for correlation
        for t in tools:
            m = result["agent_mint"].get(sim.agents[t.author].owner, 0.0)
            roll_demand[t.digest] += demand_this[t.digest]
            roll_mint[t.digest] += m
            if cur_window:
                stage_demand[cur_window][t.digest] += demand_this[t.digest]
                stage_mint[cur_window][t.digest] += m

        records.append({
            "epoch": ep, "supply": supply, "eff_beta": eff_beta,
            "newcomer_share": ncs, "window": cur_window,
            "ring_atn_share": ring_atn_share, "ring_voice": ring_voice,
            "total_mint": result["total_mint"],
        })

    # per-stage honest distortion: corr of demand vs mint within each window
    stage_corr = {w: _pearson([stage_demand[w][t.digest] for t in tools],
                              [stage_mint[w][t.digest] for t in tools])
                  for w in WINDOWS}
    return {
        "records": records,
        "stage_corr": stage_corr,
        "stage_ring_atn_share": {
            w: _window_mean(records, w, "ring_atn_share") for w in WINDOWS},
        "stage_ring_voice": {
            w: max((r["ring_voice"] for r in records if r["window"] == w),
                   default=0.0) for w in WINDOWS},
        "final_supply": records[-1]["supply"],
        "beta_trajectory": [(r["epoch"], round(r["eff_beta"], 4)) for r in records],
    }


def _window_mean(records, w, key):
    vals = [r[key] for r in records if r["window"] == w]
    return sum(vals) / len(vals) if vals else 0.0


def run(seed: int = 30) -> Dict[str, Any]:
    # baseline: uncapped, WITH rings — gives the per-stage honest corr the
    # schedule must not fall below by >0.05, and the accepted early skim.
    uncapped = _run_lifecycle(seed=seed, beta_fn=None, inject_rings=True)

    results: List[Dict[str, Any]] = []
    for form_name, form in FORMS.items():
        for S0e, S0 in zip(S0_EPOCHS, S0_VALUES):
            capped = _run_lifecycle(seed=seed, beta_fn=form(S0), inject_rings=True)
            # worst-case corr-drop across stages vs uncapped
            drops = {w: uncapped["stage_corr"][w] - capped["stage_corr"][w]
                     for w in WINDOWS}
            worst_drop = max(drops.values())
            late_skim = capped["stage_ring_atn_share"]["late"]
            max_voice = max(capped["stage_ring_voice"].values())
            results.append({
                "form": form_name, "S0_epochs": S0e, "S0": S0,
                "stage_corr": {w: round(capped["stage_corr"][w], 4) for w in WINDOWS},
                "stage_corr_drop": {w: round(drops[w], 4) for w in WINDOWS},
                "worst_corr_drop": round(worst_drop, 4),
                "stage_ring_atn_share": {
                    w: round(capped["stage_ring_atn_share"][w], 4) for w in WINDOWS},
                "late_ring_skim": round(late_skim, 4),
                "ring_voice_share_max": round(max_voice, 6),
                "final_supply": round(capped["final_supply"], 1),
                "beta_at_windows": {
                    w: round(_window_mean(capped["records"], w, "eff_beta"), 4)
                    for w in WINDOWS},
            })

    # --- (4) adversarial peg check: supply WITH vs WITHOUT rings, same
    # schedule and IDENTICAL honest behavior (dedicated honest/promotion
    # RNGs). Under D' the ring earns no rep, so it CANNOT directly move
    # supply. The only residual channel is pool DILUTION: a ring skimming
    # the fixed ATN pool leaves honest authors less ATN and thus less rep
    # per epoch, so honest supply grows slightly SLOWER with rings — which
    # keeps β slightly HIGHER (LOOSER), i.e. the only ring effect on β is
    # in the honest-FAVORABLE direction. A ring can never TIGHTEN β.
    form = hyperbolic(S0_VALUES[1])   # a representative schedule
    with_rings = _run_lifecycle(seed=seed, beta_fn=form, inject_rings=True)
    without_rings = _run_lifecycle(seed=seed, beta_fn=form, inject_rings=False)
    supply_wr = [r["supply"] for r in with_rings["records"]]
    supply_nr = [r["supply"] for r in without_rings["records"]]
    max_supply_gap = max(abs(a - b) for a, b in zip(supply_wr, supply_nr))
    rel_gap = max_supply_gap / max(1e-9, supply_nr[-1])
    beta_wr = [r["eff_beta"] for r in with_rings["records"]]
    beta_nr = [r["eff_beta"] for r in without_rings["records"]]
    # SIGNED β gap: positive = rings loosen β (honest-favorable). We report
    # the max magnitude AND the worst tightening (min signed gap) — the
    # attack direction, which should be ~0.
    beta_gaps = [a - b for a, b in zip(beta_wr, beta_nr)]
    max_beta_gap = max(abs(g) for g in beta_gaps)
    worst_tightening = min(beta_gaps)   # most-negative = ring tightened β

    # --- (3) global VOICE regression: rings must be voiceless throughout
    voice_ever = max(max(r["ring_voice"] for r in with_rings["records"]),
                     0.0)

    # recommendation: smallest worst-case corr-drop with late skim <= 0.05
    feasible = [r for r in results if r["late_ring_skim"] <= 0.05 + 1e-9]
    best = min(feasible, key=lambda r: r["worst_corr_drop"]) if feasible else \
        min(results, key=lambda r: r["worst_corr_drop"])

    verdict = {
        "uncapped_stage_corr": {w: round(uncapped["stage_corr"][w], 4) for w in WINDOWS},
        "uncapped_stage_ring_skim": {
            w: round(uncapped["stage_ring_atn_share"][w], 4) for w in WINDOWS},
        "peg_check_max_supply_gap": round(max_supply_gap, 4),
        "peg_check_rel_supply_gap": round(rel_gap, 6),
        "peg_check_max_beta_gap": round(max_beta_gap, 6),
        "peg_check_worst_beta_tightening": round(worst_tightening, 6),
        "peg_check_supply_direction": (
            "rings SLOW honest supply (pool dilution) -> β looser, "
            "honest-favorable" if supply_wr[-1] < supply_nr[-1]
            else "no adverse effect"),
        "ring_voice_ever": round(voice_ever, 6),
        "recommended": {k: best[k] for k in
                        ("form", "S0_epochs", "S0", "worst_corr_drop",
                         "late_ring_skim", "beta_at_windows")},
        "any_feasible": bool(feasible),
    }
    return {
        "scenario": "beta_supply_peg_v4_1",
        "params": {"seed": seed, "epochs": EPOCHS, "beta_min": BETA_MIN,
                   "forms": list(FORMS), "S0_epochs": S0_EPOCHS,
                   "windows": WINDOWS, "ring_K": RING_K},
        "grid": results,
        "peg_check": {
            "supply_with_rings_final": round(supply_wr[-1], 1),
            "supply_without_rings_final": round(supply_nr[-1], 1),
            "max_supply_gap": round(max_supply_gap, 4),
            "max_beta_gap": round(max_beta_gap, 6),
            "worst_beta_tightening": round(worst_tightening, 6),
        },
        "verdict": verdict,
    }
