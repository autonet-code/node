"""Fees-only + REP-from-earnings scenarios — SIM-ONLY
(see fees_only_rules.py header).

Adversarial validation of the 2026-07-10 ratified model BEFORE spec/build.
The pool is now 100% fee-funded (no base) and distributed over TOOL usage
shares; REP is claimed 1:1 on ATN EARNINGS. The heart of the study is the
USAGE-FLOOD LOOP (S2): tools are free, so a sybil ring floods usage
attestations on its own tool → captures a share of the fee pool (real ATN
paid by honest service customers) → those captures are "earnings" → the
ring claims REP on them → REP grants drift/review weight → self-reinforcing.
The ring never pays a fee.

S1 honest baseline (+ dead-start transition)
S2 usage-flood ring (one household AND K fake households; genesis/young/
   mature; K∈{5,20,100})  — THE loop
S3 wash trading (ring pays fees to its own service, reclaims via tool)
S4 whale spender (confirm zero REP, quantify externality)
S5 retroactivity (same-epoch vs carried dead-period usage)
S6 β/S0 relevance check under demand-backed REP + fee-volume growth curves
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, honest_axes
from harness_fees_only import FeesOnlySimulation
from fees_only_rules import FEE_RATE, BURN_FRACTION

DELTA = 0.7


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def _house_rep(sim, group):
    hr = {}
    for a in sim.agents.values():
        hr[a.owner or a.agent_id] = hr.get(a.owner or a.agent_id, 0.0) + a.reputation
    sup = sum(hr.values()) or 1.0
    return sum(hr.get(g.owner, 0.0) for g in group) / sup


# ===========================================================================
# S1  honest baseline + dead-start transition
# ===========================================================================

def s1_honest_baseline(*, epochs: int = 200, seed: int = 1, delta: float = DELTA,
                       n_authors: int = 15, n_providers: int = 8,
                       n_customers: int = 30, dead_epochs: int = 10,
                       calls_per_epoch: int = 200,
                       uses_per_epoch: int = 200) -> Dict[str, Any]:
    """Mixed service+tool economy, quality-correlated usage. Also the
    dead-start: `dead_epochs` epochs of ZERO service volume (pool=0, no
    mint), then demand arrives — confirm nothing pathological at the
    transition."""
    sim = FeesOnlySimulation(seed=seed, delta=delta)
    rng = sim.rng
    authors = [sim.add_agent(Agent(f"auth_{i:02d}", kind="honest", owner=f"0xauth{i:02d}"))
               for i in range(n_authors)]
    tools = []
    for i, a in enumerate(authors):
        q = -0.4 + 1.3 * (i / max(1, n_authors - 1))   # spread of true quality
        tools.append(sim.new_tool(a.agent_id, true_quality=q,
                                  topic_match=0.5 + 0.4 * rng.random()))
    # service providers (also tool authors — a household can do both)
    providers = [sim.add_agent(Agent(f"prov_{i:02d}", kind="honest", owner=f"0xprov{i:02d}"))
                 for i in range(n_providers)]
    svcs = [sim.new_service(p.agent_id, price=0.5 + rng.random())
            for p in providers]
    prov_quality = {s.service_id: 0.2 + 0.7 * rng.random() for s in svcs}
    customers = [sim.add_agent(Agent(f"cust_{i:02d}", kind="user", owner=f"0xcust{i:02d}"))
                 for i in range(n_customers)]

    records = []
    transition_ok = True
    for ep in range(epochs):
        regs = tools if ep == 0 else []
        # ---- service payments: zero during the dead period ----
        payments = []
        if ep >= dead_epochs:
            for _ in range(calls_per_epoch):
                s = max(svcs, key=lambda s: prov_quality[s.service_id] * rng.random())
                c = rng.choice(customers)
                payments.append((c.owner, s.service_id, s.price))
        # ---- tool usage: quality-correlated ----
        usages = []
        weights = [max(1e-6, sim.rank(t)) for t in tools]
        wsum = sum(weights)
        for _ in range(uses_per_epoch):
            u = rng.choice(customers)
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
        result = sim.run_epoch(registrations=regs, usages=usages, payments=payments)
        # conservation: total tool mint <= pool (fee-derived)
        assert result["total_tool_mint"] <= sim.last_pool + 1e-6, \
            f"ep{ep}: tool mint {result['total_tool_mint']} > pool {sim.last_pool}"
        # transition sanity: before first funded epoch pool must be 0 and
        # mint 0; at/after it must be >=0 with no negative rep anywhere.
        if ep < dead_epochs and (sim.last_pool > 1e-9 or result["total_tool_mint"] > 1e-9):
            transition_ok = False
        if any(a.reputation < -1e-9 for a in sim.agents.values()):
            transition_ok = False
        records.append({
            "epoch": ep, "pool": sim.last_pool, "gmv": result["service"]["gmv"],
            "burned": result["service"]["burned"],
            "tool_mint": result["total_tool_mint"],
            "supply": sim.supply(),
        })

    # correlations over cumulative earnings/rep vs true quality
    author_earn = [sim.state.cum_tool_earn.get(a.owner, 0.0) for a in authors]
    author_rep = [sim.agents[a.agent_id].reputation for a in authors]
    author_q = [t.true_quality for t in tools]
    prov_rep = {p.owner: sim.agents[p.agent_id].reputation for p in providers}
    total_gmv = sum(r["gmv"] for r in records)
    total_author_income = sum(author_earn)
    verdict = {
        "quality_vs_atn_earnings_corr": round(_pearson(author_q, author_earn), 4),
        "quality_vs_rep_corr": round(_pearson(author_q, author_rep), 4),
        "author_income_frac_of_gmv": round(total_author_income / max(1e-9, total_gmv), 5),
        "dead_start_transition_clean": transition_ok,
        "pool_zero_during_dead": all(r["pool"] < 1e-9 for r in records[:dead_epochs]),
        "pool_positive_after_dead": records[dead_epochs + 1]["pool"] > 0
        if len(records) > dead_epochs + 1 else False,
        "final_supply": round(sim.supply(), 3),
        "burn_frac_of_gmv": round(sum(r["burned"] for r in records)
                                  / max(1e-9, total_gmv), 5),
    }
    return {"scenario": "s1_honest_baseline", "params": {
        "epochs": epochs, "delta": delta, "dead_epochs": dead_epochs},
        "records": records, "verdict": verdict}


# ===========================================================================
# S2  usage-flood ring — THE loop
# ===========================================================================

def s2_usage_flood(*, epochs: int = 160, seed: int = 2, delta: float = DELTA,
                   K_values: Optional[List[int]] = None,
                   stages: Optional[List[str]] = None,
                   beta_fn=None,
                   service_rep_only: bool = False) -> Dict[str, Any]:
    """A sybil ring floods FREE usage attestations on its own ring tool to
    capture a share of the FEE-FUNDED pool, then claims REP on that capture.
    Run at three network stages (genesis/young/mature in fee-volume and
    REP-supply terms) and two ring topologies:
       - ONE household (all K sybils share a wallet)  — household damping
         should crush this to a single log1p term.
       - K FAKE households (each sybil its own wallet) — the ε·K faucet.
    Sweep K∈{5,20,100}. Measure pool-capture %, REP-share trajectory over
    120+ epochs, and whether capture COMPOUNDS via earnings→REP→weight.
    """
    if K_values is None:
        K_values = [5, 20, 100]
    if stages is None:
        stages = ["genesis", "young", "mature"]

    # stage config: how much honest fee-volume + honest rep-supply exists
    # BEFORE the ring shows up. genesis = nothing; young = some; mature =
    # lots. This sets the pool the ring is competing to skim and the rep
    # baseline its captured REP is measured against.
    STAGE = {
        "genesis": dict(honest_authors=3, honest_providers=2, honest_customers=8,
                        warmup=0, seed_rep=0.0),
        "young":   dict(honest_authors=8, honest_providers=5, honest_customers=20,
                        warmup=25, seed_rep=5.0),
        "mature":  dict(honest_authors=15, honest_providers=10, honest_customers=40,
                        warmup=80, seed_rep=10.0),
    }

    sweep = []
    for stage in stages:
        cfg = STAGE[stage]
        for K in K_values:
            for topo in ("one_house", "k_houses"):
                res = _run_flood(seed=seed + K + hash(stage + topo) % 1000,
                                 delta=delta, epochs=epochs, K=K, cfg=cfg,
                                 topo=topo, beta_fn=beta_fn,
                                 service_rep_only=service_rep_only)
                sweep.append({"stage": stage, "K": K, "topo": topo, **res})

    def _key(s):
        return f"{s['stage']}/K{s['K']}/{s['topo']}"
    worst = max(sweep, key=lambda s: s["transition_pool_capture"])
    verdict = {
        "pool_capture_late_by_cell": {_key(s): round(s["mean_late_pool_capture"], 4)
                                      for s in sweep},
        "pool_capture_transition_by_cell": {_key(s): round(s["transition_pool_capture"], 4)
                                            for s in sweep},
        "pool_capture_peak_by_cell": {_key(s): round(s["peak_pool_capture"], 4)
                                      for s in sweep},
        "ring_rep_share_final_by_cell": {_key(s): round(s["final_ring_rep_share"], 6)
                                         for s in sweep},
        "compounds_by_cell": {_key(s): s["compounds"] for s in sweep},
        "any_compounds": any(s["compounds"] for s in sweep),
        "max_transition_capture": round(max(s["transition_pool_capture"] for s in sweep), 4),
        "max_late_capture": round(max(s["mean_late_pool_capture"] for s in sweep), 4),
        "max_ring_rep_share": round(max(s["final_ring_rep_share"] for s in sweep), 6),
        "worst_transition_cell": _key(worst),
    }
    return {"scenario": "s2_usage_flood", "params": {
        "epochs": epochs, "delta": delta, "K_values": K_values, "stages": stages,
        "service_rep_only": service_rep_only,
        "beta": "supply_pegged" if beta_fn else None},
        "sweep": sweep, "verdict": verdict}


def _run_flood(*, seed, delta, epochs, K, cfg, topo, beta_fn, service_rep_only):
    sim = FeesOnlySimulation(seed=seed, delta=delta, beta_fn=beta_fn,
                             service_rep_only=service_rep_only)
    rng = sim.rng
    HA, HP, HC = cfg["honest_authors"], cfg["honest_providers"], cfg["honest_customers"]
    authors = [sim.add_agent(Agent(f"ha_{i:02d}", kind="honest", owner=f"0xha{i:02d}"))
               for i in range(HA)]
    tools = [sim.new_tool(a.agent_id, true_quality=0.2 + 0.6 * rng.random(),
                          topic_match=0.5 + 0.3 * rng.random()) for a in authors]
    providers = [sim.add_agent(Agent(f"hp_{i:02d}", kind="honest", owner=f"0xhp{i:02d}"))
                 for i in range(HP)]
    svcs = [sim.new_service(p.agent_id, price=0.5 + rng.random()) for p in providers]
    customers = [sim.add_agent(Agent(f"hc_{i:02d}", kind="user", owner=f"0xhc{i:02d}"))
                 for i in range(HC)]
    for a in authors[:max(1, HA // 2)]:
        a.reputation = cfg["seed_rep"]

    # the ring: K sybils. one_house => all share owner 0xring; k_houses =>
    # each its own wallet. Ring publishes ring tools and cross-attests.
    if topo == "one_house":
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil", owner="0xring"))
                  for i in range(K)]
    else:
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil", owner=f"0xsyb{i:03d}"))
                  for i in range(K)]
    ring_tools = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                               topic_match=0.2) for s in sybils]

    warmup = cfg["warmup"]
    records = []
    ring_started = False
    ring_start_ep = warmup
    for ep in range(epochs):
        regs = tools if ep == 0 else []
        # ring tools register at warmup (they exist but only start being
        # flooded at the ring_start).
        if ep == warmup:
            regs = regs + ring_tools

        payments = []
        for _ in range(150 if HC else 0):
            s = rng.choice(svcs) if svcs else None
            if s is None:
                break
            payments.append((rng.choice(customers).owner, s.service_id, s.price))

        usages = []
        # honest tool usage
        for _ in range(150):
            u = rng.choice(customers)
            t = rng.choice(tools)
            ok = rng.random() < 0.75
            usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
        # ring flood (free usage attestations on its own ring tools),
        # active from warmup onward.
        if ep >= warmup:
            ring_started = True
            for i, s in enumerate(sybils):
                usages.append((s.agent_id, ring_tools[(i + 1) % K], True,
                               {"correctness": 1.0}))
        result = sim.run_epoch(registrations=regs, usages=usages, payments=payments)
        assert result["total_tool_mint"] <= sim.last_pool + 1e-6

        ring_mint = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
        pool_capture = ring_mint / max(1e-9, result["total_tool_mint"]) \
            if result["total_tool_mint"] > 0 else 0.0
        ring_rep_share = _house_rep(sim, sybils)
        records.append({
            "epoch": ep, "pool": sim.last_pool,
            "tool_mint": result["total_tool_mint"],
            "ring_pool_capture": pool_capture,
            "ring_rep_share": ring_rep_share,
            "supply": sim.supply(),
            "eff_beta": result.get("eff_beta"),
        })

    # active = epochs where the ring is flooding AND a funded pool exists.
    active = [r for r in records if r["epoch"] >= ring_start_ep and r["pool"] > 1e-9]
    late = active[-20:] if len(active) >= 20 else active
    mean_late_capture = sum(r["ring_pool_capture"] for r in late) / max(1, len(late))
    # TRANSITION spike: capture in the ring's FIRST funded epoch — this is
    # the one-shot skim the model is actually exposed to (it collapses once
    # honest earners accrue rep). PEAK: the largest single-epoch capture.
    transition_capture = active[0]["ring_pool_capture"] if active else 0.0
    peak_capture = max((r["ring_pool_capture"] for r in active), default=0.0)
    # compounds := ring rep-share RISES over the active window (the feared
    # earnings→rep→weight loop). If rep-share falls, the loop self-limits.
    if len(active) >= 40:
        early_rs = sum(r["ring_rep_share"] for r in active[:20]) / 20
        late_rs = sum(r["ring_rep_share"] for r in active[-20:]) / 20
        compounds = late_rs > early_rs + 0.02
    else:
        early_rs = late_rs = active[-1]["ring_rep_share"] if active else 0.0
        compounds = False
    return {
        "mean_late_pool_capture": mean_late_capture,
        "transition_pool_capture": transition_capture,
        "peak_pool_capture": peak_capture,
        "final_ring_rep_share": records[-1]["ring_rep_share"],
        "early_active_rep_share": round(early_rs, 4),
        "late_active_rep_share": round(late_rs, 4),
        "compounds": bool(compounds),
        "records": records,
    }


# ===========================================================================
# S3  wash trading — ring pays fees to its OWN service, reclaims via tool
# ===========================================================================

def s3_wash_trading(*, epochs: int = 120, seed: int = 3, delta: float = DELTA,
                    K: int = 20, wash_gmv_per_epoch: float = 200.0) -> Dict[str, Any]:
    """Ring buys ATN, pays fees to its OWN service to inflate the pool,
    self-attests its tool to reclaim. Verify the STRICT-LOSS property (ring
    pays FEE_RATE=2.5%, reclaims at most its pro-rata share of the 1.25%
    burn) AND the REP side: does the wash generate REP the ring wouldn't
    otherwise get, and what does that VOICE cost per unit? Compare
    voice-per-dollar of washing vs honestly providing services.
    """
    # --- wash run: ring is its own customer + provider + tool author ---
    sim = FeesOnlySimulation(seed=seed, delta=delta)
    rng = sim.rng
    # an honest backdrop so the pool isn't 100% ring (realistic contention)
    h_auth = [sim.add_agent(Agent(f"ha_{i}", kind="honest", owner=f"0xha{i}"))
              for i in range(6)]
    h_tools = [sim.new_tool(a.agent_id, true_quality=0.3 + 0.5 * rng.random())
               for a in h_auth]
    h_prov = [sim.add_agent(Agent(f"hp_{i}", kind="honest", owner=f"0xhp{i}"))
              for i in range(4)]
    h_svc = [sim.new_service(p.agent_id, price=1.0) for p in h_prov]
    h_cust = [sim.add_agent(Agent(f"hc_{i}", kind="user", owner=f"0xhc{i}"))
              for i in range(15)]

    # the ring: sybil identities, a ring service, a ring tool.
    sybils = [sim.add_agent(Agent(f"syb_{i}", kind="sybil", owner=f"0xsyb{i}"))
              for i in range(K)]
    ring_provider = sim.add_agent(Agent("ring_prov", kind="sybil", owner="0xringP"))
    ring_svc = sim.new_service("ring_prov", price=1.0, service_id="ring_svc")
    ring_tools = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                               topic_match=0.2) for s in sybils]

    ring_fee_paid = 0.0
    ring_pool_reclaim = 0.0
    records = []
    for ep in range(epochs):
        regs = ([*h_tools, *ring_tools]) if ep == 0 else []
        payments = []
        # honest service demand
        for _ in range(120):
            payments.append((rng.choice(h_cust).owner,
                             rng.choice(h_svc).service_id, 1.0))
        # WASH: ring pays fees to its OWN service. wash_gmv_per_epoch of
        # self-dealing volume. Provider is the ring; customers are the ring
        # sybils. This is real ATN the ring moves to itself minus the fee.
        n_wash = int(wash_gmv_per_epoch)
        for _ in range(n_wash):
            cust = rng.choice(sybils)
            payments.append((cust.owner, "ring_svc", ring_svc.price))
        usages = []
        for _ in range(120):
            u = rng.choice(h_cust)
            t = rng.choice(h_tools)
            ok = rng.random() < 0.75
            usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
        # ring self-attests its tools to reclaim the inflated pool
        for i, s in enumerate(sybils):
            usages.append((s.agent_id, ring_tools[(i + 1) % K], True,
                           {"correctness": 1.0}))
        result = sim.run_epoch(registrations=regs, usages=usages, payments=payments)
        assert result["total_tool_mint"] <= sim.last_pool + 1e-6

        # ring fees paid this epoch = FEE_RATE * wash GMV
        wash_gmv = n_wash * ring_svc.price
        ring_fee_paid += FEE_RATE * wash_gmv
        ring_reclaim = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
        ring_pool_reclaim += ring_reclaim
        records.append({
            "epoch": ep, "pool": sim.last_pool,
            "wash_gmv": wash_gmv, "ring_reclaim": ring_reclaim,
            "ring_rep": sum(sim.agents[s.agent_id].reputation for s in sybils)
            + sim.agents["ring_prov"].reputation,
        })

    ring_rep_gained = (sum(sim.agents[s.agent_id].reputation for s in sybils)
                       + sim.agents["ring_prov"].reputation)
    # net ATN cost of the wash: fees paid MINUS pool reclaimed. (The wash
    # payments themselves are ring→ring, net zero except the fee leak.)
    net_atn_cost = ring_fee_paid - ring_pool_reclaim
    # voice-per-dollar of washing = rep gained / net ATN cost
    wash_voice_per_dollar = ring_rep_gained / max(1e-9, net_atn_cost) \
        if net_atn_cost > 0 else float("inf")

    # --- honest control: same ATN spent PROVIDING a real service ---
    # an honest provider who nets service revenue claims rep 1:1 on net.
    # voice-per-dollar honest = rep / net-revenue = 1/(fee overhead) ≈ 1
    # per the model (rep claimed 1:1 on net earnings). We measure it.
    sim2 = FeesOnlySimulation(seed=seed + 1, delta=delta)
    rng2 = sim2.rng
    hp = sim2.add_agent(Agent("honest_prov", kind="honest", owner="0xHP"))
    hsvc = sim2.new_service("honest_prov", price=1.0, service_id="hsvc")
    hcust = [sim2.add_agent(Agent(f"c_{i}", kind="user", owner=f"0xc{i}"))
             for i in range(15)]
    honest_net_rev = 0.0
    for ep in range(epochs):
        payments = [(rng2.choice(hcust).owner, "hsvc", 1.0) for _ in range(200)]
        result = sim2.run_epoch(registrations=[], usages=[], payments=payments)
        honest_net_rev += result["service"]["provider_net"].get("0xHP", 0.0)
    honest_rep = sim2.agents["honest_prov"].reputation
    honest_voice_per_dollar = honest_rep / max(1e-9, honest_net_rev)

    verdict = {
        "ring_fee_paid": round(ring_fee_paid, 3),
        "ring_pool_reclaimed": round(ring_pool_reclaim, 3),
        "net_atn_cost_of_wash": round(net_atn_cost, 3),
        "strict_loss_holds": net_atn_cost > 0,   # ring must lose ATN net
        "reclaim_frac_of_fee": round(ring_pool_reclaim / max(1e-9, ring_fee_paid), 4),
        "ring_rep_gained": round(ring_rep_gained, 3),
        "wash_voice_per_dollar": round(wash_voice_per_dollar, 4)
        if wash_voice_per_dollar != float("inf") else "inf",
        "honest_voice_per_dollar": round(honest_voice_per_dollar, 4),
        "wash_cheaper_than_honest_voice": (
            wash_voice_per_dollar > honest_voice_per_dollar
            if wash_voice_per_dollar != float("inf") else True),
    }
    return {"scenario": "s3_wash_trading", "params": {
        "epochs": epochs, "delta": delta, "K": K,
        "wash_gmv_per_epoch": wash_gmv_per_epoch},
        "records": records, "verdict": verdict}


# ===========================================================================
# S4  whale spender — confirm zero REP, quantify externality
# ===========================================================================

def s4_whale_spender(*, epochs: int = 120, seed: int = 4, delta: float = DELTA,
                     whale_calls_per_epoch: int = 500) -> Dict[str, Any]:
    """A large ATN buyer sprays service spend across honest providers.
    Confirm ZERO rep accrues to the whale (it only spends; never earns/
    authors) and quantify what its spending does to everyone else's
    pool/REP (it inflates the pool → more tool ATN for authors → more
    author REP; a positive externality by construction)."""
    def _run(with_whale):
        sim = FeesOnlySimulation(seed=seed, delta=delta)
        rng = sim.rng
        authors = [sim.add_agent(Agent(f"a_{i}", kind="honest", owner=f"0xa{i}"))
                   for i in range(10)]
        tools = [sim.new_tool(a.agent_id, true_quality=0.2 + 0.6 * rng.random())
                 for a in authors]
        provs = [sim.add_agent(Agent(f"p_{i}", kind="honest", owner=f"0xp{i}"))
                 for i in range(6)]
        svcs = [sim.new_service(p.agent_id, price=1.0) for p in provs]
        custs = [sim.add_agent(Agent(f"c_{i}", kind="user", owner=f"0xc{i}"))
                 for i in range(20)]
        whale = sim.add_agent(Agent("whale", kind="user", owner="0xwhale"))
        for ep in range(epochs):
            regs = tools if ep == 0 else []
            payments = [(rng.choice(custs).owner, rng.choice(svcs).service_id, 1.0)
                        for _ in range(100)]
            if with_whale:
                for _ in range(whale_calls_per_epoch):
                    payments.append(("0xwhale", rng.choice(svcs).service_id, 1.0))
            usages = []
            for _ in range(150):
                u = rng.choice(custs)
                t = rng.choice(tools)
                ok = rng.random() < 0.75
                usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
            sim.run_epoch(registrations=regs, usages=usages, payments=payments)
        author_rep = sum(sim.agents[a.agent_id].reputation for a in authors)
        prov_rep = sum(sim.agents[p.agent_id].reputation for p in provs)
        return sim, author_rep, prov_rep

    sim_w, author_rep_w, prov_rep_w = _run(True)
    sim_n, author_rep_n, prov_rep_n = _run(False)
    whale_rep = sim_w.agents["whale"].reputation
    verdict = {
        "whale_rep": round(whale_rep, 6),
        "whale_earns_zero_rep": whale_rep < 1e-9,
        "author_rep_with_whale": round(author_rep_w, 2),
        "author_rep_without_whale": round(author_rep_n, 2),
        "author_rep_uplift_from_whale": round(author_rep_w - author_rep_n, 2),
        "provider_rep_with_whale": round(prov_rep_w, 2),
        "provider_rep_without_whale": round(prov_rep_n, 2),
        "whale_supply_share": round(
            whale_rep / max(1e-9, sim_w.supply()), 6),
    }
    return {"scenario": "s4_whale_spender", "params": {
        "epochs": epochs, "delta": delta,
        "whale_calls_per_epoch": whale_calls_per_epoch},
        "verdict": verdict}


# ===========================================================================
# S5  retroactivity — same-epoch vs carried dead-period usage
# ===========================================================================

def s5_retroactivity(*, epochs: int = 120, seed: int = 5, delta: float = DELTA,
                     dead_epochs: int = 20, K: int = 50) -> Dict[str, Any]:
    """A ring can pre-farm usage during the FREE dead period (pool=0) at
    zero cost. Compare two accounting rules:
      - same-epoch-only: usage in a zero-pool epoch mints nothing and is
        DISCARDED (retroactive_usage=False).
      - carried: dead-period usage is banked and folded into the first
        funded epoch (retroactive_usage=True) — the ring's pre-farm cashes
        in the instant demand arrives.
    Quantify the difference in ring pool-capture at the transition.
    """
    def _run(retro, honest_active_in_dead):
        sim = FeesOnlySimulation(seed=seed, delta=delta, retroactive_usage=retro)
        rng = sim.rng
        authors = [sim.add_agent(Agent(f"a_{i}", kind="honest", owner=f"0xa{i}"))
                   for i in range(8)]
        tools = [sim.new_tool(a.agent_id, true_quality=0.3 + 0.5 * rng.random())
                 for a in authors]
        provs = [sim.add_agent(Agent(f"p_{i}", kind="honest", owner=f"0xp{i}"))
                 for i in range(5)]
        svcs = [sim.new_service(p.agent_id, price=1.0) for p in provs]
        custs = [sim.add_agent(Agent(f"c_{i}", kind="user", owner=f"0xc{i}"))
                 for i in range(20)]
        sybils = [sim.add_agent(Agent(f"s_{i}", kind="sybil", owner=f"0xs{i}"))
                  for i in range(K)]
        ring_tools = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                                   topic_match=0.2) for s in sybils]
        records = []
        for ep in range(epochs):
            regs = ([*tools, *ring_tools]) if ep == 0 else []
            payments = []
            if ep >= dead_epochs:
                payments = [(rng.choice(custs).owner, rng.choice(svcs).service_id, 1.0)
                            for _ in range(150)]
            usages = []
            # honest tool usage. During the dead period honest users may be
            # IDLE (no service demand → no reason to be around) — the case
            # where ONLY the ring pre-farms. Post-dead they always use.
            if ep >= dead_epochs or honest_active_in_dead:
                for _ in range(150):
                    u = rng.choice(custs)
                    t = rng.choice(tools)
                    ok = rng.random() < 0.75
                    usages.append((u.agent_id, t, ok,
                                   honest_axes(t, rng) if ok else None))
            # ring floods EVERY epoch including the dead period (free)
            for i, s in enumerate(sybils):
                usages.append((s.agent_id, ring_tools[(i + 1) % K], True,
                               {"correctness": 1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages, payments=payments)
            ring_mint = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
            cap = ring_mint / max(1e-9, result["total_tool_mint"]) \
                if result["total_tool_mint"] > 0 else 0.0
            records.append({"epoch": ep, "pool": sim.last_pool,
                            "ring_capture": cap,
                            "tool_mint": result["total_tool_mint"]})
        trans = next((r for r in records if r["pool"] > 1e-9), None)
        trans_capture = trans["ring_capture"] if trans else 0.0
        ss = sum(r["ring_capture"] for r in records[-20:]) / 20
        return {"trans_capture": trans_capture, "ss_capture": ss,
                "trans_epoch": trans["epoch"] if trans else None,
                "records": records}

    # two demand regimes during the dead period:
    #   busy: honest users use tools during the dead period too (so their
    #         usage also banks under carry) — carry DILUTES the ring.
    #   idle: only the ring pre-farms during the dead period (the realistic
    #         dead-start: no service volume, no honest tool traffic) — carry
    #         cashes in the ring's zero-cost pre-farm, STRICTLY worse.
    busy_same = _run(False, True)
    busy_carry = _run(True, True)
    idle_same = _run(False, False)
    idle_carry = _run(True, False)
    verdict = {
        "busy_same_epoch_transition_capture": round(busy_same["trans_capture"], 4),
        "busy_carried_transition_capture": round(busy_carry["trans_capture"], 4),
        "idle_same_epoch_transition_capture": round(idle_same["trans_capture"], 4),
        "idle_carried_transition_capture": round(idle_carry["trans_capture"], 4),
        "busy_retro_amplification": round(
            busy_carry["trans_capture"] / max(1e-9, busy_same["trans_capture"]), 3),
        "idle_retro_amplification": round(
            idle_carry["trans_capture"] / max(1e-9, idle_same["trans_capture"]), 3),
        "retro_more_capturable_when_busy":
            busy_carry["trans_capture"] > busy_same["trans_capture"] + 1e-6,
        "retro_more_capturable_when_idle":
            idle_carry["trans_capture"] > idle_same["trans_capture"] + 1e-6,
        "steady_capture_same_epoch": round(busy_same["ss_capture"], 4),
    }
    return {"scenario": "s5_retroactivity", "params": {
        "epochs": epochs, "delta": delta, "dead_epochs": dead_epochs, "K": K},
        "idle_carried_records": idle_carry["records"], "verdict": verdict}


# ===========================================================================
# S6  β/S0 relevance check under demand-backed REP + fee-volume growth
# ===========================================================================

def s6_beta_relevance(*, epochs: int = 200, seed: int = 6, delta: float = DELTA,
                      K: int = 100) -> Dict[str, Any]:
    """With REP demand-backed (claimed on earnings), is the β cap still
    load-bearing (per S2) or moot? And does the supply peg exp(−S/S0) still
    behave when REP supply now grows with FEE VOLUME (not a fixed pool per
    epoch)? Sweep fee-volume growth curves (dead→slow, dead→hot,
    hot-from-genesis) and report whether one S0 works across the envelope
    or the peg must be re-denominated (e.g. pegged to cumulative burned
    fees instead of REP supply).
    """
    BETA_MIN = 0.05

    def exp_sched(S0):
        return lambda S: max(BETA_MIN, math.exp(-S / S0))

    # fee-volume growth curves: calls_per_epoch(ep). Controls pool size and
    # thus how fast REP supply matures (rep is claimed on earnings).
    def dead_slow(ep):   return 0 if ep < 20 else int(min(300, 5 * (ep - 20)))
    def dead_hot(ep):    return 0 if ep < 20 else 300
    def hot_genesis(ep): return 300
    CURVES = {"dead_slow": dead_slow, "dead_hot": dead_hot,
              "hot_from_genesis": hot_genesis}
    S0_VALUES = {"S0_10": 10 * 100, "S0_50": 50 * 100, "S0_200": 200 * 100}

    def _run(curve_fn, beta_fn):
        sim = FeesOnlySimulation(seed=seed, delta=delta, beta_fn=beta_fn)
        rng = sim.rng
        authors = [sim.add_agent(Agent(f"a_{i}", kind="honest", owner=f"0xa{i}"))
                   for i in range(12)]
        tools = [sim.new_tool(a.agent_id, true_quality=0.2 + 0.6 * rng.random())
                 for a in authors]
        provs = [sim.add_agent(Agent(f"p_{i}", kind="honest", owner=f"0xp{i}"))
                 for i in range(8)]
        svcs = [sim.new_service(p.agent_id, price=1.0) for p in provs]
        custs = [sim.add_agent(Agent(f"c_{i}", kind="user", owner=f"0xc{i}"))
                 for i in range(30)]
        newcomers = [sim.add_agent(Agent(f"n_{i}", kind="user", owner=f"0xn{i}"))
                     for i in range(40)]
        sybils = [sim.add_agent(Agent(f"s_{i}", kind="sybil", owner=f"0xs{i}"))
                  for i in range(K)]
        ring_tools = [sim.new_tool(s.agent_id, kind="sybil", true_quality=0.0,
                                   topic_match=0.2) for s in sybils]
        records = []
        demand = {t.digest: 0.0 for t in tools}
        mint_cum = {t.digest: 0.0 for t in tools}
        for ep in range(epochs):
            regs = ([*tools, *ring_tools]) if ep == 0 else []
            n_calls = curve_fn(ep)
            payments = [(rng.choice(custs).owner, rng.choice(svcs).service_id, 1.0)
                        for _ in range(n_calls)]
            # newcomer share of honest tool demand falls as network matures
            ncs = max(0.1, 0.85 - 0.75 * ep / epochs)
            usages = []
            for _ in range(150):
                t = rng.choice(tools)
                u = rng.choice(newcomers) if rng.random() < ncs else rng.choice(custs)
                ok = rng.random() < 0.8
                usages.append((u.agent_id, t, ok, honest_axes(t, rng) if ok else None))
                if ok:
                    demand[t.digest] += 1.0
            for i, s in enumerate(sybils):
                usages.append((s.agent_id, ring_tools[(i + 1) % K], True,
                               {"correctness": 1.0}))
            result = sim.run_epoch(registrations=regs, usages=usages, payments=payments)
            for t in tools:
                mint_cum[t.digest] += result["agent_mint"].get(
                    sim.agents[t.author].owner, 0.0)
            ring_mint = sum(result["agent_mint"].get(s.owner, 0.0) for s in sybils)
            ring_cap = ring_mint / max(1e-9, result["total_tool_mint"]) \
                if result["total_tool_mint"] > 0 else 0.0
            records.append({"epoch": ep, "pool": sim.last_pool,
                            "supply": sim.supply(), "ring_capture": ring_cap,
                            "eff_beta": result.get("eff_beta")})
        honest_corr = _pearson([demand[t.digest] for t in tools],
                               [mint_cum[t.digest] for t in tools])
        late = records[-20:]
        return {"honest_corr": honest_corr,
                "late_ring_capture": sum(r["ring_capture"] for r in late) / len(late),
                "final_supply": sim.supply(),
                "beta_traj": [(r["epoch"], round(r["eff_beta"], 4) if r["eff_beta"]
                               is not None else None) for r in records],
                "records": records}

    # (a) is β load-bearing? compare uncapped vs capped ring capture per curve
    grid = []
    for cname, cfn in CURVES.items():
        uncapped = _run(cfn, None)
        for s0name, S0 in S0_VALUES.items():
            capped = _run(cfn, exp_sched(S0))
            grid.append({
                "curve": cname, "S0": s0name,
                "uncapped_ring_capture": round(uncapped["late_ring_capture"], 4),
                "capped_ring_capture": round(capped["late_ring_capture"], 4),
                "uncapped_honest_corr": round(uncapped["honest_corr"], 4),
                "capped_honest_corr": round(capped["honest_corr"], 4),
                "corr_drop": round(uncapped["honest_corr"] - capped["honest_corr"], 4),
                "capped_final_supply": round(capped["final_supply"], 1),
            })

    # does one S0 work across the envelope? A schedule "works" if it holds
    # capped ring capture < 0.3 AND corr_drop < 0.1 across all 3 curves.
    by_s0 = {}
    for g in grid:
        by_s0.setdefault(g["S0"], []).append(g)
    s0_robust = {}
    for s0, cells in by_s0.items():
        ok = all(c["capped_ring_capture"] < 0.3 and c["corr_drop"] < 0.1
                 for c in cells)
        s0_robust[s0] = {
            "robust_across_curves": ok,
            "worst_ring_capture": round(max(c["capped_ring_capture"] for c in cells), 4),
            "worst_corr_drop": round(max(c["corr_drop"] for c in cells), 4),
        }

    verdict = {
        "grid": grid,
        "beta_load_bearing": any(
            g["uncapped_ring_capture"] - g["capped_ring_capture"] > 0.1 for g in grid),
        "max_uncapped_ring_capture": round(
            max(g["uncapped_ring_capture"] for g in grid), 4),
        "s0_robustness": s0_robust,
        "any_single_s0_robust": any(v["robust_across_curves"] for v in s0_robust.values()),
    }
    return {"scenario": "s6_beta_relevance", "params": {
        "epochs": epochs, "delta": delta, "K": K,
        "curves": list(CURVES), "S0_values": list(S0_VALUES)},
        "verdict": verdict}
