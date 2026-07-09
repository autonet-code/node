"""Parameterized attack/claim scenarios over the real v3 tool economy.

Each scenario builds a Simulation (harness.py), runs ~200 epochs through
the REAL federated_epoch_close, and returns a result dict:
  {"scenario": str, "params": {...}, "epochs": [per-epoch record...],
   "verdict": {headline numbers}}.

Scenarios map to experiments/econ_attest/attacks.md:
  baseline_honest  -> claims C1 (quality ranking), C3 (bad tools buried)
  sybil_pump       -> attack 1 (sybil review ring), claim C2
  epsilon_faucet   -> attack 6 (ε-floor dust minting)
  review_nuke      -> attack 5 (competitive review nuking)
  service_clone    -> user's core hypothesis (φ moat, rediscovery cost,
                      fee-recycling coupling)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from harness import (
    Agent,
    EpochPlan,
    Simulation,
    Tool,
    conservation_check,
    discovery_ranks,
    honest_axes,
    rank_score,
    tool_head,
)

DEFAULT_EPOCHS = 200


# ---------------------------------------------------------------------------
# helpers shared across scenarios
# ---------------------------------------------------------------------------

def _mint_of(result: Dict[str, Any], digest: str) -> float:
    e = result["tool_mint"].get(digest)
    return float(e["mint"]) if e else 0.0


def _agent_mint_by_class(result: Dict[str, Any], agents: Dict[str, Agent]
                         ) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for aid, m in result["agent_mint"].items():
        cls = agents[aid].kind if aid in agents else "other"
        out[cls] = out.get(cls, 0.0) + float(m)
    return out


def _greenlight_pair(sim: Simulation, tool: Tool, plan: EpochPlan,
                     *, prefix: str = "vet") -> None:
    """Two distinct unbound vetters greenlight a tool (attack 3's cheap
    gate — used by scenarios that need tools mint-eligible so they can
    study the OTHER rails). Vetters are registered agents so they're real
    but carry no reputation weight (vetting is not voice-weighted)."""
    for i in (1, 2):
        vid = f"{prefix}_{tool.digest[:6]}_{i}"
        if vid not in sim.agents:
            sim.add_agent(Agent(agent_id=vid, kind="vetter"))
        plan.vets.append((vid, tool))


# ===========================================================================
# baseline_honest  (C1 quality ranking, C3 bad tools buried)
# ===========================================================================

def baseline_honest(*, epochs: int = DEFAULT_EPOCHS, seed: int = 1,
                    n_authors: int = 20, n_users: int = 30,
                    uses_per_epoch: int = 120) -> Dict[str, Any]:
    """N honest authors, tools of varied true quality. Users discover
    tools by real rank (topic_match x lift on drifted head) and review
    honestly. Expect: mint share and discovery rank track true quality;
    negatively-reviewed (low-quality) tools sink."""
    sim = Simulation(seed=seed)
    rng = sim.rng

    authors = [sim.add_agent(Agent(agent_id=f"author_{i:02d}", kind="honest",
                                   owner=f"0xauth{i:02d}"))
               for i in range(n_authors)]
    users = [sim.add_agent(Agent(agent_id=f"user_{i:02d}", kind="user",
                                 owner=f"0xuser{i:02d}"))
             for i in range(n_users)]

    tools: List[Tool] = []
    for i, a in enumerate(authors):
        # spread true quality across [-0.6, 0.95]
        q = -0.6 + 1.55 * (i / max(1, n_authors - 1))
        t = sim.new_tool(a.agent_id, kind="honest", true_quality=q,
                         topic_match=0.5 + 0.4 * rng.random())
        tools.append(t)

    records: List[Dict[str, Any]] = []
    for ep in range(epochs):
        plan = EpochPlan()
        # epoch 0: register + greenlight all tools
        if ep == 0:
            for t in tools:
                plan.registrations.append(t)
                _greenlight_pair(sim, t, plan)

        # discovery-driven usage: probability of picking a tool is
        # proportional to its current discovery rank_score. This is the
        # feedback loop: reviews -> drift -> rank -> usage -> mint.
        gl = sim.greenlit_digests()
        ranked = discovery_ranks(tools, sim.tool_positions,
                                 only_greenlit=gl if gl else None)
        candidates = [t for t, _ in ranked] or tools
        weights = [max(1e-6, rank_score(t, sim.tool_positions))
                   for t in candidates]
        wsum = sum(weights)
        for _ in range(uses_per_epoch):
            user = rng.choice(users)
            r = rng.random() * wsum
            acc = 0.0
            pick = candidates[0]
            for t, w in zip(candidates, weights):
                acc += w
                if acc >= r:
                    pick = t
                    break
            # honest review: quality-correlated axis scores; ok mostly true
            ok = rng.random() < 0.5 + 0.45 * (pick.true_quality + 1) / 2
            axes = honest_axes(pick, rng) if ok else None
            plan.usages.append((user.agent_id, pick, ok, axes, ""))

        result = sim.run_epoch(plan)
        pool = sim.base_emission
        conservation_check(result, pool)

        rec = {
            "epoch": ep,
            "total_mint": result["total_mint"],
            "mint_by_digest": {t.digest: _mint_of(result, t.digest)
                               for t in tools},
            "rank_by_digest": {t.digest: rank_score(t, sim.tool_positions)
                               for t in tools},
            "head_by_digest": {t.digest: tool_head(sim.tool_positions, t.digest)
                               for t in tools},
        }
        records.append(rec)

    # verdict: correlation of final cumulative mint with true quality,
    # and whether the worst-quality tools rank below the best.
    cum: Dict[str, float] = {t.digest: 0.0 for t in tools}
    for rec in records:
        for d, m in rec["mint_by_digest"].items():
            cum[d] += m
    q = [t.true_quality for t in tools]
    cm = [cum[t.digest] for t in tools]
    final_rank = records[-1]["rank_by_digest"]
    rn = [final_rank[t.digest] for t in tools]

    verdict = {
        "quality_vs_cummint_corr": _pearson(q, cm),
        "quality_vs_finalrank_corr": _pearson(q, rn),
        "top_quality_rank": _rank_of(tools, final_rank, key=lambda t: t.true_quality,
                                     reverse=True),
        "worst_quality_rank": _rank_of(tools, final_rank,
                                       key=lambda t: t.true_quality, reverse=False),
    }
    return {"scenario": "baseline_honest",
            "params": {"epochs": epochs, "seed": seed, "n_authors": n_authors,
                       "n_users": n_users, "uses_per_epoch": uses_per_epoch},
            "epochs": records, "verdict": verdict}


# ===========================================================================
# sybil_pump  (attack 1: sybil review ring; claim C2)
# ===========================================================================

def sybil_pump(*, epochs: int = 120, seed: int = 2,
               K_values: Optional[List[int]] = None,
               n_honest: int = 10, honest_users: int = 20,
               uses_per_epoch: int = 80) -> Dict[str, Any]:
    """One attacker author + K distinct-owner sybil identities that
    cross-attest (with +1 axis reviews) the attacker's tool each epoch.
    Sweep K. Measure the attacker tool's mint share vs an honest
    counterfactual tool of equal true quality.

    Sybils use DISTINCT owner wallets (same-owner would zero them via the
    owner-map exclusion) and cross-attest — but they carry ε voice until
    they earn reputation, so the pump bootstraps slowly. Compared against
    an identical honest tool that only gets organic honest usage.
    """
    if K_values is None:
        K_values = [0, 3, 10, 30, 100]

    sweep: List[Dict[str, Any]] = []
    for K in K_values:
        sim = Simulation(seed=seed + K)
        rng = sim.rng

        attacker = sim.add_agent(Agent("attacker", kind="attacker",
                                       owner="0xattacker"))
        honest_ctrl = sim.add_agent(Agent("honest_ctrl", kind="honest",
                                          owner="0xhonest"))
        users = [sim.add_agent(Agent(f"huser_{i:02d}", kind="user",
                                     owner=f"0xhu{i:02d}"))
                 for i in range(honest_users)]
        # K sybils, each its OWN household (distinct owner wallet)
        sybils = [sim.add_agent(Agent(f"sybil_{i:03d}", kind="sybil",
                                      owner=f"0xsyb{i:03d}"))
                  for i in range(K)]
        # some background honest authors so the pool has honest competition
        bg = [sim.add_agent(Agent(f"bg_{i:02d}", kind="honest",
                                  owner=f"0xbg{i:02d}"))
              for i in range(n_honest)]

        atk_tool = sim.new_tool("attacker", kind="sybil", true_quality=0.3,
                                topic_match=0.7)
        ctrl_tool = sim.new_tool("honest_ctrl", kind="honest",
                                 true_quality=0.3, topic_match=0.7)
        bg_tools = [sim.new_tool(a.agent_id, kind="honest",
                                 true_quality=0.1 + 0.7 * rng.random(),
                                 topic_match=0.4 + 0.4 * rng.random())
                    for a in bg]

        records: List[Dict[str, Any]] = []
        for ep in range(epochs):
            plan = EpochPlan()
            if ep == 0:
                for t in [atk_tool, ctrl_tool, *bg_tools]:
                    plan.registrations.append(t)
                    _greenlight_pair(sim, t, plan)

            # honest organic usage of ctrl + bg tools (equal treatment)
            for _ in range(uses_per_epoch):
                u = rng.choice(users)
                t = rng.choice([ctrl_tool, *bg_tools])
                ok = rng.random() < 0.7
                plan.usages.append((u.agent_id, t, ok,
                                    honest_axes(t, rng) if ok else None, ""))
            # honest users also organically use the attacker tool at the
            # SAME base rate as the ctrl (both q=0.3, both discoverable) so
            # the pump is the ONLY difference between them.
            for _ in range(uses_per_epoch // (n_honest + 1)):
                u = rng.choice(users)
                ok = rng.random() < 0.7
                plan.usages.append((u.agent_id, atk_tool, ok,
                                    honest_axes(atk_tool, rng) if ok else None, ""))

            # the pump: each sybil cross-attests the attacker tool with +1
            # reviews (dodging same-owner exclusion since attacker owner
            # differs from each sybil owner).
            for s in sybils:
                plan.usages.append((s.agent_id, atk_tool, True,
                                    {"correctness": 1.0, "simplicity": 1.0}, ""))

            result = sim.run_epoch(plan)
            conservation_check(result, sim.base_emission)
            records.append({
                "epoch": ep,
                "atk_mint": _mint_of(result, atk_tool.digest),
                "ctrl_mint": _mint_of(result, ctrl_tool.digest),
                "atk_rank": rank_score(atk_tool, sim.tool_positions),
                "ctrl_rank": rank_score(ctrl_tool, sim.tool_positions),
                "atk_head": tool_head(sim.tool_positions, atk_tool.digest),
                "ctrl_head": tool_head(sim.tool_positions, ctrl_tool.digest),
                "sybil_voice": _sybil_voice(sim, sybils),
            })

        atk_cum = sum(r["atk_mint"] for r in records)
        ctrl_cum = sum(r["ctrl_mint"] for r in records)
        # epoch at which attacker rank first crosses ctrl rank
        cross = next((r["epoch"] for r in records
                      if r["atk_rank"] > r["ctrl_rank"] + 1e-9), None)
        sweep.append({
            "K": K,
            "atk_cum_mint": atk_cum,
            "ctrl_cum_mint": ctrl_cum,
            "capture_ratio": (atk_cum / ctrl_cum) if ctrl_cum > 0 else float("inf"),
            "rank_cross_epoch": cross,
            "final_atk_rank": records[-1]["atk_rank"],
            "final_ctrl_rank": records[-1]["ctrl_rank"],
            "final_sybil_voice": records[-1]["sybil_voice"],
            "records": records,
        })

    verdict = {
        "capture_ratio_by_K": {s["K"]: round(s["capture_ratio"], 4) for s in sweep},
        "rank_cross_by_K": {s["K"]: s["rank_cross_epoch"] for s in sweep},
    }
    return {"scenario": "sybil_pump",
            "params": {"epochs": epochs, "seed": seed, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


def _sybil_voice(sim: Simulation, sybils: List[Agent]) -> float:
    from harness import voice_weights_from_reputation, household_of
    vw = voice_weights_from_reputation(sim.agents)
    if vw is None:
        return 1.0
    return sum(vw.get(household_of(s), 0.0) for s in sybils)


# ===========================================================================
# epsilon_faucet  (attack 6: ε-floor dust minting at scale)
# ===========================================================================

def epsilon_faucet(*, epochs: int = 120, seed: int = 3,
                   K_values: Optional[List[int]] = None,
                   n_honest: int = 20, honest_users: int = 30,
                   honest_uses_per_epoch: int = 150) -> Dict[str, Any]:
    """K minimal unbound-owner identities, each authoring one greenlit
    tool and cross-attesting the OTHERS' ring tools with dust usage.
    Measure the sybil ring's share of the FIXED emission pool vs K, and
    its epoch trajectory (does dust reputation lift them above ε?).
    """
    if K_values is None:
        K_values = [0, 5, 20, 50, 100, 200]

    sweep: List[Dict[str, Any]] = []
    for K in K_values:
        sim = Simulation(seed=seed + K)
        rng = sim.rng

        # honest economy
        h_authors = [sim.add_agent(Agent(f"hauth_{i:02d}", kind="honest",
                                         owner=f"0xha{i:02d}"))
                     for i in range(n_honest)]
        h_users = [sim.add_agent(Agent(f"huser_{i:02d}", kind="user",
                                       owner=f"0xhu{i:02d}"))
                   for i in range(honest_users)]
        h_tools = [sim.new_tool(a.agent_id, kind="honest",
                                true_quality=0.2 + 0.7 * rng.random(),
                                topic_match=0.4 + 0.4 * rng.random())
                   for a in h_authors]

        # K sybil households, each authors one ring tool
        sybils = [sim.add_agent(Agent(f"syb_{i:03d}", kind="sybil",
                                      owner=f"0xsyb{i:03d}"))
                  for i in range(K)]
        ring_tools = [sim.new_tool(s.agent_id, kind="sybil",
                                   true_quality=0.0, topic_match=0.2)
                      for s in sybils]

        records: List[Dict[str, Any]] = []
        for ep in range(epochs):
            plan = EpochPlan()
            if ep == 0:
                for t in [*h_tools, *ring_tools]:
                    plan.registrations.append(t)
                    _greenlight_pair(sim, t, plan)

            for _ in range(honest_uses_per_epoch):
                u = rng.choice(h_users)
                t = rng.choice(h_tools)
                ok = rng.random() < 0.75
                plan.usages.append((u.agent_id, t, ok,
                                    honest_axes(t, rng) if ok else None, ""))

            # ring cross-attestation: sybil i attests ring tool i+1 (dust:
            # one attested-ok receipt each). Dodges same-owner exclusion by
            # attesting a DIFFERENT sybil's tool.
            for i, s in enumerate(sybils):
                target = ring_tools[(i + 1) % K] if K > 0 else None
                if target is not None:
                    plan.usages.append((s.agent_id, target, True,
                                        {"correctness": 1.0}, ""))

            result = sim.run_epoch(plan)
            conservation_check(result, sim.base_emission)
            sybil_mint = sum(_mint_of(result, t.digest) for t in ring_tools)
            records.append({
                "epoch": ep,
                "sybil_mint": sybil_mint,
                "sybil_pool_share": (sybil_mint / result["total_mint"])
                if result["total_mint"] > 0 else 0.0,
                "sybil_voice_sum": _sybil_voice(sim, sybils),
            })

        final = records[-1]
        first = records[0]
        sweep.append({
            "K": K,
            "final_pool_share": final["sybil_pool_share"],
            "first_pool_share": first["sybil_pool_share"],
            "share_growth": final["sybil_pool_share"] - first["sybil_pool_share"],
            "final_voice_sum": final["sybil_voice_sum"],
            "records": records,
        })

    verdict = {
        "final_pool_share_by_K": {s["K"]: round(s["final_pool_share"], 5)
                                  for s in sweep},
        "share_growth_by_K": {s["K"]: round(s["share_growth"], 5)
                              for s in sweep},
    }
    return {"scenario": "epsilon_faucet",
            "params": {"epochs": epochs, "seed": seed, "K_values": K_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# review_nuke  (attack 5: down-review a young competitor)
# ===========================================================================

def review_nuke(*, epochs: int = 120, seed: int = 4,
                J_values: Optional[List[int]] = None,
                honest_users: int = 20, uses_per_epoch: int = 60,
                nuke_start: int = 1) -> Dict[str, Any]:
    """Two equal-quality tools (victim + control). Attacker directs J
    ε-households to -1-review the VICTIM each epoch. Measure rank
    divergence vs the control and the mass at which nuking stops moving
    the head (immune mass)."""
    if J_values is None:
        J_values = [0, 1, 3, 10, 30]

    sweep: List[Dict[str, Any]] = []
    for J in J_values:
        sim = Simulation(seed=seed + J)
        rng = sim.rng

        victim_author = sim.add_agent(Agent("victim_author", kind="honest",
                                            owner="0xvictim"))
        ctrl_author = sim.add_agent(Agent("ctrl_author", kind="honest",
                                          owner="0xctrl"))
        users = [sim.add_agent(Agent(f"user_{i:02d}", kind="user",
                                     owner=f"0xu{i:02d}"))
                 for i in range(honest_users)]
        nukers = [sim.add_agent(Agent(f"nuker_{i:02d}", kind="attacker",
                                      owner=f"0xnk{i:02d}"))
                  for i in range(J)]

        victim = sim.new_tool("victim_author", kind="victim",
                              true_quality=0.6, topic_match=0.7)
        ctrl = sim.new_tool("ctrl_author", kind="honest",
                            true_quality=0.6, topic_match=0.7)

        records: List[Dict[str, Any]] = []
        for ep in range(epochs):
            plan = EpochPlan()
            if ep == 0:
                for t in (victim, ctrl):
                    plan.registrations.append(t)
                    _greenlight_pair(sim, t, plan)

            # equal honest usage of both (positive honest reviews build
            # protective mass)
            for _ in range(uses_per_epoch):
                u = rng.choice(users)
                t = rng.choice([victim, ctrl])
                ok = rng.random() < 0.85
                plan.usages.append((u.agent_id, t, ok,
                                    honest_axes(t, rng) if ok else None, ""))

            # the nuke: J households -1-review the victim (must genuinely
            # invoke -> ok=True with -1 axes)
            if ep >= nuke_start:
                for nk in nukers:
                    plan.usages.append((nk.agent_id, victim, True,
                                        {"correctness": -1.0, "simplicity": -1.0},
                                        ""))

            result = sim.run_epoch(plan)
            conservation_check(result, sim.base_emission)
            vhead = tool_head(sim.tool_positions, victim.digest)
            vmass = sim.tool_positions.get(victim.digest, {}).get("mass", [1.0] * 6)
            records.append({
                "epoch": ep,
                "victim_rank": rank_score(victim, sim.tool_positions),
                "ctrl_rank": rank_score(ctrl, sim.tool_positions),
                "victim_corr_axis": vhead[4],
                "victim_mass_corr": vmass[4],
            })

        # "survival": does the victim's final rank stay within 80% of ctrl?
        final = records[-1]
        surv = final["victim_rank"] >= 0.8 * final["ctrl_rank"]
        # immune mass: first epoch after which victim rank stops dropping
        # relative to ctrl (divergence stabilizes)
        sweep.append({
            "J": J,
            "final_victim_rank": final["victim_rank"],
            "final_ctrl_rank": final["ctrl_rank"],
            "rank_ratio": (final["victim_rank"] / final["ctrl_rank"])
            if final["ctrl_rank"] > 0 else float("inf"),
            "final_victim_corr_axis": final["victim_corr_axis"],
            "survived": surv,
            "records": records,
        })

    verdict = {
        "rank_ratio_by_J": {s["J"]: round(s["rank_ratio"], 4) for s in sweep},
        "survived_by_J": {s["J"]: s["survived"] for s in sweep},
    }
    return {"scenario": "review_nuke",
            "params": {"epochs": epochs, "seed": seed, "J_values": J_values},
            "sweep": sweep, "verdict": verdict}


# ===========================================================================
# service_clone  (user's core hypothesis, refined)
# ===========================================================================

def service_clone(*, epochs: int = 200, seed: int = 5,
                  phi_values: Optional[List[float]] = None,
                  rediscovery_costs: Optional[List[float]] = None,
                  demand: float = 200.0, service_price: float = 0.5,
                  c_local: float = 0.2, clone_epoch: int = 40,
                  n_honest: int = 15, honest_users: int = 30,
                  honest_uses_per_epoch: int = 120,
                  fee_recycle: bool = True) -> Dict[str, Any]:
    """A paid remote Service with exogenous demand D and fee revenue
    R/epoch. At ``clone_epoch`` an agent publishes a FREE tool clone that
    captures at most phi (expressible fraction) of the service's demand.
    Users switch to the clone only when c_local < service_price AND
    discovery surfaces the clone.

    Service fees are recycled into the emission pool (fee_recycle), so the
    service's own revenue enlarges the pool that then pays the clone — the
    coupling the user wants priced.

    Sweeps phi and rediscovery cost. Reports:
      - clone author's cumulative mint vs rediscovery cost (does cloning pay?)
      - surviving service revenue (should converge to moat rent (1-phi)*R)
      - fee-recycling acceleration of clone payback (fee_recycle on vs off)
    """
    if phi_values is None:
        phi_values = [0.3, 0.7, 1.0]
    if rediscovery_costs is None:
        rediscovery_costs = [0.0, 5.0, 20.0, 50.0]

    # Service fee model: each unit of service demand pays `service_price`;
    # a fixed bps is burned and recycled into the pool (Substrate.sol
    # SERVICE_FEE_BPS ~ 2.5%, half to treasury -> ~1.25% recycled).
    RECYCLE_FRACTION = 0.0125

    sweep: List[Dict[str, Any]] = []
    for phi in phi_values:
        for rcost in rediscovery_costs:
            res = _run_service_clone(
                seed=seed, epochs=epochs, phi=phi, rediscovery_cost=rcost,
                demand=demand, service_price=service_price, c_local=c_local,
                clone_epoch=clone_epoch, n_honest=n_honest,
                honest_users=honest_users,
                honest_uses_per_epoch=honest_uses_per_epoch,
                recycle_fraction=RECYCLE_FRACTION if fee_recycle else 0.0)
            res["phi"] = phi
            res["rediscovery_cost"] = rcost
            sweep.append(res)

    # Isolate the fee-recycling coupling. Recycling adds ~1.25% of
    # service fees to the pool, which lifts EVERY author's absolute mint
    # (the clone's share is unchanged, but its ATN payout rises with the
    # bigger pie). Use a demanding rediscovery cost so payback spans many
    # epochs where the small per-epoch lift can compound into a
    # measurable payback-epoch difference. Report BOTH the payback epoch
    # and the clone's absolute cumulative mint at the horizon.
    coupling = []
    for rec_on in (True, False):
        res = _run_service_clone(
            seed=seed, epochs=epochs, phi=0.7, rediscovery_cost=400.0,
            demand=demand, service_price=service_price, c_local=c_local,
            clone_epoch=clone_epoch, n_honest=n_honest,
            honest_users=honest_users,
            honest_uses_per_epoch=honest_uses_per_epoch,
            recycle_fraction=RECYCLE_FRACTION if rec_on else 0.0)
        res["recycle_on"] = rec_on
        coupling.append(res)

    verdict = {
        "clone_pays_by_phi_rcost": {
            f"phi={s['phi']},rcost={s['rediscovery_cost']}":
            (s["clone_cum_mint"] > s["rediscovery_cost"])
            for s in sweep},
        "surviving_service_rev_frac_by_phi": _surviving_rev_by_phi(sweep, demand,
                                                                   service_price),
        "payback_epoch_recycle_on": coupling[0]["payback_epoch"],
        "payback_epoch_recycle_off": coupling[1]["payback_epoch"],
        "clone_cum_mint_recycle_on": round(coupling[0]["clone_cum_mint"], 3),
        "clone_cum_mint_recycle_off": round(coupling[1]["clone_cum_mint"], 3),
    }
    return {"scenario": "service_clone",
            "params": {"epochs": epochs, "seed": seed, "phi_values": phi_values,
                       "rediscovery_costs": rediscovery_costs, "demand": demand,
                       "service_price": service_price, "c_local": c_local,
                       "clone_epoch": clone_epoch, "fee_recycle": fee_recycle},
            "sweep": sweep, "coupling": coupling, "verdict": verdict}


def _surviving_rev_by_phi(sweep, demand, price):
    out: Dict[float, float] = {}
    for s in sweep:
        phi = s["phi"]
        full = demand * price
        frac = s["final_service_rev"] / full if full > 0 else 0.0
        out.setdefault(phi, frac)   # first rcost entry per phi is fine
    return {k: round(v, 4) for k, v in out.items()}


def _run_service_clone(*, seed, epochs, phi, rediscovery_cost, demand,
                       service_price, c_local, clone_epoch, n_honest,
                       honest_users, honest_uses_per_epoch, recycle_fraction
                       ) -> Dict[str, Any]:
    sim = Simulation(seed=seed + int(phi * 100) + int(rediscovery_cost))
    rng = sim.rng

    # honest background economy (competes for the pool)
    h_authors = [sim.add_agent(Agent(f"hauth_{i:02d}", kind="honest",
                                     owner=f"0xha{i:02d}"))
                 for i in range(n_honest)]
    h_users = [sim.add_agent(Agent(f"huser_{i:02d}", kind="user",
                                   owner=f"0xhu{i:02d}"))
               for i in range(honest_users)]
    h_tools = [sim.new_tool(a.agent_id, kind="honest",
                            true_quality=0.2 + 0.7 * rng.random(),
                            topic_match=0.4 + 0.4 * rng.random())
               for a in h_authors]

    # the clone author + its user base (the demand that could switch)
    clone_author = sim.add_agent(Agent("clone_author", kind="clone",
                                       owner="0xclone"))
    demand_users = [sim.add_agent(Agent(f"duser_{i:02d}", kind="user",
                                        owner=f"0xdu{i:02d}"))
                    for i in range(max(1, honest_users))]

    clone_tool: Optional[Tool] = None
    # switch decision: users move to the local tool only if local exec
    # is cheaper than the service price (c_local < service_price). The
    # expressible fraction phi caps how much demand the tool can serve.
    users_switch = c_local < service_price
    switch_frac = phi if users_switch else 0.0

    records: List[Dict[str, Any]] = []
    clone_cum_mint = 0.0
    payback_epoch: Optional[int] = None

    for ep in range(epochs):
        plan = EpochPlan()
        if ep == 0:
            for t in h_tools:
                plan.registrations.append(t)
                _greenlight_pair(sim, t, plan)

        # service demand this epoch: pre-clone = full demand; post-clone,
        # switch_frac of demand migrates to the free tool, so the service
        # keeps (1 - switch_frac) of demand (the moat rent).
        served_by_service = demand
        served_by_clone = 0.0
        if clone_tool is not None:
            # discovery gate: does the clone actually surface? (greenlit +
            # positive rank). If rank <= a floor, users can't find it.
            gl = sim.greenlit_digests()
            surfaced = clone_tool.digest in gl and \
                rank_score(clone_tool, sim.tool_positions) > 0.1
            eff_switch = switch_frac if surfaced else 0.0
            served_by_clone = demand * eff_switch
            served_by_service = demand * (1.0 - eff_switch)

        service_rev = served_by_service * service_price

        # recycle a fraction of service fees into next epoch's pool
        sim.recycled_fees = service_rev * recycle_fraction

        # honest usage of background tools
        for _ in range(honest_uses_per_epoch):
            u = rng.choice(h_users)
            t = rng.choice(h_tools)
            ok = rng.random() < 0.75
            plan.usages.append((u.agent_id, t, ok,
                                honest_axes(t, rng) if ok else None, ""))

        # publish the clone at clone_epoch
        if ep == clone_epoch:
            clone_tool = sim.new_tool("clone_author", kind="clone",
                                      true_quality=0.7, topic_match=0.75)
            plan.registrations.append(clone_tool)
            _greenlight_pair(sim, clone_tool, plan)

        # clone usage: each switched demand user attests the clone tool.
        # scale receipts to the migrated demand (bounded so the sim stays
        # fast); more switched demand -> more attesters -> more mint.
        if clone_tool is not None and served_by_clone > 0:
            n_clone_uses = max(1, int(round(served_by_clone / 5.0)))
            for k in range(n_clone_uses):
                u = demand_users[k % len(demand_users)]
                ok = rng.random() < 0.9
                plan.usages.append((u.agent_id, clone_tool, ok,
                                    honest_axes(clone_tool, rng) if ok else None,
                                    ""))

        result = sim.run_epoch(plan)
        conservation_check(result, sim.last_pool)

        clone_mint = _mint_of(result, clone_tool.digest) if clone_tool else 0.0
        clone_cum_mint += clone_mint
        if payback_epoch is None and clone_cum_mint >= rediscovery_cost \
                and clone_tool is not None:
            payback_epoch = ep

        records.append({
            "epoch": ep,
            "service_rev": service_rev,
            "clone_mint": clone_mint,
            "clone_cum_mint": clone_cum_mint,
            "clone_rank": rank_score(clone_tool, sim.tool_positions)
            if clone_tool else 0.0,
            "pool": result.get("emission_pool", sim.base_emission),
        })

    return {
        "clone_cum_mint": clone_cum_mint,
        "final_service_rev": records[-1]["service_rev"],
        "payback_epoch": payback_epoch,
        "records": records,
    }


# ---------------------------------------------------------------------------
# tiny stats
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _rank_of(tools, rank_map, *, key, reverse):
    """1-based discovery rank of the tool that is extreme by ``key``."""
    target = sorted(tools, key=key, reverse=reverse)[0]
    order = sorted(tools, key=lambda t: -rank_map[t.digest])
    return order.index(target) + 1
