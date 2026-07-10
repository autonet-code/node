"""Fees-only + REP-from-earnings simulation driver — SIM-ONLY
(see fees_only_rules.py header).

Threads the two carry channels the model needs on top of v4.1:
  1. the FEE BURN: fees burned in epoch t fund the tool-mint pool in
     epoch t+1 (a one-epoch lag — burn is observed at close, pool is set
     at the next open). Epoch 0 opens with a zero pool by construction
     [FO-1]. A `retroactive_usage` mode carries usage weight from the
     zero-pool "dead period" into the first funded epoch (S5).
  2. REP accrual from EARNINGS, not from the rep-holder-attributable
     mint (that was v4.1 D'). Both service providers and tool authors
     gain reputation; pure spenders never do.

Reputation is bound to the AUTHOR/PROVIDER agent; a household's rep is
the sum over its agents (household_of). The service market is exogenous:
scenarios supply the per-epoch payment list.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of
from fees_only_rules import (
    FeesOnlyState, Review, Service,
    compute_fees_only_epoch, run_service_market, fees_only_rank_score,
)


class FeesOnlySimulation:
    def __init__(self, *, seed: int, delta: float = 0.7,
                 epsilon: float = 0.05, beta=None, beta_fn=None,
                 rep_claim_fraction: float = 1.0,
                 service_rep_only: bool = False,
                 retroactive_usage: bool = False):
        self.rng = random.Random(seed)
        self.delta = delta
        self.epsilon = epsilon
        self.beta = beta
        self.beta_fn = beta_fn
        self.rep_claim_fraction = rep_claim_fraction
        self.service_rep_only = service_rep_only
        self.retroactive_usage = retroactive_usage
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.services: Dict[str, Service] = {}
        self.state = FeesOnlyState()
        self.epoch = 0
        # [FO-1] carry: fees burned this epoch fund NEXT epoch's pool.
        self.pending_burn = 0.0
        self.last_pool = 0.0
        self._last_eff_beta = None
        # retroactive-usage accumulator (S5): usage seen while the pool was
        # zero, optionally carried into the first funded epoch.
        self._carried_usage: Dict[str, Dict[str, float]] = {}

    # -- registration --------------------------------------------------
    def add_agent(self, agent: Agent) -> Agent:
        self.agents[agent.agent_id] = agent
        return agent

    def new_tool(self, author: str, *, kind: str = "honest",
                 true_quality: float = 0.5, topic_match: float = 0.7,
                 digest: Optional[str] = None) -> Tool:
        dg = digest or "".join(self.rng.choice("0123456789abcdef")
                               for _ in range(64))
        t = Tool(digest=dg, author=author, kind=kind, true_quality=true_quality,
                 topic_match=topic_match, born_epoch=self.epoch,
                 node_id=f"tmnode_{len(self.tools):05d}")
        self.tools[dg] = t
        return t

    def new_service(self, provider_agent: str, *, price: float = 1.0,
                    service_id: Optional[str] = None) -> Service:
        sid = service_id or f"svc_{len(self.services):05d}"
        a = self.agents.get(provider_agent)
        house = household_of(a) if a else provider_agent
        s = Service(service_id=sid, provider_house=house, price=price)
        self.services[sid] = s
        return s

    def rank(self, tool: Tool) -> float:
        return fees_only_rank_score(tool, self.state.v41.positions)

    def tool_rating(self, tool: Tool) -> float:
        pos = self.state.v41.positions.get(tool.digest)
        if not pos:
            return 0.0
        h = pos["head"]
        return (h[4] + h[5]) / 2.0

    def supply(self) -> float:
        return sum(a.reputation for a in self.agents.values())

    # -- one epoch -----------------------------------------------------
    def run_epoch(self, *,
                  registrations: List[Tool],
                  usages: List[Tuple[str, Tool, bool, Optional[Dict[str, float]]]],
                  payments: Optional[List[Tuple[str, str, float]]] = None,
                  ) -> Dict[str, Any]:
        payments = payments or []
        for t in registrations:
            self.state.v41.registrations.setdefault(t.digest, {"author": t.author})

        # ---- tool usage → usage_counts + reviews (as v4.1 driver) -----
        usage_counts: Dict[str, Dict[str, float]] = {}
        reviews: Dict[str, List[Review]] = {}
        author_house: Dict[str, str] = {}
        for t in self.tools.values():
            a = self.agents.get(t.author)
            author_house[t.digest] = household_of(a) if a else t.author
        for (caller, tool, ok, axes) in usages:
            a = self.agents.get(caller)
            house = household_of(a) if a else caller
            if ok:
                usage_counts.setdefault(tool.digest, {})
                usage_counts[tool.digest][house] = \
                    usage_counts[tool.digest].get(house, 0.0) + 1.0
                if axes:
                    reviews.setdefault(tool.digest, []).append(
                        Review(household=house, axes=dict(axes)))

        # ---- [FO-1] pool = fees burned LAST epoch (0 at genesis) ------
        pool = self.pending_burn
        self.pending_burn = 0.0

        # ---- retroactive-usage (S5): if the pool was 0 last epoch, carry
        # that usage forward and fold it in when a real pool arrives -----
        if self.retroactive_usage:
            if pool <= 0.0:
                # dead period: bank this epoch's usage, mint nothing.
                for d, hc in usage_counts.items():
                    bank = self._carried_usage.setdefault(d, {})
                    for h, c in hc.items():
                        bank[h] = bank.get(h, 0.0) + c
            elif self._carried_usage:
                # first funded epoch: add the banked dead-period usage.
                for d, hc in self._carried_usage.items():
                    live = usage_counts.setdefault(d, {})
                    for h, c in hc.items():
                        live[h] = live.get(h, 0.0) + c
                self._carried_usage = {}

        # ---- resolve β (supply-pegged schedule wins) ------------------
        if self.beta_fn is not None:
            eff_beta = float(self.beta_fn(self.supply()))
        else:
            eff_beta = self.beta
        self._last_eff_beta = eff_beta
        self.last_pool = pool

        # ---- settle the service market this epoch ---------------------
        settlement = run_service_market(payments=payments, services=self.services)

        # ---- the fees-only close --------------------------------------
        result = compute_fees_only_epoch(
            agents=self.agents, state=self.state, tools=self.tools,
            usage_counts=usage_counts, reviews=reviews,
            author_house=author_house, burned_pool=pool,
            service_settlement=settlement, delta=self.delta,
            epsilon=self.epsilon, beta=eff_beta,
            rep_claim_fraction=self.rep_claim_fraction,
            service_rep_only=self.service_rep_only,
        )

        # ---- [FO-2] accrue REPUTATION from earnings claims ------------
        # rep_claim is by household; bind it to the earning agent(s). For
        # tool earnings we credit the tool's author agent; for service
        # earnings we credit the provider agent. To keep it simple and
        # deterministic we map a household's claim onto its highest-rep
        # agent (or first agent), which is how a household would self-
        # custody its earned voice. This does NOT change totals — only who
        # in the household holds the rep — and household_of collapses it
        # back for all weight math.
        rep_claim = result["rep_claim"]
        house_to_agent: Dict[str, Agent] = {}
        for a in self.agents.values():
            h = household_of(a)
            cur = house_to_agent.get(h)
            if cur is None or a.reputation > cur.reputation:
                house_to_agent[h] = a
        for house, claim in rep_claim.items():
            a = house_to_agent.get(house)
            if a is not None:
                a.reputation += float(claim)

        # ---- [FO-1] this epoch's burn funds NEXT epoch's pool ---------
        self.pending_burn = settlement["burned"]

        self.epoch += 1
        result["eff_beta"] = eff_beta
        result["pool"] = pool
        return result
