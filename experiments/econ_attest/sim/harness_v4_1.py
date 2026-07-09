"""v4.1 simulation driver — SIM-ONLY (see v4_1_rules.py header).

Same scaffolding as harness_v4.py, but drives v4_1_rules.compute_v41_epoch
and — critically for [v4.1-D'] — accrues reputation from the DECOUPLED
``rep_increment`` (rep-holder-attributable mint only), not from total ATN
mint. This is what makes a dust ring skim ATN forever while never gaining
voice.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of
from v4_1_rules import Review, V41State, compute_v41_epoch, v41_rank_score

BASE_EMISSION = 100.0


class V41Simulation:
    def __init__(self, *, seed: int, delta: float = 0.7,
                 epsilon: float = 0.05, base_emission: float = BASE_EMISSION,
                 beta=None, adaptive_beta: bool = False,
                 beta_ceiling: float = 0.2, beta_fn=None):
        self.rng = random.Random(seed)
        self.delta = delta
        self.epsilon = epsilon
        self.base_emission = base_emission
        # β cap (accepted, rep-independent, ATN-side only). beta=None ->
        # uncapped pure-v4.1. adaptive_beta -> peg β to last epoch's
        # observed zero-rep weight share, clamped at beta_ceiling.
        # beta_fn(supply) -> β : the SUPPLY-PEGGED schedule (maturity proxy);
        # supply is total reputation, read fresh each epoch. Takes priority
        # over `beta` / `adaptive_beta` when set.
        self.beta = beta
        self.adaptive_beta = adaptive_beta
        self.beta_ceiling = beta_ceiling
        self.beta_fn = beta_fn
        self._last_observed_zero_share = 0.0
        self._last_eff_beta = None
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.state = V41State()
        self.epoch = 0
        self.recycled_fees = 0.0
        self.last_pool = base_emission

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

    def rank(self, tool: Tool) -> float:
        return v41_rank_score(tool, self.state.positions)

    def tool_rating(self, tool: Tool) -> float:
        pos = self.state.positions.get(tool.digest)
        if not pos:
            return 0.0
        h = pos["head"]
        return (h[4] + h[5]) / 2.0

    def run_epoch(self, *,
                  registrations: List[Tool],
                  usages: List[Tuple[str, Tool, bool, Optional[Dict[str, float]]]],
                  inspections: Optional[List[Tuple[str, Tool, Dict[str, float]]]] = None,
                  ) -> Dict[str, Any]:
        inspections = inspections or []
        for t in registrations:
            self.state.registrations.setdefault(t.digest, {"author": t.author})

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
        for (agent_id, tool, axes) in inspections:
            a = self.agents.get(agent_id)
            house = household_of(a) if a else agent_id
            reviews.setdefault(tool.digest, []).append(
                Review(household=house, axes=dict(axes), inspection=True))

        pool = self.base_emission + self.recycled_fees
        self.last_pool = pool

        # β resolution priority: supply-pegged schedule > adaptive-peg >
        # fixed value. The supply schedule reads TOTAL REPUTATION SUPPLY
        # (the maturity proxy) — under D' this cannot be inflated by dust
        # usage (zero-rep mint grants no rep), so a ring can't move β.
        if self.beta_fn is not None:
            supply = sum(a.reputation for a in self.agents.values())
            eff_beta = float(self.beta_fn(supply))
        elif self.adaptive_beta:
            eff_beta = min(self.beta_ceiling,
                           max(self.epsilon, self._last_observed_zero_share))
        else:
            eff_beta = self.beta

        result = compute_v41_epoch(
            agents=self.agents, state=self.state, tools=self.tools,
            usage_counts=usage_counts, reviews=reviews,
            author_house=author_house, emission_pool=pool,
            delta=self.delta, epsilon=self.epsilon, beta=eff_beta,
        )
        self._last_observed_zero_share = result.get("observed_zero_share", 0.0)
        self._eff_beta = eff_beta
        self._last_eff_beta = eff_beta

        # [v4.1-D'] accrue REPUTATION from rep_increment only (decoupled);
        # ATN payout (agent_mint) is tracked but does NOT feed voice.
        rep_inc = result["rep_increment"]
        raw_rep = result["raw_rep"]
        rep_basis_by_house: Dict[str, float] = {}
        digests_by_house: Dict[str, List[str]] = {}
        for digest, term in raw_rep.items():
            t = self.tools.get(digest)
            if not t:
                continue
            a = self.agents.get(t.author)
            house = household_of(a) if a else t.author
            rep_basis_by_house[house] = rep_basis_by_house.get(house, 0.0) + term
            digests_by_house.setdefault(house, []).append(digest)
        for house, inc in rep_inc.items():
            hbasis = rep_basis_by_house.get(house, 0.0)
            if hbasis <= 0:
                continue
            for digest in digests_by_house.get(house, []):
                share = raw_rep[digest] / hbasis
                author_id = self.tools[digest].author
                a = self.agents.get(author_id)
                if a is not None:
                    a.reputation += float(inc) * share

        # track cumulative ATN payout separately (for D' ATN-share metric)
        self.recycled_fees = 0.0
        self.epoch += 1
        return result
