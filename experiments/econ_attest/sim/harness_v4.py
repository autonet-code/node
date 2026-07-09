"""v4 simulation driver — SIM-ONLY (see v4_rules.py header).

Wraps v4_rules.compute_v4_epoch with the same population/discovery-loop
scaffolding as harness.py, but drives the v4 ruleset instead of the real
close. Reputation still accrues 1:1 from mint and feeds next epoch's
weights. Emission pool is the same fixed pie.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of
from v4_rules import Review, V4State, compute_v4_epoch, v4_rank_score

BASE_EMISSION = 100.0


class V4Simulation:
    def __init__(self, *, seed: int, beta: float = 0.1, delta: float = 0.5,
                 Q: float = 5.0, epsilon: float = 0.05,
                 base_emission: float = BASE_EMISSION):
        self.rng = random.Random(seed)
        self.beta = beta
        self.delta = delta
        self.Q = Q
        self.epsilon = epsilon
        self.base_emission = base_emission
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.state = V4State()
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
        return v4_rank_score(tool, self.state.positions)

    def run_epoch(self, *,
                  registrations: List[Tool],
                  # (caller_agent_id, tool, ok, axes|None) — usage receipts
                  usages: List[Tuple[str, Tool, bool, Optional[Dict[str, float]]]],
                  # (agent_id, tool, axes) — inspection reviews (rule B):
                  # move position, mint nothing, no usage receipt
                  inspections: Optional[List[Tuple[str, Tool, Dict[str, float]]]] = None,
                  ) -> Dict[str, Any]:
        inspections = inspections or []

        # register (carry author map)
        for t in registrations:
            self.state.registrations.setdefault(
                t.digest, {"author": t.author})

        # collapse usage counts per household, exclude author household &
        # zero-ok. Build reviews list (usage-linked + inspection).
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
                        Review(household=house, axes=dict(axes),
                               inspection=False))

        # [v4-B] inspection reviews: drift only, no usage/mint
        for (agent_id, tool, axes) in inspections:
            a = self.agents.get(agent_id)
            house = household_of(a) if a else agent_id
            reviews.setdefault(tool.digest, []).append(
                Review(household=house, axes=dict(axes), inspection=True))

        pool = self.base_emission + self.recycled_fees
        self.last_pool = pool

        result = compute_v4_epoch(
            agents=self.agents, state=self.state, tools=self.tools,
            usage_counts=usage_counts, reviews=reviews,
            author_house=author_house, emission_pool=pool,
            beta=self.beta, delta=self.delta, Q=self.Q, epsilon=self.epsilon,
        )

        # accrue reputation 1:1 from mint. compute_v4_epoch returns
        # agent_mint keyed by household (pool-normalized). Credit each
        # household's ATN to its authoring agent(s) split evenly among the
        # tools that household authored this epoch — for these populations
        # each author is its own household so the split is trivial.
        raw = result["raw_mint"]            # digest -> raw usage_term
        raw_by_house: Dict[str, float] = {}
        digests_by_house: Dict[str, List[str]] = {}
        for digest, term in raw.items():
            t = self.tools.get(digest)
            if not t:
                continue
            a = self.agents.get(t.author)
            house = household_of(a) if a else t.author
            raw_by_house[house] = raw_by_house.get(house, 0.0) + term
            digests_by_house.setdefault(house, []).append(digest)
        for house, minted in result["agent_mint"].items():
            # distribute the household's minted ATN across its authored
            # digests proportional to each digest's raw term, credited to
            # the digest's author agent.
            hraw = raw_by_house.get(house, 0.0)
            if hraw <= 0:
                continue
            for digest in digests_by_house.get(house, []):
                share = raw[digest] / hraw
                author_id = self.tools[digest].author
                a = self.agents.get(author_id)
                if a is not None:
                    a.reputation += float(minted) * share

        self.recycled_fees = 0.0
        self.epoch += 1
        result["total_mint"] = sum(result["agent_mint"].values())
        return result
