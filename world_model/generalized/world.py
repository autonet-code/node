"""World: coalition of tendencies, observation stream, stake graph.

A World holds:
  - tendencies: id -> GeneralizedTendency
  - observations: id -> Observation (the current observation set)
  - history: cumulative observations seen so far (for novelty floors)

The World provides the operations that actually mutate state:
  - add_observation: record a fact arriving from outside
  - apply_stakes: walk every tendency's last_stakes and write Stake
    objects onto the corresponding nodes (own + cross-staked)

Cross-tendency staking is what creates child nodes in other tendencies'
trees. When tendency A stakes CON on a node owned by tendency B, that
stake is recorded on B's node directly. When the staking pressure on
some position becomes contested enough (multiple tendencies disagree
strongly), the growth rule (in grow.py) sprouts new children to
accommodate the dispute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..models.tree import Node, Position, Stake
from .observation import Observation


@dataclass
class World:
    """A coalition of tendencies and the observation stream they share."""

    tendencies: Dict[str, "GeneralizedTendency"] = field(default_factory=dict)  # type: ignore[name-defined]
    observations: Dict[str, Observation] = field(default_factory=dict)
    history: List[Observation] = field(default_factory=list)

    def add_tendency(self, tendency: "GeneralizedTendency") -> None:  # type: ignore[name-defined]
        self.tendencies[tendency.id] = tendency

    def add_observation(self, obs: Observation) -> None:
        self.observations[obs.id] = obs
        self.history.append(obs)

    def clear_observations(self) -> None:
        self.observations.clear()

    def apply_stakes(self) -> None:
        """Walk every tendency's last_stakes and write unit-weight
        posts onto the relevant nodes.

        Under the post-only refactor, every stored Stake has weight=1.
        Positive intents become a single PRO post by the tendency on
        the target node; negative intents are dropped here because
        cross-tendency disagreement is expressed structurally (via
        CON-position children in the disagreer's own tree, possibly
        co-parented into the target tendency's tree at sprout time)
        rather than via signed weight.

        Existing posts attributed to any tendency in `self.tendencies`
        are removed first to avoid double-counting across rounds.
        """
        # 1. Remove existing posts attributed to any tendency in self.tendencies
        all_tendency_ids = set(self.tendencies.keys())
        for tendency in self.tendencies.values():
            for node in tendency.tree.all_nodes():
                node.stakes = [s for s in node.stakes if s.agent_id not in all_tendency_ids]
                node.invalidate_cache()

        # 2. Apply each tendency's recorded intents as unit-weight posts.
        for tendency in self.tendencies.values():
            for (target_tendency_id, node_id), signed in tendency.last_stakes.items():
                if signed <= 0.0:
                    continue   # disagreement is structural, not weight-based
                target = self.tendencies.get(target_tendency_id)
                if target is None:
                    continue
                node = target.tree.get_node(node_id)
                if node is None:
                    continue
                node.add_post(agent_id=tendency.id)

    def total_stake_on(self, target_tendency_id: str, node_id: str) -> float:
        """Number of posts on a node (count of unit-weight stakes from
        all tendencies). Always non-negative under the post-only
        model. Returns 0.0 if the node or tendency is unknown.
        """
        target = self.tendencies.get(target_tendency_id)
        if target is None:
            return 0.0
        node = target.tree.get_node(node_id)
        if node is None:
            return 0.0
        return float(len(node.stakes))

    def intrinsic_score(self, node) -> float:
        """How strongly this node is supported by its full subtree
        across all tendencies that share it. See
        `tendency._intrinsic_score`.
        """
        from .tendency import _intrinsic_score
        return _intrinsic_score(node)

    def root_scores(self) -> Dict[str, float]:
        """Net score of each tendency's thesis (its tree's root
        net_score). The net_score recursion picks up unit-weight
        posts and signed children naturally; for co-parented nodes,
        the child's contribution is signed by the edge polarity in
        each parent's tree.
        """
        return {tid: t.tree.score for tid, t in self.tendencies.items()}
