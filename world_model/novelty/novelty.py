"""
Novelty - Anchored Implementation

This module provides a CONCRETE implementation of novelty measurement.
It requires an anchored reference frame (a world model snapshot) to compute.

All scores produced here are RELATIVE to the chosen anchor. Different
anchors will produce different (equally valid) novelty scores.

See core.py for the theoretical definition and abstract interfaces.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
import math

from .core import (
    Concept,
    Stance,
    Claim as AbstractClaim,
    ClaimHierarchy as AbstractClaimHierarchy,
    Observation as AbstractObservation,
    TendencyProfile as AbstractTendencyProfile,
    ReferenceFrame,
    NoveltyMeasure,
    NoveltyScore,
    NoveltyComponents,
)

from world_model import (
    Observation,
    ObservationStore,
    Tendency,
    Agent,
    AgentSet,
    Position,
    Stake,
    Node,
    Tree,
    TreeStore,
)


# =============================================================================
# Concrete implementations of abstract core types
# =============================================================================

class ConcreteObservation(AbstractObservation):
    """Wraps world_model.Observation to implement core.Observation interface."""

    def __init__(self, obs: Observation):
        self._obs = obs

    @property
    def content(self) -> Any:
        return self._obs.content


class ConcreteClaim(AbstractClaim):
    """Wraps world_model.Node to implement core.Claim interface."""

    def __init__(self, node: Node, depth: int = 0):
        self._node = node
        self._depth = depth

    @property
    def content(self) -> Any:
        return self._node.content

    @property
    def children(self) -> List["ConcreteClaim"]:
        return [ConcreteClaim(child, self._depth + 1) for child in self._node.all_children]

    @property
    def depth(self) -> int:
        return self._depth

    def get_stakes(self) -> Dict[str, float]:
        return {stake.agent_id: stake.weight for stake in self._node.stakes}

    def get_positioned_observations(self) -> List[tuple[AbstractObservation, Stance]]:
        # In our model, child nodes are positioned observations
        result = []
        for child in self._node.pro_children:
            obs = ConcreteObservation(Observation(content=child.content))
            result.append((obs, Stance.PRO))
        for child in self._node.con_children:
            obs = ConcreteObservation(Observation(content=child.content))
            result.append((obs, Stance.CON))
        return result


class ConcreteClaimHierarchy(AbstractClaimHierarchy):
    """Wraps world_model.Tree to implement core.ClaimHierarchy interface."""

    def __init__(self, tree: Tree):
        self._tree = tree

    @property
    def root(self) -> ConcreteClaim:
        return ConcreteClaim(self._tree.root_node, depth=0)

    def all_claims(self) -> List[ConcreteClaim]:
        """All claims depth-first."""
        claims = []
        def collect(node: Node, depth: int):
            claims.append(ConcreteClaim(node, depth))
            for child in node.all_children:
                collect(child, depth + 1)
        if self._tree.root_node:
            collect(self._tree.root_node, 0)
        return claims

    def max_depth(self) -> int:
        return self._tree.depth()

    def total_stake(self) -> float:
        total = 0.0
        for claim in self.all_claims():
            total += sum(claim.get_stakes().values())
        return total


class ConcreteTendencyProfile(AbstractTendencyProfile):
    """Wraps world_model.AgentSet to implement core.TendencyProfile interface."""

    def __init__(self, agents: AgentSet):
        self._agents = agents

    def get_allocation(self, tendency_id: str) -> float:
        try:
            tendency = Tendency(tendency_id)
            agent = self._agents.agents.get(tendency)
            return agent.allocation if agent else 0.0
        except ValueError:
            return 0.0

    def all_tendencies(self) -> List[str]:
        return [t.value for t in Tendency]


# =============================================================================
# World Model as Reference Frame
# =============================================================================

class WorldModelReference(ReferenceFrame):
    """
    A world model snapshot serving as the reference frame for novelty.

    This is the ANCHOR - the arbitrary but necessary reference point
    against which novelty is measured. The world model consists of:

    - TreeStore: claim hierarchies (what the agent believes)
    - AgentSet: tendency allocations (what the agent cares about)
    - ObservationStore: integrated facts (what the agent has absorbed)

    IMPORTANT: This is a snapshot. It represents the agent's state at a
    particular moment. The agent IS this configuration - not something
    separate that "has" this configuration.
    """

    def __init__(
        self,
        trees: TreeStore,
        agents: AgentSet,
        observations: ObservationStore,
    ):
        self._trees = trees
        self._agents = agents
        self._observations = observations

    @property
    def hierarchies(self) -> List[ConcreteClaimHierarchy]:
        """The belief hierarchies in this reference frame."""
        return [ConcreteClaimHierarchy(tree) for tree in self._trees.all()]

    @property
    def tendencies(self) -> ConcreteTendencyProfile:
        """The motivational structure of this reference frame."""
        return ConcreteTendencyProfile(self._agents)

    @property
    def observations(self) -> List[ConcreteObservation]:
        """Observations already integrated into this frame."""
        return [ConcreteObservation(obs) for obs in self._observations.all()]

    # Keep direct access for implementation convenience
    @property
    def trees(self) -> TreeStore:
        return self._trees

    @property
    def agents(self) -> AgentSet:
        return self._agents

    @property
    def observation_store(self) -> ObservationStore:
        return self._observations

    def contains(self, concept: Concept) -> float:
        """
        How much does this world model already contain this concept?

        Measured by how well the concept fits into existing trees.
        """
        if not self._trees.all():
            return 0.0

        # Check if concept content matches any existing observation
        if isinstance(concept.content, str):
            for obs in self._observations.all():
                if obs.content == concept.content:
                    return 1.0  # Exact match - fully contained

        # Otherwise, measure fit across trees (inverse of novelty)
        # This is a simplified containment check
        return 0.0  # Detailed check happens in novelty computation

    def absorb(self, concept: Concept, weight: float = 1.0) -> "WorldModelReference":
        """
        Return a new world model with this concept integrated.

        This is how novelty becomes familiar over time.
        The original reference is unchanged (immutable snapshot).
        """
        # Create copies of all components
        new_observations = ObservationStore()
        for obs in self._observations.all():
            new_observations.add(obs)

        # Add the new concept as an observation
        if isinstance(concept.content, str):
            new_obs = Observation(content=concept.content)
            new_observations.add(new_obs)

        # TODO: Also update trees to integrate the concept
        # This would involve staking it in appropriate locations

        return WorldModelReference(
            trees=self._trees,  # For now, trees unchanged
            agents=self._agents,  # Allocations unchanged
            observations=new_observations,
        )


# =============================================================================
# Detailed Novelty Score
# =============================================================================

@dataclass
class StakeAttempt:
    """Result of attempting to stake a concept in a tree."""
    tree: Tree
    best_node: Optional[Node] = None
    best_position: Optional[Position] = None
    best_stance: Optional[Stance] = None
    fit_score: float = 0.0
    stance_confidence: float = 0.0
    contradiction_depth: int = 0
    creates_contradiction: bool = False


@dataclass
class AnchoredNoveltyScore(NoveltyScore):
    """
    Detailed novelty score with component breakdown.

    Extends the abstract NoveltyScore with implementation-specific details
    about how the score was computed.
    """
    # Detailed breakdown
    stake_attempts: list[StakeAttempt] = field(default_factory=list)
    trees_with_fit: int = 0
    trees_total: int = 0
    max_contradiction_depth: int = 0
    projected_allocation_shift: dict = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Combined novelty score using geometric mean."""
        return self.components.composite

    # Convenience accessors for component values
    @property
    def integration_resistance(self) -> float:
        return self.components.integration_resistance

    @property
    def contradiction_depth(self) -> float:
        return self.components.contradiction_depth

    @property
    def tree_coverage_gap(self) -> float:
        return self.components.coverage_gap

    @property
    def allocation_disruption(self) -> float:
        return self.components.allocation_disruption

    def __repr__(self):
        return (
            f"AnchoredNoveltyScore(composite={self.composite_score:.3f}, "
            f"integration={self.integration_resistance:.2f}, "
            f"depth={self.contradiction_depth:.2f}, "
            f"coverage_gap={self.tree_coverage_gap:.2f}, "
            f"disruption={self.allocation_disruption:.2f})"
        )


# =============================================================================
# Anchored Novelty Measure (concrete implementation)
# =============================================================================

class AnchoredNoveltyMeasure(NoveltyMeasure):
    """
    Concrete novelty measurement using a world model as anchor.

    This is the practical implementation that actually computes novelty.
    It requires:
    - A world model snapshot (the anchor/reference frame)
    - A similarity function (default: word overlap, or use neural models)

    All results are relative to the chosen anchor.
    """

    def __init__(
        self,
        reference: WorldModelReference,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize with an anchored reference frame.

        Args:
            reference: The world model snapshot to measure against
            similarity_fn: Function to compare text similarity (default: word overlap)
                          For better results, use embeddings.relation_fit_score
        """
        self.reference = reference
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity (0-1)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def detect_stance(self, concept: Concept, claim: AbstractClaim) -> tuple[Stance, float]:
        """
        Detect what stance a concept takes relative to a claim.

        Returns (stance, confidence) where:
        - stance is PRO, CON, or NEUTRAL
        - confidence is 0-1 indicating certainty of stance detection

        Uses the similarity function - if it's a neural model with NLI,
        this will detect entailment/contradiction. Otherwise falls back
        to heuristic based on similarity score.
        """
        concept_text = str(concept.content)
        claim_text = str(claim.content)

        similarity = self.similarity_fn(concept_text, claim_text)

        # The similarity function may encode stance in its score
        # High similarity (>0.5) suggests PRO
        # Very low similarity (<0.2) might suggest CON (topically related but opposing)
        # Medium similarity suggests NEUTRAL

        if similarity > 0.5:
            return Stance.PRO, similarity
        elif similarity < 0.2:
            # Low similarity could mean unrelated OR contradicting
            # Without NLI, we can't distinguish well
            return Stance.NEUTRAL, 1.0 - similarity
        else:
            return Stance.NEUTRAL, 0.5

    def measure(self, concept: Concept, reference: ReferenceFrame = None) -> AnchoredNoveltyScore:
        """
        Measure novelty of a concept against the anchored reference.

        Args:
            concept: The concept to evaluate
            reference: Optional override reference (uses self.reference if None)

        Returns:
            Detailed novelty score with component breakdown
        """
        ref = reference or self.reference
        if not isinstance(ref, WorldModelReference):
            raise TypeError("AnchoredNoveltyMeasure requires WorldModelReference")

        # Convert concept to observation if needed
        if isinstance(concept, Observation):
            obs = concept
        elif isinstance(concept.content, str):
            obs = Observation(content=concept.content)
        else:
            raise TypeError("Concept content must be string")

        # Build score with components
        components = NoveltyComponents()
        score = AnchoredNoveltyScore(
            value=0.0,  # Will be set to composite_score
            concept=concept,
            reference=ref,
            components=components,
        )

        # Attempt to stake in each tree
        for tree in ref.trees.all():
            attempt = self._attempt_stake(obs, tree, concept)
            score.stake_attempts.append(attempt)
            if attempt.fit_score > 0.3:
                score.trees_with_fit += 1

        score.trees_total = len(ref.trees)

        # Compute component scores
        components.integration_resistance = self._compute_integration_resistance(score)
        components.contradiction_depth = self._compute_contradiction_depth(score)
        components.coverage_gap = self._compute_tree_coverage_gap(score)
        components.allocation_disruption = self._compute_allocation_disruption(score, ref)

        # Set the main value
        score.value = score.composite_score

        return score

    def _attempt_stake(self, obs: Observation, tree: Tree, concept: Concept) -> StakeAttempt:
        """Attempt to find where a concept would stake in a tree."""
        attempt = StakeAttempt(tree=tree)

        best_fit = 0.0
        best_node = None
        best_position = None
        best_stance = None
        best_confidence = 0.0
        max_depth = 0

        def search_node(node: Node, depth: int):
            nonlocal best_fit, best_node, best_position, best_stance, best_confidence, max_depth

            node_content = node.content or ""

            # Detect stance using the concept against this node as a claim
            claim = ConcreteClaim(node, depth)
            stance, confidence = self.detect_stance(concept, claim)

            similarity = self.similarity_fn(obs.content, node_content)

            # Compute fit based on stance
            if stance == Stance.PRO:
                fit = similarity * 0.8 * confidence
                position = Position.PRO
            elif stance == Stance.CON:
                fit = (1 - similarity) * 0.7 * confidence
                position = Position.CON
            else:
                fit = similarity * 0.3  # Neutral has lower fit
                position = Position.PRO  # Default to PRO for neutral

            if fit > best_fit:
                best_fit = fit
                best_node = node
                best_position = position
                best_stance = stance
                best_confidence = confidence
                max_depth = depth

            for child in node.all_children:
                search_node(child, depth + 1)

        if tree.root_node:
            search_node(tree.root_node, 0)

        attempt.best_node = best_node
        attempt.best_position = best_position
        attempt.best_stance = best_stance
        attempt.fit_score = best_fit
        attempt.stance_confidence = best_confidence
        attempt.contradiction_depth = max_depth if best_stance == Stance.CON else 0

        if best_node and best_stance == Stance.CON:
            existing_pro_stakes = sum(s.weight for s in best_node.stakes)
            attempt.creates_contradiction = existing_pro_stakes > 0

        return attempt

    def _compute_integration_resistance(self, score: AnchoredNoveltyScore) -> float:
        """How hard is it to stake this concept in existing trees?"""
        if not score.stake_attempts:
            return 1.0

        total_fit = sum(a.fit_score for a in score.stake_attempts)
        avg_fit = total_fit / len(score.stake_attempts)
        return 1.0 - avg_fit

    def _compute_contradiction_depth(self, score: AnchoredNoveltyScore) -> float:
        """How deep do contradictions go?"""
        if not score.stake_attempts:
            return 0.0

        max_depth = 0
        max_tree_depth = 1

        for attempt in score.stake_attempts:
            if attempt.creates_contradiction:
                max_depth = max(max_depth, attempt.contradiction_depth)
            tree_depth = attempt.tree.depth()
            if tree_depth > max_tree_depth:
                max_tree_depth = tree_depth

        score.max_contradiction_depth = max_depth

        if max_tree_depth > 0:
            # Higher depth score when contradicting SHALLOWER (more foundational) claims
            # Depth 0 = root = most foundational
            # So we want: contradiction at depth 0 → high score
            # contradiction at max depth → low score
            if any(a.creates_contradiction for a in score.stake_attempts):
                # Invert: shallower contradiction = higher novelty
                depth_ratio = max_depth / max_tree_depth
                return 1.0 - depth_ratio  # Contradiction at root (0) → 1.0
            return 0.0

        return 0.0

    def _compute_tree_coverage_gap(self, score: AnchoredNoveltyScore) -> float:
        """How many trees fail to accommodate this concept?"""
        if score.trees_total == 0:
            return 1.0

        coverage = score.trees_with_fit / score.trees_total
        return 1.0 - coverage

    def _compute_allocation_disruption(
        self,
        score: AnchoredNoveltyScore,
        ref: WorldModelReference
    ) -> float:
        """Would integrating this concept shift tendency allocations?"""
        if not score.stake_attempts:
            return 0.0

        tendency_claims: dict[Tendency, float] = {t: 0.0 for t in Tendency}

        for attempt in score.stake_attempts:
            if attempt.best_node and attempt.fit_score > 0.2:
                for stake in attempt.best_node.stakes:
                    try:
                        tendency = Tendency(stake.agent_id)
                        tendency_claims[tendency] += stake.weight * attempt.fit_score
                    except ValueError:
                        pass

        total_claims = sum(tendency_claims.values())
        if total_claims > 0:
            for t in tendency_claims:
                tendency_claims[t] /= total_claims

        current_allocations = {t: a.allocation for t, a in ref.agents.agents.items()}

        shift = 0.0
        for t in Tendency:
            current = current_allocations.get(t, 0.0)
            claimed = tendency_claims.get(t, 0.0)
            shift += abs(claimed - current)

        normalized_shift = min(shift / 2.0, 1.0)

        score.projected_allocation_shift = {
            t: tendency_claims.get(t, 0.0) - current_allocations.get(t, 0.0)
            for t in Tendency
        }

        return normalized_shift


# =============================================================================
# Convenience functions
# =============================================================================

def compute_novelty(
    concept: Observation,
    trees: TreeStore,
    agents: AgentSet,
    observations: ObservationStore,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> AnchoredNoveltyScore:
    """
    Convenience function to compute novelty of a concept.

    Creates a world model reference from the provided components and
    measures the concept's novelty against it.

    Args:
        concept: The observation/concept to evaluate
        trees: Agent's value hierarchies (claim hierarchies)
        agents: Agent's tendency allocations
        observations: Known observations already integrated
        similarity_fn: Optional custom similarity function
                      For best results, use embeddings.relation_fit_score

    Returns:
        AnchoredNoveltyScore with component breakdowns

    Note: All scores are relative to this specific world model anchor.
    """
    reference = WorldModelReference(trees, agents, observations)
    measure = AnchoredNoveltyMeasure(reference, similarity_fn)

    # Wrap observation in Concept if needed
    if isinstance(concept, Observation):
        c = Concept(content=concept.content, metadata=concept.metadata)
    else:
        c = concept

    return measure.measure(c)


# Legacy aliases for backwards compatibility
NoveltyComputer = AnchoredNoveltyMeasure
