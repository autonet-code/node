"""
Alignment Pricing — economic pressure toward aligned operations.

Implements the pricing function from Section 2.4 of emergent_alignment.md:

    alignment = geometric_mean(user_to_jurisdiction, task_to_user, task_to_jurisdiction)
    high alignment -> subsidized (toward free)
    neutral        -> base cost
    low alignment  -> premium (funds subsidies)

In V1 this is **advisory only**: the score is computed and displayed but
pricing is not enforced. The same mechanism drives both inference pricing
and training reward multipliers.

Usage:
    pricing = AlignmentPricing()
    result = pricing.compute_price(
        task_description="translate this document",
        user_standards="professional translation, accuracy",
        jurisdiction_standards="language services, quality assurance",
        base_cost=100,
    )
    # result.user_pays, result.treasury_pays, result.alignment_score
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PricingResult:
    """Result of an alignment pricing computation."""
    alignment_score: float       # 0.0-1.0 composite alignment
    user_pays: float             # Amount the user pays
    treasury_pays: float         # Amount subsidized by treasury
    premium_collected: float     # Extra premium above base cost (feeds treasury)
    base_cost: float             # Original base cost
    tier: str                    # "subsidized", "neutral", or "premium"
    breakdown: Dict[str, float]  # Individual alignment components

    @property
    def effective_cost(self) -> float:
        """Total cost (user_pays should equal this)."""
        return self.user_pays

    @property
    def discount_rate(self) -> float:
        """Fraction of base cost covered by treasury (0.0-1.0)."""
        if self.base_cost <= 0:
            return 0.0
        return self.treasury_pays / self.base_cost

    @property
    def premium_rate(self) -> float:
        """Premium multiplier above base cost (0.0 = no premium)."""
        if self.base_cost <= 0:
            return 0.0
        return self.premium_collected / self.base_cost


@dataclass
class TrainingIncentiveResult:
    """Result of a training reward multiplier computation."""
    capability_gap: float        # 0.0-1.0, how much the network needs this
    reward_multiplier: float     # Multiplier on base training reward
    base_reward: float           # Original base reward
    effective_reward: float      # base_reward * reward_multiplier
    module_id: str               # Which module is being trained


def _cosine_similarity_keywords(text_a: str, text_b: str) -> float:
    """
    Simple keyword-overlap similarity between two text descriptions.

    Uses normalized word-set overlap as a proxy for semantic similarity.
    In production, this would be replaced with embedding-based similarity
    (e.g., sentence transformers). For V1 advisory mode, keyword overlap
    is sufficient to demonstrate the mechanism.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    if not text_a or not text_b:
        return 0.0

    # Normalize and tokenize
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    # Remove very common words that don't carry meaning
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "about", "this", "that", "it",
    }
    words_a -= stop_words
    words_b -= stop_words

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    # Jaccard similarity
    return len(intersection) / len(union)


class AlignmentPricing:
    """
    Computes alignment-based pricing for inference and training.

    The core pricing function creates economic pressure toward alignment:
    - Highly aligned operations are subsidized (toward free)
    - Neutral operations pay base cost
    - Misaligned operations pay a premium (funds subsidies)

    This is the same mechanism for both inference pricing and training
    reward multipliers, applied to different inputs.

    Parameters are governable — thresholds, max rates, and subsidy caps
    can be adjusted through DAO governance.
    """

    def __init__(
        self,
        high_alignment_threshold: float = 0.7,
        low_alignment_threshold: float = 0.3,
        max_subsidy_rate: float = 0.8,
        max_premium_rate: float = 0.5,
        network_maturity: float = 0.1,
        treasury_balance: float = 0.0,
        treasury_threshold: float = 100000.0,
        similarity_fn=None,
    ):
        """
        Args:
            high_alignment_threshold: Score above which subsidies apply
            low_alignment_threshold: Score below which premiums apply
            max_subsidy_rate: Maximum fraction of cost that can be subsidized
            max_premium_rate: Maximum premium above base cost (0.5 = up to 1.5x)
            network_maturity: 0.0 = early network, 1.0 = mature (affects subsidy capacity)
            treasury_balance: Current treasury balance (affects subsidy capacity)
            treasury_threshold: Treasury balance at which full subsidy is available
            similarity_fn: Custom text similarity function (default: keyword overlap)
        """
        self.high_threshold = high_alignment_threshold
        self.low_threshold = low_alignment_threshold
        self.max_subsidy_rate = max_subsidy_rate
        self.max_premium_rate = max_premium_rate
        self.network_maturity = network_maturity
        self.treasury_balance = treasury_balance
        self.treasury_threshold = treasury_threshold
        self._similarity = similarity_fn or _cosine_similarity_keywords

        self.logger = logging.getLogger("AlignmentPricing")

    def compute_alignment(
        self,
        task_description: str,
        user_standards: str,
        jurisdiction_standards: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite alignment score.

        Uses geometric mean of three pairwise similarities:
        - user_to_jurisdiction: How aligned the user is with their jurisdiction
        - task_to_user: How aligned the task is with the user's standards
        - task_to_jurisdiction: How aligned the task is with jurisdiction standards

        Returns:
            (composite_score, breakdown_dict)
        """
        user_to_jurisdiction = self._similarity(user_standards, jurisdiction_standards)
        task_to_user = self._similarity(task_description, user_standards)
        task_to_jurisdiction = self._similarity(task_description, jurisdiction_standards)

        # Geometric mean (handles zeros gracefully)
        product = user_to_jurisdiction * task_to_user * task_to_jurisdiction
        if product <= 0:
            composite = 0.0
        else:
            composite = product ** (1.0 / 3.0)

        breakdown = {
            "user_to_jurisdiction": user_to_jurisdiction,
            "task_to_user": task_to_user,
            "task_to_jurisdiction": task_to_jurisdiction,
        }

        return composite, breakdown

    def compute_price(
        self,
        task_description: str,
        user_standards: str,
        jurisdiction_standards: str,
        base_cost: float,
    ) -> PricingResult:
        """
        Compute alignment-adjusted price for an inference operation.

        This is the core pricing function from the emergent_alignment paper
        (Section 2.4), adapted for V1 advisory mode.

        Args:
            task_description: Description of the task/operation
            user_standards: User's published personal standards
            jurisdiction_standards: Jurisdiction's published standards
            base_cost: Base cost of the operation (in ATN or tokens)

        Returns:
            PricingResult with adjusted pricing.
        """
        alignment, breakdown = self.compute_alignment(
            task_description, user_standards, jurisdiction_standards
        )

        # Subsidy capacity depends on network maturity and treasury
        if self.treasury_threshold > 0:
            treasury_factor = min(
                1.0, self.treasury_balance / self.treasury_threshold
            )
        else:
            treasury_factor = 0.0
        effective_subsidy_capacity = min(
            self.network_maturity, treasury_factor
        ) * self.max_subsidy_rate

        if alignment >= self.high_threshold:
            # Subsidized tier: alignment above high threshold
            # Subsidy scales linearly within the high-alignment range
            alignment_range = 1.0 - self.high_threshold
            if alignment_range > 0:
                subsidy_fraction = (alignment - self.high_threshold) / alignment_range
            else:
                subsidy_fraction = 1.0
            subsidy_rate = effective_subsidy_capacity * subsidy_fraction
            user_pays = base_cost * (1.0 - subsidy_rate)
            treasury_pays = base_cost * subsidy_rate
            premium_collected = 0.0
            tier = "subsidized"

        elif alignment <= self.low_threshold:
            # Premium tier: alignment below low threshold
            # Premium scales linearly within the low-alignment range
            if self.low_threshold > 0:
                misalignment_fraction = (self.low_threshold - alignment) / self.low_threshold
            else:
                misalignment_fraction = 1.0
            premium_rate = self.max_premium_rate * misalignment_fraction
            user_pays = base_cost * (1.0 + premium_rate)
            treasury_pays = 0.0
            premium_collected = base_cost * premium_rate
            tier = "premium"

        else:
            # Neutral tier: between thresholds
            user_pays = base_cost
            treasury_pays = 0.0
            premium_collected = 0.0
            tier = "neutral"

        return PricingResult(
            alignment_score=alignment,
            user_pays=user_pays,
            treasury_pays=treasury_pays,
            premium_collected=premium_collected,
            base_cost=base_cost,
            tier=tier,
            breakdown=breakdown,
        )

    def compute_training_incentive(
        self,
        module_id: str,
        current_score: int,
        target_score: int,
        base_reward: float,
    ) -> TrainingIncentiveResult:
        """
        Compute training reward multiplier based on capability gap.

        The same pricing mechanism applied to training: modules that the
        network lacks (low capability score) get higher training rewards.
        Modules at or above target quality get reduced rewards.

        Args:
            module_id: Module identifier (e.g., "visual_encoder")
            current_score: Current capability score (0-10000 bps)
            target_score: Target capability score (0-10000 bps)
            base_reward: Base training reward in ATN

        Returns:
            TrainingIncentiveResult with reward multiplier.
        """
        if target_score <= 0:
            # No target set — use base reward
            return TrainingIncentiveResult(
                capability_gap=0.0,
                reward_multiplier=1.0,
                base_reward=base_reward,
                effective_reward=base_reward,
                module_id=module_id,
            )

        # Capability gap: how far from target (0.0 = at target, 1.0 = no capability)
        gap = max(0.0, min(1.0, 1.0 - (current_score / target_score)))

        # Reward multiplier:
        #   gap=1.0 (no capability) -> 3.0x reward (strong incentive)
        #   gap=0.5 (halfway)       -> 2.0x reward
        #   gap=0.0 (at target)     -> 1.0x reward (base)
        #   gap<0   (above target)  -> 0.5x reward (diminishing returns)
        if current_score >= target_score:
            # Above target: diminishing returns
            overshoot = (current_score - target_score) / target_score
            multiplier = max(0.5, 1.0 - overshoot * 0.5)
        else:
            # Below target: proportional incentive
            multiplier = 1.0 + gap * 2.0  # 1x to 3x

        effective_reward = base_reward * multiplier

        return TrainingIncentiveResult(
            capability_gap=gap,
            reward_multiplier=multiplier,
            base_reward=base_reward,
            effective_reward=effective_reward,
            module_id=module_id,
        )

    def update_network_state(
        self,
        maturity: Optional[float] = None,
        treasury_balance: Optional[float] = None,
    ):
        """Update network state parameters that affect pricing."""
        if maturity is not None:
            self.network_maturity = max(0.0, min(1.0, maturity))
        if treasury_balance is not None:
            self.treasury_balance = max(0.0, treasury_balance)
