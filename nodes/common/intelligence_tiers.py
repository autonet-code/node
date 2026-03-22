"""
Intelligence Tiers for Native Inference.

Defines three tiers of inference quality based on how many pipeline
modules are involved. Higher tiers use more modules, produce richer
results, and cost more ATN.

Story 9.3: Intelligence tiers (more modules = deeper model = higher price).
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

from .model_module import MODULE_NAMES

logger = logging.getLogger(__name__)


class InferenceTier(IntEnum):
    """
    Intelligence tiers for inference pricing.

    Tier 1: Encoder only — visual features, embeddings, classification
    Tier 2: Encoder + reasoning — cross-modal fusion, semantic prediction
    Tier 3: Full pipeline + decode — complete generation from latent plan
    """
    TIER_1 = 1  # Encoder only
    TIER_2 = 2  # Encoder + cross-modal + predictor
    TIER_3 = 3  # Full pipeline including local decode


# Module sets per tier
TIER_MODULES: Dict[InferenceTier, List[str]] = {
    InferenceTier.TIER_1: [
        "visual_encoder",
    ],
    InferenceTier.TIER_2: [
        "visual_encoder",
        "text_encoder",
        "cross_modal_fusion",
        "semantic_predictor",
    ],
    InferenceTier.TIER_3: MODULE_NAMES.copy(),  # All 5 modules
}


@dataclass
class TierPricing:
    """Pricing configuration for a tier."""
    tier: InferenceTier
    base_cost_atn: float      # Base ATN cost per request
    cost_per_token: float     # Additional cost per output token
    modules: List[str]        # Modules involved
    description: str = ""

    @property
    def module_count(self) -> int:
        return len(self.modules)


# Default pricing (in ATN units, not wei)
DEFAULT_PRICING: Dict[InferenceTier, TierPricing] = {
    InferenceTier.TIER_1: TierPricing(
        tier=InferenceTier.TIER_1,
        base_cost_atn=1.0,
        cost_per_token=0.0,
        modules=TIER_MODULES[InferenceTier.TIER_1],
        description="Encoder only: visual features and embeddings",
    ),
    InferenceTier.TIER_2: TierPricing(
        tier=InferenceTier.TIER_2,
        base_cost_atn=5.0,
        cost_per_token=0.01,
        modules=TIER_MODULES[InferenceTier.TIER_2],
        description="Encoder + reasoning: cross-modal fusion and semantic prediction",
    ),
    InferenceTier.TIER_3: TierPricing(
        tier=InferenceTier.TIER_3,
        base_cost_atn=10.0,
        cost_per_token=0.05,
        modules=TIER_MODULES[InferenceTier.TIER_3],
        description="Full pipeline: complete generation from latent plan",
    ),
}


class InferenceTierManager:
    """
    Manages intelligence tier pricing and module selection.

    Usage:
        manager = InferenceTierManager()
        pricing = manager.get_pricing(InferenceTier.TIER_2)
        cost = manager.compute_cost(InferenceTier.TIER_3, output_tokens=100)
        modules = manager.get_modules(InferenceTier.TIER_1)
    """

    def __init__(self, pricing: Optional[Dict[InferenceTier, TierPricing]] = None):
        self._pricing = pricing or DEFAULT_PRICING.copy()

    def get_pricing(self, tier: InferenceTier) -> TierPricing:
        """Get pricing info for a tier."""
        if tier not in self._pricing:
            raise ValueError(f"Unknown tier: {tier}")
        return self._pricing[tier]

    def get_modules(self, tier: InferenceTier) -> List[str]:
        """Get the list of modules used by a tier."""
        return self.get_pricing(tier).modules

    def compute_cost(self, tier: InferenceTier, output_tokens: int = 0) -> float:
        """
        Compute ATN cost for an inference request.

        Args:
            tier: Intelligence tier
            output_tokens: Number of output tokens (for Tier 2/3)

        Returns:
            Total ATN cost
        """
        p = self.get_pricing(tier)
        return p.base_cost_atn + (p.cost_per_token * output_tokens)

    def recommend_tier(self, task_type: str) -> InferenceTier:
        """
        Recommend a tier based on task type.

        Args:
            task_type: One of "embedding", "classification", "generation", "reasoning"

        Returns:
            Recommended InferenceTier
        """
        task_map = {
            "embedding": InferenceTier.TIER_1,
            "classification": InferenceTier.TIER_1,
            "similarity": InferenceTier.TIER_1,
            "reasoning": InferenceTier.TIER_2,
            "qa": InferenceTier.TIER_2,
            "captioning": InferenceTier.TIER_2,
            "generation": InferenceTier.TIER_3,
            "translation": InferenceTier.TIER_3,
            "summarization": InferenceTier.TIER_3,
        }
        return task_map.get(task_type, InferenceTier.TIER_2)

    def tier_summary(self) -> str:
        """Human-readable summary of all tiers."""
        lines = ["Intelligence Tiers", "=" * 50]
        for tier in InferenceTier:
            p = self._pricing[tier]
            lines.append(
                f"  Tier {tier.value}: {p.description}"
            )
            lines.append(
                f"    Modules: {', '.join(p.modules)}"
            )
            lines.append(
                f"    Base cost: {p.base_cost_atn} ATN + {p.cost_per_token} ATN/token"
            )
        return "\n".join(lines)
