"""
Tests for AlignmentPricing — alignment-based economic pricing.

Run: python -m pytest tests/test_alignment_pricing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.common.alignment_pricing import (
    AlignmentPricing,
    PricingResult,
    TrainingIncentiveResult,
    _cosine_similarity_keywords,
)


# =============================================================================
# Tests: Keyword Similarity
# =============================================================================


class TestKeywordSimilarity:
    def test_identical_text(self):
        """Identical text has similarity 1.0."""
        score = _cosine_similarity_keywords(
            "machine learning training", "machine learning training"
        )
        assert score == 1.0

    def test_no_overlap(self):
        """Completely different text has similarity 0.0."""
        score = _cosine_similarity_keywords(
            "machine learning", "cooking recipes"
        )
        assert score == 0.0

    def test_partial_overlap(self):
        """Partial overlap produces intermediate score."""
        score = _cosine_similarity_keywords(
            "machine learning training data",
            "machine learning model inference"
        )
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        """Empty strings produce 0.0."""
        assert _cosine_similarity_keywords("", "something") == 0.0
        assert _cosine_similarity_keywords("something", "") == 0.0
        assert _cosine_similarity_keywords("", "") == 0.0

    def test_stop_words_filtered(self):
        """Stop words are filtered and don't inflate similarity."""
        # "the" and "a" are stop words
        score = _cosine_similarity_keywords("the", "a")
        assert score == 0.0

    def test_case_insensitive(self):
        """Similarity is case-insensitive."""
        score = _cosine_similarity_keywords("Machine Learning", "machine learning")
        assert score == 1.0


# =============================================================================
# Tests: AlignmentPricing — Alignment Computation
# =============================================================================


class TestAlignmentComputation:
    def test_high_alignment(self):
        """Highly aligned task/user/jurisdiction produces high score."""
        pricing = AlignmentPricing()
        score, breakdown = pricing.compute_alignment(
            task_description="translation accuracy quality",
            user_standards="translation accuracy quality assurance",
            jurisdiction_standards="translation accuracy quality standards",
        )
        assert score > 0.5
        assert "user_to_jurisdiction" in breakdown
        assert "task_to_user" in breakdown
        assert "task_to_jurisdiction" in breakdown

    def test_zero_alignment(self):
        """Completely unrelated texts produce zero alignment."""
        pricing = AlignmentPricing()
        score, _ = pricing.compute_alignment(
            task_description="cooking recipes",
            user_standards="quantum physics research",
            jurisdiction_standards="marine biology conservation",
        )
        assert score == 0.0

    def test_geometric_mean(self):
        """Composite score is geometric mean of three similarities."""
        # Use a custom similarity function for precise testing
        call_results = iter([0.8, 0.6, 0.4])
        pricing = AlignmentPricing(
            similarity_fn=lambda a, b: next(call_results)
        )

        score, _ = pricing.compute_alignment("a", "b", "c")
        expected = (0.8 * 0.6 * 0.4) ** (1.0 / 3.0)
        assert abs(score - expected) < 1e-10

    def test_one_zero_component_zeros_all(self):
        """If any component is zero, the geometric mean is zero."""
        call_results = iter([0.9, 0.0, 0.7])
        pricing = AlignmentPricing(
            similarity_fn=lambda a, b: next(call_results)
        )

        score, _ = pricing.compute_alignment("a", "b", "c")
        assert score == 0.0


# =============================================================================
# Tests: AlignmentPricing — Price Computation
# =============================================================================


class TestPriceComputation:
    def test_neutral_tier(self):
        """Moderate alignment produces base cost (neutral tier)."""
        # Similarity that always returns 0.5 -> alignment = 0.5
        pricing = AlignmentPricing(
            high_alignment_threshold=0.7,
            low_alignment_threshold=0.3,
            similarity_fn=lambda a, b: 0.5,
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        assert result.tier == "neutral"
        assert result.user_pays == 100.0
        assert result.treasury_pays == 0.0
        assert result.premium_collected == 0.0

    def test_premium_tier(self):
        """Low alignment produces premium pricing."""
        # Similarity that returns very low values
        pricing = AlignmentPricing(
            high_alignment_threshold=0.7,
            low_alignment_threshold=0.3,
            max_premium_rate=0.5,
            similarity_fn=lambda a, b: 0.01,  # Very low alignment
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        assert result.tier == "premium"
        assert result.user_pays > 100.0  # User pays more
        assert result.treasury_pays == 0.0
        assert result.premium_collected > 0.0

    def test_subsidized_tier(self):
        """High alignment produces subsidized pricing."""
        # High similarity + mature network with treasury
        pricing = AlignmentPricing(
            high_alignment_threshold=0.7,
            low_alignment_threshold=0.3,
            max_subsidy_rate=0.8,
            network_maturity=1.0,
            treasury_balance=100000.0,
            treasury_threshold=100000.0,
            similarity_fn=lambda a, b: 0.95,  # Very high alignment
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        assert result.tier == "subsidized"
        assert result.user_pays < 100.0  # User pays less
        assert result.treasury_pays > 0.0  # Treasury subsidizes
        assert result.premium_collected == 0.0

    def test_no_subsidy_early_network(self):
        """Early network (low maturity) doesn't subsidize even for high alignment."""
        pricing = AlignmentPricing(
            network_maturity=0.0,
            treasury_balance=100000.0,
            treasury_threshold=100000.0,
            similarity_fn=lambda a, b: 0.95,
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        # Subsidized tier but subsidy capacity is 0
        assert result.tier == "subsidized"
        assert result.user_pays == pytest.approx(100.0)
        assert result.treasury_pays == pytest.approx(0.0)

    def test_no_subsidy_empty_treasury(self):
        """Empty treasury doesn't subsidize even for high alignment."""
        pricing = AlignmentPricing(
            network_maturity=1.0,
            treasury_balance=0.0,
            treasury_threshold=100000.0,
            similarity_fn=lambda a, b: 0.95,
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        assert result.treasury_pays == pytest.approx(0.0)

    def test_max_premium(self):
        """Maximum premium is bounded by max_premium_rate."""
        pricing = AlignmentPricing(
            max_premium_rate=0.5,
            low_alignment_threshold=0.3,
            similarity_fn=lambda a, b: 0.0,  # Zero alignment -> max premium
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)

        # Premium should be at most 50% (max_premium_rate=0.5)
        assert result.user_pays <= 150.0
        assert result.premium_collected <= 50.0

    def test_base_cost_zero(self):
        """Zero base cost produces zero pricing."""
        pricing = AlignmentPricing(similarity_fn=lambda a, b: 0.5)
        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=0.0)

        assert result.user_pays == 0.0
        assert result.treasury_pays == 0.0

    def test_self_funding_property(self):
        """Premium revenue > 0 when alignment is low, can fund subsidies."""
        pricing = AlignmentPricing(
            max_premium_rate=0.5,
            similarity_fn=lambda a, b: 0.01,
        )

        result = pricing.compute_price("task", "user", "jurisdiction", base_cost=100.0)
        assert result.premium_collected > 0.0

    def test_pricing_result_properties(self):
        """PricingResult properties work correctly."""
        result = PricingResult(
            alignment_score=0.8,
            user_pays=80.0,
            treasury_pays=20.0,
            premium_collected=0.0,
            base_cost=100.0,
            tier="subsidized",
            breakdown={"user_to_jurisdiction": 0.8},
        )

        assert result.effective_cost == 80.0
        assert result.discount_rate == pytest.approx(0.2)
        assert result.premium_rate == pytest.approx(0.0)


# =============================================================================
# Tests: AlignmentPricing — Training Incentives
# =============================================================================


class TestTrainingIncentives:
    def test_zero_capability_max_reward(self):
        """Zero capability -> maximum reward multiplier (3x)."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="visual_encoder",
            current_score=0,
            target_score=8000,
            base_reward=100.0,
        )

        assert result.capability_gap == 1.0
        assert result.reward_multiplier == pytest.approx(3.0)
        assert result.effective_reward == pytest.approx(300.0)

    def test_at_target_base_reward(self):
        """At target capability -> base reward (1x)."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="text_encoder",
            current_score=8000,
            target_score=8000,
            base_reward=100.0,
        )

        assert result.capability_gap == 0.0
        assert result.reward_multiplier == pytest.approx(1.0)
        assert result.effective_reward == pytest.approx(100.0)

    def test_above_target_reduced_reward(self):
        """Above target -> reduced reward multiplier."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="text_encoder",
            current_score=10000,
            target_score=8000,
            base_reward=100.0,
        )

        assert result.reward_multiplier < 1.0
        assert result.reward_multiplier >= 0.5  # Never below 0.5x
        assert result.effective_reward < 100.0

    def test_halfway_to_target(self):
        """Halfway to target -> 2x reward."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="predictor",
            current_score=4000,
            target_score=8000,
            base_reward=100.0,
        )

        assert result.capability_gap == pytest.approx(0.5)
        assert result.reward_multiplier == pytest.approx(2.0)
        assert result.effective_reward == pytest.approx(200.0)

    def test_no_target_base_reward(self):
        """No target set -> base reward (1x)."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="unknown_module",
            current_score=5000,
            target_score=0,
            base_reward=100.0,
        )

        assert result.reward_multiplier == 1.0
        assert result.effective_reward == 100.0

    def test_minimum_multiplier(self):
        """Multiplier never drops below 0.5x even far above target."""
        pricing = AlignmentPricing()
        result = pricing.compute_training_incentive(
            module_id="saturated_module",
            current_score=10000,
            target_score=2000,  # Way above target
            base_reward=100.0,
        )

        assert result.reward_multiplier >= 0.5
        assert result.effective_reward >= 50.0


# =============================================================================
# Tests: AlignmentPricing — Network State Updates
# =============================================================================


class TestNetworkState:
    def test_update_maturity(self):
        pricing = AlignmentPricing(network_maturity=0.1)
        pricing.update_network_state(maturity=0.5)
        assert pricing.network_maturity == 0.5

    def test_update_treasury(self):
        pricing = AlignmentPricing(treasury_balance=0.0)
        pricing.update_network_state(treasury_balance=50000.0)
        assert pricing.treasury_balance == 50000.0

    def test_maturity_clamped(self):
        pricing = AlignmentPricing()
        pricing.update_network_state(maturity=1.5)
        assert pricing.network_maturity == 1.0

        pricing.update_network_state(maturity=-0.1)
        assert pricing.network_maturity == 0.0

    def test_treasury_not_negative(self):
        pricing = AlignmentPricing()
        pricing.update_network_state(treasury_balance=-100.0)
        assert pricing.treasury_balance == 0.0


# =============================================================================
# Tests: AlignmentPricing — Continuous Pricing (no binary allow/deny)
# =============================================================================


class TestContinuousPricing:
    def test_pricing_is_continuous(self):
        """Pricing varies continuously, never binary allow/deny."""
        results = []
        for sim in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            pricing = AlignmentPricing(
                network_maturity=1.0,
                treasury_balance=100000.0,
                treasury_threshold=100000.0,
                similarity_fn=lambda a, b, s=sim: s,
            )
            result = pricing.compute_price("task", "user", "jurisdiction", 100.0)
            results.append(result)

        # All results should have user_pays > 0 (never free)
        for r in results:
            assert r.user_pays >= 0.0

        # Premium results should cost more than neutral
        premium_results = [r for r in results if r.tier == "premium"]
        neutral_results = [r for r in results if r.tier == "neutral"]
        subsidized_results = [r for r in results if r.tier == "subsidized"]

        assert len(premium_results) > 0
        assert len(neutral_results) > 0
        assert len(subsidized_results) > 0

        for p in premium_results:
            assert p.user_pays >= 100.0
        for n in neutral_results:
            assert n.user_pays == pytest.approx(100.0)
        for s in subsidized_results:
            assert s.user_pays <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
