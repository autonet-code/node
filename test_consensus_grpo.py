"""
Tests for the GRPO module and consensus-as-truth logic.

Verifies:
- Difficulty reward bell curve
- Majority vote computation
- Group-relative advantage computation
- Proposer/solver reward calculations
- Anti-collusion incentive properties
"""

import math
import pytest
from nodes.common.grpo import (
    DifficultyTarget,
    RolloutResult,
    compute_difficulty_reward,
    compute_majority_vote,
    compute_group_advantages,
    compute_proposer_reward,
    compute_solver_rewards,
    select_frontier_inputs,
    filter_by_difficulty,
)


class TestDifficultyReward:
    """Tests for the difficulty reward bell curve."""

    def test_peak_solvability_gives_max_reward(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.50, target)
        assert reward == pytest.approx(1.0, abs=0.01)

    def test_min_boundary_gives_reduced_reward(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.25, target)
        assert 0 < reward < 1.0

    def test_max_boundary_gives_reduced_reward(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.75, target)
        assert 0 < reward < 1.0

    def test_outside_min_gives_zero(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.20, target)
        assert reward == 0.0

    def test_outside_max_gives_zero(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.80, target)
        assert reward == 0.0

    def test_full_agreement_gives_zero(self):
        """100% agreement = trivially easy = anti-collusion zero reward."""
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(1.0, target)
        assert reward == 0.0

    def test_zero_agreement_gives_zero(self):
        """0% agreement = impossible task = zero reward."""
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward = compute_difficulty_reward(0.0, target)
        assert reward == 0.0

    def test_reward_is_symmetric_around_peak(self):
        """Reward should be equal at equal distances from peak."""
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward_above = compute_difficulty_reward(0.60, target)
        reward_below = compute_difficulty_reward(0.40, target)
        assert reward_above == pytest.approx(reward_below, abs=0.01)

    def test_closer_to_peak_gives_higher_reward(self):
        """Points closer to peak should yield higher reward."""
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.50)
        reward_close = compute_difficulty_reward(0.45, target)
        reward_far = compute_difficulty_reward(0.30, target)
        assert reward_close > reward_far


class TestMajorityVote:
    """Tests for majority vote computation."""

    def test_unanimous_vote(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.9),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s3", answer_hash="A", confidence=0.7),
        ]
        answer, solvability, count = compute_majority_vote(rollouts)
        assert answer == "A"
        assert solvability == pytest.approx(1.0)
        assert count == 3

    def test_split_vote(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.9),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s3", answer_hash="B", confidence=0.7),
            RolloutResult(solver_id="s4", answer_hash="B", confidence=0.6),
        ]
        answer, solvability, count = compute_majority_vote(rollouts)
        assert answer == "A"  # A has 2 votes, B has 2, A appears first
        assert solvability == pytest.approx(0.5)
        assert count == 2

    def test_clear_majority(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.9),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s3", answer_hash="A", confidence=0.7),
            RolloutResult(solver_id="s4", answer_hash="B", confidence=0.6),
        ]
        answer, solvability, count = compute_majority_vote(rollouts)
        assert answer == "A"
        assert solvability == pytest.approx(0.75)
        assert count == 3

    def test_empty_rollouts(self):
        answer, solvability, count = compute_majority_vote([])
        assert answer == ""
        assert solvability == 0.0
        assert count == 0

    def test_single_rollout(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="X", confidence=0.5),
        ]
        answer, solvability, count = compute_majority_vote(rollouts)
        assert answer == "X"
        assert solvability == 1.0
        assert count == 1


class TestGroupAdvantages:
    """Tests for GRPO group-relative advantages."""

    def test_normalized_advantages(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        advantages = compute_group_advantages(rewards)
        assert len(advantages) == 5
        # Mean should be ~0
        assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
        # Lowest reward -> most negative advantage
        assert advantages[0] < advantages[4]

    def test_equal_rewards_give_zero_advantages(self):
        rewards = [5.0, 5.0, 5.0]
        advantages = compute_group_advantages(rewards)
        assert all(a == 0.0 for a in advantages)

    def test_empty_rewards(self):
        advantages = compute_group_advantages([])
        assert advantages == []

    def test_single_reward(self):
        advantages = compute_group_advantages([42.0])
        assert advantages == [0.0]


class TestProposerReward:
    """Tests for proposer reward calculation."""

    def test_peak_difficulty_gives_full_reward(self):
        target = DifficultyTarget(peak_solvability=0.5)
        reward = compute_proposer_reward(0.5, target, base_reward=100.0)
        assert reward == pytest.approx(100.0, abs=1.0)

    def test_outside_range_gives_zero(self):
        target = DifficultyTarget(min_solvability=0.25, max_solvability=0.75, peak_solvability=0.5)
        reward = compute_proposer_reward(1.0, target, base_reward=100.0)
        assert reward == 0.0


class TestSolverRewards:
    """Tests for per-solver reward computation."""

    def test_majority_solvers_rewarded(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.7),
            RolloutResult(solver_id="s3", answer_hash="B", confidence=0.6),
        ]
        rewards = compute_solver_rewards(rollouts, "A", 0.8, base_reward=10.0)
        # s1 and s2 should get rewards, s3 should get 0
        assert rewards[0][1] > 0  # s1
        assert rewards[1][1] > 0  # s2
        assert rewards[2][1] == 0  # s3

    def test_higher_confidence_gets_bonus(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.9),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.3),
        ]
        rewards = compute_solver_rewards(rollouts, "A", 1.0, base_reward=10.0)
        # s1 has confidence > 0.5, so gets bonus
        # s2 has confidence < 0.5, so no bonus
        assert rewards[0][1] > rewards[1][1]

    def test_zero_difficulty_means_no_reward(self):
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.8),
        ]
        rewards = compute_solver_rewards(rollouts, "A", 0.0, base_reward=10.0)
        assert rewards[0][1] == 0.0


class TestFrontierSelection:
    """Tests for frontier input selection."""

    def test_selects_near_target_uncertainty(self):
        candidates = list(range(10))
        # Uncertainties: some near 0.5 (frontier), some far
        uncertainties = [0.1, 0.2, 0.3, 0.48, 0.49, 0.50, 0.51, 0.52, 0.8, 0.9]
        selected = select_frontier_inputs(uncertainties, candidates, target_uncertainty=0.5, top_k=3)
        assert len(selected) == 3
        # Should select indices 5 (0.50), 4 (0.49), 6 (0.51)
        assert set(selected) == {5, 4, 6}

    def test_empty_inputs(self):
        assert select_frontier_inputs([], [], top_k=5) == []


class TestDifficultyFilter:
    """Tests for difficulty-based filtering."""

    def test_keeps_rollouts_in_range(self):
        # 2 out of 3 agree -> solvability = 66.7%
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.7),
            RolloutResult(solver_id="s3", answer_hash="B", confidence=0.6),
        ]
        result = filter_by_difficulty(rollouts, min_retention=0.25, max_retention=0.75)
        assert len(result) == 3  # 66.7% is within [25%, 75%]

    def test_filters_out_trivial_tasks(self):
        # All agree -> solvability = 100%
        rollouts = [
            RolloutResult(solver_id="s1", answer_hash="A", confidence=0.8),
            RolloutResult(solver_id="s2", answer_hash="A", confidence=0.7),
        ]
        result = filter_by_difficulty(rollouts, min_retention=0.25, max_retention=0.75)
        assert len(result) == 0  # 100% is outside [25%, 75%]


class TestAntiCollusion:
    """Integration tests verifying anti-collusion properties."""

    def test_collusion_yields_less_than_honest_disagreement(self):
        """
        Core anti-collusion property: when all solvers give the same answer
        (possible collusion), the reward should be LESS than when solvers
        naturally disagree at ~50%.
        """
        target = DifficultyTarget(
            min_solvability=0.25,
            max_solvability=0.75,
            peak_solvability=0.50,
        )

        # Collusion scenario: 100% agreement
        collusion_reward = compute_difficulty_reward(1.0, target)

        # Honest scenario: 50% agreement (natural frontier difficulty)
        honest_reward = compute_difficulty_reward(0.5, target)

        assert collusion_reward == 0.0
        assert honest_reward > 0.0
        assert honest_reward > collusion_reward

    def test_proposer_incentive_aligned_with_difficulty(self):
        """Proposer earns more by creating harder (well-calibrated) tasks."""
        target = DifficultyTarget(
            min_solvability=0.25,
            max_solvability=0.75,
            peak_solvability=0.50,
        )
        base = 100.0

        easy_reward = compute_proposer_reward(0.9, target, base)    # Too easy
        medium_reward = compute_proposer_reward(0.5, target, base)  # Perfect
        hard_reward = compute_proposer_reward(0.1, target, base)    # Too hard

        assert easy_reward == 0.0     # Outside range
        assert hard_reward == 0.0     # Outside range
        assert medium_reward > 0.0    # Peak reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
