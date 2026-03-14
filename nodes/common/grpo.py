"""
Group Relative Policy Optimization (GRPO) for Autonet.

Implements the reward computation and advantage estimation from MM-Zero's
self-evolving training framework, adapted for decentralized consensus-as-truth.

Key concepts:
- Difficulty-calibrated rewards: peak reward when solver agreement is ~50%
- Group-relative advantages: normalize rewards within rollout groups
- Solvability scoring: bell-curve reward centered on target difficulty
"""

import math
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DifficultyTarget:
    """Mirrors AutonetLib.DifficultyTarget from Solidity."""
    min_solvability: float = 0.25   # 25% minimum agreement
    max_solvability: float = 0.75   # 75% maximum agreement
    peak_solvability: float = 0.50  # 50% optimal difficulty


@dataclass
class RolloutResult:
    """Result of a single solver's rollout."""
    solver_id: str
    answer_hash: str
    confidence: float        # 0-1
    weight_delta: Optional[dict] = None
    metrics: Optional[dict] = None


def compute_difficulty_reward(
    solvability: float,
    target: DifficultyTarget,
) -> float:
    """
    Compute reward using a bell-curve centered on peak solvability.

    This is the core anti-collusion mechanism: tasks where all solvers agree
    (solvability ~1.0) yield near-zero reward, making collusion unprofitable.

    Args:
        solvability: Fraction of solvers who gave the majority answer (0-1)
        target: Difficulty target with min/max/peak solvability

    Returns:
        Reward multiplier in [0, 1]. 1.0 at peak, 0 outside range.
    """
    if solvability < target.min_solvability or solvability > target.max_solvability:
        return 0.0

    half_range = target.max_solvability - target.min_solvability
    if half_range <= 0:
        return 1.0

    distance = abs(solvability - target.peak_solvability)
    normalized = distance / half_range

    # Quadratic falloff (matches Solidity implementation)
    score = 1.0 - normalized ** 2
    return max(0.0, score)


def compute_majority_vote(rollouts: List[RolloutResult]) -> Tuple[str, float, int]:
    """
    Compute majority vote across solver rollouts.

    Returns:
        Tuple of (majority_answer_hash, solvability_ratio, majority_count)
    """
    if not rollouts:
        return ("", 0.0, 0)

    # Count occurrences of each answer
    answer_counts: Dict[str, int] = {}
    for rollout in rollouts:
        answer_counts[rollout.answer_hash] = answer_counts.get(rollout.answer_hash, 0) + 1

    # Find majority
    majority_answer = max(answer_counts, key=answer_counts.get)
    majority_count = answer_counts[majority_answer]
    solvability = majority_count / len(rollouts)

    return majority_answer, solvability, majority_count


def compute_group_advantages(rewards: List[float]) -> List[float]:
    """
    Compute group-relative advantages (GRPO).

    Normalizes rewards within the group to yield response-level advantages.
    This replaces the need for a separate value function.

    Args:
        rewards: List of raw reward values for each rollout in the group

    Returns:
        List of normalized advantage values (mean=0, std=1)
    """
    if not rewards:
        return []

    n = len(rewards)
    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / n
    std_reward = math.sqrt(variance) if variance > 0 else 0.0

    if std_reward < 1e-8:
        return [0.0] * n

    return [(r - mean_reward) / std_reward for r in rewards]


def compute_proposer_reward(
    solvability: float,
    target: DifficultyTarget,
    base_reward: float,
) -> float:
    """
    Compute proposer reward based on task difficulty calibration.

    The proposer is rewarded for generating tasks at the frontier of
    model capability. Well-calibrated tasks (solvability near peak)
    earn full reward; poorly calibrated ones earn less or nothing.

    Args:
        solvability: Actual solver agreement ratio
        target: The difficulty target the proposer committed to
        base_reward: Maximum possible reward

    Returns:
        Actual reward amount
    """
    difficulty_score = compute_difficulty_reward(solvability, target)
    return base_reward * difficulty_score


def compute_solver_rewards(
    rollouts: List[RolloutResult],
    majority_answer: str,
    difficulty_score: float,
    base_reward: float,
) -> List[Tuple[str, float]]:
    """
    Compute per-solver rewards for consensus-mode tasks.

    Solvers who matched the majority answer receive a reward scaled by
    the difficulty score. Solvers who disagreed get nothing (no slashing).

    Args:
        rollouts: All solver rollouts for the task
        majority_answer: The answer hash with most votes
        difficulty_score: Difficulty score from compute_difficulty_reward()
        base_reward: Base solver reward amount

    Returns:
        List of (solver_id, reward_amount) tuples
    """
    rewards = []
    for rollout in rollouts:
        if rollout.answer_hash == majority_answer:
            reward = base_reward * difficulty_score

            # Confidence bonus: up to 20% extra for high confidence
            if rollout.confidence > 0.5:
                bonus = reward * (rollout.confidence - 0.5) / 5.0
                reward += bonus

            rewards.append((rollout.solver_id, reward))
        else:
            rewards.append((rollout.solver_id, 0.0))

    return rewards


def select_frontier_inputs(
    uncertainties: List[float],
    candidates: List,
    target_uncertainty: float = 0.5,
    top_k: int = 100,
) -> List:
    """
    Select inputs near the frontier of model capability.

    Used by the Proposer to generate tasks at the optimal difficulty.
    Selects inputs where the model is most uncertain (~50% confidence).

    Args:
        uncertainties: Model uncertainty for each candidate (0-1)
        candidates: Corresponding input data
        target_uncertainty: Target uncertainty level (0.5 = maximum info)
        top_k: Number of frontier inputs to select

    Returns:
        List of selected frontier inputs
    """
    if not uncertainties or not candidates:
        return []

    # Sort by distance from target uncertainty
    indexed = list(enumerate(uncertainties))
    indexed.sort(key=lambda x: abs(x[1] - target_uncertainty))

    # Select top-k closest to frontier
    selected_indices = [idx for idx, _ in indexed[:top_k]]
    return [candidates[i] for i in selected_indices]


def filter_by_difficulty(
    rollouts: List[RolloutResult],
    min_retention: float = 0.25,
    max_retention: float = 0.75,
) -> List[RolloutResult]:
    """
    Filter training data by difficulty (MM-Zero stage-specific filtering).

    Retains only rollouts where solvability falls within the retention range.
    This prevents training on trivially easy or impossibly hard tasks.

    Args:
        rollouts: All rollouts to filter
        min_retention: Minimum solvability for retention
        max_retention: Maximum solvability for retention

    Returns:
        Filtered list of rollouts
    """
    if not rollouts:
        return []

    _, solvability, _ = compute_majority_vote(rollouts)

    if min_retention <= solvability <= max_retention:
        return rollouts
    else:
        logger.info(
            f"Filtered out rollouts: solvability={solvability:.2f} "
            f"outside [{min_retention}, {max_retention}]"
        )
        return []
