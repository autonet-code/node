"""Solution verification: did the solver's stake delta improve the
equilibrium?

Same protocol contract as ``verify_jepa_solution``: returns a numeric
score in [0, 100]. Above some threshold = valid contribution.

Metric
------

  - Build the seed world (or load the pre-solver global model).
  - Equilibrate. Record per-tendency root scores and the GAP between
    each tendency's root score and the next-largest opposing tendency's
    root score (a measure of how decisively the equilibrium has
    resolved).
  - Apply the solver's stake delta.
  - Equilibrate again.
  - Compute the average GAP after vs before. Improvement = positive
    increase in average gap (more decisive resolution).

Score: improvement * 100, clipped to [0, 100]. Negative improvement
(degradation) returns 0.

The metric is hash-deterministic: any verifier replaying the same
seed and the same delta gets the same score. That's what the
contract layer needs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World, equilibrate

from .adapter import build_charter_world, deserialize_world
from .aggregate import apply_stake_delta


logger = logging.getLogger(__name__)


def _average_resolution_gap(world: World) -> float:
    """For each tendency, compute the gap between its root score and
    the maximum root score of OTHER tendencies. Take the absolute
    value (sign-agnostic; we want decisiveness, not direction).
    Average across tendencies.
    """
    scores = world.root_scores()
    if len(scores) < 2:
        return 0.0
    gaps: List[float] = []
    for tid, s in scores.items():
        other_max = max(v for k, v in scores.items() if k != tid)
        gaps.append(abs(s - other_max))
    return sum(gaps) / len(gaps)


def verify_world_model_solution(
    delta: Dict[str, Any],
    seed_world_payload: Optional[Dict[str, Any]] = None,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Replay the solver's contribution; return a score 0-100.

    Args:
      delta: the solver's stake delta (from train_world_model_on_task).
      seed_world_payload: the world the solver started from. If None,
        uses a fresh charter world.
      threshold: minimum score to consider valid (0-100).

    Returns dict with keys: score, gap_before, gap_after, valid.
    """
    if seed_world_payload is not None:
        before = deserialize_world(seed_world_payload)
        after = deserialize_world(seed_world_payload)
    else:
        before = build_charter_world()
        after = build_charter_world()

    equilibrate(before, max_rounds=8, tolerance=1e-3)
    gap_before = _average_resolution_gap(before)

    apply_stake_delta(after, delta)
    equilibrate(after, max_rounds=8, tolerance=1e-3)
    gap_after = _average_resolution_gap(after)

    improvement = gap_after - gap_before
    # Score: 100 * improvement, but normalized by the scale we expect
    # (~0.5 of root-score units = full score). Clip to [0, 100].
    score = max(0.0, min(100.0, 200.0 * improvement))

    result = {
        "score": score,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "improvement": improvement,
        "valid": score >= threshold,
    }
    logger.info(
        "verification: gap %.4f -> %.4f (Δ %+.4f), score %.1f, valid=%s",
        gap_before, gap_after, improvement, score, result["valid"],
    )
    return result
