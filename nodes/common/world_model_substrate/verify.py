"""Solution verification: did replaying the solver's events improve
the equilibrium?

Same protocol contract as ``verify_jepa_solution``: returns a numeric
score in [0, 100]. Above some threshold = valid contribution.

Metric
------

  - Take the seed world (or load the pre-solver global model).
  - Equilibrate. Compute average resolution gap.
  - Replay the contribution's events. Equilibrate.
  - Compute new average resolution gap.
  - Score: 200 * (gap_after - gap_before), clipped to [0, 100].

Hash-deterministic: same seed + same events => same score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World, equilibrate

from .adapter import build_charter_world, deserialize_world
from .aggregate import apply_events


logger = logging.getLogger(__name__)


def _average_resolution_gap(world: World) -> float:
    scores = world.root_scores()
    if len(scores) < 2:
        return 0.0
    gaps: List[float] = []
    for tid, s in scores.items():
        other_max = max(v for k, v in scores.items() if k != tid)
        gaps.append(abs(s - other_max))
    return sum(gaps) / len(gaps)


def verify_world_model_solution(
    contribution: Dict[str, Any],
    seed_world_payload: Optional[Dict[str, Any]] = None,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    if seed_world_payload is not None:
        before = deserialize_world(seed_world_payload)
        after = deserialize_world(seed_world_payload)
    else:
        before = build_charter_world()
        after = build_charter_world()

    equilibrate(before, max_rounds=8, tolerance=1e-3)
    gap_before = _average_resolution_gap(before)

    apply_events(after, contribution.get("events", []))
    gap_after = _average_resolution_gap(after)

    improvement = gap_after - gap_before
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
