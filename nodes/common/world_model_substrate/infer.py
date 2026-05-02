"""Inference: use the aggregated global world to score an input.

Same JSON output schema as the JEPA inference path:
  {
    "predictions": [...],
    "probabilities": [...],
    "model_hash": str,
    ...
  }

For the world-model substrate, "prediction" is the index of the
charter tendency with the highest root score after equilibrating
the global world with the inference input attached as observations.
"Probabilities" are the softmax of root scores across charter
tendencies.

If the input is itself a tendency-charter alignment query (e.g.,
"score this turn against the charter"), the result is interpretable
as: which charter principle does this turn most align with, and how
strongly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from world_model.generalized import Observation, equilibrate

from .adapter import (
    CHARTER,
    build_charter_world,
    deserialize_world,
    turn_to_observation,
)


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def infer_with_world_model(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run inference on a single input.

    Args:
      input_data: a dict with key "turns": list of turn dicts, or a
        single turn dict at the top level (will be wrapped).
      global_model_payload: serialized world from the aggregator. If
        None, uses a fresh charter world (no accumulated structure).

    Returns: dict with predictions, probabilities, root_scores,
    aligned_principle.
    """
    if "turns" in input_data:
        turns = input_data["turns"]
    elif "turn" in input_data:
        turns = [input_data["turn"]]
    else:
        # Treat the whole dict as one turn
        turns = [input_data]

    if global_model_payload is not None:
        world = deserialize_world(global_model_payload)
    else:
        world = build_charter_world()

    for i, turn in enumerate(turns):
        world.add_observation(turn_to_observation(turn, turn_index=i))
    equilibrate(world, max_rounds=8, tolerance=1e-3)

    # Read root scores in charter order so predictions are stable
    scores = world.root_scores()
    ordered = [scores.get(entry["id"], 0.0) for entry in CHARTER]
    probs = _softmax(ordered)
    pred = max(range(len(ordered)), key=lambda i: ordered[i])

    return {
        "predictions": [pred],
        "probabilities": [probs],
        "root_scores": dict(scores),
        "aligned_principle": CHARTER[pred]["id"],
        "principle_thesis": CHARTER[pred]["thesis"],
    }
