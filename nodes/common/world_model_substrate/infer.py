"""Inference: query the global world for the region relevant to a
request, return its structure.

Two query modes:

  1. Charter-alignment query (default for turn-shaped inputs):
     evaluate the input against the four charter tendencies, return
     which root principle dominates and the softmax of root scores.
     This is the original alignment-scoring path.

  2. General query (any other input):
     locate the query in the graph, render the located region's
     structure, return the rendered output. The decoder (LLM at the
     I/O boundary, or the calling pipeline) consumes this structure
     to produce its final answer in whatever format it needs.

The two paths share the same backing graph and the same locate
primitive. Charter-alignment is just locate restricted to the four
root tendencies plus a softmax readout.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from world_model.generalized import (
    Observation,
    default_locator,
    equilibrate,
    render,
)

from .adapter import (
    CHARTER,
    build_charter_world,
    deserialize_world,
    turn_to_observation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def _looks_like_turn(input_data: Dict[str, Any]) -> bool:
    """Heuristic: is this input a charter-alignment query (i.e., does
    it look like an agent turn we'd score against the four roots)?

    Markers: presence of 'turn', 'turns', or any of the four
    *_impact fields anywhere in the dict.
    """
    if not isinstance(input_data, dict):
        return False
    if "turn" in input_data or "turns" in input_data:
        return True
    impact_keys = {
        "life_impact", "self_pres_impact",
        "intelligence_impact", "evolution_impact",
    }
    if any(k in input_data for k in impact_keys):
        return True
    # Inner inspection: 'tool_calls', 'tool_call', 'tool', 'command'
    # are turn-shaped. 'query', 'question', 'request' are general.
    if any(k in input_data for k in ("tool_calls", "tool_call", "tool", "command")):
        return True
    return False


# ---------------------------------------------------------------------------
# Charter-alignment inference (original path)
# ---------------------------------------------------------------------------


def _infer_alignment(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if "turns" in input_data:
        turns = input_data["turns"]
    elif "turn" in input_data:
        turns = [input_data["turn"]]
    else:
        turns = [input_data]

    if global_model_payload is not None:
        world = deserialize_world(global_model_payload)
    else:
        world = build_charter_world()

    for i, turn in enumerate(turns):
        world.add_observation(turn_to_observation(turn, turn_index=i))
    equilibrate(world, max_rounds=8, tolerance=1e-3)

    scores = world.root_scores()
    ordered = [scores.get(entry["id"], 0.0) for entry in CHARTER]
    probs = _softmax(ordered)
    pred = max(range(len(ordered)), key=lambda i: ordered[i])

    return {
        "mode": "alignment",
        "predictions": [pred],
        "probabilities": [probs],
        "root_scores": dict(scores),
        "aligned_principle": CHARTER[pred]["id"],
        "principle_thesis": CHARTER[pred]["thesis"],
    }


# ---------------------------------------------------------------------------
# General inference (locate + render)
# ---------------------------------------------------------------------------


def _infer_general(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if global_model_payload is not None:
        world = deserialize_world(global_model_payload)
    else:
        world = build_charter_world()

    # Equilibrate once so existing structure has settled scores
    equilibrate(world, max_rounds=4, tolerance=1e-3)

    locator = default_locator()
    region = locator(world, input_data)
    structure = render(world, region, descendants_per_node=3)

    return {
        "mode": "general",
        "structure": structure,
        # Compatibility fields so downstream callers expecting the
        # alignment shape don't crash. They're empty/default for
        # general queries.
        "predictions": [],
        "probabilities": [],
        "root_scores": world.root_scores(),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def infer_with_world_model(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference. Two modes auto-selected unless overridden:

      mode='alignment' or input looks turn-shaped:
        Score against the four charter tendencies. Return the
        dominant principle and softmax probabilities. Original path.

      mode='general' or input is anything else:
        locate(input) -> render(region). Return the rendered region's
        structure for downstream rendering.
    """
    if mode is None:
        mode = "alignment" if _looks_like_turn(input_data) else "general"

    if mode == "alignment":
        return _infer_alignment(input_data, global_model_payload)
    elif mode == "general":
        return _infer_general(input_data, global_model_payload)
    else:
        raise ValueError(f"unknown inference mode: {mode!r}")
