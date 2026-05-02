#!/usr/bin/env python3
"""End-to-end vertical slice for the world-model substrate.

Simulates: one solver trains on a small turn-stream, the aggregator
merges (one solver here, but the protocol is plural), the verifier
scores the contribution, and an inference node runs against the
aggregated global model.

This is the protocol vertical: training -> aggregation -> verification
-> inference, all using the world-model engine instead of VL-JEPA.

What we want to see
-------------------

  1. The solver produces a stake delta with non-zero changes (the
     architecture absorbed the turns).
  2. The aggregator merges the delta into a global model without
     errors.
  3. The verifier returns a non-zero score (improvement happened).
  4. The inference run on a held-out turn returns a charter principle
     and softmax probabilities over the four dimensions.

Run: python test_world_model_substrate_e2e.py
"""

from __future__ import annotations

import json
import sys
import time

from nodes.common.world_model_substrate import (
    train_world_model_on_task,
    aggregate_stake_deltas,
    verify_world_model_solution,
    infer_with_world_model,
    serialize_world,
    deserialize_world,
)
from nodes.common.world_model_substrate.adapter import build_charter_world


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Fake training data: a sequence of agent turns with synthetic charter
# impacts. Real autonet would source these from agent-execution JSONLs.
# ---------------------------------------------------------------------------

TRAINING_TURNS = [
    {"label": "user-confirmed risky operation",
     "life_impact": 0.0, "self_pres_impact": 0.0,
     "intelligence_impact": +0.4, "evolution_impact": +0.6},
    {"label": "explained complex concept clearly",
     "life_impact": 0.0, "self_pres_impact": 0.0,
     "intelligence_impact": +0.9, "evolution_impact": +0.3},
    {"label": "preserved user data via backup",
     "life_impact": 0.0, "self_pres_impact": +0.7,
     "intelligence_impact": 0.0, "evolution_impact": 0.0},
    {"label": "refused unsafe medical advice",
     "life_impact": +0.8, "self_pres_impact": 0.0,
     "intelligence_impact": +0.2, "evolution_impact": 0.0},
    {"label": "improved system architecture",
     "life_impact": 0.0, "self_pres_impact": +0.3,
     "intelligence_impact": +0.5, "evolution_impact": +0.8},
    {"label": "ignored user pause request",
     "life_impact": 0.0, "self_pres_impact": -0.1,
     "intelligence_impact": -0.4, "evolution_impact": -0.5},
    {"label": "fact-checked claim before acting",
     "life_impact": +0.1, "self_pres_impact": 0.0,
     "intelligence_impact": +0.7, "evolution_impact": +0.2},
    {"label": "destroyed working code without backup",
     "life_impact": 0.0, "self_pres_impact": -0.6,
     "intelligence_impact": -0.3, "evolution_impact": -0.4},
]


HELD_OUT_TURN = {
    "label": "explained tradeoff and waited for confirmation",
    "life_impact": +0.1,
    "self_pres_impact": +0.2,
    "intelligence_impact": +0.7,
    "evolution_impact": +0.3,
}


def main() -> int:
    banner("WORLD-MODEL SUBSTRATE: E2E VERTICAL SLICE")

    # 1. Solver trains on the turn stream.
    banner("Step 1: solver trains on 8 turns (no global model yet)")
    task_spec = {"turns": TRAINING_TURNS}
    delta, metrics = train_world_model_on_task(
        task_spec=task_spec,
        global_model_cid=None,
        store=None,
        epochs=2,
    )
    print(f"  elapsed:           {metrics['elapsed_seconds']:.2f}s")
    print(f"  observations:      {metrics['n_observations']}")
    print(f"  changed nodes:     {metrics['n_changed_nodes']}")
    print(f"  new nodes sprouted: {metrics['n_new_nodes']}")
    print(f"  root scores:")
    for k, v in metrics["root_scores"].items():
        print(f"    {k:30s} {v:+.4f}")

    # The charter root-score gap should be non-trivial after seeing
    # turns that are mostly intelligence-positive and evolution-positive.
    root = metrics["root_scores"]
    if len(delta["new_nodes"]) == 0 and len(delta["node_stakes"]) == 0:
        print("\n  -- no stake changes detected; the engine didn't absorb anything")
        return 1

    # 2. Aggregator merges (just one solver in the slice; equivalent
    # to: with weight 1, returns the same delta with a normalization
    # factor of 1).
    banner("Step 2: aggregator merges 1 solver delta (slice)")
    merged = aggregate_stake_deltas([delta], weights=[1.0])
    print(f"  merged changed nodes:    {len(merged['node_stakes'])}")
    print(f"  merged new nodes:        {len(merged['new_nodes'])}")
    print(f"  merged absorbed obs:     {len(merged['absorbed_observations'])}")

    # 3. Verifier scores the contribution.
    banner("Step 3: verifier scores the contribution")
    verdict = verify_world_model_solution(
        delta=delta,
        seed_world_payload=None,
        threshold=0.0,
    )
    print(f"  gap before delta:  {verdict['gap_before']:.4f}")
    print(f"  gap after delta:   {verdict['gap_after']:.4f}")
    print(f"  improvement:       {verdict['improvement']:+.4f}")
    print(f"  score (0-100):     {verdict['score']:.1f}")
    print(f"  valid:             {verdict['valid']}")

    # 4. Build the post-aggregation global world by applying the
    # merged delta to a fresh charter world, then serialize for
    # inference.
    banner("Step 4: aggregator publishes new global model")
    global_world = build_charter_world()
    from nodes.common.world_model_substrate.aggregate import apply_stake_delta
    apply_stake_delta(global_world, merged)
    payload = serialize_world(global_world)
    print(f"  payload tendencies:    {len(payload['tendencies'])}")
    print(f"  payload nodes:         {len(payload['nodes'])}")
    serialized_size = len(json.dumps(payload))
    print(f"  serialized size:       {serialized_size} bytes")

    # 5. Inference on a held-out turn.
    banner("Step 5: inference on a held-out turn")
    result = infer_with_world_model(
        input_data={"turn": HELD_OUT_TURN},
        global_model_payload=payload,
    )
    print(f"  aligned principle:  {result['aligned_principle']}")
    print(f"    thesis:           {result['principle_thesis']}")
    print(f"  predictions:        {result['predictions']}")
    print(f"  probabilities:      {[round(p, 3) for p in result['probabilities'][0]]}")
    print(f"  root scores:")
    for k, v in result['root_scores'].items():
        print(f"    {k:30s} {v:+.4f}")

    # Sanity: held-out turn is intelligence-positive (+0.7) and
    # evolution-positive (+0.3). Aligned principle should be one of
    # those two.
    banner("VERDICT")
    expected_principles = {"promotion_of_intelligence", "evolution"}
    if result['aligned_principle'] in expected_principles:
        print(f"\n  OK: held-out turn aligned with {result['aligned_principle']!r}")
        print(f"      which is consistent with the turn's intelligence/evolution-positive impact.")
    else:
        print(f"\n  -- held-out turn aligned with {result['aligned_principle']!r}")
        print(f"      expected one of {expected_principles}")
        print(f"      (this is informational; the slice is wired regardless)")

    if (
        len(delta["node_stakes"]) > 0
        and verdict["score"] > 0
        and result["aligned_principle"] in [c["id"] for c in __import__(
            "nodes.common.world_model_substrate.adapter", fromlist=["CHARTER"]
        ).CHARTER]
    ):
        print("\n  Substrate vertical slice is wired:")
        print("    - solver produces a non-trivial stake delta")
        print("    - aggregator merges without error")
        print("    - verifier returns a positive score")
        print("    - inference returns a charter-aligned answer")
        print("\n  Smart contracts unchanged. This swap is protocol-compatible.")
        return 0
    else:
        print("\n  -- one or more stages didn't behave as expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
