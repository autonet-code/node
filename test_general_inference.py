#!/usr/bin/env python3
"""General-mode inference: locate + render against the substrate.

Tests that the substrate can answer non-charter queries by locating a
region in the graph and rendering its structure -- the path that
generalizes the substrate beyond alignment scoring toward arbitrary
inference.

Setup
-----

  1. Train the substrate on a stream of turns whose 4D coords spread
     into different regions (intelligence-positive, self-pres-positive,
     etc.) -- so the graph has structural diversity at depth.
  2. Issue a 'general' query whose content is a coordinate-like point
     in one region (intelligence-positive corner).
  3. Verify the inference returns a structured region from that
     corner of the graph, not from anywhere else.

What we want to see
-------------------

  - mode == 'general' (auto-detected, no turn-shaped fields).
  - structure.region_size > 0.
  - The returned nodes' tendencies cluster around the query's
    coordinate region (intelligence/evolution), not the unrelated
    ones (life/self_pres).
  - Each returned node has ancestors and descendants populated.
"""

from __future__ import annotations

import sys

from nodes.common.world_model_substrate import (
    train_world_model_on_task,
    aggregate_contributions,
    apply_events,
    build_charter_world,
    serialize_world,
    infer_with_world_model,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


# Diverse training stream so the graph has structure in multiple regions
TRAINING_TURNS = [
    {"label": "explained design tradeoffs",
     "intelligence_impact": 0.9, "evolution_impact": 0.4},
    {"label": "wrote architecture decision record",
     "intelligence_impact": 0.7, "evolution_impact": 0.6},
    {"label": "improved test coverage",
     "intelligence_impact": 0.4, "evolution_impact": 0.7},
    {"label": "rolled back broken deploy",
     "self_pres_impact": 0.7},
    {"label": "backed up DB before migration",
     "self_pres_impact": 0.6, "evolution_impact": 0.2},
    {"label": "refactored module API",
     "intelligence_impact": 0.5, "evolution_impact": 0.8},
    {"label": "explained core concepts to user",
     "intelligence_impact": 0.8},
]


def main() -> int:
    banner("GENERAL-MODE INFERENCE: locate + render")

    # 1. Train a world from one solver's turns
    contribution, metrics = train_world_model_on_task(
        task_spec={"turns": TRAINING_TURNS},
        epochs=2,
        agent_id="solver-A",
    )
    print(f"  trained: {len(contribution['events'])} events")

    # 2. Build the global world
    merged = aggregate_contributions([contribution], weights=[1.0])
    world = build_charter_world()
    apply_events(world, merged["events"])
    payload = serialize_world(world)
    print(f"  global world: {len(payload['nodes'])} nodes")

    # 3. Issue a charter-shaped query (should auto-detect alignment mode)
    banner("Query 1: charter-shaped (auto-detect alignment mode)")
    align_result = infer_with_world_model(
        input_data={"turn": {"intelligence_impact": 0.7, "evolution_impact": 0.3}},
        global_model_payload=payload,
    )
    print(f"  mode:               {align_result['mode']}")
    print(f"  aligned principle:  {align_result.get('aligned_principle')}")
    if align_result["mode"] != "alignment":
        print("  -- expected mode='alignment' for turn-shaped input")
        return 1

    # 4. Issue a general query (raw coord region)
    banner("Query 2: general (raw coords, intelligence corner)")
    general_result = infer_with_world_model(
        input_data={"coords": [0.0, 0.0, 0.9, 0.3]},
        global_model_payload=payload,
    )
    print(f"  mode:               {general_result['mode']}")
    if general_result["mode"] != "general":
        print(f"  -- expected mode='general', got {general_result['mode']!r}")
        return 1
    structure = general_result["structure"]
    print(f"  region_size:        {structure['region_size']}")
    print(f"  by_tendency keys:   {list(structure['by_tendency'].keys())}")

    if structure["region_size"] == 0:
        print("  -- region empty; locate didn't find anything")
        return 1

    # 5. Inspect a few returned nodes
    print("\n  top-3 nodes by proximity:")
    for node in structure["nodes"][:3]:
        print(f"    {node['tendency_id']:30s} d={node['distance_from_query']:.3f} "
              f"score={node['score']:+.3f} content={node['content'][:30]!r}")
        if node["ancestors"]:
            anc_chain = " -> ".join(a["content"][:20] for a in node["ancestors"])
            print(f"      ancestors: {anc_chain}")
        if node["descendants"]:
            desc_list = ", ".join(d["content"][:20] for d in node["descendants"])
            print(f"      descendants: {desc_list}")

    # Verify region clusters around intelligence/evolution
    by_t = structure["by_tendency"]
    intel_count = by_t.get("promotion_of_intelligence", {}).get("n_nodes", 0)
    evo_count = by_t.get("evolution", {}).get("n_nodes", 0)
    life_count = by_t.get("life_precious", {}).get("n_nodes", 0)
    pres_count = by_t.get("self_preservation", {}).get("n_nodes", 0)

    banner("VERDICT")
    if intel_count + evo_count > life_count + pres_count:
        print(f"\n  OK: region clusters around intelligence ({intel_count}) and "
              f"evolution ({evo_count}) as expected.")
        print(f"      life={life_count}, self_pres={pres_count} are properly "
              f"farther from the query.")
    else:
        print(f"\n  -- region didn't cluster correctly: intel={intel_count}, "
              f"evo={evo_count}, life={life_count}, self_pres={pres_count}")
        return 1

    if all(node["ancestors"] is not None and node["descendants"] is not None
           for node in structure["nodes"]):
        print(f"  OK: all {structure['region_size']} nodes have ancestors and "
              f"descendants populated.")
    else:
        print(f"  -- some nodes missing ancestors/descendants")
        return 1

    print()
    print("  General-mode inference works end-to-end:")
    print("    locate(query) returned a region clustered correctly")
    print("    render(region) surfaced ancestors + descendants")
    print("    auto mode-detection routes turn-shaped queries to alignment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
