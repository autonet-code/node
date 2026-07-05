#!/usr/bin/env python3
"""Multi-solver convergence: two solvers, same content, one node.

Tests that content-addressed sub-claim ids dedupe naturally when two
solvers independently propose the same claim. Without convergence, the
aggregator would end up with parallel sub-claims for the same content;
with convergence, both solvers' events resolve to the same node.

Setup
-----

Two solvers train on overlapping turn streams. Their non-overlapping
turns produce distinct sub-claims; their overlapping turns must
produce the *same* sub-claim by content-addressed id.

Verifies
--------

  1. Each solver produces events independently.
  2. Aggregator merges the event streams.
  3. After replay, the global world has fewer total nodes than the
     naive sum of solver-side nodes -- because the overlapping turn
     consolidates.
  4. Mint is attributed to the agents who actually emitted events
     for each contribution.
"""

from __future__ import annotations

import sys

from nodes.common.world_model_substrate import (
    train_world_model_on_task,
    aggregate_contributions,
    apply_events,
    build_charter_world,
    EpochSnapshots,
    reconcile_epoch,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


# Shared turn (both solvers see this)
SHARED_TURN = {
    "label": "wrote a comprehensive ADR",
    "intelligence_impact": 0.8, "evolution_impact": 0.6,
}

SOLVER_A_TURNS = [
    SHARED_TURN,
    {"label": "fixed a regression",
     "intelligence_impact": 0.3, "evolution_impact": 0.5},
]

SOLVER_B_TURNS = [
    SHARED_TURN,
    {"label": "added integration tests",
     "intelligence_impact": 0.5, "evolution_impact": 0.7},
]


def main() -> int:
    banner("MULTI-SOLVER CONVERGENCE")
    print("\n  Two solvers, one shared turn. The shared turn must produce")
    print("  the SAME sub-claim id from both solvers (content-addressed).\n")

    # Train each solver
    contrib_a, _ = train_world_model_on_task(
        task_spec={"turns": SOLVER_A_TURNS}, epochs=1,
        agent_id="solver-A",
    )
    contrib_b, _ = train_world_model_on_task(
        task_spec={"turns": SOLVER_B_TURNS}, epochs=1,
        agent_id="solver-B",
    )

    a_sprout_ids = {
        e["node_id"] for e in contrib_a["events"]
        if e["kind"] == "sub_claim_sprouted"
    }
    b_sprout_ids = {
        e["node_id"] for e in contrib_b["events"]
        if e["kind"] == "sub_claim_sprouted"
    }
    shared_ids = a_sprout_ids & b_sprout_ids
    print(f"  solver-A sprout ids:      {len(a_sprout_ids)}")
    print(f"  solver-B sprout ids:      {len(b_sprout_ids)}")
    print(f"  shared (content-matched): {len(shared_ids)}")

    # Aggregator merges
    merged = aggregate_contributions([contrib_a, contrib_b], weights=[1.0, 1.0])
    print(f"\n  merged events:            {len(merged['events'])}")

    # Replay onto a fresh world
    world = build_charter_world()
    snapshots = EpochSnapshots()
    snapshots.record_start(world)
    apply_events(world, merged["events"])
    snapshots.record_close(world)

    # Count nodes after replay
    total_nodes_after = sum(
        len(t.tree.all_nodes()) for t in world.tendencies.values()
    )
    print(f"  global world nodes after: {total_nodes_after}")

    # Sum of nodes-each-solver-would-have-produced-alone, ignoring roots
    naive_total = (len(a_sprout_ids) + len(b_sprout_ids)) + 4  # 4 root tendencies
    print(f"  naive sum if no convergence: {naive_total}")

    # Reconcile
    recon = reconcile_epoch(
        world=world, snapshots=snapshots,
        events=merged["events"],
        agent_weights=merged["agent_weights"],
    )

    print(f"\n  per-agent mint:")
    for agent, mint in sorted(recon["agent_mint"].items()):
        print(f"    {agent:20s} {mint:+.4f}")

    # Verdict
    banner("VERDICT")
    success = True
    if len(shared_ids) > 0:
        print(f"\n  OK: {len(shared_ids)} shared sub-claim ids "
              f"(content-addressed convergence)")
    else:
        print(f"\n  -- no shared ids; convergence isn't happening")
        success = False
    if total_nodes_after < naive_total:
        print(f"  OK: global world has {total_nodes_after} nodes vs naive {naive_total} "
              f"-- {naive_total - total_nodes_after} nodes deduplicated")
    else:
        print(f"  -- no deduplication observed")
        success = False

    a_mint = recon["agent_mint"].get("solver-A", 0.0)
    b_mint = recon["agent_mint"].get("solver-B", 0.0)
    if a_mint > 0 and b_mint > 0:
        print(f"  OK: both solvers minted (A: {a_mint:.4f}, B: {b_mint:.4f})")
    else:
        print(f"  -- mint attribution: A={a_mint}, B={b_mint}")
        success = False

    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
