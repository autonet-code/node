#!/usr/bin/env python3
"""End-to-end epoch reconciliation: events -> world -> snapshots -> mint.

Two solvers train on different turn streams. The aggregator merges
their event streams. The reconciler computes per-agent novelty (a
descriptive measure of surprise) and per-agent mint (the rewarded
subset, paid only for positive movement landing positive).

What we want to see
-------------------

  1. Both solvers produce events.
  2. Aggregator merges deterministically.
  3. Snapshots capture epoch-start (empty) and epoch-close (post-merge).
  4. Reconciler returns:
     - non-zero per-agent mint for the solver whose events drove
       upward score movement landing positive
     - zero or near-zero mint for events that didn't survive or
       landed negative
     - per-agent novelty that includes all surprise, not just the
       rewarded subset
  5. Total mint reasonably tracks total positive score movement.
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


# Solver A: turns mostly intelligence- and evolution-positive
SOLVER_A_TURNS = [
    {"label": "explained API design",
     "intelligence_impact": 0.8, "evolution_impact": 0.5},
    {"label": "improved test coverage",
     "intelligence_impact": 0.4, "evolution_impact": 0.7},
    {"label": "wrote architectural ADR",
     "intelligence_impact": 0.7, "evolution_impact": 0.6},
    {"label": "refactored module for clarity",
     "intelligence_impact": 0.6, "evolution_impact": 0.5},
]

# Solver B: turns mostly self-preservation-positive
SOLVER_B_TURNS = [
    {"label": "backed up DB before migration", "self_pres_impact": 0.7},
    {"label": "rejected --force flag", "self_pres_impact": 0.6},
    {"label": "rolled back broken deploy", "self_pres_impact": 0.5},
]

# Solver C: turns that should NOT mint (life-negative)
SOLVER_C_TURNS = [
    {"label": "ignored safety check", "self_pres_impact": -0.6},
    {"label": "destroyed code without backup", "self_pres_impact": -0.8},
]


def main() -> int:
    banner("EPOCH RECONCILIATION: events -> world -> mint")

    # ---- Phase 1: each solver trains independently ----
    banner("Phase 1: three solvers train independently")
    contrib_a, _ = train_world_model_on_task(
        task_spec={"turns": SOLVER_A_TURNS}, epochs=2,
        agent_id="solver-A",
    )
    print(f"  solver-A: {len(contrib_a['events'])} events")
    contrib_b, _ = train_world_model_on_task(
        task_spec={"turns": SOLVER_B_TURNS}, epochs=2,
        agent_id="solver-B",
    )
    print(f"  solver-B: {len(contrib_b['events'])} events")
    contrib_c, _ = train_world_model_on_task(
        task_spec={"turns": SOLVER_C_TURNS}, epochs=2,
        agent_id="solver-C",
    )
    print(f"  solver-C: {len(contrib_c['events'])} events (negative-impact)")

    # ---- Phase 2: aggregator merges ----
    banner("Phase 2: aggregator merges contributions")
    merged = aggregate_contributions(
        [contrib_a, contrib_b, contrib_c],
        weights=[1.0, 1.0, 1.0],
    )
    print(f"  merged events:  {len(merged['events'])}")
    print(f"  agent weights:  {merged['agent_weights']}")

    # ---- Phase 3: epoch snapshot at start (fresh world), then replay ----
    banner("Phase 3: epoch snapshots at start and close")
    world = build_charter_world()
    snapshots = EpochSnapshots()
    snapshots.record_start(world)
    print(f"  start scores: {sum(snapshots.start.values()):+.3f} (sum across roots)")

    apply_events(world, merged["events"])
    # No intermediate checkpoint -- single open/close pair
    snapshots.record_close(world)
    print(f"  close scores:")
    for root_id, s in world.root_scores().items():
        print(f"    {root_id:30s} {s:+.4f}")

    # ---- Phase 4: reconcile ----
    banner("Phase 4: reconcile per-agent mint and novelty")
    result = reconcile_epoch(
        world=world,
        snapshots=snapshots,
        events=merged["events"],
        agent_weights=merged["agent_weights"],
    )

    print(f"\n  total novelty: {result['total_novelty']:.4f}")
    print(f"  total mint:    {result['total_mint']:.4f}")
    print(f"\n  per-agent novelty (descriptive):")
    for agent, nov in sorted(result["agent_novelty"].items()):
        print(f"    {agent:20s} {nov:+.4f}")
    print(f"\n  per-agent mint (rewarded):")
    for agent, mint in sorted(result["agent_mint"].items()):
        print(f"    {agent:20s} {mint:+.4f}")

    # ---- Verdict ----
    banner("VERDICT")
    a_mint = result["agent_mint"].get("solver-A", 0.0)
    b_mint = result["agent_mint"].get("solver-B", 0.0)
    c_mint = result["agent_mint"].get("solver-C", 0.0)
    a_nov = result["agent_novelty"].get("solver-A", 0.0)
    b_nov = result["agent_novelty"].get("solver-B", 0.0)
    c_nov = result["agent_novelty"].get("solver-C", 0.0)

    success = True
    print()
    if a_mint > 0:
        print(f"  OK: solver-A minted {a_mint:.4f} (intelligence/evolution-positive)")
    else:
        print(f"  -- solver-A minted nothing despite positive turns")
        success = False
    if b_mint > 0:
        print(f"  OK: solver-B minted {b_mint:.4f} (self-preservation-positive)")
    else:
        print(f"  -- solver-B minted nothing despite positive turns")
        success = False
    if c_mint <= 0.001:
        print(f"  OK: solver-C minted {c_mint:.4f} (negative turns -> negligible mint)")
    else:
        print(f"  -- solver-C minted {c_mint:.4f}; expected ~0 for negative turns")
        success = False

    # Novelty should be present even where mint isn't (descriptive vs rewarded)
    if c_nov > 0:
        print(f"  OK: solver-C generated novelty ({c_nov:.4f}) without minting")
        print(f"      (novelty is descriptive of surprise; mint requires endorsed direction)")
    else:
        print(f"  -- solver-C produced no novelty either; nothing was surprising")

    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
