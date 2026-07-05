#!/usr/bin/env python3
"""Test the mint gate: training-agent CON debate suppresses mint.

Scenario: a worker agent posts a contribution that mints normally.
A training agent then posts a CON sub-claim under one of the charter
tendencies, flagging the contribution as a violation. When that CON
sub-claim accumulates positive net_score, the gate suppresses the
worker's mint.
"""

from __future__ import annotations

import sys

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)
from world_model.models.tree import Position

from nodes.common.world_model_substrate import (
    DEFAULT_CHARTER_IDS,
    EpochSnapshots,
    apply_mint_gate,
    build_charter_world,
    charter_violation_score,
    reconcile_epoch,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> int:
    banner("MINT GATE: training-agent debate suppresses mint")

    # Build a charter world (this gives us the four root tendencies)
    world = build_charter_world()

    # Worker contribution: a node added under good_resolution / similar.
    # For this minimal test, manually sprout a sub-claim under one of
    # the charter tendencies pretending it represents worker output.
    intel = world.tendencies["promotion_of_intelligence"]
    worker_node = intel.sprout_child(
        parent_node_id=intel.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.0, 0.0, 0.8, 0.2),
        polarity_axis=(0.0, 0.0, 1.0, 0.0),
        content="worker resolved task X by approach Y",
    )
    worker_node.add_stake("worker-A", 1.0)

    # Phase 1: initial reconciliation, no gate
    snapshots = EpochSnapshots()
    snapshots.record_start(world)
    equilibrate(world, max_rounds=4, tolerance=1e-3)
    snapshots.record_close(world)

    pseudo_events = [{
        "kind": "sub_claim_sprouted",
        "seq": 1,
        "author_agent": "worker-A",
        "tendency_id": "promotion_of_intelligence",
        "parent_id": intel.tree.root_node.id,
        "node_id": worker_node.id,
        "position": "pro",
        "coords": [0.0, 0.0, 0.8, 0.2],
        "polarity_axis": [0.0, 0.0, 1.0, 0.0],
        "content": "worker resolved task X by approach Y",
    }]
    recon = reconcile_epoch(
        world=world,
        snapshots=snapshots,
        events=pseudo_events,
        agent_weights={"worker-A": 1.0},
    )
    mint_before_gate = recon["agent_mint"].get("worker-A", 0.0)
    print(f"\n  Phase 1 (no gate): worker-A mint = {mint_before_gate:.4f}")

    # Phase 2: training agent posts CON sub-claim flagging the worker
    # node as a violation. The CON sub-claim's content references the
    # worker_node.id, which is how the gate finds it.
    life = world.tendencies["life_precious"]
    flag_node = life.sprout_child(
        parent_node_id=life.tree.root_node.id,
        position=Position.CON,
        anchor=(0.5, 0.0, 0.0, 0.0),
        polarity_axis=(-1.0, 0.0, 0.0, 0.0),
        content=f"approach Y violates life_precious; refers to {worker_node.id}",
    )
    # Training agent stakes the flag heavily so it has positive
    # net_score (the network agrees with the violation flag).
    flag_node.add_stake("training-A", 1.5)
    equilibrate(world, max_rounds=4, tolerance=1e-3)

    violation = charter_violation_score(world, worker_node.id, DEFAULT_CHARTER_IDS)
    print(f"\n  Phase 2: charter_violation_score = {violation:.3f}")

    # Re-reconcile + apply gate
    snapshots2 = EpochSnapshots()
    snapshots2.record_start(world)
    snapshots2.record_close(world)
    recon2 = reconcile_epoch(
        world=world,
        snapshots=snapshots2,
        events=pseudo_events,
        agent_weights={"worker-A": 1.0},
    )
    # We need scores to have moved during the epoch for there to be
    # any mint to gate. For this test, fall back to the original recon
    # if the new one has no mint (the test is about the gate, not the
    # epoch reconciliation).
    if recon2["total_mint"] == 0:
        recon2 = recon
    apply_mint_gate(recon2, world, gate_strength=1.0)
    mint_after_gate = recon2["agent_mint"].get("worker-A", 0.0)
    print(f"\n  Phase 3 (gate applied): worker-A mint = {mint_after_gate:.4f}")

    # Verdict
    banner("VERDICT")
    success = True

    # 1. Initial mint > 0
    if mint_before_gate > 0:
        print(f"\n  OK: initial mint > 0 ({mint_before_gate:.4f})")
    else:
        print(f"\n  -- initial mint was 0; can't test gate")
        success = False

    # 2. Violation score > 0
    if violation > 0:
        print(f"  OK: violation score > 0 ({violation:.3f})")
    else:
        print(f"  -- violation score is 0; gate detection didn't fire")
        success = False

    # 3. Gated mint < initial mint
    if success and mint_after_gate < mint_before_gate:
        print(f"  OK: gate reduced mint ({mint_before_gate:.4f} -> {mint_after_gate:.4f})")
    elif success:
        print(f"  -- gate didn't reduce mint")
        success = False

    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
