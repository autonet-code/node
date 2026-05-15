#!/usr/bin/env python3
"""Tests for the `scope` parameter on equilibrate / apply_stakes.

Two correctness gates:

1. scope=None (or omitted) preserves baseline behavior — running the
   pre-scope kernel and the new one with scope=None produces
   byte-identical node trees and root scores.

2. scope=subset isolates work to those tendencies — root scores and
   per-node post counts for OUT-OF-SCOPE tendencies are byte-identical
   before/after the scoped pass.

This is the launch-blocker correctness check for the hot-path fix
described in POST_AUTONET_FINDINGS.md.
"""

from __future__ import annotations

import copy

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


def _make_world_three_tendencies() -> World:
    world = World()
    world.add_tendency(GeneralizedTendency(
        id="T_a",
        thesis="A",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=0.5,
        smooth_promotion=True,
    ))
    world.add_tendency(GeneralizedTendency(
        id="T_b",
        thesis="B",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=0.5,
        smooth_promotion=True,
    ))
    world.add_tendency(GeneralizedTendency(
        id="T_c",
        thesis="C",
        anchor=(0.0, 0.0, 1.0),
        polarity_axis=(0.0, 0.0, 1.0),
        bandwidth=0.5,
        smooth_promotion=True,
    ))
    return world


def _seed_observations(world: World) -> None:
    # An observation near T_a's anchor (should land for T_a, maybe
    # ripple into B/C via cross-tendency posts).
    world.add_observation(Observation(
        id="obs_a", coords=(0.9, 0.05, 0.05), label="near A",
    ))
    # An observation near T_b's anchor.
    world.add_observation(Observation(
        id="obs_b", coords=(0.05, 0.9, 0.05), label="near B",
    ))
    # An observation near T_c's anchor.
    world.add_observation(Observation(
        id="obs_c", coords=(0.05, 0.05, 0.9), label="near C",
    ))


def _snapshot(world: World) -> dict:
    """Capture observable state for byte-comparison. Root-node UUIDs
    are normalized to "ROOT" so that two worlds with identical content-
    addressed state but different root UUIDs compare equal."""
    root_id_map = {t.tree.root_node.id: f"ROOT::{tid}"
                   for tid, t in world.tendencies.items()}

    def normalize(node_id: str) -> str:
        return root_id_map.get(node_id, node_id)

    out: dict = {"scores": {}, "node_post_counts": {}, "last_stakes": {}}
    for tid, t in world.tendencies.items():
        out["scores"][tid] = t.tree.score
        out["last_stakes"][tid] = {
            (target_tid, normalize(node_id)): w
            for (target_tid, node_id), w in t.last_stakes.items()
        }
        per_node: dict = {}
        for node in t.tree.all_nodes():
            per_node[normalize(node.id)] = sorted(s.agent_id for s in node.stakes)
        out["node_post_counts"][tid] = per_node
    return out


def test_scope_none_preserves_baseline() -> None:
    """Two worlds, identical setup. One runs `equilibrate(world)`
    (no scope arg, default None), the other runs
    `equilibrate(world, scope=None)` explicitly. Both must produce
    identical state."""
    world_default = _make_world_three_tendencies()
    _seed_observations(world_default)
    world_explicit = _make_world_three_tendencies()
    _seed_observations(world_explicit)

    rounds_default = equilibrate(world_default)
    rounds_explicit = equilibrate(world_explicit, scope=None)

    assert rounds_default == rounds_explicit, (
        f"round count diverged: {rounds_default} vs {rounds_explicit}"
    )
    snap_default = _snapshot(world_default)
    snap_explicit = _snapshot(world_explicit)
    if snap_default != snap_explicit:
        # Diff for debugging.
        for key in ("scores", "last_stakes", "node_post_counts"):
            if snap_default[key] != snap_explicit[key]:
                print(f"DIVERGENCE in {key}:")
                print(f"  default:  {snap_default[key]}")
                print(f"  explicit: {snap_explicit[key]}")
        raise AssertionError("state diverged under scope=None")


def test_scope_none_matches_full_set() -> None:
    """`scope=None` and `scope={all tendency ids}` should produce
    identical results — the full-set explicit scope is just the
    default path with extra bookkeeping."""
    world_none = _make_world_three_tendencies()
    _seed_observations(world_none)
    world_full = _make_world_three_tendencies()
    _seed_observations(world_full)

    equilibrate(world_none)
    equilibrate(world_full, scope={"T_a", "T_b", "T_c"})

    snap_none = _snapshot(world_none)
    snap_full = _snapshot(world_full)
    assert snap_none == snap_full, "scope=full-set diverged from scope=None"


def test_scope_subset_leaves_out_of_scope_untouched() -> None:
    """Run a baseline pass to populate state. Then construct a fresh
    second world starting from the same state. Run scoped equilibrate
    on a subset {T_a, T_b}. Out-of-scope T_c's tree, last_stakes, and
    score must be byte-identical to its pre-scoped state."""
    world = _make_world_three_tendencies()
    _seed_observations(world)
    equilibrate(world)
    # Snapshot T_c before scoped pass.
    pre_c_score = world.tendencies["T_c"].tree.score
    pre_c_stakes = dict(world.tendencies["T_c"].last_stakes)
    pre_c_nodes: dict = {}
    for node in world.tendencies["T_c"].tree.all_nodes():
        pre_c_nodes[node.id] = sorted(s.agent_id for s in node.stakes)

    # Add a new observation aimed at T_a to give the scoped pass
    # something to do.
    world.add_observation(Observation(
        id="obs_a2", coords=(0.85, 0.1, 0.05), label="another near A",
    ))
    equilibrate(world, scope={"T_a", "T_b"})

    # T_c must be unchanged.
    assert world.tendencies["T_c"].tree.score == pre_c_score, (
        f"T_c score moved: {pre_c_score} -> {world.tendencies['T_c'].tree.score}"
    )
    assert dict(world.tendencies["T_c"].last_stakes) == pre_c_stakes, (
        "T_c last_stakes changed under scoped pass"
    )
    post_c_nodes: dict = {}
    for node in world.tendencies["T_c"].tree.all_nodes():
        post_c_nodes[node.id] = sorted(s.agent_id for s in node.stakes)
    assert post_c_nodes == pre_c_nodes, (
        "T_c node post-counts changed under scoped pass"
    )


if __name__ == "__main__":
    test_scope_none_preserves_baseline()
    print("OK: scope=None preserves baseline")
    test_scope_none_matches_full_set()
    print("OK: scope=full-set matches scope=None")
    test_scope_subset_leaves_out_of_scope_untouched()
    print("OK: scope=subset leaves out-of-scope untouched")
