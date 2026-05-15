#!/usr/bin/env python3
"""Tests for cross-tendency co-parenting via sprout_child edge discovery.

Under the post-and-coparent refactor, when a node is sprouted at a
coordinate that falls within the locality bandwidth of another
tendency's anchor, sprout_child appends a parent edge into that
tendency's tree automatically. This is what creates "work items":
emergent multi-parented nodes, no explicit type.
"""

from __future__ import annotations

import sys

from world_model.generalized.tendency import GeneralizedTendency
from world_model.generalized.world import World
from world_model.models.tree import Position


def _make_world_two_tendencies() -> tuple[World, GeneralizedTendency, GeneralizedTendency]:
    world = World()
    t_a = GeneralizedTendency(
        id="T_a",
        thesis="Tendency A",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=0.7,
    )
    t_b = GeneralizedTendency(
        id="T_b",
        thesis="Tendency B",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=0.7,
    )
    world.add_tendency(t_a)
    world.add_tendency(t_b)
    return world, t_a, t_b


def test_sprout_far_from_other_no_coparent() -> None:
    """A node sprouted far from another tendency's anchor stays
    single-parented."""
    world, t_a, t_b = _make_world_two_tendencies()
    # Anchor at (1, 0, 0) -- on top of A, far from B at (0, 1, 0)
    # (Euclidean distance sqrt(2) ~ 1.41, while B's bandwidth*1.5 = 1.05).
    new_node = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        world=world,
    )
    parent_tendencies = {p.tendency_id for p in new_node.parents}
    assert parent_tendencies == {"T_a"}, (
        f"expected only T_a parent, got {parent_tendencies}"
    )


def test_sprout_near_other_acquires_coparent() -> None:
    """A node sprouted at a coordinate inside another tendency's
    locality bandwidth gets a parent edge in that tendency's tree."""
    world, t_a, t_b = _make_world_two_tendencies()
    # Anchor at (0.6, 0.6, 0) -- close to both anchors. Distance to
    # A = sqrt(0.16 + 0.36) ~ 0.72; to B = sqrt(0.36 + 0.16) ~ 0.72.
    # Both are within bandwidth*1.5 = 1.05.
    new_node = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.6, 0.6, 0.0),
        polarity_axis=(1.0, 1.0, 0.0),
        world=world,
    )
    parent_tendencies = {p.tendency_id for p in new_node.parents}
    assert "T_a" in parent_tendencies and "T_b" in parent_tendencies, (
        f"expected co-parent in T_a and T_b, got {parent_tendencies}"
    )


def test_coparented_node_appears_in_both_trees() -> None:
    """A co-parented node is reachable from both tendencies' trees
    via tree.get_node, not just the tendency that sprouted it."""
    world, t_a, t_b = _make_world_two_tendencies()
    new_node = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.6, 0.6, 0.0),
        polarity_axis=(1.0, 1.0, 0.0),
        world=world,
    )
    assert t_a.tree.get_node(new_node.id) is not None, "missing from T_a"
    assert t_b.tree.get_node(new_node.id) is not None, "missing from T_b"


def test_coparenting_is_idempotent() -> None:
    """Calling sprout_child twice with the same world doesn't create
    duplicate parent links."""
    world, t_a, t_b = _make_world_two_tendencies()
    new_node = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.6, 0.6, 0.0),
        polarity_axis=(1.0, 1.0, 0.0),
        world=world,
    )
    parents_first = list(new_node.parents)
    # Sprout again -- should hit the existing-node branch, no new
    # parent links should be appended.
    same = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.6, 0.6, 0.0),
        polarity_axis=(1.0, 1.0, 0.0),
        world=world,
    )
    assert same.id == new_node.id
    assert list(same.parents) == parents_first, (
        f"parent set should be stable; got {same.parents} vs {parents_first}"
    )


def main() -> int:
    tests = [
        test_sprout_far_from_other_no_coparent,
        test_sprout_near_other_acquires_coparent,
        test_coparented_node_appears_in_both_trees,
        test_coparenting_is_idempotent,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  OK  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"  {failed}/{len(tests)} test(s) failed")
        return 1
    print(f"  {len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
