#!/usr/bin/env python3
"""Tests for federation parent-merge under post-and-coparent semantics.

Two solvers can post a sub-claim at the same coordinates, hung under
different parents (or different tendencies). When their event streams
are replayed onto the same live world, the content-addressed id (over
{anchor, axis} only) collapses them to a single node, and the parent
links from both solvers accumulate.

This is the autonet-side merge story.
"""

from __future__ import annotations

import sys
from typing import List

# autonet's substrate adapter lives in the autonet repo
sys.path.insert(0, r"C:\code\autonet")

from world_model.generalized.tendency import GeneralizedTendency
from world_model.generalized.world import World
from world_model.models.tree import Position

from nodes.common.world_model_substrate.aggregate import apply_events  # type: ignore
from nodes.common.world_model_substrate.events import SubClaimSprouted  # type: ignore


def _build_world() -> World:
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
    return world


def test_two_solvers_same_anchor_same_id() -> None:
    """Solver 1 posts a sub-claim at coords c with parent in T_a;
    solver 2 posts at the same coords with parent in T_b. After
    replay, the live world has ONE node with parent edges in both
    tendencies.
    """
    world = _build_world()
    coords = [0.7, 0.0, 0.0]
    axis = [1.0, 0.0, 0.0]

    events: List[dict] = [
        SubClaimSprouted(
            seq=1,
            author_agent="solver_1",
            tendency_id="T_a",
            parent_id="solver1_root_T_a",
            node_id="solver1_node_X",
            position="pro",
            coords=coords,
            polarity_axis=axis,
            content="claim X",
        ).to_dict(),
        SubClaimSprouted(
            seq=2,
            author_agent="solver_2",
            tendency_id="T_b",
            parent_id="solver2_root_T_b",
            node_id="solver2_node_X_alt",
            position="pro",
            coords=coords,
            polarity_axis=axis,
            content="claim X (rephrased)",
        ).to_dict(),
    ]
    apply_events(world, events)

    t_a = world.tendencies["T_a"]
    t_b = world.tendencies["T_b"]
    a_nonroot = [n for n in t_a.tree.all_nodes() if n.id != t_a.tree.root_node.id]
    b_nonroot = [n for n in t_b.tree.all_nodes() if n.id != t_b.tree.root_node.id]

    # The two events should resolve to the SAME live node id.
    assert len(a_nonroot) == 1, (
        f"T_a should have exactly one non-root node, got {len(a_nonroot)}"
    )
    assert len(b_nonroot) == 1, (
        f"T_b should have exactly one non-root node, got {len(b_nonroot)}"
    )
    assert a_nonroot[0].id == b_nonroot[0].id, (
        "the two solvers' nodes should consolidate to one id "
        f"(got {a_nonroot[0].id} vs {b_nonroot[0].id})"
    )

    # The node's parent set should accumulate edges from both tendencies.
    node = a_nonroot[0]
    parent_tendencies = {p.tendency_id for p in node.parents}
    assert "T_a" in parent_tendencies and "T_b" in parent_tendencies, (
        f"expected parents in both T_a and T_b, got {parent_tendencies}"
    )


def main() -> int:
    tests = [test_two_solvers_same_anchor_same_id]
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
