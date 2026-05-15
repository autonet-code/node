#!/usr/bin/env python3
"""Tests for correctness-as-veto dynamics.

Two pieces:
  1. Per-tendency rate constants (novelty_gamma_*) let a deployer give
     the correctness root a higher gamma_con so CON evidence settles
     fast on it.
  2. Asymmetric pruning (prune_veto_negatives) drops subtrees rooted
     under a veto-shaped tendency's children when their intrinsic
     score falls below a hard floor, regardless of n.

These tests exercise (2) directly: hand-build a substrate where
under the correctness root one sub-claim has accumulated negative
intrinsic_score (CON children dragging it down) and another sub-claim
has positive intrinsic_score (PRO children supporting it). Then call
prune_veto_negatives; the negative one should disappear while the
positive one survives.
"""

from __future__ import annotations

import sys

from world_model.generalized.tendency import GeneralizedTendency, _intrinsic_score
from world_model.generalized.world import World
from world_model.generalized.prune import prune_veto_negatives
from world_model.models.tree import Position


def _make_world_with_veto_root() -> tuple[World, GeneralizedTendency]:
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="Code is correct",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=0.7,
        veto_shaped=True,
        veto_score_floor=-1.0,
    )
    world.add_tendency(correctness)
    return world, correctness


def test_negative_subtree_pruned() -> None:
    """A direct-child sub-claim of the correctness root, dragged into
    deeply negative intrinsic_score by CON children, gets pruned by
    prune_veto_negatives."""
    world, t = _make_world_with_veto_root()
    bad = t.sprout_child(
        parent_node_id=t.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.5, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        content="bad_claim",
    )
    # Add CON children whose own posts drag bad's intrinsic_score below
    # the floor. Each CON child carries 5 unit-weight posts; bad has
    # 0 direct posts and 0 PRO children. Intrinsic_score(bad) = 0 - 5 - 5 = -10.
    for i in range(2):
        c = t.sprout_child(
            parent_node_id=bad.id,
            position=Position.CON,
            anchor=(1.0, 0.5 + 0.1 * i, 0.1),
            polarity_axis=(-1.0, 0.0, 0.0),
            content=f"counterexample_{i}",
        )
        for k in range(5):
            c.add_post(agent_id=f"agent_{k}")

    # Sanity: bad's intrinsic_score is well below the floor.
    score = _intrinsic_score(bad)
    assert score < t.veto_score_floor, (
        f"bad's intrinsic_score {score} should be < floor {t.veto_score_floor}"
    )

    pruned = prune_veto_negatives(world)
    assert bad.id in pruned, f"expected bad ({bad.id}) to be pruned, got {pruned}"
    # bad should be gone from the tree.
    assert t.tree.get_node(bad.id) is None, "pruned node still in tree index"


def test_positive_subtree_survives() -> None:
    """A direct-child sub-claim with positive intrinsic_score is
    NOT pruned even on a veto-shaped root."""
    world, t = _make_world_with_veto_root()
    good = t.sprout_child(
        parent_node_id=t.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.5, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        content="good_claim",
    )
    # Pile on PRO posts + a PRO child with posts so intrinsic > floor.
    for k in range(3):
        good.add_post(agent_id=f"agent_{k}")

    pruned = prune_veto_negatives(world)
    assert good.id not in pruned, f"good ({good.id}) should survive; got {pruned}"
    assert t.tree.get_node(good.id) is not None, "good node was removed unexpectedly"


def test_con_child_accumulates_pruned() -> None:
    """A CON-position direct child of a veto root, with enough
    accumulated posts that its negative contribution clears the
    floor, gets pruned. This is the structural-veto case: the
    substrate has classified the work item as anti-correctness
    (CON-position), and the evidence has accumulated."""
    world, t = _make_world_with_veto_root()
    # Sprout a CON child of the correctness root with 3 posts on it.
    # Contribution to correctness = -intrinsic_score(child) = -3,
    # well below floor=-1.0 -> should be pruned.
    bad = t.sprout_child(
        parent_node_id=t.tree.root_node.id,
        position=Position.CON,
        anchor=(-1.0, 0.5, 0.0),
        polarity_axis=(-1.0, 0.0, 0.0),
        content="anti_correctness_claim",
    )
    for k in range(3):
        bad.add_post(agent_id=f"agent_{k}")

    pruned = prune_veto_negatives(world)
    assert bad.id in pruned, (
        f"CON child with intrinsic={_intrinsic_score(bad)} contributing "
        f"{-_intrinsic_score(bad):+.2f} (floor={t.veto_score_floor}) "
        f"should have been pruned; got {pruned}"
    )
    assert t.tree.get_node(bad.id) is None, "pruned node still in tree index"


def test_non_veto_tendency_not_affected() -> None:
    """prune_veto_negatives only acts on tendencies tagged
    veto_shaped=True; other tendencies' subtrees are untouched."""
    world, t = _make_world_with_veto_root()
    other = GeneralizedTendency(
        id="other",
        thesis="Other",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=0.7,
        veto_shaped=False,
    )
    world.add_tendency(other)
    bad_in_other = other.sprout_child(
        parent_node_id=other.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.0, 0.5, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        content="negative_in_other",
    )
    # Make bad_in_other's intrinsic_score deeply negative.
    for i in range(2):
        c = other.sprout_child(
            parent_node_id=bad_in_other.id,
            position=Position.CON,
            anchor=(0.0, 0.5 + 0.1 * i, 0.1),
            polarity_axis=(0.0, -1.0, 0.0),
            content=f"counter_{i}",
        )
        for k in range(5):
            c.add_post(agent_id=f"agent_{k}")

    pruned = prune_veto_negatives(world)
    assert bad_in_other.id not in pruned, (
        "non-veto tendency's subtrees should not be pruned by veto-prune"
    )


def main() -> int:
    tests = [
        test_negative_subtree_pruned,
        test_positive_subtree_survives,
        test_con_child_accumulates_pruned,
        test_non_veto_tendency_not_affected,
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
