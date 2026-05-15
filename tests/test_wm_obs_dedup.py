#!/usr/bin/env python3
"""Test for cross-tendency observation-id dedup at sprout time.

Without dedup, when two tendencies' acts evaluate the same
observation with different polarity classifications, each sprouts
its own node (different content hash because polarity_axis
differs). The two parallel nodes then end up linked as children of
each other in cross-tendency edge discovery, and their intrinsic
contributions can cancel out -- the source of the S11 sub-child
accretion in Tier 1A.

With dedup (cross-tendency search by observation_id),
`_ensure_obs_child` reuses an existing same-obs node and just
appends the appropriate parent link in the calling tendency's
tree.

Test scenario: an observation at coords (-1, +0.8, +0.5) where
correctness classifies it CON, simplicity PRO, idiom PRO. After
all three tendencies have acted, there should be ONE node carrying
this observation, with parent links in all three tendencies at
their respective positions. Crucially, the node should NOT be a
sub-child of itself in any tree.
"""

from __future__ import annotations

import sys

from world_model.generalized.tendency import GeneralizedTendency
from world_model.generalized.world import World
from world_model.generalized import Observation


def _build_three_tendencies():
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="Code is correct.",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=1.5,
    )
    simplicity = GeneralizedTendency(
        id="simplicity",
        thesis="Code is simple.",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=1.5,
    )
    idiom = GeneralizedTendency(
        id="idiom",
        thesis="Code is idiomatic.",
        anchor=(0.0, 0.0, 1.0),
        polarity_axis=(0.0, 0.0, 1.0),
        bandwidth=1.5,
    )
    world.add_tendency(correctness)
    world.add_tendency(simplicity)
    world.add_tendency(idiom)
    return world


def test_disagreeing_tendencies_share_one_obs_node() -> None:
    """An observation at coords that three tendencies classify
    differently should produce ONE node with three parent links,
    not multiple parallel nodes."""
    world = _build_three_tendencies()
    obs = Observation(id="obs_test", coords=(-1.0, 0.8, 0.5),
                      label="ambiguous_obs")
    world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()

    # Collect every node carrying obs.id across the world's tendencies.
    obs_nodes = []
    seen_ids = set()
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            if node.observation_id == obs.id and node.id not in seen_ids:
                obs_nodes.append(node)
                seen_ids.add(node.id)

    assert len(obs_nodes) == 1, (
        f"expected 1 obs node after dedup, got {len(obs_nodes)}: "
        f"{[n.id for n in obs_nodes]}"
    )

    node = obs_nodes[0]
    # All three tendencies should have a parent link on this node.
    parent_tendencies = {p.tendency_id for p in node.parents}
    assert "correctness" in parent_tendencies, (
        f"correctness missing from parent links: {parent_tendencies}"
    )
    assert "simplicity" in parent_tendencies, (
        f"simplicity missing from parent links: {parent_tendencies}"
    )
    assert "idiom" in parent_tendencies, (
        f"idiom missing from parent links: {parent_tendencies}"
    )

    # The node must not list itself as a parent in any tendency.
    for link in node.parents:
        assert link.parent_id != node.id, (
            f"self-referential parent link: {link}"
        )


def test_dedup_records_per_tendency_position() -> None:
    """The shared node should record the correct PRO/CON position
    per tendency: CON for correctness (coords[0]=-1), PRO for
    simplicity (coords[1]=+0.8 > 0), PRO for idiom (coords[2]=+0.5 > 0).
    """
    world = _build_three_tendencies()
    obs = Observation(id="obs_test", coords=(-1.0, 0.8, 0.5),
                      label="ambiguous_obs")
    world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()

    correctness = world.tendencies["correctness"]
    obs_node = None
    for node in correctness.tree.all_nodes():
        if node.observation_id == obs.id:
            obs_node = node
            break
    assert obs_node is not None, "obs node not found in correctness's tree"

    # Find the parent links in each tendency and verify position.
    by_tendency = {p.tendency_id: p.position for p in obs_node.parents
                   if p.tendency_id in world.tendencies}
    from world_model.models.tree import Position
    assert by_tendency.get("correctness") == Position.CON, (
        f"correctness should be CON, got {by_tendency.get('correctness')}"
    )
    # simplicity and idiom: positions assigned by the cross-tendency
    # edge discovery (sign of dot product with polarity_axis).
    # simplicity polarity (0,1,0) dot (-1, 0.8, 0.5) = 0.8 > 0 -> PRO
    # idiom polarity (0,0,1) dot (-1, 0.8, 0.5) = 0.5 > 0 -> PRO
    assert by_tendency.get("simplicity") == Position.PRO, (
        f"simplicity should be PRO, got {by_tendency.get('simplicity')}"
    )
    assert by_tendency.get("idiom") == Position.PRO, (
        f"idiom should be PRO, got {by_tendency.get('idiom')}"
    )


def main() -> int:
    tests = [
        test_disagreeing_tendencies_share_one_obs_node,
        test_dedup_records_per_tendency_position,
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
