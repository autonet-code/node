#!/usr/bin/env python3
"""Phase 3.3: the slow-path Lindblad pass must surface cross-domain
links that the discrete kernel's locality gate skipped.

Scenario: two tendencies A and B with bandwidth=0.5 each, anchors
placed so their bandwidths don't overlap. Sub-claims sprouted under
each at coords that are coord-close but still OUTSIDE each other's
bandwidth-1.5 gate. The discrete kernel's cross-tendency edge
discovery never links them. The slow Lindblad pass — with boosted mu
and longer t_total — DOES link them via the J_ab coupling matrix,
posts `_lindblad_cross_link` stakes, and exposes the connection to
the discrete kernel's next pass.

Pass criteria:
  - After discrete equilibrate: no `_lindblad_cross_link` stakes exist.
  - After equilibrate_continuous_exploration: at least one
    `_lindblad_cross_link` stake on each of the two sub-claims that
    the Hamiltonian identified as the dominant coupling pair.
"""

from __future__ import annotations

import sys

from world_model.generalized import (
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.equilibrate import equilibrate_continuous_exploration
from world_model.models.tree import Position


def _make_world() -> World:
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
    return world


def _has_cross_link_stake(node) -> bool:
    return any(s.agent_id == "_lindblad_cross_link" for s in node.stakes)


def test_discrete_kernel_does_not_emit_cross_link() -> None:
    """Sanity: equilibrate alone never produces _lindblad_cross_link
    stakes. They only come from the Lindblad path."""
    world = _make_world()
    t_a = world.tendencies["T_a"]
    t_b = world.tendencies["T_b"]

    # Sub-claims close-but-outside each other's bandwidth.
    # T_a's anchor=(1,0,0), bandwidth=0.5 => gate (bw*1.5)=0.75.
    # T_b's anchor=(0,1,0).
    # Sub-claims at (1.0, 0.3, 0) under T_a and (0.3, 1.0, 0) under T_b.
    # Cross distance ~ sqrt((0.7)^2+(0.7)^2) ~ 0.99 — outside both gates.
    # Distance to T_b anchor for first sub-claim: sqrt(1+0.49) ~ 1.22 > 0.75.
    sub_a = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.3, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        world=world,
    )
    sub_b = t_b.sprout_child(
        parent_node_id=t_b.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.3, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        world=world,
    )

    # Confirm bandwidth gate did NOT co-parent them.
    a_parents = {p.tendency_id for p in sub_a.parents}
    b_parents = {p.tendency_id for p in sub_b.parents}
    assert a_parents == {"T_a"}, f"sub_a should only have T_a parent: {a_parents}"
    assert b_parents == {"T_b"}, f"sub_b should only have T_b parent: {b_parents}"

    # Seed an observation for each tendency so the sub-claims accumulate
    # capacity (J_ab depends on capacity products).
    world.add_observation(Observation(id="oa", coords=(1.0, 0.0, 0.0), label="a"))
    world.add_observation(Observation(id="ob", coords=(0.0, 1.0, 0.0), label="b"))
    equilibrate(world, max_rounds=5)

    # Discrete kernel must not emit _lindblad_cross_link stakes.
    for tid in ("T_a", "T_b"):
        for node in world.tendencies[tid].tree.all_nodes():
            assert not _has_cross_link_stake(node), (
                f"discrete kernel unexpectedly produced cross_link on "
                f"{tid}/{node.id}"
            )


def test_slow_lindblad_surfaces_cross_link() -> None:
    """The architectural integrity test: slow Lindblad pass posts
    _lindblad_cross_link on the dominant coupled sub-claim pair."""
    world = _make_world()
    t_a = world.tendencies["T_a"]
    t_b = world.tendencies["T_b"]

    sub_a = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.3, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        world=world,
    )
    sub_b = t_b.sprout_child(
        parent_node_id=t_b.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.3, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        world=world,
    )

    # Seed observations + discrete equilibrate to accumulate capacity.
    world.add_observation(Observation(id="oa", coords=(1.0, 0.0, 0.0), label="a"))
    world.add_observation(Observation(id="ob", coords=(0.0, 1.0, 0.0), label="b"))
    equilibrate(world, max_rounds=5)

    # Confirm baseline: no cross_links present.
    pre = [_has_cross_link_stake(n)
           for tid in ("T_a", "T_b")
           for n in world.tendencies[tid].tree.all_nodes()]
    assert not any(pre), "test setup contaminated: cross_links exist pre-Lindblad"

    # Run the slow Lindblad exploration pass with a permissive
    # threshold. bandwidth=0.5 (the substrate's value); mu and t_total
    # are the exploration defaults.
    result = equilibrate_continuous_exploration(
        world,
        bandwidth=0.5,
        cross_link_threshold=0.01,
    )

    # Inspect emitted descriptors.
    print(f"  Lindblad emitted {len(result['cross_links'])} cross-links: "
          f"{[(c['root_a'], c['root_b'], c['J']) for c in result['cross_links']]}")
    assert len(result["cross_links"]) > 0, (
        f"slow Lindblad failed to surface a cross-link; "
        f"Js={result['Js']}"
    )

    # The link must connect a node in T_a's tree to a node in T_b's tree.
    link = result["cross_links"][0]
    assert {link["root_a"], link["root_b"]} == {"T_a", "T_b"}, (
        f"cross-link expected between T_a and T_b: {link}"
    )
    node_a = world.tendencies[link["root_a"]].tree.get_node(link["node_a"])
    node_b = world.tendencies[link["root_b"]].tree.get_node(link["node_b"])
    assert node_a is not None and node_b is not None, (
        f"cross-link references unknown nodes: {link}"
    )

    # The dominant pair carries the cross-link stake.
    assert _has_cross_link_stake(node_a), (
        f"node_a {link['node_a']} missing _lindblad_cross_link"
    )
    assert _has_cross_link_stake(node_b), (
        f"node_b {link['node_b']} missing _lindblad_cross_link"
    )


def test_discrete_kernel_preserves_cross_link_stakes() -> None:
    """After Lindblad posts cross-link stakes, a subsequent discrete
    equilibrate must NOT strip them. world.apply_stakes only removes
    stakes attributed to tendencies registered in world.tendencies;
    `_lindblad_cross_link` isn't a tendency, so the stake survives."""
    world = _make_world()
    t_a = world.tendencies["T_a"]
    t_b = world.tendencies["T_b"]

    sub_a = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.3, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        world=world,
    )
    sub_b = t_b.sprout_child(
        parent_node_id=t_b.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.3, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        world=world,
    )
    world.add_observation(Observation(id="oa", coords=(1.0, 0.0, 0.0), label="a"))
    world.add_observation(Observation(id="ob", coords=(0.0, 1.0, 0.0), label="b"))
    equilibrate(world, max_rounds=5)

    result = equilibrate_continuous_exploration(
        world, bandwidth=0.5, cross_link_threshold=0.01,
    )
    assert len(result["cross_links"]) > 0

    # Snapshot which nodes received the cross-link stake.
    linked_node_ids = set()
    for tid in ("T_a", "T_b"):
        for n in world.tendencies[tid].tree.all_nodes():
            if _has_cross_link_stake(n):
                linked_node_ids.add(n.id)
    assert linked_node_ids, "expected at least one node with cross_link"

    # Now run discrete equilibrate again — cross_link stakes survive.
    equilibrate(world, max_rounds=5)
    surviving = set()
    for tid in ("T_a", "T_b"):
        for n in world.tendencies[tid].tree.all_nodes():
            if _has_cross_link_stake(n):
                surviving.add(n.id)
    assert linked_node_ids.issubset(surviving), (
        f"discrete equilibrate stripped cross_links: "
        f"lost {linked_node_ids - surviving}"
    )


if __name__ == "__main__":
    test_discrete_kernel_does_not_emit_cross_link()
    print("OK: discrete kernel does not produce cross_link stakes")
    test_slow_lindblad_surfaces_cross_link()
    print("OK: slow Lindblad surfaces dominant cross_link pair")
    test_discrete_kernel_preserves_cross_link_stakes()
    print("OK: cross_link stakes survive subsequent discrete equilibrate")
