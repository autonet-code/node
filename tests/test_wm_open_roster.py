#!/usr/bin/env python3
"""Smoke test for the open-roster Tendency redesign.

Exercises:
- TendencyFactory instantiates open-roster tendencies from specs
- TendencySet manages allocations summing to 1.0 under add/remove
- Trees and Nodes still accept stakes from arbitrary tendency ids
- AttentionState reads new TendencySet shape correctly

This intentionally does not use the legacy Arena (which requires LLM
calls and is being rebuilt against the open roster). It validates the
data-layer redesign in isolation.
"""

from __future__ import annotations

from world_model import (
    Tendency,
    TendencySet,
    TendencySpec,
    DefaultTendencyFactory,
    build_legacy_personality_set,
    Observation,
    ObservationStore,
    Tree,
    Node,
    Position,
    Stake,
    AttentionState,
    BALANCED_CURVE,
)


def banner(text: str) -> None:
    print()
    print("-" * 60)
    print(text)
    print("-" * 60)


def test_legacy_roster_through_factory() -> None:
    banner("test: legacy seven-tendency roster via factory")
    ts = build_legacy_personality_set()
    assert len(ts) == 7
    assert abs(ts.total_allocation - 1.0) < 1e-9
    assert all(t.allocation > 0 for t in ts)
    print(f"  OK -- {ts}")


def test_arbitrary_roster_for_physics() -> None:
    banner("test: arbitrary roster for categorical-physics")
    factory = DefaultTendencyFactory()
    specs = [
        TendencySpec(id="cohesive_topos", initial_allocation=0.25,
                     description="Cohesive infinity-topos doctrine",
                     initial_claim="All physics lives in a single ambient topos."),
        TendencySpec(id="dagger_compact", initial_allocation=0.25,
                     description="Reversibility / CPT structure",
                     initial_claim="Every process has a structural reverse."),
        TendencySpec(id="non_cartesian_monoidal", initial_allocation=0.25,
                     description="Non-copyable resources",
                     initial_claim="Information cannot be freely duplicated."),
        TendencySpec(id="higher_categorical", initial_allocation=0.25,
                     description="2-morphisms, gauge transformations",
                     initial_claim="Morphisms between morphisms matter."),
    ]
    ts = factory.build_set(specs)
    assert len(ts) == 4
    assert abs(ts.total_allocation - 1.0) < 1e-9
    assert ts.has("cohesive_topos")
    assert "initial_claim" in ts.get("dagger_compact").metadata
    print(f"  OK -- {ts}")
    print(f"  cohesive_topos claim: {ts.get('cohesive_topos').metadata['initial_claim']}")


def test_dynamic_roster_birth_and_death() -> None:
    banner("test: tendencies can be born and culled mid-run")
    ts = build_legacy_personality_set()
    initial_count = len(ts)

    # Birth: factory adds a new tendency mid-run
    factory = DefaultTendencyFactory()
    novel = factory.instantiate(TendencySpec(
        id="creative_expression", initial_allocation=0.05,
        description="Drive to make/share artifacts",
    ))
    ts.add(novel)
    ts.normalize()
    assert len(ts) == initial_count + 1
    assert abs(ts.total_allocation - 1.0) < 1e-9
    assert ts.has("creative_expression")

    # Death: cull a tendency whose allocation collapsed
    ts.set_allocation("status", 0.0)
    removed = ts.remove("status")
    ts.normalize()
    assert removed is not None
    assert removed.id == "status"
    assert not ts.has("status")
    assert abs(ts.total_allocation - 1.0) < 1e-9
    print(f"  OK -- {ts}")


def test_tree_accepts_arbitrary_tendency_ids() -> None:
    banner("test: trees and stakes accept arbitrary tendency ids")
    factory = DefaultTendencyFactory()
    ts = factory.build_set([
        TendencySpec(id="alpha", initial_allocation=0.5),
        TendencySpec(id="beta", initial_allocation=0.5),
    ])

    obs_store = ObservationStore()
    obs, _ = obs_store.add(Observation(content="some observation"))

    tree = Tree(root_value="some root claim")
    node = Node(observation_id=obs.id, content=obs.content, tree_id=tree.id)
    tree.add_node(tree.root_node.id, node, Position.PRO)

    # Stakes carry arbitrary string ids; nothing in tree.py constrains them.
    node.add_stake("alpha", weight=ts.get("alpha").allocation * 0.8)
    node.add_stake("beta", weight=ts.get("beta").allocation * 0.3)

    assert len(node.stakes) == 2
    assert {s.agent_id for s in node.stakes} == {"alpha", "beta"}
    assert tree.score > 0
    print(f"  OK -- tree.score={tree.score:.3f}, stakes={node.stakes}")


def test_attention_state_with_open_roster() -> None:
    banner("test: AttentionState consumes the new TendencySet shape")
    ts = build_legacy_personality_set()
    state = AttentionState(agent_set=ts, curve=BALANCED_CURVE)

    state.update_novelty(0.0)
    base_alloc = dict(state.effective_allocations)
    base_curiosity = base_alloc.get("curiosity", 0.0)

    state.update_novelty(0.95)  # very surprising
    high_alloc = dict(state.effective_allocations)
    high_curiosity = high_alloc.get("curiosity", 0.0)

    assert high_curiosity > base_curiosity, (
        f"high-novelty curiosity ({high_curiosity:.3f}) should exceed "
        f"baseline curiosity ({base_curiosity:.3f})"
    )
    assert state.dominant_tendency in ts.ids()
    print(f"  base curiosity:  {base_curiosity:.3f}")
    print(f"  high curiosity:  {high_curiosity:.3f}")
    print(f"  dominant now:    {state.dominant_tendency}")
    print(f"  describe:        {state.describe()}")


def test_serialization_round_trip() -> None:
    banner("test: TendencySet round-trips through to_dict/from_dict")
    ts1 = build_legacy_personality_set()
    payload = ts1.to_dict()
    ts2 = TendencySet.from_dict(payload)
    assert len(ts2) == len(ts1)
    for tid in ts1.ids():
        assert ts2.has(tid)
        assert abs(ts2.get(tid).allocation - ts1.get(tid).allocation) < 1e-9
    print("  OK")


def test_legacy_dict_compat() -> None:
    banner("test: TendencySet.from_dict accepts the legacy 'agents' shape")
    legacy_payload = {
        "agents": {
            "survival":  {"tendency": "survival",  "allocation": 0.5},
            "curiosity": {"tendency": "curiosity", "allocation": 0.5},
        },
        "calibrated": True,
    }
    ts = TendencySet.from_dict(legacy_payload)
    assert len(ts) == 2
    assert ts.calibrated is True
    assert ts.has("survival") and ts.has("curiosity")
    print(f"  OK -- {ts}")


if __name__ == "__main__":
    test_legacy_roster_through_factory()
    test_arbitrary_roster_for_physics()
    test_dynamic_roster_birth_and_death()
    test_tree_accepts_arbitrary_tendency_ids()
    test_attention_state_with_open_roster()
    test_serialization_round_trip()
    test_legacy_dict_compat()
    print()
    print("All open-roster smoke tests passed.")
