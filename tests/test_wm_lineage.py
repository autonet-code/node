#!/usr/bin/env python3
"""Smoke tests for the producer-consumer lineage subsystem.

Validates:
  - emit() writes to the origin outbox only (no propagation at emit-time)
  - pull_view() reads neighbors per stake-weight cutoff and radius
  - Pulls are read-only; repeated pulls don't change state
  - Retention policies (DropOldest, RefuseWhenFull, Unbounded, BoundedRingPlusCompaction)
  - Allocation rolling window tracks origin-only shifts
  - Visibility scales with sparsity (the cleanest result)
  - Round-trip serialization across all policy kinds
"""

from __future__ import annotations

from world_model import (
    BoundedRingPlusCompactionPolicy,
    CompactionTier,
    DropOldestPolicy,
    EngineClock,
    Event,
    EventType,
    Lineage,
    LineageRecorder,
    OutboxFullError,
    RefuseWhenFullPolicy,
    StakeWeightGraph,
    UnboundedPolicy,
    policy_from_dict,
)


def banner(text: str) -> None:
    print()
    print("-" * 60)
    print(text)
    print("-" * 60)


def make_recorder(
    edges: list[tuple[str, str, float]],
    radius: int = 2,
    min_weight: float = 0.01,
    policy_factory=None,
) -> LineageRecorder:
    graph = StakeWeightGraph()
    nodes: set[str] = set()
    for a, b, w in edges:
        graph.add_edge(a, b, w)
        nodes.add(a)
        nodes.add(b)
    rec = LineageRecorder(
        clock=EngineClock(),
        graph=graph,
        default_pull_radius=radius,
        default_pull_min_weight=min_weight,
    )
    for n in nodes:
        ln = Lineage(outbox=policy_factory()) if policy_factory else Lineage()
        rec.register(n, ln)
    return rec


def test_engine_clock_monotonic() -> None:
    banner("test: EngineClock ticks monotonically")
    clock = EngineClock()
    assert [clock.tick() for _ in range(3)] == [1, 2, 3]
    assert clock.now() == 3
    print("  OK -- ticks: 1, 2, 3, now()=3")


def test_emit_writes_to_origin_only() -> None:
    banner("test: emit() writes to origin outbox only, no propagation")
    rec = make_recorder([("alpha", "beta", 0.9)])  # strong link, but emit doesn't propagate
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o1"})
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o2"})

    alpha_events = rec.lineage_of("alpha").events()
    beta_events = rec.lineage_of("beta").events()
    assert len(alpha_events) == 2
    assert len(beta_events) == 0, "emit must not write to neighbors"
    print(f"  alpha (origin): {len(alpha_events)} event(s) in own outbox")
    print(f"  beta  (strong link, but emit is origin-only): {len(beta_events)} event(s)")


def test_pull_view_walks_graph_to_radius() -> None:
    banner("test: pull_view walks graph to radius along edges >= cutoff")
    # Chain: alpha - beta - gamma - delta, all weights above cutoff
    rec = make_recorder([
        ("alpha", "beta",  0.5),
        ("beta",  "gamma", 0.5),
        ("gamma", "delta", 0.5),
    ], radius=2, min_weight=0.01)

    # Each tendency emits one event into its own outbox.
    rec.emit(EventType.STAKED, origin_id="alpha", payload={})
    rec.emit(EventType.STAKED, origin_id="beta",  payload={})
    rec.emit(EventType.STAKED, origin_id="gamma", payload={})
    rec.emit(EventType.STAKED, origin_id="delta", payload={})

    # Alpha's pull at radius 2 should see alpha, beta, gamma but not delta.
    alpha_view = rec.pull_view("alpha")
    origins = {e.origin_id for e in alpha_view}
    assert origins == {"alpha", "beta", "gamma"}, origins
    assert "delta" not in origins, "delta is 3 hops away, outside radius"
    print(f"  alpha pulls (radius=2) sees: {sorted(origins)}")
    print(f"  delta excluded as expected")


def test_pull_respects_min_weight() -> None:
    banner("test: pull_view filters by min_weight")
    rec = make_recorder([
        ("alpha", "beta", 0.5),    # strong
        ("alpha", "gamma", 0.005), # below default cutoff (0.01)
    ], radius=1, min_weight=0.01)
    rec.emit(EventType.STAKED, origin_id="alpha")
    rec.emit(EventType.STAKED, origin_id="beta")
    rec.emit(EventType.STAKED, origin_id="gamma")

    view = rec.pull_view("alpha")
    origins = {e.origin_id for e in view}
    assert origins == {"alpha", "beta"}, origins
    print(f"  alpha pulls: {sorted(origins)}; gamma excluded (weight 0.005 < cutoff 0.01)")


def test_pull_is_read_only() -> None:
    banner("test: pull_view is read-only (does not mutate state)")
    rec = make_recorder([("alpha", "beta", 0.9)])
    rec.emit(EventType.STAKED, origin_id="alpha")
    rec.emit(EventType.STAKED, origin_id="beta")

    clock_before = rec.clock.now()
    counts_before = {tid: len(ln.events()) for tid, ln in rec.lineages.items()}

    # Pull twice; nothing should change.
    rec.pull_view("alpha")
    rec.pull_view("alpha")
    rec.pull_view("beta", radius=5, min_weight=0.0)

    clock_after = rec.clock.now()
    counts_after = {tid: len(ln.events()) for tid, ln in rec.lineages.items()}

    assert clock_before == clock_after, "pulls must not advance the clock"
    assert counts_before == counts_after, "pulls must not modify outboxes"
    print(f"  clock unchanged ({clock_after}), outboxes unchanged ({counts_after})")


def test_pull_parameters_override_defaults() -> None:
    banner("test: per-pull radius/min_weight override the recorder defaults")
    rec = make_recorder([
        ("alpha", "beta",  0.5),
        ("beta",  "gamma", 0.5),
    ], radius=1, min_weight=0.01)
    rec.emit(EventType.STAKED, origin_id="alpha")
    rec.emit(EventType.STAKED, origin_id="beta")
    rec.emit(EventType.STAKED, origin_id="gamma")

    # Default radius=1: alpha sees alpha and beta only.
    default_view = {e.origin_id for e in rec.pull_view("alpha")}
    assert default_view == {"alpha", "beta"}

    # Override radius=2: alpha now reaches gamma too.
    wide_view = {e.origin_id for e in rec.pull_view("alpha", radius=2)}
    assert wide_view == {"alpha", "beta", "gamma"}
    print(f"  radius=1 (default): {sorted(default_view)}")
    print(f"  radius=2 (override): {sorted(wide_view)}")


def test_drop_oldest_policy() -> None:
    banner("test: DropOldestPolicy evicts oldest at capacity")
    rec = make_recorder(
        [("alpha", "beta", 0.5)],
        policy_factory=lambda: DropOldestPolicy(capacity=3),
    )
    for i in range(5):
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})

    events = rec.lineage_of("alpha").events()
    assert len(events) == 3
    assert [e.payload["i"] for e in events] == [2, 3, 4]
    print(f"  emitted 5 into ring of 3; kept i={[e.payload['i'] for e in events]}")


def test_refuse_when_full_policy() -> None:
    banner("test: RefuseWhenFullPolicy raises on overflow")
    rec = make_recorder(
        [("alpha", "beta", 0.5)],
        policy_factory=lambda: RefuseWhenFullPolicy(capacity=2),
    )
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": 0})
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": 1})
    try:
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": 2})
    except OutboxFullError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("expected OutboxFullError on third emission")


def test_unbounded_policy() -> None:
    banner("test: UnboundedPolicy retains everything")
    rec = make_recorder(
        [("alpha", "beta", 0.5)],
        policy_factory=lambda: UnboundedPolicy(),
    )
    for i in range(50):
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})
    events = rec.lineage_of("alpha").events()
    assert len(events) == 50
    print(f"  emitted 50, retained {len(events)} (no eviction)")


def test_bounded_ring_plus_compaction_policy() -> None:
    banner("test: BoundedRingPlusCompactionPolicy spills evicted events to lower tiers")

    # Demote function: just stamp the event with a new lod. (Real protocols
    # would use LODProtocol.reduce() to actually compress payload fields.)
    def demote_to_lod(event: Event, target_lod: int) -> Event:
        return Event(
            engine_time=event.engine_time,
            type=event.type,
            origin_id=event.origin_id,
            payload=dict(event.payload),
            wall_clock=event.wall_clock,
            lod=target_lod,
        )

    policy = BoundedRingPlusCompactionPolicy(
        tiers=[
            CompactionTier(capacity=2, target_lod=3),  # tier 0: 2 slots, full LOD
            CompactionTier(capacity=2, target_lod=2),  # tier 1: 2 slots, demoted to LOD 2
            CompactionTier(capacity=2, target_lod=1),  # tier 2: 2 slots, demoted to LOD 1
        ],
        demote_fn=demote_to_lod,
    )
    rec = make_recorder([("alpha", "beta", 0.5)], policy_factory=lambda: policy)

    # Emit 6 events. Layout:
    #   tier 2 (LOD 1):  i=0, i=1
    #   tier 1 (LOD 2):  i=2, i=3
    #   tier 0 (LOD 3):  i=4, i=5
    for i in range(6):
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})

    events = rec.lineage_of("alpha").events()
    assert len(events) == 6, f"expected 6 retained, got {len(events)}"
    # Sorted by engine_time
    by_i = {e.payload["i"]: e for e in events}
    assert by_i[0].lod == 1, f"oldest demoted to LOD 1, got {by_i[0].lod}"
    assert by_i[1].lod == 1
    assert by_i[2].lod == 2
    assert by_i[3].lod == 2
    assert by_i[4].lod == 3, f"newest at LOD 3, got {by_i[4].lod}"
    assert by_i[5].lod == 3
    print(f"  6 events retained across 3 tiers with LOD demotion:")
    for i in range(6):
        print(f"    i={i}: lod={by_i[i].lod}")

    # Emit 4 more -- the 2 oldest (currently at LOD 1) get pushed off the
    # bottom. The 2 next-oldest (LOD 2) move down to LOD 1. Etc.
    for i in range(6, 10):
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})

    events = rec.lineage_of("alpha").events()
    surviving_is = sorted(e.payload["i"] for e in events)
    assert surviving_is == [4, 5, 6, 7, 8, 9], surviving_is
    print(f"  after 4 more emissions, surviving i={surviving_is} (oldest 4 dropped)")


def test_allocation_window_origin_only() -> None:
    banner("test: allocation rolling window only updates on origin")
    rec = make_recorder([("alpha", "beta", 0.5)])
    rec.emit_allocation_shift("alpha", new_allocation=0.4, delta=+0.1)
    rec.emit_allocation_shift("alpha", new_allocation=0.5, delta=+0.1)

    assert rec.lineage_of("alpha").recent_allocations() == [0.4, 0.5]
    assert rec.lineage_of("beta").recent_allocations() == []
    print(f"  alpha window: {rec.lineage_of('alpha').recent_allocations()}")
    print(f"  beta  window: {rec.lineage_of('beta').recent_allocations()} (correctly empty)")


def test_global_event_view_total_order() -> None:
    banner("test: global_event_view returns events sorted by engine_time")
    rec = make_recorder([
        ("alpha", "beta", 0.5),
        ("gamma", "delta", 0.5),
    ])
    rec.emit(EventType.STAKED, origin_id="alpha")    # t=1
    rec.emit(EventType.STAKED, origin_id="gamma")    # t=2
    rec.emit(EventType.STAKED, origin_id="beta")     # t=3
    rec.emit(EventType.STAKED, origin_id="delta")    # t=4

    times = [e.engine_time for e in rec.global_event_view()]
    assert times == [1, 2, 3, 4]
    print(f"  global view engine-times: {times}")


def test_visibility_scales_with_sparsity() -> None:
    banner("test: pull-visibility tracks stake-weight sparsity")
    rec = make_recorder([
        ("alpha", "beta",     0.8),
        ("alpha", "gamma",    0.001),  # below cutoff
        ("beta",  "epsilon",  0.7),
    ], radius=2, min_weight=0.01)
    rec.register("zeta")  # isolated

    for tid in ["alpha", "beta", "gamma", "epsilon", "zeta"]:
        rec.emit(EventType.STAKED, origin_id=tid)

    visible_to_alpha = {e.origin_id for e in rec.pull_view("alpha")}
    assert visible_to_alpha == {"alpha", "beta", "epsilon"}, visible_to_alpha
    print(f"  alpha sees: {sorted(visible_to_alpha)}")
    print(f"  hidden:     {sorted(set(rec.lineages) - visible_to_alpha)}")


def test_serialization_round_trip_each_policy() -> None:
    banner("test: each retention policy round-trips through to_dict/from_dict")

    for label, factory in [
        ("DropOldest",        lambda: DropOldestPolicy(capacity=8)),
        ("RefuseWhenFull",    lambda: RefuseWhenFullPolicy(capacity=8)),
        ("Unbounded",         lambda: UnboundedPolicy()),
        ("BoundedRingPlusCompaction",
            lambda: BoundedRingPlusCompactionPolicy(
                tiers=[
                    CompactionTier(capacity=4, target_lod=3),
                    CompactionTier(capacity=4, target_lod=2),
                ],
                demote_fn=None,
            )),
    ]:
        rec = make_recorder([("alpha", "beta", 0.5)], policy_factory=factory)
        for i in range(3):
            rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})

        ln = rec.lineage_of("alpha")
        payload = ln.to_dict()
        ln2 = Lineage.from_dict(payload)

        assert len(ln2.events()) == len(ln.events()), label
        for e1, e2 in zip(ln.events(), ln2.events()):
            assert e1.engine_time == e2.engine_time
            assert e1.payload == e2.payload
        print(f"  {label}: {len(ln.events())} event(s) round-trip OK")


def test_legacy_lineage_dict_compat() -> None:
    banner("test: Lineage.from_dict accepts legacy v1 shape")
    legacy = {
        "allocation_window_capacity": 16,
        "event_ring_capacity": 4,
        "allocation_window": [0.1, 0.2],
        "event_ring": [
            {"engine_time": 1, "type": "staked", "origin_id": "alpha",
             "payload": {}, "wall_clock": None},
        ],
    }
    ln = Lineage.from_dict(legacy)
    assert ln.recent_allocations() == [0.1, 0.2]
    assert len(ln.events()) == 1
    assert isinstance(ln.outbox, DropOldestPolicy)
    print(f"  legacy v1 dict loads as Lineage with DropOldestPolicy")


if __name__ == "__main__":
    test_engine_clock_monotonic()
    test_emit_writes_to_origin_only()
    test_pull_view_walks_graph_to_radius()
    test_pull_respects_min_weight()
    test_pull_is_read_only()
    test_pull_parameters_override_defaults()
    test_drop_oldest_policy()
    test_refuse_when_full_policy()
    test_unbounded_policy()
    test_bounded_ring_plus_compaction_policy()
    test_allocation_window_origin_only()
    test_global_event_view_total_order()
    test_visibility_scales_with_sparsity()
    test_serialization_round_trip_each_policy()
    test_legacy_lineage_dict_compat()
    print()
    print("All lineage smoke tests passed.")
