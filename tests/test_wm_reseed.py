#!/usr/bin/env python3
"""Smoke tests for reseed-and-equilibrate.

Validates:
  - Pure function: inputs are not mutated
  - Trivial reseed (no substitutions) is a near-identity
  - Substitution adds, replaces, and removes tendencies
  - Calibration converges (allocations sum to 1, deltas below tolerance)
  - Iteration cap is respected
  - Caller-supplied edges replace the substituted tendency's neighborhood
  - Recorder integration emits one ALLOCATION_SHIFTED per affected tendency
  - Reseeding the same input twice produces the same output (determinism)
"""

from __future__ import annotations

import copy

from world_model import (
    DefaultTendencyFactory,
    EngineClock,
    EventType,
    Lineage,
    LineageRecorder,
    PresentState,
    ReseedResult,
    StakeWeightGraph,
    Substitution,
    Tendency,
    TendencySet,
    TendencySpec,
    reseed_and_equilibrate,
)


def banner(text: str) -> None:
    print()
    print("-" * 60)
    print(text)
    print("-" * 60)


def make_simple_state() -> PresentState:
    """A 4-tendency starting state with arbitrary edges and lineage slots."""
    factory = DefaultTendencyFactory()
    ts = factory.build_set([
        TendencySpec(id="alpha", initial_allocation=0.25),
        TendencySpec(id="beta",  initial_allocation=0.25),
        TendencySpec(id="gamma", initial_allocation=0.25),
        TendencySpec(id="delta", initial_allocation=0.25),
    ])
    lineages = {tid: Lineage() for tid in ts.ids()}
    graph = StakeWeightGraph()
    graph.add_edge("alpha", "beta",  0.5)
    graph.add_edge("beta",  "gamma", 0.4)
    graph.add_edge("gamma", "delta", 0.3)
    return PresentState(tendencies=ts, lineages=lineages, graph=graph)


def test_inputs_are_not_mutated() -> None:
    banner("test: reseed is pure -- inputs not mutated")
    state = make_simple_state()
    snapshot = copy.deepcopy(state)

    reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(
                id="alpha",
                new_tendency=Tendency(id="alpha", allocation=0.5),
            ),
        ],
    )

    # The original state must be byte-identical after the call.
    assert state.tendencies.to_dict() == snapshot.tendencies.to_dict()
    assert {k: v.to_dict() for k, v in state.lineages.items()} == \
           {k: v.to_dict() for k, v in snapshot.lineages.items()}
    assert state.graph.weights == snapshot.graph.weights
    print("  OK -- input state unchanged after reseed")


def test_trivial_reseed_is_near_identity() -> None:
    banner("test: empty substitution list is a near-identity")
    state = make_simple_state()
    pre = {t.id: t.allocation for t in state.tendencies.all()}

    result = reseed_and_equilibrate(state, substitutions=[])

    post = {t.id: t.allocation for t in result.state.tendencies.all()}
    for tid in pre:
        assert abs(pre[tid] - post[tid]) < 1e-9, (tid, pre[tid], post[tid])
    assert result.converged
    print(f"  iterations={result.iterations} converged={result.converged}")
    print(f"  allocations preserved within 1e-9")


def test_substitute_replaces_tendency() -> None:
    banner("test: substitute swaps a tendency and re-equilibrates")
    state = make_simple_state()

    # Replace alpha with a new tendency claiming 0.5 allocation.
    result = reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(
                id="alpha",
                new_tendency=Tendency(id="alpha", allocation=0.5,
                                      description="reseeded alpha"),
            ),
        ],
        learning_rate=0.5,
    )

    new_alloc = {t.id: t.allocation for t in result.state.tendencies.all()}
    total = sum(new_alloc.values())
    assert abs(total - 1.0) < 1e-9, total
    assert new_alloc["alpha"] > new_alloc["beta"], (
        "alpha should dominate after being reseeded high"
    )
    assert "alpha" in result.affected_ids
    print(f"  iterations={result.iterations} converged={result.converged}")
    print(f"  allocations: {new_alloc}")


def test_substitute_adds_new_tendency() -> None:
    banner("test: substituting an unknown id adds a new tendency")
    state = make_simple_state()

    result = reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(
                id="epsilon",
                new_tendency=Tendency(id="epsilon", allocation=0.4),
                edges={"alpha": 0.6, "beta": 0.2},
            ),
        ],
    )

    assert result.state.tendencies.has("epsilon")
    assert "epsilon" in result.state.lineages
    assert result.state.graph.has("epsilon")
    assert result.state.graph.edge_weight("epsilon", "alpha") == 0.6
    assert result.state.graph.edge_weight("epsilon", "beta") == 0.2
    total = sum(t.allocation for t in result.state.tendencies.all())
    assert abs(total - 1.0) < 1e-9
    print(f"  added 'epsilon' with edges {result.state.graph.weights['epsilon']}")
    print(f"  total allocation: {total:.6f}")


def test_substitute_removes_tendency() -> None:
    banner("test: substitution with new_tendency=None removes the tendency")
    state = make_simple_state()

    result = reseed_and_equilibrate(
        state,
        substitutions=[Substitution(id="delta", new_tendency=None)],
    )

    assert not result.state.tendencies.has("delta")
    assert "delta" not in result.state.lineages
    assert not result.state.graph.has("delta")
    total = sum(t.allocation for t in result.state.tendencies.all())
    assert abs(total - 1.0) < 1e-9
    print(f"  removed 'delta'; remaining: {result.state.tendencies.ids()}")
    print(f"  total allocation re-normalized to: {total:.6f}")


def test_iteration_cap_respected() -> None:
    banner("test: max_iterations cap is honored")
    state = make_simple_state()

    # With learning_rate=0.5 and a substantial substitution, convergence
    # to 1e-12 normally takes ~40+ iterations. Cap at 3 and check
    # converged=False.
    result = reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(id="alpha",
                         new_tendency=Tendency(id="alpha", allocation=0.9)),
        ],
        tolerance=1e-12,
        max_iterations=3,
        learning_rate=0.5,
    )
    assert result.iterations == 3
    assert not result.converged
    print(f"  iterations={result.iterations} converged={result.converged}")
    print(f"  final_max_delta={result.final_max_delta:.6f}")


def test_recorder_emits_one_event_per_affected() -> None:
    banner("test: recorder emits one ALLOCATION_SHIFTED per affected tendency")
    state = make_simple_state()
    recorder = LineageRecorder(clock=EngineClock(), graph=state.graph)
    for tid in state.tendencies.ids():
        recorder.register(tid, state.lineages[tid])

    clock_before = recorder.clock.now()
    result = reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(id="alpha",
                         new_tendency=Tendency(id="alpha", allocation=0.7)),
        ],
        recorder=recorder,
    )

    n_emitted = recorder.clock.now() - clock_before
    assert n_emitted == len(result.affected_ids), (
        f"expected one event per affected ({len(result.affected_ids)}), "
        f"clock advanced by {n_emitted}"
    )
    # Each affected tendency's outbox got exactly one new event of the
    # right type.
    for tid in result.affected_ids:
        ev = state.lineages[tid].latest_event()
        assert ev is not None
        assert ev.type == EventType.ALLOCATION_SHIFTED
        assert ev.origin_id == tid
    print(f"  affected ids: {result.affected_ids}")
    print(f"  clock advanced by {n_emitted} (== len(affected_ids))")


def test_reseed_is_deterministic() -> None:
    banner("test: same input -> same output, twice in a row")
    s1 = make_simple_state()
    s2 = make_simple_state()  # fresh deep-equal copy

    subs = [Substitution(id="beta",
                         new_tendency=Tendency(id="beta", allocation=0.6))]
    r1 = reseed_and_equilibrate(s1, substitutions=subs)
    r2 = reseed_and_equilibrate(s2, substitutions=subs)

    a1 = {t.id: t.allocation for t in r1.state.tendencies.all()}
    a2 = {t.id: t.allocation for t in r2.state.tendencies.all()}
    for tid in a1:
        assert abs(a1[tid] - a2[tid]) < 1e-12, (tid, a1[tid], a2[tid])
    assert r1.iterations == r2.iterations
    assert r1.converged == r2.converged
    print(f"  identical: iterations={r1.iterations}, "
          f"max_delta={r1.final_max_delta:.2e}")


def test_substitution_with_preserved_edges() -> None:
    banner("test: substitution without explicit edges keeps prior neighborhood")
    state = make_simple_state()
    pre_alpha_edges = dict(state.graph.weights["alpha"])

    result = reseed_and_equilibrate(
        state,
        substitutions=[
            Substitution(id="alpha",
                         new_tendency=Tendency(id="alpha", allocation=0.3)),
            # edges=None -> preserve existing
        ],
    )

    post_alpha_edges = dict(result.state.graph.weights["alpha"])
    assert post_alpha_edges == pre_alpha_edges, (
        f"expected preserved edges {pre_alpha_edges}, got {post_alpha_edges}"
    )
    print(f"  alpha's edges preserved: {post_alpha_edges}")


def test_invalid_learning_rate_rejected() -> None:
    banner("test: learning_rate outside (0,1) is rejected")
    state = make_simple_state()
    for bad_lr in [0.0, 1.0, -0.1, 1.5]:
        try:
            reseed_and_equilibrate(state, substitutions=[], learning_rate=bad_lr)
        except ValueError:
            print(f"  lr={bad_lr}: rejected as expected")
            continue
        raise AssertionError(f"learning_rate={bad_lr} should have been rejected")


if __name__ == "__main__":
    test_inputs_are_not_mutated()
    test_trivial_reseed_is_near_identity()
    test_substitute_replaces_tendency()
    test_substitute_adds_new_tendency()
    test_substitute_removes_tendency()
    test_iteration_cap_respected()
    test_recorder_emits_one_event_per_affected()
    test_reseed_is_deterministic()
    test_substitution_with_preserved_edges()
    test_invalid_learning_rate_rejected()
    print()
    print("All reseed smoke tests passed.")
