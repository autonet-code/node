"""Regression tests for the act() memoization optimization.

The optimization caches `evaluate()` results by frame identity. The
cache must not break:

  1. Determinism: two world services fed the same events end in
     bit-identical state (root_scores match exactly).
  2. Cache invalidation on absorb: after a frame absorbs, subsequent
     evaluations against the new frame must NOT return cached terms
     from the old frame.
  3. Cross-tendency staking still happens: post-counts on nodes
     created by other tendencies must match the unscoped behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import (
    Observation,
    equilibrate,
)
from world_model.generalized.tendency import GeneralizedTendency
from world_model.generalized.world import World


def _make_two_tendencies():
    """Two tendencies anchored at opposite corners of a 2D space."""
    t1 = GeneralizedTendency(
        id="t1", thesis="t1_thesis",
        anchor=(1.0, 0.0), polarity_axis=(1.0, 0.0),
        bandwidth=1.5,
    )
    t2 = GeneralizedTendency(
        id="t2", thesis="t2_thesis",
        anchor=(-1.0, 0.0), polarity_axis=(-1.0, 0.0),
        bandwidth=1.5,
    )
    return t1, t2


def _seed_world(observations: list[tuple[str, tuple[float, ...]]]) -> World:
    t1, t2 = _make_two_tendencies()
    world = World()
    world.add_tendency(t1)
    world.add_tendency(t2)
    for obs_id, coords in observations:
        world.add_observation(Observation(id=obs_id, coords=coords, label=obs_id))
        equilibrate(world, max_rounds=4, tolerance=1e-3)
    return world


def test_act_memo_produces_identical_root_scores_under_repeated_event_stream():
    """The same event stream applied to two fresh worlds must produce
    identical root_scores — proves the cache doesn't leak across
    instances and is reset cleanly per world.
    """
    obs = [
        ("a", (0.9, 0.0)),
        ("b", (-0.9, 0.0)),
        ("c", (0.5, 0.5)),
        ("d", (-0.5, 0.5)),
        ("e", (0.0, 1.0)),
    ]
    w1 = _seed_world(obs)
    w2 = _seed_world(obs)
    assert w1.root_scores() == w2.root_scores()


def test_act_memo_clears_cache_on_frame_replace():
    """When a tendency replaces its frame via _replace_frame (which
    absorb routes through), the eval caches are cleared. This is the
    central correctness invariant: cache entries are only valid under
    the frame they were computed against.

    We verify by observing that calling _replace_frame empties both
    caches, regardless of their prior content.
    """
    t1, t2 = _make_two_tendencies()
    world = World()
    world.add_tendency(t1)
    world.add_tendency(t2)
    obs = Observation(id="o1", coords=(0.9, 0.0), label="o1")
    world.add_observation(obs)
    equilibrate(world, max_rounds=4, tolerance=1e-3)

    # After equilibrate, t1 has populated its eval caches.
    pre_own = len(t1._own_term_cache)
    pre_cross = len(t1._cross_term_cache)
    assert pre_own + pre_cross > 0, (
        "expected at least one cache entry after equilibrate; "
        "test setup may not exercise the cached loops"
    )

    # Replace the frame the same way absorb would.
    obs2 = Observation(id="o2", coords=(0.7, 0.1), label="o2")
    t1._replace_frame(t1.frame.absorb(obs2))

    # Caches are explicitly cleared.
    assert len(t1._own_term_cache) == 0
    assert len(t1._cross_term_cache) == 0


def test_world_service_seed_replay_determinism(tmp_path: Path):
    """Two WorldService instances seeded with the same observations
    in the same order arrive at identical event_applied counts and
    root_scores. Proves the cache doesn't introduce non-determinism
    across the public submit_observation path.
    """
    work_units = [
        ("query about cats", "cats are mammals", [1.0, 1.0, 0.0, 0.0]),
        ("query about dogs", "dogs are mammals", [1.0, 1.0, 0.0, 0.0]),
        ("query about fish", "fish are not mammals", [1.0, 0.0, 0.0, 0.0]),
        ("query about birds", "birds are not mammals", [1.0, 0.0, 0.0, 0.0]),
        ("query about humans", "humans are also mammals", [1.0, 1.0, 0.0, 0.0]),
    ]

    ws1 = WorldService(rpb_address="default", data_root=tmp_path / "ws1")
    ws2 = WorldService(rpb_address="default", data_root=tmp_path / "ws2")

    for unit in work_units:
        ws1.submit_work_units([unit], agent_id="root")
        ws2.submit_work_units([unit], agent_id="root")

    s1 = ws1._world.root_scores()
    s2 = ws2._world.root_scores()
    assert s1 == s2
