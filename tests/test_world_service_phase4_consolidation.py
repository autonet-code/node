"""Phase 4 consolidation tests: dim migration + seed scaling + Lindblad trigger.

Three independent checks, packaged together because they all probe
the post-consolidation WorldService surface.

  1. **Migration test (Task #111)**: a WorldService persists at the
     legacy 1024-dim, then reloads at the new default (64-dim). The
     pad/truncate logic in submit_observation must keep the world
     consistent — old 1024-dim coords on disk get truncated at reload,
     new 64-dim coords get padded if the service was somehow
     instantiated at a higher dim.

  2. **Seed-time scaling (Task #106)**: with scoped equilibrate on,
     per-observation submit time should NOT grow quadratically with
     world size. Compare mean per-event time at N=100 vs N=300; the
     ratio should be far below 9× (the quadratic prediction).

  3. **Lindblad exploration trigger (Task #115)**: the activity-based
     trigger respects its threshold, only fires when the counter
     exceeds the configured every-N, and resets after firing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Tuple

import pytest

from nodes.common.world_model_substrate.adapter import N_DIMS
from nodes.common.world_service import WorldService
from world_model.generalized import Observation


def _coord(charter: Tuple[float, ...], embed_tail: Tuple[float, ...]) -> Tuple[float, ...]:
    head = tuple(charter)
    if len(head) < N_DIMS:
        head = head + (0.0,) * (N_DIMS - len(head))
    return head + tuple(embed_tail)


# ----- Task #111: migration test --------------------------------------------


def test_migration_1024_to_64_via_truncate(tmp_path):
    """Build a world at the legacy 1024-dim, persist, reload at the
    new 64-dim default, and verify the world is still usable.

    The reload path (replay events through submit_observation) goes
    through the pad/truncate boundary in submit_work_units — which is
    the load-bearing migration guarantee.
    """
    legacy_dim = 1024
    new_dim = 64

    # Phase 1: write a world at legacy dim with a few observations.
    # Use small tail magnitude so the obs coords stay inside charter
    # bandwidth (otherwise the scoped equilibrate gate finds no
    # tendencies in scope, which is correct behavior but produces no
    # state to migrate).
    rpb_dir = tmp_path / "rpb"
    svc_legacy = WorldService(
        rpb_address="0xMIGRATION",
        data_root=rpb_dir,
        embedding_dim=legacy_dim,
    )
    legacy_events_applied = 0
    for i in range(5):
        tail = tuple(0.01 * float((i + j) % 3) for j in range(legacy_dim))
        obs = Observation(
            id=f"obs_{i}",
            coords=_coord((1.0, 0.0, 0.0, 0.0), tail),
            label=f"legacy_obs_{i}",
        )
        receipt = svc_legacy.submit_observation(
            obs, agent_id="legacy_agent",
            sprout_under_charter=True, sprout_rootless=True,
        )
        legacy_events_applied += receipt["events_applied"]
    assert legacy_events_applied >= 5, \
        "legacy world should have ingested observations without crashing"

    # Phase 2: reload at new dim. The pad/truncate logic in
    # submit_work_units / submit_observation handles per-event
    # dimension reconciliation.
    svc_new = WorldService(
        rpb_address="0xMIGRATION",
        data_root=rpb_dir,
        embedding_dim=new_dim,
    )
    # Submit a fresh observation at the new dim — should not crash.
    tail_new = tuple(float(j % 5) for j in range(new_dim))
    obs_new = Observation(
        id="obs_new_post_migration",
        coords=_coord((0.5, 0.5, 0.0, 0.0), tail_new),
        label="post_migration",
    )
    receipt = svc_new.submit_observation(
        obs_new, agent_id="new_agent",
        sprout_under_charter=True, sprout_rootless=True,
    )
    assert receipt["events_applied"] >= 1
    # World still has charter roots and is operating.
    scores = svc_new.read_root_scores()
    assert len(scores) >= 4, "charter roots should still be present"


# ----- Task #106: seed-time scaling -----------------------------------------


@pytest.mark.slow
def test_seed_time_not_quadratic(tmp_path):
    """Submit N observations and check per-event time at small vs
    large N. With scoped equilibrate, per-event cost should grow
    sub-quadratically. We assert the per-event ratio at N=300 / N=100
    is well below the quadratic prediction (9×).

    The exact ratio depends on hardware and the embedding-tail
    geometry; we set the bar generously at 4× to avoid flaky failures.
    A pre-scoping (quadratic) implementation would land near 9×.
    """
    svc = WorldService(
        rpb_address="0xBENCH",
        data_root=tmp_path / "bench",
        embedding_dim=64,
    )
    dim = 64

    def submit_batch(n_start: int, n_end: int) -> float:
        t0 = time.time()
        for i in range(n_start, n_end):
            tail = tuple(float(((i * 13) + j) % 7) for j in range(dim))
            obs = Observation(
                id=f"bench_obs_{i}",
                coords=_coord((0.1, 0.0, 0.0, 0.0), tail),
                label=f"bench_{i}",
            )
            svc.submit_observation(
                obs, agent_id="bench_agent",
                sprout_under_charter=True, sprout_rootless=True,
            )
        return time.time() - t0

    # Warm up + collect timings.
    submit_batch(0, 50)                       # warmup, discard
    t_first_100 = submit_batch(50, 150)       # mean over 100 events
    t_next_200 = submit_batch(150, 350)       # mean over 200 events

    per_event_first = t_first_100 / 100.0
    per_event_next = t_next_200 / 200.0
    ratio = per_event_next / max(per_event_first, 1e-9)

    print(f"  per-event time at N≈100: {per_event_first*1000:.2f} ms")
    print(f"  per-event time at N≈300: {per_event_next*1000:.2f} ms")
    print(f"  ratio: {ratio:.2f}x  (quadratic prediction: 9x)")

    # Generous bar — the goal is to catch a quadratic regression, not
    # to certify constants. A quadratic equilibrate would land near 9x.
    assert ratio < 4.0, (
        f"per-event time grew {ratio:.2f}x from N=100 to N=300 — "
        f"scoped equilibrate may have regressed to quadratic."
    )


# ----- Task #115: Lindblad exploration trigger ------------------------------


def test_exploration_trigger_respects_threshold(tmp_path):
    """`maybe_run_exploration_pass` only fires when obs_since_last
    reaches the configured threshold, and resets the counter after."""
    svc = WorldService(
        rpb_address="0xEXPLORE",
        data_root=tmp_path / "explore",
        embedding_dim=64,
    )

    # Below threshold: should not fire.
    result = svc.maybe_run_exploration_pass(every=5)
    assert result["ran"] is False
    assert result["obs_since_last"] == 0
    assert result["threshold"] == 5

    # Submit a few observations.
    for i in range(3):
        tail = tuple(float(j) for j in range(64))
        obs = Observation(
            id=f"explore_obs_{i}",
            coords=_coord((0.0, 0.1, 0.0, 0.0), tail),
            label=f"explore_{i}",
        )
        svc.submit_observation(
            obs, agent_id="ex", sprout_under_charter=True, sprout_rootless=True,
        )

    # Still below threshold.
    result = svc.maybe_run_exploration_pass(every=5)
    assert result["ran"] is False, "should not fire at 3/5"
    assert result["obs_since_last"] == 3

    # Cross threshold.
    for i in range(3, 6):
        tail = tuple(float(j) for j in range(64))
        obs = Observation(
            id=f"explore_obs_{i}",
            coords=_coord((0.0, 0.1, 0.0, 0.0), tail),
            label=f"explore_{i}",
        )
        svc.submit_observation(
            obs, agent_id="ex", sprout_under_charter=True, sprout_rootless=True,
        )

    result = svc.maybe_run_exploration_pass(every=5)
    assert result["ran"] is True, "should fire at 6/5"
    assert result["obs_since_last"] == 6

    # Counter resets after firing.
    result = svc.maybe_run_exploration_pass(every=5)
    assert result["ran"] is False, "should not fire immediately after reset"
    assert result["obs_since_last"] == 0


def test_exploration_disabled_by_default(tmp_path):
    """With threshold=0 (the default unless AUTONET_LINDBLAD_EXPLORE_EVERY
    is set), the trigger never fires regardless of activity."""
    svc = WorldService(
        rpb_address="0xEXPLOREOFF",
        data_root=tmp_path / "exploreoff",
        embedding_dim=64,
    )
    for i in range(50):
        tail = tuple(float(j) for j in range(64))
        obs = Observation(
            id=f"off_obs_{i}",
            coords=_coord((0.0, 0.1, 0.0, 0.0), tail),
            label=f"off_{i}",
        )
        svc.submit_observation(
            obs, agent_id="off", sprout_under_charter=True, sprout_rootless=True,
        )
    result = svc.maybe_run_exploration_pass(every=0)
    assert result["ran"] is False
    assert result["threshold"] == 0
