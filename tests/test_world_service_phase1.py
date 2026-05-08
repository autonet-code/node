"""Phase 1 acceptance tests for the persistent WorldService.

What this proves
----------------

  1. A single ``WorldService`` survives multiple submit_events calls and
     accumulates state across them — i.e. it is NOT a per-task fresh-world.
  2. Snapshots are written atomically and a restart restores the world
     from disk, recovering both the snapshot and any tail events.
  3. submit_events with empty input is a no-op (does not corrupt state).
  4. The persistence layout puts files exactly where the plan says: under
     ``<root>/<rpb>/{snapshot.json, events.jsonl}``.

These tests are deliberately substrate-only: no chain, no ATN, no
network, no probes — those are Phases 4–6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nodes.common.world_model_substrate.adapter import (
    _EventRecorder,
    _all_node_ids,
    build_charter_world,
    turn_to_observation,
)
from nodes.common.world_service import WorldService
from world_model.generalized import equilibrate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events_for_turns(
    turns: List[Dict[str, Any]],
    *,
    agent_id: str = "test-agent",
    epoch_index: int = 0,
) -> List[Dict[str, Any]]:
    """Generate a deterministic event stream by running the engine in
    isolation against a charter world. We do this rather than hand-crafting
    events to ensure the events are realistic — same shape the adapter
    produces in production.
    """
    world = build_charter_world()
    recorder = _EventRecorder(agent_id=agent_id)
    for i, turn in enumerate(turns):
        obs = turn_to_observation(turn, turn_index=epoch_index * 10000 + i)
        before_ids = _all_node_ids(world)
        world.add_observation(obs)
        recorder.observation_added(obs)
        equilibrate(world, max_rounds=4, tolerance=1e-3)
        recorder.sub_claims_after_equilibrate(world, before_ids)
    return [e.to_dict() for e in recorder.events]


def _make_turns(scale: float = 1.0, count: int = 3) -> List[Dict[str, Any]]:
    """Synthetic turns with explicit charter-axis impacts, so the score
    movement is deterministic and visible without depending on the
    score_turn heuristic."""
    return [
        {
            "label": f"t{i}_scale{scale}",
            "life_impact": 0.6 * scale,
            "self_pres_impact": 0.4 * scale,
            "intelligence_impact": 0.5 * scale,
            "evolution_impact": 0.3 * scale,
            "_seed": i,
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_service_constructs_with_charter_roots(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_test_a", data_root=tmp_path)
    try:
        roots = svc.read_root_scores()
        assert set(roots.keys()) == {
            "life_precious",
            "self_preservation",
            "promotion_of_intelligence",
            "evolution",
            "correctness",
            "simplicity",
        }
    finally:
        svc.shutdown()


def test_submit_events_is_idempotent_on_empty(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_test_b", data_root=tmp_path)
    try:
        before = svc.read_root_scores()
        receipt = svc.submit_events([])
        assert receipt["events_applied"] == 0
        after = svc.read_root_scores()
        assert before == after
    finally:
        svc.shutdown()


def test_world_persists_across_submit_events_calls(tmp_path: Path):
    """Three sequential probes accumulate; the service is NOT a
    per-task fresh world. The number of nodes must increase across calls,
    not reset."""
    svc = WorldService(rpb_address="rpb_test_c", data_root=tmp_path)
    try:
        node_counts = []
        for batch in range(3):
            events = _events_for_turns(
                _make_turns(scale=1.0 + batch * 0.5, count=3),
                agent_id=f"agent-batch{batch}",
                epoch_index=batch,
            )
            svc.submit_events(events)
            node_counts.append(svc.stats()["n_nodes"])

        # Strictly monotone — each probe added at least one node.
        # (Content-addressed dedupe means equal observations across batches
        # would NOT add new nodes; we use distinct scales/labels above so
        # they don't collide.)
        assert node_counts[0] < node_counts[1] < node_counts[2], node_counts
    finally:
        svc.shutdown()


def test_restart_restores_state_from_snapshot(tmp_path: Path):
    """Build a world, snapshot it, restart, observe that the world is
    restored from disk rather than rebuilt fresh."""
    rpb = "rpb_restart"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    events = _events_for_turns(_make_turns(count=4), agent_id="agent-pre-snap")
    svc.submit_events(events)
    pre_root_scores = svc.read_root_scores()
    pre_nodes = svc.stats()["n_nodes"]
    svc.snapshot_now()
    svc.shutdown()

    # New service instance, same RPB & data root.
    svc2 = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        post_root_scores = svc2.read_root_scores()
        post_nodes = svc2.stats()["n_nodes"]
        # The fresh charter alone has 4 root nodes (one per axis).
        # Restoring from snapshot must preserve the additional nodes.
        assert post_nodes == pre_nodes
        assert post_root_scores == pytest.approx(pre_root_scores, rel=1e-9, abs=1e-9)
    finally:
        svc2.shutdown()


def test_restart_with_extra_events_after_snapshot(tmp_path: Path):
    """Snapshot caches scores but the event log is the source of truth.
    Events from multiple submit_events calls must all be replayed on
    restart, preserving batch boundaries (so equilibration history
    reproduces bit-for-bit)."""
    rpb = "rpb_tail"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    pre_events = _events_for_turns(_make_turns(count=3), agent_id="agent-snap")
    svc.submit_events(pre_events)
    svc.snapshot_now()

    tail_events = _events_for_turns(
        _make_turns(scale=2.0, count=2),
        agent_id="agent-tail",
        epoch_index=1,
    )
    svc.submit_events(tail_events)
    expected_nodes = svc.stats()["n_nodes"]
    expected_scores = svc.read_root_scores()
    svc.shutdown()

    # The events log must contain both batches' events, plus the
    # synthetic equilibrate markers that record where each
    # ``submit_events`` call ended.
    base = tmp_path / rpb
    events_log = base / "events.jsonl"
    assert events_log.exists()
    lines = [l for l in events_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(pre_events) + len(tail_events) + 2  # +2 markers

    svc2 = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        assert svc2.stats()["n_nodes"] == expected_nodes
        # With batch boundaries preserved on disk, scores reproduce
        # exactly. (They are deterministic given event order +
        # equilibrate call structure.)
        assert svc2.read_root_scores() == pytest.approx(
            expected_scores, rel=1e-9, abs=1e-9,
        )
    finally:
        svc2.shutdown()


def test_persistence_layout_matches_plan(tmp_path: Path):
    rpb = "rpb_layout"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        events = _events_for_turns(_make_turns(count=2), agent_id="agent-layout")
        svc.submit_events(events)
        svc.snapshot_now()
    finally:
        svc.shutdown()

    base = tmp_path / rpb
    scores_file = base / "scores.json"
    events_file = base / "events.jsonl"
    assert scores_file.exists(), f"missing {scores_file}"
    assert events_file.exists(), f"missing {events_file}"

    payload = json.loads(scores_file.read_text(encoding="utf-8"))
    assert payload["schema"] == 2
    assert payload["rpb_address"] == rpb
    assert "root_scores" in payload
    assert "events_applied" in payload


def test_unsafe_rpb_address_is_sanitized(tmp_path: Path):
    """RPB addresses can come from chain output; an attacker-influenced
    string mustn't escape the data root."""
    rpb = "../etc/passwd\x00:: weird"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        # The actual on-disk dir lives under tmp_path, not anywhere
        # outside it.
        snapshot_dir = svc._persistence._dir
        snapshot_dir_resolved = snapshot_dir.resolve()
        assert tmp_path.resolve() in snapshot_dir_resolved.parents, \
            f"escape detected: {snapshot_dir_resolved} not under {tmp_path}"
    finally:
        svc.shutdown()


def test_two_services_different_rpbs_dont_share_world(tmp_path: Path):
    """A node hosting two RPBs (later-phase scenario) must not bleed
    state between them."""
    rpb_a = "rpb_alpha"
    rpb_b = "rpb_beta"
    svc_a = WorldService(rpb_address=rpb_a, data_root=tmp_path)
    svc_b = WorldService(rpb_address=rpb_b, data_root=tmp_path)
    try:
        events_a = _events_for_turns(_make_turns(count=3), agent_id="agent-a")
        svc_a.submit_events(events_a)
        # B should have only the fresh charter (one node per root).
        from nodes.common.world_model_substrate.adapter import N_DIMS
        assert svc_b.stats()["n_nodes"] == N_DIMS
        # A should have more.
        assert svc_a.stats()["n_nodes"] > N_DIMS
    finally:
        svc_a.shutdown()
        svc_b.shutdown()
