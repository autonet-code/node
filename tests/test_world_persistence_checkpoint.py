"""Checkpoint-restore boot path in WorldPersistence.

Simulates the daemon's live flow (apply batch -> append to log ->
periodically write_snapshot) and verifies that try_restore's
checkpoint fast path lands on EXACTLY the world a full replay
produces — including when the checkpoint is stale (taken several
batches before shutdown) — and that every inconsistency falls back
to full replay instead of failing the boot.
"""

import json

import pytest

from nodes.common.world_model_substrate.adapter import (
    build_charter_world,
    train_world_model_on_task,
)
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_persistence import PersistenceConfig, WorldPersistence
from world_model.generalized import worlds_equal


def _event_batches():
    turns = []
    for i in range(10):
        turns.append({
            "label": f"turn_{i}",
            "life_impact": 0.7 if i % 3 == 0 else -0.3,
            "self_pres_impact": 0.4 * ((-1) ** i),
            "intelligence_impact": 0.5 if i % 2 else 0.0,
            "evolution_impact": 0.2,
            "correctness_impact": -0.4 if i % 4 == 0 else 0.3,
            "simplicity_impact": 0.1,
        })
    contribution, _ = train_world_model_on_task(
        {"turns": turns}, agent_id="ckpt-agent",
    )
    events = contribution["events"]
    return [events[i:i + 5] for i in range(0, len(events), 5)]


@pytest.fixture()
def persistence(tmp_path):
    cfg = PersistenceConfig(rpb_address="ckpt-test", data_root=tmp_path)
    p = WorldPersistence(cfg)
    yield p
    p.close()


def _live_run(persistence, batches, checkpoint_after=None):
    """Replicates WorldService's flow: apply, append, maybe snapshot.
    Returns the final live world and total event count.
    """
    world = build_charter_world(bandwidth=persistence.config.bandwidth,
                                embedding_dim=persistence.config.embedding_dim)
    applied = 0
    for i, batch in enumerate(batches):
        apply_events(world, batch)
        persistence.append_events(batch)
        applied += len(batch)
        if checkpoint_after is not None and i == checkpoint_after:
            persistence.write_snapshot(world, events_applied=applied)
    return world, applied


def test_checkpoint_fast_path_is_exact(persistence):
    batches = _event_batches()
    assert len(batches) >= 3
    # Checkpoint midway -> stale by the time the "daemon" stops.
    live, total = _live_run(persistence, batches,
                            checkpoint_after=len(batches) // 2)

    restored = persistence.try_restore()
    assert restored is not None
    assert restored.from_checkpoint
    assert restored.tail_events > 0
    assert restored.events_replayed == total
    assert worlds_equal(restored.world, live)


def test_checkpoint_at_tip_replays_nothing(persistence):
    batches = _event_batches()
    live, total = _live_run(persistence, batches,
                            checkpoint_after=len(batches) - 1)

    restored = persistence.try_restore()
    assert restored.from_checkpoint
    assert restored.tail_events == 0
    assert restored.events_replayed == total
    assert worlds_equal(restored.world, live)


def test_corrupt_checkpoint_falls_back_to_full_replay(persistence):
    batches = _event_batches()
    live, total = _live_run(persistence, batches, checkpoint_after=0)
    persistence.checkpoint_path.write_text("{not json", encoding="utf-8")

    restored = persistence.try_restore()
    assert restored is not None
    assert not restored.from_checkpoint
    assert restored.events_replayed == total
    assert worlds_equal(restored.world, live)


def test_foreign_checkpoint_rejected(persistence):
    batches = _event_batches()
    live, total = _live_run(persistence, batches, checkpoint_after=0)
    payload = json.loads(
        persistence.checkpoint_path.read_text(encoding="utf-8"))
    payload["rpb_address"] = "someone-else"
    persistence.checkpoint_path.write_text(
        json.dumps(payload), encoding="utf-8")

    restored = persistence.try_restore()
    assert not restored.from_checkpoint
    assert worlds_equal(restored.world, live)


def test_checkpoint_ahead_of_log_falls_back(persistence):
    batches = _event_batches()
    live, total = _live_run(persistence, batches, checkpoint_after=0)
    payload = json.loads(
        persistence.checkpoint_path.read_text(encoding="utf-8"))
    payload["events_applied"] = total + 999   # claims more than the log
    persistence.checkpoint_path.write_text(
        json.dumps(payload), encoding="utf-8")

    restored = persistence.try_restore()
    assert not restored.from_checkpoint
    assert worlds_equal(restored.world, live)


def test_no_log_no_checkpoint_returns_none(persistence):
    assert persistence.try_restore() is None
