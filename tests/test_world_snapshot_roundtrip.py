"""Snapshot round-trip verification harness (task: fast-boot prerequisite).

Proves, on the production replay rail (``apply_events`` over substrate
event dicts), that ``snapshot_world``/``restore_world`` is exact:

  replay(head) -> snapshot -> restore        == replay(head)
  then replay(tail) onto BOTH worlds         -> still bit-equal

If these hold, a checkpoint written at event offset K plus replay of
events K..N is indistinguishable from a full replay of 0..N — which is
the safety argument for checkpoint-restore boot in world_persistence.

The last test replays the real archived daemon event log if present
(skipped elsewhere); the synthetic tests carry the CI guarantee.
"""

import json
from pathlib import Path

import pytest

from nodes.common.world_model_substrate.adapter import (
    build_charter_world,
    train_world_model_on_task,
)
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.events import snapshot_node_scores
from world_model.generalized import restore_world, snapshot_world, worlds_equal


def _synthetic_events():
    """Generate substrate events the same way the live feed does:
    turns -> observations -> equilibrate -> recorded sprouts.
    Explicit impact fields keep this independent of embedders.
    """
    turns = []
    for i in range(12):
        turns.append({
            "label": f"turn_{i}",
            "life_impact": 0.8 if i % 3 == 0 else -0.4,
            "self_pres_impact": 0.5 * ((-1) ** i),
            "intelligence_impact": 0.6 if i % 2 else 0.0,
            "evolution_impact": 0.3,
            "correctness_impact": -0.5 if i % 4 == 0 else 0.4,
            "simplicity_impact": 0.2,
        })
    contribution, _metrics = train_world_model_on_task(
        {"turns": turns}, agent_id="harness-agent",
    )
    events = contribution["events"]
    assert len(events) >= 12, "fixture produced too few events to be meaningful"
    return events


def _replay(events_batches, embedding_dim=0):
    world = build_charter_world(bandwidth=1.5, embedding_dim=embedding_dim)
    for batch in events_batches:
        if batch:
            apply_events(world, batch)
    return world


@pytest.fixture(scope="module")
def event_split():
    events = _synthetic_events()
    cut = len(events) // 2
    # Split into batches of 4 to mimic feed-cycle batching.
    def batched(evs):
        return [evs[i:i + 4] for i in range(0, len(evs), 4)]
    return batched(events[:cut]), batched(events[cut:])


def test_snapshot_restore_exact_after_replay(event_split):
    head, _tail = event_split
    world = _replay(head)
    restored = restore_world(snapshot_world(world))
    assert worlds_equal(world, restored)
    assert snapshot_node_scores(world) == snapshot_node_scores(restored)


def test_restored_world_replays_tail_identically(event_split):
    head, tail = event_split
    world = _replay(head)
    restored = restore_world(snapshot_world(world))

    for batch in tail:
        apply_events(world, batch)
        apply_events(restored, batch)
        assert worlds_equal(world, restored)

    assert snapshot_node_scores(world) == snapshot_node_scores(restored)
    assert world.root_scores() == restored.root_scores()


def test_checkpoint_equals_full_replay(event_split):
    """The actual fast-boot claim: snapshot-at-K + tail replay equals
    a single full replay from scratch."""
    head, tail = event_split
    checkpointed = restore_world(snapshot_world(_replay(head)))
    for batch in tail:
        apply_events(checkpointed, batch)

    full = _replay(head + tail)
    assert worlds_equal(checkpointed, full)


def test_snapshot_payload_is_json_safe(event_split):
    head, _tail = event_split
    world = _replay(head)
    payload = json.loads(json.dumps(snapshot_world(world)))
    assert worlds_equal(world, restore_world(payload))


_ARCHIVED_LOG = (
    Path.home() / ".autonet" / "world" / "default"
    / "events.jsonl.archived-20260610-194930"
)


@pytest.mark.skipif(not _ARCHIVED_LOG.exists(), reason="no archived daemon log")
def test_real_log_checkpoint_matches_full_replay():
    """Replay a slice of the real daemon event log, checkpoint midway,
    and verify the restored world finishes the replay identically.
    Honors __equilibrate__ markers the way try_restore does.
    """
    lines = _ARCHIVED_LOG.read_text(encoding="utf-8").splitlines()[:120]
    batches, batch = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("kind") == "__equilibrate__":
            if batch:
                batches.append(batch)
                batch = []
        else:
            batch.append(entry)
    if batch:
        batches.append(batch)
    assert len(batches) >= 2, "need at least two batches to checkpoint midway"

    cut = len(batches) // 2
    world = _replay(batches[:cut], embedding_dim=64)
    restored = restore_world(snapshot_world(world))

    for b in batches[cut:]:
        apply_events(world, b)
        apply_events(restored, b)

    assert worlds_equal(world, restored)
    assert snapshot_node_scores(world) == snapshot_node_scores(restored)
