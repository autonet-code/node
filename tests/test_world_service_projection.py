"""Phase 4 tests: daemon-side epoch projection surface.

Validates:
  1. ``close_epoch`` writes a durable JSON record under
     ``<data_root>/<rpb>/epochs/<epoch_id>.json``.
  2. Records are tagged ``authoritative=False`` (Phase 5 will stamp
     true on the federated canonical version).
  3. Subscribers registered via ``subscribe_epoch_closed`` are invoked
     with each close. Subscriber failures don't break the close path
     or other subscribers.
  4. ``read_epoch`` and ``read_recent_epochs`` return both in-memory
     and on-disk records; ``read_agent_projection`` aggregates per
     agent.
  5. Records survive daemon restart — a fresh WorldService can read
     epoch records written by a prior process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import Observation


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _obs(label, charter=(0.0, 0.0, 0.0, 0.0), embed_idx=0, mag=0.5):
    return Observation(
        id=f"obs_{label}",
        coords=_coord(charter, embed_idx, mag),
        label=label,
    )


# ---------------------------------------------------------------------------
# On-disk persistence
# ---------------------------------------------------------------------------


def test_close_writes_durable_record_to_disk(tmp_path: Path):
    rpb = "rpb_proj_a"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        svc.open_epoch("e_one")
        svc.submit_observation(_obs("foo", charter=(0, 0, 0.6, 0)), agent_id="A")
        svc.close_epoch()
    finally:
        svc.shutdown()

    record_path = tmp_path / rpb / "epochs" / "e_one.json"
    assert record_path.exists(), f"missing record at {record_path}"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["epoch_id"] == "e_one"
    assert payload["rpb_address"] == rpb
    assert payload["authoritative"] is False
    assert payload["scope"] == "local"
    assert "agent_mint" in payload


def test_record_tagged_not_authoritative(tmp_path: Path):
    """Phase 4's records are projection-level. Authoritative flag only
    flips when Phase 5 federation lands the canonical version."""
    svc = WorldService(rpb_address="rpb_proj_b", data_root=tmp_path)
    try:
        svc.open_epoch()
        svc.close_epoch()
        recent = svc.read_recent_epochs(n=1)
        assert len(recent) == 1
        assert recent[0]["authoritative"] is False
    finally:
        svc.shutdown()


def test_records_survive_restart(tmp_path: Path):
    """A fresh WorldService can read epoch records that a prior process
    wrote. Important for agents that connect after-the-fact."""
    rpb = "rpb_proj_c"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        for i in range(3):
            svc.open_epoch(f"e_{i}")
            svc.submit_observation(
                _obs(f"x{i}", charter=(0, 0, 0.6, 0), embed_idx=i),
                agent_id="A",
            )
            svc.close_epoch()
    finally:
        svc.shutdown()

    # Restart.
    svc2 = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        # The new instance hasn't seen any closes in-memory, but should
        # find the prior process's records on disk.
        assert svc2.read_epoch("e_0") is not None
        recent = svc2.read_recent_epochs(n=10)
        assert len(recent) == 3
        ids = {r["epoch_id"] for r in recent}
        assert ids == {"e_0", "e_1", "e_2"}
    finally:
        svc2.shutdown()


# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------


def test_subscriber_invoked_on_close(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_proj_d", data_root=tmp_path)
    received: List[Dict[str, Any]] = []
    try:
        svc.subscribe_epoch_closed(received.append)
        svc.open_epoch("eA")
        svc.submit_observation(_obs("x", charter=(0, 0, 0.6, 0)), agent_id="A")
        svc.close_epoch()

        assert len(received) == 1
        assert received[0]["epoch_id"] == "eA"
        assert received[0]["authoritative"] is False
        assert "agent_mint" in received[0]
    finally:
        svc.shutdown()


def test_subscriber_failure_isolated(tmp_path: Path):
    """A bad subscriber must not break the close, nor stop other
    subscribers from being notified."""
    svc = WorldService(rpb_address="rpb_proj_e", data_root=tmp_path)
    saw: List[Dict[str, Any]] = []

    def _bad(_):
        raise RuntimeError("nope")

    try:
        svc.subscribe_epoch_closed(_bad)
        svc.subscribe_epoch_closed(saw.append)
        svc.open_epoch("eX")
        svc.close_epoch()
        # The bad handler ran first and raised, but the close didn't
        # crash and the second subscriber still got the record.
        assert len(saw) == 1
        assert saw[0]["epoch_id"] == "eX"
    finally:
        svc.shutdown()


def test_unsubscribe_stops_notifications(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_proj_f", data_root=tmp_path)
    saw: List[Dict[str, Any]] = []
    try:
        svc.subscribe_epoch_closed(saw.append)
        svc.open_epoch()
        svc.close_epoch()
        assert len(saw) == 1

        svc.unsubscribe_epoch_closed(saw.append)
        svc.open_epoch()
        svc.close_epoch()
        # No new notification.
        assert len(saw) == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def test_read_agent_projection_aggregates_across_epochs(tmp_path: Path):
    """Sum of an agent's per-epoch novelty across the last N epochs.
    Prose activity earns no mint (one rail: tools only) — it shows up
    as the novelty diagnostic."""
    svc = WorldService(rpb_address="rpb_proj_g", data_root=tmp_path)
    try:
        for i in range(4):
            svc.open_epoch(f"e_{i}")
            svc.submit_observation(
                _obs(f"a{i}", charter=(0, 0, 0.6, 0), embed_idx=i),
                agent_id="A",
            )
            svc.close_epoch()

        proj = svc.read_agent_projection("A", last_n_epochs=10)
        assert proj["agent_id"] == "A"
        assert proj["epochs_considered"] == 4
        # Prose never mints; the diagnostic (novelty) aggregates.
        assert proj["total_mint_projection"] == 0.0
        assert proj["total_novelty_projection"] > 0
        per_epoch_sum = sum(e["novelty"] for e in proj["per_epoch"])
        assert proj["total_novelty_projection"] == pytest.approx(per_epoch_sum)
        # All projections are non-authoritative until Phase 5.
        for e in proj["per_epoch"]:
            assert e["authoritative"] is False
    finally:
        svc.shutdown()


def test_read_agent_projection_zero_for_unknown_agent(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_proj_h", data_root=tmp_path)
    try:
        svc.open_epoch()
        svc.submit_observation(_obs("known", charter=(0, 0, 0.6, 0)), agent_id="A")
        svc.close_epoch()

        proj = svc.read_agent_projection("non-existent")
        assert proj["total_mint_projection"] == 0.0
        # Still returns the per-epoch breakdown, with mint=0 entries.
        assert proj["epochs_considered"] >= 1
        assert all(e["mint"] == 0.0 for e in proj["per_epoch"])
    finally:
        svc.shutdown()


def test_unsafe_epoch_id_doesnt_escape_directory(tmp_path: Path):
    """Phase 5 will accept epoch ids over the wire — the persistence
    layer must sanitize them so a malicious/malformed id can't write
    outside the rpb's epochs dir."""
    rpb = "rpb_proj_safety"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    try:
        svc.open_epoch("../../etc/passwd\x00")
        svc.close_epoch()
    finally:
        svc.shutdown()

    base = tmp_path / rpb / "epochs"
    # The record landed under base, not somewhere else.
    files = list(base.glob("*.json"))
    assert files, "no record was written"
    # Resolve to absolute and make sure they're all under base.
    base_resolved = base.resolve()
    for f in files:
        assert base_resolved in f.resolve().parents, \
            f"escape: {f.resolve()} not under {base_resolved}"
