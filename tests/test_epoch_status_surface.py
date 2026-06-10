"""Epoch observability surfaces (EPOCH_OBSERVABILITY_SPEC.md):

  - EpochScheduler.status()  — timing half (mode, T_max, seed source)
  - WorldService.epoch_status() — world half (epoch id, buffer, emission)
  - the combined dict the ws `epoch_status` handler assembles

The federated-close push event payload is covered implicitly by
test_state_sync (FederatedCloseResult fields) — here we only pin the
service-level halves the ws handlers read.
"""

import time

import pytest

from nodes.common.epoch_scheduler import EpochScheduler, EpochSchedulerConfig
from nodes.common.world_service import WorldService


class _StubWS:
    current_epoch_id = "e_stub"
    epoch_history: list = []

    def open_epoch(self, *a, **k):
        return {}

    def close_epoch(self, *a, **k):
        return {"epoch_id": "e_stub"}


def _candle_scheduler(seed_source=None):
    return EpochScheduler(
        _StubWS(),
        config=EpochSchedulerConfig(
            candle_min_seconds=100.0,
            candle_window_seconds=50.0,
            open_first_epoch_on_start=False,
        ),
        candle_seed_source=seed_source,
    )


class TestSchedulerStatus:
    def test_candle_mode_fields(self):
        sched = _candle_scheduler()
        sched._last_open_at = 1000.0
        s = sched.status()
        assert s["mode"] == "candle"
        assert s["opened_at"] == 1000.0
        assert s["t_max"] == 1150.0          # open + min + window
        assert s["candle_min_seconds"] == 100.0
        assert s["candle_window_seconds"] == 50.0
        assert s["interval_seconds"] is None
        assert s["seed_source"] == "local"

    def test_chain_seed_reported(self):
        sched = _candle_scheduler(seed_source=lambda window_end: None)
        assert sched.status()["seed_source"] == "chain"

    def test_interval_mode_fields(self):
        sched = EpochScheduler(
            _StubWS(),
            config=EpochSchedulerConfig(
                interval_seconds=60.0, open_first_epoch_on_start=False,
            ),
        )
        sched._last_open_at = 2000.0
        s = sched.status()
        assert s["mode"] == "interval"
        assert s["t_max"] == 2060.0
        assert s["interval_seconds"] == 60.0
        assert s["candle_min_seconds"] is None

    def test_no_open_epoch_yet(self):
        sched = _candle_scheduler()
        s = sched.status()
        assert s["opened_at"] is None
        assert s["t_max"] is None


class TestWorldServiceEpochStatus:
    def _service(self, tmp_path, rate=None):
        return WorldService(
            rpb_address="status-test",
            data_root=tmp_path,
            embedding_dim=0,
            epoch_emission_rate=rate,
        )

    def test_open_epoch_visible(self, tmp_path):
        svc = self._service(tmp_path, rate=1.0)
        svc.open_epoch("e_status_1")
        with svc._lock:
            svc._buffer_epoch_events_locked([{"kind": "a"}, {"kind": "b"}])
        s = svc.epoch_status()
        assert s["epoch_id"] == "e_status_1"
        assert s["buffered_events"] == 2
        assert s["opened_at"] == pytest.approx(time.time(), abs=5)
        assert s["emission_rate"] == 1.0
        assert s["emission_clock"] is not None
        assert s["rpb_address"] == "status-test"

    def test_between_epochs(self, tmp_path):
        svc = self._service(tmp_path)
        svc.open_epoch("e_status_2")
        svc.close_epoch(apply_gate=False)
        s = svc.epoch_status()
        assert s["epoch_id"] is None
        assert s["opened_at"] is None
        assert s["buffered_events"] == 0
        assert s["emission_rate"] is None

    def test_history_survives_restart(self, tmp_path):
        """Closed-epoch records rehydrate from disk on construction —
        with multi-day candle epochs, since-boot-only history would
        almost always render empty."""
        svc = self._service(tmp_path)
        svc.open_epoch("e_persist_1")
        svc.close_epoch(apply_gate=False)
        svc.open_epoch("e_persist_2")
        svc.close_epoch(apply_gate=False)
        svc.shutdown()

        reborn = self._service(tmp_path)
        ids = [r["epoch_id"] for r in reborn.epoch_history]
        assert ids == ["e_persist_1", "e_persist_2"]
        assert reborn.epoch_history[-1]["closed_at"] >= \
            reborn.epoch_history[0]["closed_at"]

    def test_combined_shape_matches_spec(self, tmp_path):
        """The ws handler merges the two halves; the merged dict must
        cover every field EPOCH_OBSERVABILITY_SPEC.md §2 documents."""
        svc = self._service(tmp_path, rate=0.5)
        svc.open_epoch("e_status_3")
        sched = _candle_scheduler()
        sched._last_open_at = svc.epoch_status()["opened_at"]

        epoch = svc.epoch_status()
        sched_half = sched.status()
        sched_half.pop("opened_at", None)
        epoch.update(sched_half)

        for field in (
            "epoch_id", "opened_at", "mode",
            "candle_min_seconds", "candle_window_seconds", "t_max",
            "interval_seconds", "buffered_events",
            "emission_rate", "emission_clock", "seed_source",
        ):
            assert field in epoch, field
        assert epoch["mode"] == "candle"
        assert epoch["epoch_id"] == "e_status_3"
        # T_max derives from the SAME opened_at the world reports.
        assert epoch["t_max"] == pytest.approx(epoch["opened_at"] + 150.0)
