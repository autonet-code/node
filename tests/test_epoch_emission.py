"""Fixed-emission epoch normalization (apply_emission_pool).

The economics flip: with a pool, an epoch mints exactly
``emission_rate x duration`` tokens TOTAL, split pro-rata by post-gate
contribution — junk takes share from co-contributors instead of
printing new supply.
"""
from __future__ import annotations

import pytest

from nodes.common.federated_reconcile import apply_emission_pool


def _result(agent_mint):
    return {
        "agent_mint": dict(agent_mint),
        "total_mint": sum(agent_mint.values()),
    }


class TestApplyEmissionPool:
    def test_total_equals_pool(self):
        out = apply_emission_pool(_result({"a": 3.0, "b": 1.0}), 100.0)
        assert out["total_mint"] == pytest.approx(100.0)
        assert sum(out["agent_mint"].values()) == pytest.approx(100.0)

    def test_proportions_preserved(self):
        out = apply_emission_pool(_result({"a": 3.0, "b": 1.0}), 100.0)
        assert out["agent_mint"]["a"] == pytest.approx(75.0)
        assert out["agent_mint"]["b"] == pytest.approx(25.0)

    def test_zero_activity_no_division(self):
        out = apply_emission_pool(_result({}), 100.0)
        assert out["agent_mint"] == {}
        assert out["raw_mint_total"] == 0.0
        assert out["emission_pool"] == 100.0

    def test_raw_total_recorded(self):
        out = apply_emission_pool(_result({"a": 3.0, "b": 1.0}), 100.0)
        assert out["raw_mint_total"] == pytest.approx(4.0)

    def test_spam_takes_share_not_supply(self):
        """The economics flip in one assertion: adding a spammer leaves
        total emission unchanged and reduces honest agents' mint."""
        honest = apply_emission_pool(_result({"a": 3.0, "b": 1.0}), 100.0)
        spammed = apply_emission_pool(
            _result({"a": 3.0, "b": 1.0, "spam": 4.0}), 100.0,
        )
        assert spammed["total_mint"] == pytest.approx(honest["total_mint"])
        assert spammed["agent_mint"]["a"] < honest["agent_mint"]["a"]

    def test_deterministic_across_key_order(self):
        r1 = apply_emission_pool(_result({"a": 3.0, "b": 1.0, "c": 2.0}), 7.0)
        r2 = apply_emission_pool(_result({"c": 2.0, "b": 1.0, "a": 3.0}), 7.0)
        assert r1["agent_mint"] == r2["agent_mint"]
        assert list(r1["agent_mint"]) == sorted(r1["agent_mint"])


class TestFederatedPoolDeterminism:
    def test_quiet_epoch_carries_the_whole_pool(self):
        from nodes.common.canonical_ordering import canonical_order
        from nodes.common.federated_reconcile import federated_epoch_close

        out = federated_epoch_close(canonical_order([]), emission_pool=25.0)
        assert out["total_mint"] == 0.0
        assert out["agent_mint"] == {}
        assert out["tool_rounds"]["pool_carry"] == 25_000_000


# ---------------------------------------------------------------------------
# Candle close
# ---------------------------------------------------------------------------

class _FakeLibrary:
    """Minimal stand-in for scheduler tests."""

    def __init__(self):
        self.epoch_history = []
        self.current_epoch_id = None
        self.closed_with = []
        self._n = 0

    def open_epoch(self, epoch_id=None):
        self._n += 1
        self.current_epoch_id = f"e{self._n}"

    def close_epoch(self, *, cutoff_ts=None):
        self.closed_with.append(cutoff_ts)
        record = {"epoch_id": self.current_epoch_id, "closed_at": 123.0,
                  "n_events": 0}
        self.epoch_history.append(record)
        self.current_epoch_id = None
        return record


class TestCandleScheduler:
    def _scheduler(self, min_s, window_s):
        from nodes.common.epoch_scheduler import EpochScheduler, EpochSchedulerConfig
        ws = _FakeLibrary()
        sched = EpochScheduler(
            ws,
            config=EpochSchedulerConfig(
                interval_seconds=1.0,
                candle_min_seconds=min_s,
                candle_window_seconds=window_s,
            ),
        )
        return ws, sched

    def test_no_close_before_full_window(self):
        import time
        ws, sched = self._scheduler(min_s=3600, window_s=1800)
        # Epoch just opened; even past interval_seconds nothing closes.
        sched._last_open_at = time.time() - 3599
        assert sched.maybe_tick() is None
        sched._last_open_at = time.time() - (3600 + 1799)
        assert sched.maybe_tick() is None
        assert ws.closed_with == []

    def test_close_after_window_with_cut_in_bounds(self):
        import time
        ws, sched = self._scheduler(min_s=3600, window_s=1800)
        open_at = time.time() - (3600 + 1800 + 1)
        sched._last_open_at = open_at
        result = sched.maybe_tick()
        assert result is not None
        assert len(ws.closed_with) == 1
        t_cut = ws.closed_with[0]
        assert open_at + 3600 <= t_cut <= open_at + 3600 + 1800

    def test_cut_is_deterministic_for_same_history(self):
        import time
        ws1, s1 = self._scheduler(3600, 1800)
        ws2, s2 = self._scheduler(3600, 1800)
        t = time.time() - 7200
        s1._last_open_at = t
        s2._last_open_at = t
        s1.maybe_tick()
        s2.maybe_tick()
        # Same (prev history, epoch id, open time) -> same cut.
        assert ws1.closed_with == ws2.closed_with

    def test_plain_mode_unaffected(self):
        import time
        from nodes.common.epoch_scheduler import EpochScheduler, EpochSchedulerConfig
        ws = _FakeLibrary()
        sched = EpochScheduler(ws, config=EpochSchedulerConfig(interval_seconds=60))
        sched._last_open_at = time.time() - 61
        result = sched.maybe_tick()
        assert result is not None
        assert ws.closed_with == [None]    # no cutoff in plain mode


class TestLibraryCandleCutoff:
    def test_post_cut_events_roll_forward(self, tmp_path):
        import time
        from nodes.common.library_service import LibraryService
        svc = LibraryService(rpb_address="candle-test", data_root=tmp_path)
        svc.open_epoch("e1")
        with svc._lock:
            svc._buffer_locked([{"kind": "a"}, {"kind": "b"}, {"kind": "c"}])
        now = time.time()
        # Backdate the first two events; the third stays "after the cut".
        svc._epoch_event_ts = [now - 100, now - 90, now + 100]
        record = svc.close_epoch(cutoff_ts=now)
        assert record["n_events"] == 2
        # Rolled event seeds the next epoch's buffer.
        svc.open_epoch("e2")
        assert len(svc._epoch_events) == 1
        assert svc._epoch_events[0]["kind"] == "c"
        assert svc.epoch_history[-1]["events_rolled_forward"] == 1
