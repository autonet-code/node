"""The basket split (federated_reconcile.split_pools).

Each key takes ``pool * w // W`` of every token's pool: the same fraction
of each, in pure integers, so the amounts never add up to more than a pool
and adding a spammer takes share from the others instead of creating
money.
"""
from __future__ import annotations

from nodes.common.federated_reconcile import (
    basket_value,
    integer_weights,
    split_pools,
    value_of,
)

T6, T18 = "0x" + "a6" * 20, "0x" + "1b" * 20


class TestSplitPools:
    def test_never_more_than_the_pool_and_dust_is_small(self):
        pools = {T6: 1_000_003, T18: 10 ** 18 + 1}
        out = split_pools(pools, {"a": 3, "b": 1, "c": 3})
        for t, p in pools.items():
            paid = sum(b.get(t, 0) for b in out.values())
            assert p - len(out) < paid <= p

    def test_proportions_are_the_same_in_every_token(self):
        out = split_pools({T6: 4_000_000, T18: 4 * 10 ** 18}, {"a": 3, "b": 1})
        assert out["a"] == {T6: 3_000_000, T18: 3 * 10 ** 18}
        assert out["b"] == {T6: 1_000_000, T18: 10 ** 18}

    def test_no_weight_pays_nothing(self):
        assert split_pools({T6: 100}, {}) == {}
        assert split_pools({}, {"a": 1}) == {}

    def test_spam_takes_share_not_supply(self):
        honest = split_pools({T6: 100_000}, {"a": 3, "b": 1})
        spammed = split_pools({T6: 100_000}, {"a": 3, "b": 1, "spam": 4})
        assert sum(b[T6] for b in spammed.values()) <= 100_000
        assert spammed["a"][T6] < honest["a"][T6]

    def test_deterministic_across_key_order(self):
        w1 = integer_weights({"a": 3.0, "b": 1.0, "c": 2.0}, 10)
        w2 = integer_weights({"c": 2.0, "b": 1.0, "a": 3.0}, 10)
        assert w1 == w2 and list(w1) == sorted(w1)
        assert split_pools({T6: 7}, w1) == split_pools({T6: 7}, w2)

    def test_integer_weights_drop_zero_and_negative(self):
        assert integer_weights({"a": 1e-12, "b": -1.0, "c": 0.5}, 10) == {"c": 5_000_000_000}


class TestValue:
    def test_one_whole_token_of_either_decimals(self):
        assert value_of(10 ** 6, 10 ** 30) == 10 ** 6
        assert value_of(10 ** 18, 10 ** 18) == 10 ** 6

    def test_basket_values_each_token_then_adds(self):
        assert basket_value({T6: 1_500_000, T18: 5 * 10 ** 17},
                            {T6: 10 ** 30, T18: 10 ** 18}) == 2_000_000
        assert basket_value({T6: 7}, {}) == 0


class TestFederatedPoolDeterminism:
    def test_quiet_epoch_carries_the_whole_pool(self):
        from nodes.common.canonical_ordering import canonical_order
        from nodes.common.federated_reconcile import federated_epoch_close

        out = federated_epoch_close(canonical_order([]),
                                    tool_share_window={T6: 25_000_000, T18: 3})
        assert out["total_value"] == 0
        assert out["claims"] == {}
        assert out["tool_rounds"]["pool_carry"] == {T18: 3, T6: 25_000_000}


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
