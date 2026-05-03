"""Tests for the post-orchestration reconciliation pass (issue #21 / task #19).

After each bridge orchestration we compare the predicted subscription burn
(computed turn-by-turn from the per-class estimator) against the actual
util_5h delta from Anthropic's headers. Discrepancies > 10% nudge the
dominant class's rate via EMA.
"""
from __future__ import annotations

import pytest

from atn.providers.bridge import BridgeProvider


def _make_stub() -> BridgeProvider:
    s = BridgeProvider.__new__(BridgeProvider)
    s._cum_tokens_by_class = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._last_snapshot = None
    s._tokens_per_pct_by_class = {}
    s._estimator_obs_count = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._bootstrap_relative = {"haiku": 5.0, "sonnet": 1.0, "opus": 0.2, "other": 1.0}
    s._bootstrap_sonnet_rate = 60_000.0
    s._predicted_pct_since_refresh = 0.0
    s._predicted_tokens_by_class_since_refresh = {
        "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
    }
    s._last_reconciliation = None
    s._rate_limits = {}
    return s


@pytest.mark.asyncio
async def test_reconcile_no_op_when_no_prior_reading(monkeypatch):
    s = _make_stub()
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")

    # Stub refresh_usage so reconcile doesn't try to hit the network.
    async def _fake_refresh():
        s._rate_limits["five_hour"] = {"utilization": 0.05}
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    report = await s.reconcile_after_orchestration()
    # No prior reading, so actual_pct is None; no adjustment.
    assert report["actual_pct"] is None
    assert report["adjusted_class"] is None
    # Counters reset.
    assert s._predicted_pct_since_refresh == 0


@pytest.mark.asyncio
async def test_reconcile_no_adjustment_when_within_tolerance(monkeypatch):
    s = _make_stub()
    s._rate_limits["five_hour"] = {"utilization": 0.10}  # prior reading
    s.record_turn_tokens(60_000, "claude-sonnet-4-6")  # predicted 1pp at sonnet=60k

    async def _fake_refresh():
        s._rate_limits["five_hour"] = {"utilization": 0.1095}  # +0.95pp actual
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    report = await s.reconcile_after_orchestration()
    assert report["predicted_pct"] == pytest.approx(1.0)
    assert report["actual_pct"] == pytest.approx(0.95, rel=0.01)
    # Within 10% → no adjustment.
    assert report["adjusted_class"] is None


@pytest.mark.asyncio
async def test_reconcile_adjusts_dominant_class_on_mismatch(monkeypatch):
    s = _make_stub()
    s._rate_limits["five_hour"] = {"utilization": 0.10}
    # Predict 1pp at the bootstrap sonnet rate of 60_000.
    s.record_turn_tokens(60_000, "claude-sonnet-4-6")

    async def _fake_refresh():
        # Actual burn was 2pp — twice what we predicted. Rate was too high.
        s._rate_limits["five_hour"] = {"utilization": 0.12}
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    report = await s.reconcile_after_orchestration()
    assert report["adjusted_class"] == "sonnet"
    # ratio = 2.0 → target = 60_000 / 2.0 = 30_000
    # adjusted = 0.7 * 60_000 + 0.3 * 30_000 = 51_000
    assert s._tokens_per_pct_by_class["sonnet"] == pytest.approx(51_000)


@pytest.mark.asyncio
async def test_reconcile_picks_dominant_class(monkeypatch):
    """If a turn used both Sonnet and Opus, the one with more tokens gets adjusted."""
    s = _make_stub()
    s._rate_limits["five_hour"] = {"utilization": 0.10}
    s.record_turn_tokens(20_000, "claude-sonnet-4-6")
    s.record_turn_tokens(80_000, "claude-opus-4-7")  # opus dominates

    async def _fake_refresh():
        s._rate_limits["five_hour"] = {"utilization": 0.30}  # big mismatch
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    report = await s.reconcile_after_orchestration()
    assert report["adjusted_class"] == "opus"


@pytest.mark.asyncio
async def test_reconcile_skips_when_actual_pct_negligible(monkeypatch):
    s = _make_stub()
    s._rate_limits["five_hour"] = {"utilization": 0.10}
    s.record_turn_tokens(100, "claude-sonnet-4-6")  # tiny prediction

    async def _fake_refresh():
        s._rate_limits["five_hour"] = {"utilization": 0.1001}  # 0.01pp — noise
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    report = await s.reconcile_after_orchestration()
    # actual_pct < 0.5 threshold → no adjustment.
    assert report["adjusted_class"] is None


@pytest.mark.asyncio
async def test_reconcile_resets_counters_after(monkeypatch):
    s = _make_stub()
    s._rate_limits["five_hour"] = {"utilization": 0.10}
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")

    async def _fake_refresh():
        s._rate_limits["five_hour"] = {"utilization": 0.11}
        return dict(s._rate_limits)
    monkeypatch.setattr(s, "refresh_usage", _fake_refresh)

    await s.reconcile_after_orchestration()
    assert s._predicted_pct_since_refresh == 0
    assert all(v == 0 for v in s._predicted_tokens_by_class_since_refresh.values())
    assert s._last_reconciliation is not None
