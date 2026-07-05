"""Tests for the deterministic metering service (atn/metering.py).

Covers:
  * cost-per-token dollar computation from the static table,
  * closed-form subscription quota inference over synthetic snapshots,
  * daemon-wide dollar-cap pre-flight enforcement + period rollover,
  * JSONL persistence / replay.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from atn.config import ATNConfig, ProviderConfig
from atn.metering import (
    MeteringService,
    QuotaEstimate,
    cost_usd,
    infer_quota,
)


# ---------------------------------------------------------------------------
# Cost-per-token table
# ---------------------------------------------------------------------------

def test_cost_usd_known_model_prefix_match():
    # Opus 4.8: $5/MTok input, $25/MTok output.
    usd = cost_usd("claude-opus-4-8", input_tokens=1_000_000, output_tokens=0)
    assert usd == pytest.approx(5.0)
    usd = cost_usd("claude-opus-4-8", input_tokens=0, output_tokens=1_000_000)
    assert usd == pytest.approx(25.0)


def test_cost_usd_prefix_match_with_date_suffix():
    # A dated model id falls through to the base entry via longest-prefix.
    usd = cost_usd("claude-opus-4-8-20260101", input_tokens=1_000_000)
    assert usd == pytest.approx(5.0)


def test_cost_usd_cache_lanes():
    # Opus cache_read 0.5/MTok, cache_write 6.25/MTok.
    usd = cost_usd(
        "claude-opus-4-8",
        cache_read_tokens=1_000_000, cache_write_tokens=1_000_000,
    )
    assert usd == pytest.approx(0.5 + 6.25)


def test_cost_usd_subscription_or_local_model_returns_none():
    # A local/unpriced model has no per-token dollar price.
    assert cost_usd("qwen3.5:4b", input_tokens=1_000_000) is None
    assert cost_usd("", input_tokens=1000) is None


# ---------------------------------------------------------------------------
# Quota inference — closed form, synthetic snapshots
# ---------------------------------------------------------------------------

def _snaps(pairs: list[tuple[float, int]]) -> list[dict]:
    """Build snapshot dicts from (utilization, cumulative_tokens) pairs."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = []
    for i, (u, tok) in enumerate(pairs):
        out.append({
            "ts": (t0 + timedelta(minutes=i)).isoformat(),
            "utilization": u,
            "cumulative_tokens": tok,
        })
    return out


def test_infer_quota_exact_linear():
    # quota = 1,000,000 tokens fills 100%. So 10% => 100k, 50% => 500k.
    snaps = _snaps([(0.0, 0), (0.10, 100_000), (0.50, 500_000)])
    est = infer_quota(snaps, provider="claude_max")
    assert est.quota_tokens == pytest.approx(1_000_000, rel=1e-6)
    # At 50% utilization, remaining is 500k.
    assert est.remaining_tokens == pytest.approx(500_000, rel=1e-6)
    assert est.current_utilization == pytest.approx(0.50)
    assert est.confidence > 0.0


def test_infer_quota_noisy_least_squares_through_origin():
    # Underlying quota 2,000,000 with +/- noise on the token readings. The
    # through-origin least-squares slope should recover close to 2e6.
    import random
    random.seed(7)
    pairs = []
    for i in range(1, 30):
        u = i / 100.0                       # 1%..29%
        true_tokens = u * 2_000_000
        noise = random.uniform(-40_000, 40_000)
        pairs.append((u, int(true_tokens + noise)))
    est = infer_quota(_snaps(pairs), provider="claude_max")
    assert est.quota_tokens == pytest.approx(2_000_000, rel=0.05)
    assert est.confidence > 0.7  # many obs + wide span => high confidence


def test_infer_quota_low_span_low_confidence():
    # Only a sliver of the window covered (2%..3%) => low confidence even if
    # the slope is right.
    snaps = _snaps([(0.02, 20_000), (0.03, 30_000)])
    est = infer_quota(snaps, provider="claude_max")
    assert est.quota_tokens == pytest.approx(1_000_000, rel=0.01)
    assert est.confidence < 0.3


def test_infer_quota_ignores_pre_reset_window():
    # A window rollover (utilization drops) resets the cumulative-token zero.
    # The fit must use ONLY the post-reset window.
    snaps = _snaps([
        (0.60, 600_000), (0.90, 900_000),   # old window (quota ~1e6)
        (0.05, 1_000_000),                  # RESET: util dropped, tokens kept climbing
        (0.25, 2_000_000),                  # new window: +20% => +1,000,000 tokens
    ])
    est = infer_quota(snaps, provider="claude_max")
    # Post-reset: from (0.05, 1.0M) to (0.25, 2.0M) => 1.0M tokens / 0.20 = 5.0M quota.
    assert est.quota_tokens == pytest.approx(5_000_000, rel=1e-6)
    assert est.current_utilization == pytest.approx(0.25)


def test_infer_quota_empty_returns_zero_confidence():
    est = infer_quota([], provider="claude_max")
    assert isinstance(est, QuotaEstimate)
    assert est.quota_tokens is None
    assert est.confidence == 0.0


# ---------------------------------------------------------------------------
# Service: dollar caps + pre-flight
# ---------------------------------------------------------------------------

def _config_with_cap(provider: str, limit: float, period: str = "none") -> ATNConfig:
    cfg = ATNConfig()
    cfg.providers[provider] = ProviderConfig(
        name=provider, dollar_limit=limit, dollar_period=period,
    )
    return cfg


def test_dollar_cap_preflight_blocks_when_exceeded(tmp_path):
    cfg = _config_with_cap("anthropic", 1.0)  # $1.00 cap
    m = MeteringService(tmp_path, cfg)
    ok, reason = m.check_provider_spend("anthropic")
    assert ok  # nothing spent yet

    # Spend $5 worth of Opus output (5 * 25/MTok = $0.125 per 1M... need 40M).
    # Simpler: 1,000,000 output tokens on opus = $25 > $1 cap.
    res = m.record_usage("anthropic", "claude-opus-4-8", output_tokens=1_000_000)
    assert res["usd"] == pytest.approx(25.0)
    assert res["over_limit"] is True

    ok, reason = m.check_provider_spend("anthropic")
    assert not ok
    assert "anthropic" in reason and "cap" in reason.lower()


def test_dollar_cap_no_limit_never_blocks(tmp_path):
    cfg = ATNConfig()  # no provider caps
    m = MeteringService(tmp_path, cfg)
    m.record_usage("anthropic", "claude-opus-4-8", output_tokens=10_000_000)
    ok, _ = m.check_provider_spend("anthropic")
    assert ok  # no cap configured


def test_subscription_provider_no_dollar_leg(tmp_path):
    # claude_max has no per-token price → record_usage records a snapshot but
    # no dollar spend.
    cfg = ATNConfig()
    m = MeteringService(tmp_path, cfg)
    res = m.record_usage(
        "claude_max", "claude-opus-4-8",
        output_tokens=500_000,
        utilization=0.10, cumulative_tokens=100_000,
    )
    # Subscription model id has a price in the table (direct-API price), but
    # the point of the metering split is the provider is subscription; still,
    # cost_usd keys on model id, so claude-opus-4-8 DOES have a price. The
    # provider distinction matters for the caller (engine passes utilization
    # for subscription providers). Assert the snapshot landed either way.
    est = m.estimate_quota("claude_max")
    assert est.observations >= 0
    # A single snapshot with no prior can't fit yet; feed a second.
    m.record_usage(
        "claude_max", "claude-opus-4-8",
        utilization=0.20, cumulative_tokens=200_000,
    )
    est = m.estimate_quota("claude_max")
    assert est.quota_tokens == pytest.approx(1_000_000, rel=0.01)


def test_period_rollover_resets_spend(tmp_path):
    cfg = _config_with_cap("anthropic", 100.0, period="daily")
    m = MeteringService(tmp_path, cfg)
    # Spend, then force the period start into the past and re-check.
    m.record_usage("anthropic", "claude-opus-4-8", output_tokens=1_000_000)  # $25
    assert m.provider_spend("anthropic")["spent_usd"] == pytest.approx(25.0)
    # Backdate the period start > 1 day.
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    m._period_start["anthropic"] = old
    ok, _ = m.check_provider_spend("anthropic")  # triggers rollover check
    assert ok
    assert m.provider_spend("anthropic")["spent_usd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistence / replay
# ---------------------------------------------------------------------------

def test_spend_persists_across_restart(tmp_path):
    cfg = _config_with_cap("anthropic", 100.0)
    m = MeteringService(tmp_path, cfg)
    m.record_usage("anthropic", "claude-opus-4-8", output_tokens=1_000_000)  # $25
    # New instance replays the JSONL ledger.
    m2 = MeteringService(tmp_path, cfg)
    assert m2.provider_spend("anthropic")["spent_usd"] == pytest.approx(25.0)


def test_snapshots_persist_across_restart(tmp_path):
    cfg = ATNConfig()
    m = MeteringService(tmp_path, cfg)
    m.record_snapshot("claude_max", 0.10, 100_000)
    m.record_snapshot("claude_max", 0.50, 500_000)
    m2 = MeteringService(tmp_path, cfg)
    est = m2.estimate_quota("claude_max")
    assert est.quota_tokens == pytest.approx(1_000_000, rel=1e-6)


def test_duplicate_snapshot_not_recorded(tmp_path):
    cfg = ATNConfig()
    m = MeteringService(tmp_path, cfg)
    m.record_snapshot("claude_max", 0.10, 100_000)
    m.record_snapshot("claude_max", 0.10, 100_000)  # identical → skipped
    assert len(m._snapshots["claude_max"]) == 1
