"""Tests for per-model rate lookup on the bridge estimator (task #22).

The estimator stores rates keyed by class (Anthropic doesn't expose
per-version utilization), but exposes ``tokens_per_pct_for_model`` so callers
can ask in model-id terms. The store routes through model_specs to find the
class.
"""
from __future__ import annotations

import pytest


def _make_stub():
    from atn.providers.bridge import BridgeProvider
    s = BridgeProvider.__new__(BridgeProvider)
    s._cum_tokens_by_class = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._tokens_per_pct_by_class = {}
    s._bootstrap_relative = {"haiku": 5.0, "sonnet": 1.0, "opus": 0.2, "other": 1.0}
    s._bootstrap_sonnet_rate = 60_000.0
    return s


def test_per_model_lookup_routes_through_class():
    s = _make_stub()
    # No observations yet — bootstrap rates by class.
    assert s.tokens_per_pct_for_model("claude-opus-4-7") == pytest.approx(12_000)
    assert s.tokens_per_pct_for_model("claude-sonnet-4-7") == pytest.approx(60_000)
    assert s.tokens_per_pct_for_model("claude-haiku-4-5") == pytest.approx(300_000)


def test_versioned_model_id_picks_class_rate():
    """Versioned IDs (e.g. ...-20260320 suffix) resolve to the canonical spec."""
    s = _make_stub()
    s._tokens_per_pct_by_class["sonnet"] = 50_000
    # The model store handles version → canonical id, so the rate matches sonnet.
    assert s.tokens_per_pct_for_model("claude-sonnet-4-7-20260320") == pytest.approx(50_000)


def test_unknown_model_falls_back_to_bootstrap():
    s = _make_stub()
    # Made-up model → 'other' class → bootstrap_sonnet_rate * 1.0 = 60_000.
    assert s.tokens_per_pct_for_model("totally-fake-model-9000") == pytest.approx(60_000)


def test_observed_class_rate_wins_over_bootstrap():
    s = _make_stub()
    s._tokens_per_pct_by_class["opus"] = 8_000  # observed
    assert s.tokens_per_pct_for_model("claude-opus-4-7") == pytest.approx(8_000)
    # Sonnet still bootstrap.
    assert s.tokens_per_pct_for_model("claude-sonnet-4-7") == pytest.approx(60_000)


def test_bulk_lookup_for_list_of_models():
    s = _make_stub()
    s._tokens_per_pct_by_class["opus"] = 10_000
    rates = s.tokens_per_pct_by_model([
        "claude-opus-4-7",
        "claude-sonnet-4-7",
        "claude-haiku-4-5",
    ])
    assert rates["claude-opus-4-7"] == pytest.approx(10_000)
    assert rates["claude-sonnet-4-7"] == pytest.approx(60_000)
    assert rates["claude-haiku-4-5"] == pytest.approx(300_000)
