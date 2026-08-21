"""Tests for tokens-per-percent estimator + get_my_budget_status tool
(issue #21 / task #15)."""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.agent_tools import _get_my_budget_status
from atn.runtime import Runtime


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id: str, parent_id: str | None, *, mode=AgentMode.COGNITIVE,
           budgets=None) -> AgentDefinition:
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=mode,
        parent_id=parent_id,
        budgets=budgets or {},
    )


# ---------------------------------------------------------------------------
# Per-model-class estimator
# ---------------------------------------------------------------------------

def _make_stub():
    """Build a stub matching the per-class estimator's expected attributes."""
    from atn.providers.bridge import BridgeProvider
    s = BridgeProvider.__new__(BridgeProvider)
    s._cum_tokens_by_class = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._last_snapshot = None
    s._tokens_per_pct_by_class = {}
    s._estimator_obs_count = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._bootstrap_relative = {"haiku": 5.0, "sonnet": 1.0, "opus": 0.2, "other": 1.0}
    s._bootstrap_sonnet_rate = 60_000.0
    # Reconciliation counters (added in task #19).
    s._predicted_pct_since_refresh = 0.0
    s._predicted_tokens_by_class_since_refresh = {
        "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
    }
    s._last_reconciliation = None
    s._rate_limits = {}
    return s


def test_estimator_bootstrap_returns_rates_for_all_classes():
    s = _make_stub()
    rates = s.tokens_per_pct_by_class
    assert set(rates.keys()) == {"haiku", "sonnet", "opus", "other"}
    # Sonnet bootstrap = 60_000; Haiku = 5x; Opus = 0.2x.
    assert rates["sonnet"] == 60_000
    assert rates["haiku"] == pytest.approx(300_000)
    assert rates["opus"] == pytest.approx(12_000)


def test_estimator_single_class_observation_locks_in_rate():
    """Sonnet-only delta of 50,000 tokens between two snapshots at +1pp."""
    s = _make_stub()
    s._record_usage_snapshot(0.05)  # baseline: 0 tokens at 5%
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")  # tokens accrue
    s._record_usage_snapshot(0.06)  # +1pp, +50k tokens → 50k tokens/pct
    rates = s.tokens_per_pct_by_class
    assert rates["sonnet"] == pytest.approx(50_000)


def test_estimator_ema_smooths_subsequent_observations():
    s = _make_stub()
    s._record_usage_snapshot(0.05)
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")
    s._record_usage_snapshot(0.06)  # rate observation 1: 50k/pct → locks in
    s.record_turn_tokens(100_000, "claude-sonnet-4-6")
    s._record_usage_snapshot(0.07)  # rate observation 2: 100k/pct
    # EMA: 0.7 * 50_000 + 0.3 * 100_000 = 65_000.
    assert s.tokens_per_pct_by_class["sonnet"] == pytest.approx(65_000)


def test_estimator_handles_window_rollover_without_crash():
    s = _make_stub()
    s._record_usage_snapshot(0.50)
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")
    s._record_usage_snapshot(0.80)
    # Window rolled over.
    s._record_usage_snapshot(0.05)
    # New post-rollover observation works.
    s.record_turn_tokens(50_000, "claude-sonnet-4-6")
    s._record_usage_snapshot(0.06)
    assert "sonnet" in s.tokens_per_pct_by_class


def test_estimator_classifies_model_strings():
    from atn.providers.base import classify_model
    assert classify_model("claude-haiku-4-5") == "haiku"
    assert classify_model("claude-sonnet-4-6") == "sonnet"
    assert classify_model("claude-opus-4-7") == "opus"
    assert classify_model("gpt-5") == "other"
    assert classify_model("") == "other"


# ---------------------------------------------------------------------------
# get_my_budget_status tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_tool_returns_own_caps(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    await rt.registry.register_agent(
        _agent("root.child", "root", budgets={"claude_max": 1000}),
    )
    result = await _get_my_budget_status(rt, {"_caller_id": "root.child"})
    assert result["caller_id"] == "root.child"
    assert result["own"]["claude_max"]["limit"] == 1000


@pytest.mark.asyncio
async def test_status_tool_reports_ancestor_headroom(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    await rt.registry.register_agent(
        _agent("root.child", "root", budgets={"claude_max": 1000}),
    )
    rt.registry.record_token_usage("root.child", "claude_max", 200)

    result = await _get_my_budget_status(rt, {"_caller_id": "root.child"})
    # Own usage shows the increment.
    assert result["own"]["claude_max"]["used"] == 200
    # Ancestor (root) shows it cascaded up.
    ancestor_ids = [a["agent_id"] for a in result["ancestors"]]
    assert "root" in ancestor_ids
    root_info = next(a for a in result["ancestors"] if a["agent_id"] == "root")
    assert root_info["budgets"]["claude_max"]["used"] == 200
    assert root_info["budgets"]["claude_max"]["remaining"] == 9800


@pytest.mark.asyncio
async def test_status_tool_subscription_data_when_available(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )

    # Stub a provider with rate_limits populated.
    class _StubProv:
        name = "claude_max"
        _rate_limits = {"five_hour": {"utilization": 0.42, "status": "ok"}}
        tokens_per_percent = 1234.5

    rt.providers._active_providers["root"] = _StubProv()

    result = await _get_my_budget_status(rt, {"_caller_id": "root"})
    assert "claude_max" in result["subscription"]
    assert result["subscription"]["claude_max"]["tokens_per_percent"] == 1234.5
    assert result["subscription"]["claude_max"]["rate_limits"]["five_hour"]["utilization"] == 0.42
