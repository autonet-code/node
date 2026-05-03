"""Tests for period-based budget rollover (issue #21 / task #12)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.runtime.agent_registry import _resolve_budget


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _bare_agent(agent_id: str, parent_id: str | None, budgets=None) -> AgentDefinition:
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.PIPELINE,
        parent_id=parent_id,
        budgets=budgets or {},
    )


# ---------------------------------------------------------------------------
# _resolve_budget shape parsing
# ---------------------------------------------------------------------------

class _D:
    """Minimal stand-in for AgentDefinition.budgets attribute."""
    def __init__(self, budgets):
        self.budgets = budgets


def test_resolve_budget_int_form():
    assert _resolve_budget(_D({"claude_max": 1000}), "claude_max") == (1000.0, "none", "tokens")


def test_resolve_budget_dict_form():
    assert _resolve_budget(
        _D({"claude_max": {"limit": 5000, "period": "monthly"}}),
        "claude_max",
    ) == (5000.0, "monthly", "tokens")


def test_resolve_budget_unknown_period_falls_back_to_none():
    assert _resolve_budget(
        _D({"claude_max": {"limit": 5000, "period": "fortnightly"}}),
        "claude_max",
    ) == (5000.0, "none", "tokens")


def test_resolve_budget_missing_provider():
    assert _resolve_budget(_D({"gemini": 100}), "claude_max") == (0.0, "none", "tokens")


def test_resolve_budget_no_definition():
    assert _resolve_budget(None, "claude_max") == (0.0, "none", "tokens")


def test_resolve_budget_pct_subscription_form():
    assert _resolve_budget(
        _D({"claude_max": {"pct": 30, "of": "subscription_5h", "period": "weekly"}}),
        "claude_max",
    ) == (30.0, "weekly", "pct_subscription_5h")


# ---------------------------------------------------------------------------
# Period rollover on record_token_usage / check_budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_period_does_not_reset_within_window(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent(
        "a", None,
        budgets={"claude_max": {"limit": 10_000, "period": "daily"}},
    ))
    rt.registry.record_token_usage("a", "claude_max", 3000)
    rt.registry.record_token_usage("a", "claude_max", 2000)
    info = rt.registry.get_budget_info("a")
    assert info["claude_max"]["used"] == 5000
    assert info["claude_max"]["period"] == "daily"


@pytest.mark.asyncio
async def test_period_resets_when_window_elapsed(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent(
        "a", None,
        budgets={"claude_max": {"limit": 10_000, "period": "daily"}},
    ))
    rt.registry.record_token_usage("a", "claude_max", 8000)

    # Backdate the period stamp so the window has elapsed.
    yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    rt.registry._budget_period_start["a"]["claude_max"] = yesterday

    # Next record should observe rollover and zero the counter before adding.
    rt.registry.record_token_usage("a", "claude_max", 500)
    info = rt.registry.get_budget_info("a")
    assert info["claude_max"]["used"] == 500


@pytest.mark.asyncio
async def test_check_budget_clears_blocked_state_after_rollover(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent(
        "a", None,
        budgets={"claude_max": {"limit": 1000, "period": "hourly"}},
    ))
    rt.registry.record_token_usage("a", "claude_max", 1500)
    ok, blocker = rt.registry.check_budget("a", "claude_max")
    assert ok is False and blocker == "a"

    # Backdate; check_budget should auto-reset and now allow.
    rt.registry._budget_period_start["a"]["claude_max"] = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).isoformat()
    ok, blocker = rt.registry.check_budget("a", "claude_max")
    assert ok is True and blocker is None
    # And the counter is zeroed.
    assert rt.registry._budget_used["a"]["claude_max"] == 0


@pytest.mark.asyncio
async def test_period_none_never_resets(tmp_path):
    """Lifetime budgets (int form, no period) should never roll over."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent(
        "a", None, budgets={"claude_max": 5000},
    ))
    rt.registry.record_token_usage("a", "claude_max", 3000)
    # Even if we manually backdate, period == "none" → no reset.
    rt.registry._budget_period_start.setdefault("a", {})["claude_max"] = (
        datetime.now(timezone.utc) - timedelta(days=400)
    ).isoformat()
    rt.registry.record_token_usage("a", "claude_max", 1000)
    assert rt.registry._budget_used["a"]["claude_max"] == 4000


@pytest.mark.asyncio
async def test_period_state_persists_across_restart(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent(
        "a", None,
        budgets={"claude_max": {"limit": 10_000, "period": "weekly"}},
    ))
    rt.registry.record_token_usage("a", "claude_max", 1000)
    stamp = rt.registry._budget_period_start["a"]["claude_max"]

    del rt
    rt2 = _make_runtime(tmp_path)
    assert rt2.registry._budget_period_start["a"]["claude_max"] == stamp
    assert rt2.registry._budget_used["a"]["claude_max"] == 1000
