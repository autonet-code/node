"""Tests for per-(provider, model) budget keys (task #21).

Schema:
    budgets = {
        "claude_max":                          # provider-wide cap
            {"pct": 80, "of": "subscription_5h", "period": "weekly"},
        "claude_max:claude-opus-4-7":          # per-model cap
            {"pct": 20, "of": "subscription_5h", "period": "weekly"},
    }

The recorder updates BOTH the provider-wide and per-model counters on each
turn (when model_id is supplied). The first cap to overflow blocks the agent.
"""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.runtime.agent_registry import _resolve_budget, budget_key


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id, parent_id, *, mode=AgentMode.COGNITIVE, budgets=None):
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=mode,
        parent_id=parent_id, budgets=budgets or {},
    )


def test_budget_key_format():
    assert budget_key("claude_max") == "claude_max"
    assert budget_key("claude_max", "claude-opus-4-7") == "claude_max:claude-opus-4-7"


def test_resolve_separate_caps():
    """Provider-wide and per-model caps are independent on resolve()."""
    defn = type("X", (), {"budgets": {
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 20_000,
    }})()
    assert _resolve_budget(defn, "claude_max")[0] == 100_000
    assert _resolve_budget(defn, "claude_max:claude-opus-4-7")[0] == 20_000


@pytest.mark.asyncio
async def test_recorder_updates_both_keys(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 20_000,
    }))
    rt.registry.record_token_usage(
        "root", "claude_max", 5000,
        model_id="claude-opus-4-7",
    )
    info = rt.registry.get_budget_info("root")
    assert info["claude_max"]["used"] == 5000
    assert info["claude_max:claude-opus-4-7"]["used"] == 5000


@pytest.mark.asyncio
async def test_per_model_cap_can_block_before_provider_cap(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 1000,  # tight Opus cap
    }))
    # Burn against Opus.
    exceeded = rt.registry.record_token_usage(
        "root", "claude_max", 1500,
        model_id="claude-opus-4-7",
    )
    assert exceeded == "root"
    info = rt.registry.get_budget_info("root")
    # Provider-wide cap not blown.
    assert info["claude_max"]["used"] == 1500
    assert info["claude_max"]["remaining"] == 98_500
    # Per-model cap blown.
    assert info["claude_max:claude-opus-4-7"]["used"] == 1500
    assert info["claude_max:claude-opus-4-7"]["remaining"] == 0


@pytest.mark.asyncio
async def test_other_models_dont_consume_per_model_cap(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 1000,
    }))
    # Run on Sonnet — should not touch the Opus cap.
    rt.registry.record_token_usage(
        "root", "claude_max", 5000,
        model_id="claude-sonnet-4-7",
    )
    info = rt.registry.get_budget_info("root")
    assert info["claude_max"]["used"] == 5000
    assert info["claude_max:claude-opus-4-7"]["used"] == 0


@pytest.mark.asyncio
async def test_check_budget_examines_both_keys(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 1000,
    }))
    rt.registry.record_token_usage(
        "root", "claude_max", 1200,
        model_id="claude-opus-4-7",
    )
    # Per-model cap blown → check_budget for that model fails.
    ok, blocker = rt.registry.check_budget(
        "root", "claude_max", model_id="claude-opus-4-7",
    )
    assert ok is False
    assert blocker == "root"
    # But Sonnet still has headroom (provider-wide not blown).
    ok, blocker = rt.registry.check_budget(
        "root", "claude_max", model_id="claude-sonnet-4-7",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_get_budget_info_carries_scope_and_model_id(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 1000,
    }))
    info = rt.registry.get_budget_info("root")
    assert info["claude_max"]["scope"] == "provider"
    assert info["claude_max"]["provider"] == "claude_max"
    assert info["claude_max"]["model_id"] == ""
    assert info["claude_max:claude-opus-4-7"]["scope"] == "model"
    assert info["claude_max:claude-opus-4-7"]["provider"] == "claude_max"
    assert info["claude_max:claude-opus-4-7"]["model_id"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_versioned_model_id_resolves_to_canonical(tmp_path):
    """A turn ran with claude-opus-4-7-20260320 should hit the claude-opus-4-7 cap."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max:claude-opus-4-7": 1000,
    }))
    # The engine resolves the spec id before calling the recorder; simulate that here.
    from atn.model_specs import resolve
    spec = resolve("claude-opus-4-7-20260320")
    rt.registry.record_token_usage(
        "root", "claude_max", 500,
        model_id=spec.id,
    )
    info = rt.registry.get_budget_info("root")
    assert info["claude_max:claude-opus-4-7"]["used"] == 500


@pytest.mark.asyncio
async def test_per_model_cap_persists_across_restart(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None, budgets={
        "claude_max": 100_000,
        "claude_max:claude-opus-4-7": 5000,
    }))
    rt.registry.record_token_usage(
        "root", "claude_max", 1500, model_id="claude-opus-4-7",
    )
    del rt
    rt2 = _make_runtime(tmp_path)
    assert (
        rt2.registry._budget_used.get("root", {}).get("claude_max:claude-opus-4-7")
        == 1500
    )
