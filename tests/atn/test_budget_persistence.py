"""Tests for AgentRegistry budget state persistence.

Covers issue #21 / task #11 — `_budget_used` survives daemon restart.
"""
from __future__ import annotations

import json

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
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


def _bare_agent(agent_id: str, parent_id: str | None, budgets=None) -> AgentDefinition:
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.PIPELINE,
        parent_id=parent_id,
        budgets=budgets or {},
    )


@pytest.mark.asyncio
async def test_budget_state_file_created_on_first_record(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent("a", None))

    state_path = tmp_path / "data" / "budget_state.json"
    assert not state_path.exists()

    rt.registry.record_token_usage("a", "claude_max", 1000)

    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["used"]["a"]["claude_max"] == 1000


@pytest.mark.asyncio
async def test_budget_state_reloaded_across_restart(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent("a", None))
    await rt.registry.register_agent(_bare_agent("b", "a"))
    rt.registry.record_token_usage("b", "claude_max", 4200)

    # Simulate a daemon restart: drop runtime, build fresh.
    del rt
    rt2 = _make_runtime(tmp_path)

    # Counters survived — no need to re-register agents to read state.
    assert rt2.registry._budget_used.get("b", {}).get("claude_max") == 4200
    assert rt2.registry._budget_used.get("a", {}).get("claude_max") == 4200


@pytest.mark.asyncio
async def test_budget_state_atomic_write_no_partial(tmp_path):
    """A poisoned write shouldn't corrupt the existing file."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent("a", None))
    rt.registry.record_token_usage("a", "claude_max", 100)

    state_path = tmp_path / "data" / "budget_state.json"
    original = state_path.read_text()

    # Confirm no .tmp leftover after a normal write.
    siblings = list(state_path.parent.iterdir())
    assert all(not s.name.startswith(".budget_state.") for s in siblings), \
        "Atomic write left a .tmp file behind"
    assert state_path.read_text() == original


@pytest.mark.asyncio
async def test_unregister_clears_budget_state(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_bare_agent("a", None))
    await rt.registry.register_agent(_bare_agent("b", "a"))
    rt.registry.record_token_usage("b", "claude_max", 500)

    await rt.registry.unregister_agent("b")

    state_path = tmp_path / "data" / "budget_state.json"
    data = json.loads(state_path.read_text())
    assert "b" not in data.get("used", {})
    # Parent's roll-up retained.
    assert data["used"]["a"]["claude_max"] == 500


@pytest.mark.asyncio
async def test_corrupt_state_file_does_not_crash(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "budget_state.json").write_text("{ this is not valid json")

    # Constructing the runtime must not raise.
    rt = _make_runtime(tmp_path)
    assert rt.registry._budget_used == {}
