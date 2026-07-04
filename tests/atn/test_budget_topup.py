"""Budget top-ups — the blessed semantics (docs/tool_substrate.md,
reference default 3): budgets are parent-updateable within the parent's
own headroom. Owner unconstrained; no self-raise; no stranger-raise;
cascade re-checked against the new limits on update.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator.tools import execute_tool


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _budget(limit: int) -> dict:
    return {"claude_max": {"limit": limit, "unit": "tokens",
                           "period": "none"}}


async def _register(rt, agent_id, parent_id=None, limit=None):
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5",
        parent_id=parent_id,
        budgets=_budget(limit) if limit else {},
    )
    await rt.register_agent(defn)
    return defn


@pytest.mark.asyncio
async def test_parent_tops_up_child_within_headroom(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "budgets": _budget(50_000)},
        rt, caller_id="parent")
    assert "budgets" in res.get("changed", []), res
    assert rt.get_agent("kid").budgets["claude_max"]["limit"] == 50_000


@pytest.mark.asyncio
async def test_topup_beyond_parent_headroom_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "budgets": _budget(200_000)},
        rt, caller_id="parent")
    assert "error" in res
    assert rt.get_agent("kid").budgets["claude_max"]["limit"] == 10_000


@pytest.mark.asyncio
async def test_self_raise_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "budgets": _budget(90_000)},
        rt, caller_id="kid")
    assert "own budget" in res["error"]


@pytest.mark.asyncio
async def test_non_parent_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)
    await _register(rt, "uncle", limit=100_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "budgets": _budget(20_000)},
        rt, caller_id="uncle")
    assert "not the parent" in res["error"]


@pytest.mark.asyncio
async def test_owner_unconstrained_by_relationship(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "budgets": _budget(60_000)},
        rt, caller_id="")  # owner
    assert "budgets" in res.get("changed", [])


@pytest.mark.asyncio
async def test_non_budget_updates_unaffected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", limit=100_000)
    await _register(rt, "kid", parent_id="parent", limit=10_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "kid", "description": "helper"},
        rt, caller_id="kid")
    assert "description" in res.get("changed", [])
