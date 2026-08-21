"""Regression: an agent created with a 40k token budget must surface a flat
integer limit through the config surface (get_agent / list_agents), not the
nested {limit, period} authoring shape that a scalar-expecting client coerces
to 0.
"""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.agent_tools import _get_agent, _list_agents
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


@pytest.mark.asyncio
async def test_get_agent_flattens_scalar_budget(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(AgentDefinition(
        id="a", name="a", mode=AgentMode.COGNITIVE,
        budgets={"claude_max": 40_000},
    ))
    res = await _get_agent(rt, {"agent_id": "a"})
    assert res["budgets"] == {"claude_max": 40_000}
    assert isinstance(res["budgets"]["claude_max"], int)


@pytest.mark.asyncio
async def test_get_agent_flattens_nested_budget(tmp_path):
    # The nested {limit, period} authoring shape (what the create form sends
    # when a period is chosen) must ALSO surface as a flat 40000, not a dict
    # the frontend would coerce to 0.
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(AgentDefinition(
        id="a", name="a", mode=AgentMode.COGNITIVE,
        budgets={"claude_max": {"limit": 40_000, "period": "monthly"}},
    ))
    res = await _get_agent(rt, {"agent_id": "a"})
    assert res["budgets"]["claude_max"] == 40_000
    assert isinstance(res["budgets"]["claude_max"], int)


@pytest.mark.asyncio
async def test_list_agents_flattens_budget(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(AgentDefinition(
        id="a", name="a", mode=AgentMode.COGNITIVE,
        budgets={"claude_max": {"limit": 40_000, "period": "monthly"}},
    ))
    res = await _list_agents(rt, {})
    entry = next(a for a in res["agents"] if a["id"] == "a")
    assert entry["budgets"]["claude_max"] == 40_000


@pytest.mark.asyncio
async def test_get_agent_empty_budget_stays_empty(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(AgentDefinition(
        id="a", name="a", mode=AgentMode.COGNITIVE, budgets={},
    ))
    res = await _get_agent(rt, {"agent_id": "a"})
    assert res["budgets"] == {}
