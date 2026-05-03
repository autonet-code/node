"""Tests for mandatory budgets + cascading clamp at register_agent
(issue #21 / task #14)."""
from __future__ import annotations

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
# Mandatory budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root_cognitive_no_budget_allowed(tmp_path):
    rt = _make_runtime(tmp_path)
    # Root sets the cap — no budget required.
    await rt.registry.register_agent(_agent("root", None))


@pytest.mark.asyncio
async def test_pipeline_no_budget_allowed(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("p", None, mode=AgentMode.PIPELINE),
    )
    # Pipeline child also exempt.
    await rt.registry.register_agent(
        _agent("p.1", "p", mode=AgentMode.PIPELINE),
    )


@pytest.mark.asyncio
async def test_cognitive_child_without_budget_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    with pytest.raises(ValueError, match="must declare a budget"):
        await rt.registry.register_agent(_agent("root.child", "root"))


@pytest.mark.asyncio
async def test_cognitive_child_with_zero_limit_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    with pytest.raises(ValueError, match="positive limit"):
        await rt.registry.register_agent(
            _agent("root.child", "root", budgets={"claude_max": 0}),
        )


@pytest.mark.asyncio
async def test_cognitive_child_with_period_form_accepted(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    await rt.registry.register_agent(
        _agent("root.child", "root",
               budgets={"claude_max": {"limit": 1000, "period": "daily"}}),
    )


# ---------------------------------------------------------------------------
# Cascading clamp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_child_limit_within_parent_headroom(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    await rt.registry.register_agent(
        _agent("root.child", "root", budgets={"claude_max": 5000}),
    )


@pytest.mark.asyncio
async def test_child_limit_exceeds_parent_cap_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 1000}),
    )
    with pytest.raises(ValueError, match="cascade violation"):
        await rt.registry.register_agent(
            _agent("root.child", "root", budgets={"claude_max": 5000}),
        )


@pytest.mark.asyncio
async def test_child_limit_exceeds_parent_remaining_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    # Burn most of root's headroom.
    rt.registry.record_token_usage("root", "claude_max", 9000)
    with pytest.raises(ValueError, match="cascade violation"):
        await rt.registry.register_agent(
            _agent("root.child", "root", budgets={"claude_max": 5000}),
        )
    # 800 fits in remaining 1000.
    await rt.registry.register_agent(
        _agent("root.child", "root", budgets={"claude_max": 800}),
    )


@pytest.mark.asyncio
async def test_grandchild_clamped_by_nearest_capping_ancestor(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 100_000}),
    )
    await rt.registry.register_agent(
        _agent("root.a", "root", budgets={"claude_max": 5000}),
    )
    # Grandchild caps at 4000 — fits in parent's 5000.
    await rt.registry.register_agent(
        _agent("root.a.b", "root.a", budgets={"claude_max": 4000}),
    )
    # But 6000 would blow the nearest cap (root.a's 5000).
    with pytest.raises(ValueError, match="cascade violation"):
        await rt.registry.register_agent(
            _agent("root.a.c", "root.a", budgets={"claude_max": 6000}),
        )


@pytest.mark.asyncio
async def test_uncapped_provider_does_not_clamp(tmp_path):
    """If no ancestor caps the provider the child names, the child sets its own ceiling."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    # Child caps gemini — root only caps claude_max → no clamp on gemini.
    await rt.registry.register_agent(
        _agent("root.child", "root", budgets={"gemini": 999_999}),
    )
