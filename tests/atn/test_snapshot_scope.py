"""Scope-aware snapshot: a connection rooted at agent X sees only X's subtree.

Default (scope_ids=None) must remain the full fleet, byte-compatible with the
pre-scoping behavior so the localhost / orchestrator-root path is unchanged."""
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


def _agent(agent_id: str, parent_id: str | None) -> AgentDefinition:
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        parent_id=parent_id, budgets={},
    )


async def _fleet(tmp_path) -> Runtime:
    # orchestrator -> a -> {a.1, a.2} ; orchestrator -> b
    rt = _make_runtime(tmp_path)
    for aid, pid in [
        ("orchestrator", None), ("a", "orchestrator"),
        ("a.1", "a"), ("a.2", "a"), ("b", "orchestrator"),
    ]:
        await rt.registry.register_agent(_agent(aid, pid))
    return rt


@pytest.mark.asyncio
async def test_full_fleet_default(tmp_path):
    rt = await _fleet(tmp_path)
    snap = rt.snapshot()
    assert set(snap["agents"].keys()) == {"orchestrator", "a", "a.1", "a.2", "b"}


@pytest.mark.asyncio
async def test_scoped_to_subtree_excludes_others(tmp_path):
    rt = await _fleet(tmp_path)
    scope = rt.registry.get_subtree_ids("a")  # {a, a.1, a.2}
    snap = rt.snapshot(scope)
    assert set(snap["agents"].keys()) == {"a", "a.1", "a.2"}
    # The sibling b and the orchestrator are NOT visible to an 'a'-rooted view.
    assert "b" not in snap["agents"]
    assert "orchestrator" not in snap["agents"]


@pytest.mark.asyncio
async def test_scoped_children_count_respects_scope(tmp_path):
    rt = await _fleet(tmp_path)
    # Full fleet: 'a' has 2 children.
    assert rt.snapshot()["agents"]["a"].get("children_count") == 2
    # Scope to just {a, a.1}: 'a' should count only the in-scope child.
    snap = rt.snapshot({"a", "a.1"})
    assert snap["agents"]["a"].get("children_count") == 1
    assert set(snap["agents"].keys()) == {"a", "a.1"}


@pytest.mark.asyncio
async def test_scoped_leaf_only_self(tmp_path):
    rt = await _fleet(tmp_path)
    snap = rt.snapshot(rt.registry.get_subtree_ids("a.1"))
    assert set(snap["agents"].keys()) == {"a.1"}
    assert "children_count" not in snap["agents"]["a.1"]


@pytest.mark.asyncio
async def test_global_sections_present_when_scoped(tmp_path):
    """Scoping filters agents/executions but leaves the global daemon sections
    (the snapshot must still be a well-formed dashboard payload)."""
    rt = await _fleet(tmp_path)
    snap = rt.snapshot(rt.registry.get_subtree_ids("a"))
    # These keys exist in the full snapshot and should survive scoping.
    for key in ("agents", "executions"):
        assert key in snap
