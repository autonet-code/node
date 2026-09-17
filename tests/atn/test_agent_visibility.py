"""Observation tools are scoped to the caller's subtree.

An agent sees itself, its descendants, and its parent's id. Peers (siblings
and other trees) are invisible: not listed, and not resolvable by id. The
owner surface (no caller id) stays unrestricted.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.agent_tools import (
    _get_agent,
    _get_history,
    _get_snapshot,
    _list_agents,
    _visible_ids,
)
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode


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


async def _fleet(rt):
    """Two trees: boss -> worker -> intern, and a lone peer."""
    for aid, parent in (("boss", None), ("worker", "boss"),
                        ("intern", "worker"), ("peer", None)):
        await rt.register_agent(AgentDefinition(
            id=aid, name=aid.title(), mode=AgentMode.COGNITIVE,
            system_prompt="x", cognitive_model="claude-sonnet-5",
            parent_id=parent,
        ))


@pytest.mark.asyncio
async def test_visible_ids_is_subtree_plus_parent(tmp_path):
    rt = _make_runtime(tmp_path)
    await _fleet(rt)
    assert _visible_ids(rt, {}) is None
    assert _visible_ids(rt, {"_caller_id": "worker"}) == {"worker", "intern", "boss"}
    assert _visible_ids(rt, {"_caller_id": "peer"}) == {"peer"}


@pytest.mark.asyncio
async def test_list_agents_hides_peers(tmp_path):
    rt = _make_runtime(tmp_path)
    await _fleet(rt)
    owner = await _list_agents(rt, {})
    assert {a["id"] for a in owner["agents"]} == {"boss", "worker", "intern", "peer"}
    seen = await _list_agents(rt, {"_caller_id": "peer"})
    assert {a["id"] for a in seen["agents"]} == {"peer"}
    seen = await _list_agents(rt, {"_caller_id": "worker"})
    assert {a["id"] for a in seen["agents"]} == {"boss", "worker", "intern"}


@pytest.mark.asyncio
async def test_get_agent_and_history_refuse_peers_like_missing(tmp_path):
    rt = _make_runtime(tmp_path)
    await _fleet(rt)
    hidden = await _get_agent(rt, {"agent_id": "boss", "_caller_id": "peer"})
    missing = await _get_agent(rt, {"agent_id": "nope", "_caller_id": "peer"})
    assert "error" in hidden and hidden["error"].replace("boss", "X") == missing["error"].replace("nope", "X")
    ok = await _get_agent(rt, {"agent_id": "intern", "_caller_id": "boss"})
    assert ok.get("id") == "intern"
    hist = await _get_history(rt, {"agent_id": "peer", "_caller_id": "worker"})
    assert "error" in hist


@pytest.mark.asyncio
async def test_agent_snapshot_is_scoped_and_trimmed(tmp_path):
    rt = _make_runtime(tmp_path)
    await _fleet(rt)
    full = await _get_snapshot(rt, {})
    assert "available_models" in full and "providers" in full
    scoped = await _get_snapshot(rt, {"_caller_id": "worker"})
    assert set(scoped["agents"]) == {"boss", "worker", "intern"}  # parent id is visible
    for ui_only in ("available_models", "providers", "voice", "input", "update"):
        assert ui_only not in scoped
