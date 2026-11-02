"""External agents (atn/external_agents.md): an outside harness drives the
agent over the daemon's WS port; the daemon holds its key and never runs it."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from atn.agent_tools import _create_agent, _review_due, _use_tool
from atn.config import ATNConfig
from atn.events import EventBus
from atn.provider_identity import EXTERNAL_PROVIDER, is_external, routing_error
from atn.runtime import Runtime


def _runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def test_external_needs_no_model():
    assert routing_error(EXTERNAL_PROVIDER, "") is None


@pytest.mark.asyncio
async def test_create_external_agent(tmp_path):
    rt = _runtime(tmp_path)
    out = await _create_agent(rt, {"id": "cc-session", "provider": "external",
                                   "harness": "claude-code"})
    assert out.get("external") is True, out
    assert out["address"].startswith("0x")
    defn = rt.get_agent("cc-session")
    assert is_external(defn)
    assert "unified_tools" in defn.tools and "publishing" in defn.tools
    # The daemon never runs it.
    assert await rt.trigger_run("cc-session") is None


@pytest.mark.asyncio
async def test_external_agent_refuses_a_prompt(tmp_path):
    rt = _runtime(tmp_path)
    out = await _create_agent(rt, {"id": "x", "provider": "external",
                                   "prompt": "do things"})
    assert "error" in out


@pytest.mark.asyncio
async def test_use_tool_reports_review_due(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc", "provider": "external"})

    async def call_tool(name, arguments, caller_id=None):
        return {"value": 1}

    monkeypatch.setattr(rt.tool_registry, "call_tool", call_tool)
    monkeypatch.setattr(rt.tool_registry, "get_tool",
                        lambda name: SimpleNamespace(digest="d" * 64))
    origin = {"value": "adopted"}
    monkeypatch.setattr(rt.tool_store, "get",
                        lambda d: SimpleNamespace(origin=origin["value"]))
    out = await _use_tool(rt, {"name": "word_count", "_caller_id": "cc"})
    assert out["review_due"] == ["word_count"]
    assert _review_due(rt, "cc") == {"d" * 64: "word_count"}

    # The household's own tools are never due: nobody reviews their own.
    rt.__dict__["_external_review_due"] = {}
    origin["value"] = "authored"
    out = await _use_tool(rt, {"name": "word_count", "_caller_id": "cc"})
    assert "review_due" not in out


def test_local_owner_key_is_stable(tmp_path):
    rt = _runtime(tmp_path)
    k1 = rt.registry.local_owner_key()
    k2 = rt.registry.local_owner_key()
    assert k1 == k2 and k1.startswith("0x") and len(k1) == 66
