"""The native-agent mode (atn/native_mode.py)."""
from types import SimpleNamespace

import pytest

from atn.native_mode import native_agents_enabled, set_native_agents


def _rt(tmp_path, agents=None):
    return SimpleNamespace(_config=SimpleNamespace(data_dir=tmp_path),
                           registry=SimpleNamespace(_agents=agents or {}))


def test_default_follows_what_the_daemon_hosts(tmp_path, monkeypatch):
    monkeypatch.delenv("ATN_NATIVE_AGENTS", raising=False)
    assert native_agents_enabled(_rt(tmp_path)) is False
    ext = {"cc": SimpleNamespace(provider="external")}
    assert native_agents_enabled(_rt(tmp_path, ext)) is False
    native = {**ext, "kevin": SimpleNamespace(provider="claude_max")}
    assert native_agents_enabled(_rt(tmp_path, native)) is True


def test_the_owner_choice_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("ATN_NATIVE_AGENTS", raising=False)
    rt = _rt(tmp_path, {"kevin": SimpleNamespace(provider="claude_max")})
    set_native_agents(rt, False)
    assert native_agents_enabled(rt) is False
    set_native_agents(rt, True)
    assert native_agents_enabled(_rt(tmp_path)) is True


@pytest.mark.asyncio
async def test_create_refuses_native_agents_when_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_NATIVE_AGENTS", "0")
    from atn.agent_tools import execute_tool
    from tests.atn.test_tool_store import _make_runtime
    rt = _make_runtime(tmp_path)
    out = await execute_tool("create_agent", {
        "id": "n1", "name": "N", "prompt": "p", "provider": "claude_max",
        "model": "claude-sonnet-5"}, rt, caller_id="")
    assert "native agents are off" in out["error"]
    out = await execute_tool("create_agent", {
        "id": "e1", "name": "E", "provider": "external"}, rt, caller_id="")
    assert out.get("external") is True
