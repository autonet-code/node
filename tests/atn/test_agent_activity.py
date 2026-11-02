"""External agents' activity log (atn/agent_activity.py) and the ws path
that fills it."""
import asyncio
from types import SimpleNamespace

from atn import agent_activity
from atn.agent_activity import ActivityLog


def test_record_and_read_newest_first(tmp_path):
    log = ActivityLog(tmp_path)
    log.record("ext-1", "probe_tools", {"query": "csv", "caller_id": "ext-1"},
               True, {"tools": []}, 1.0, 1.5)
    log.record("ext-1", "use_tool", {"tool_id": "t"}, False, "boom", 2.0, 2.1)
    out = log.read("ext-1")
    assert [e["tool"] for e in out] == ["use_tool", "probe_tools"]
    assert out[0]["ok"] is False and out[0]["result"] == "boom"
    assert "caller_id" not in out[1]["input"]
    assert out[1]["duration_ms"] == 500
    assert log.read("nobody") == []


def test_trims_and_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_activity, "KEEP", 3)
    log = ActivityLog(tmp_path)
    for i in range(10):
        log.record("a", f"t{i}", {}, True, "x" * 5000, 0, 0)
    out = log.read("a", limit=100)
    assert [e["tool"] for e in out] == ["t9", "t8", "t7"]
    assert len(out[0]["result"]) <= agent_activity.PREVIEW_CHARS + 3


def test_ws_logs_external_calls_only(tmp_path, monkeypatch):
    from atn import ws_server
    emitted = []

    async def emit(ev):
        emitted.append(ev)

    async def fake_execute(tool, args, runtime, caller_id=""):
        return {"ok": True, "tool": tool}

    monkeypatch.setattr(ws_server, "execute_tool", fake_execute)
    agents = {"ext": SimpleNamespace(provider="external"),
              "nat": SimpleNamespace(provider="anthropic")}
    bridge = ws_server.WebSocketBridge.__new__(ws_server.WebSocketBridge)
    bridge.runtime = SimpleNamespace(
        registry=SimpleNamespace(_agents=agents),
        events=SimpleNamespace(emit=emit),
        _config=SimpleNamespace(data_dir=tmp_path))
    assert bridge._is_external_agent("ext")
    assert not bridge._is_external_agent("nat")
    assert not bridge._is_external_agent("")

    res = asyncio.run(bridge._execute_logged("probe_tools", {"query": "q"}, "ext"))
    assert res["tool"] == "probe_tools"
    entries = bridge._activity_log().read("ext")
    assert entries[0]["tool"] == "probe_tools" and entries[0]["ok"]
    assert [e.type.value for e in emitted] == [
        "agent.tool_use_start", "agent.tool_use_result"]
