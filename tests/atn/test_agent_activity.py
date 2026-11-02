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


def _bridge(tmp_path, emitted):
    from atn import ws_server

    async def emit(ev):
        emitted.append(ev)

    agents = {"ext": SimpleNamespace(provider="external"),
              "nat": SimpleNamespace(provider="anthropic")}
    bridge = ws_server.WebSocketBridge.__new__(ws_server.WebSocketBridge)
    bridge.runtime = SimpleNamespace(
        registry=SimpleNamespace(_agents=agents),
        events=SimpleNamespace(emit=emit),
        _config=SimpleNamespace(data_dir=tmp_path))
    return bridge


def test_external_calls_are_logged_once(tmp_path, monkeypatch):
    from atn import ws_server
    from atn.ws_auth import ClientSession
    emitted = []
    bridge = _bridge(tmp_path, emitted)
    assert bridge._is_external_agent("ext")
    assert not bridge._is_external_agent("nat")
    assert not bridge._is_external_agent("")

    seen = []
    real = ws_server.WebSocketBridge._handle_message

    async def handle(self, msg, session):
        if not msg.get("_activity_logged"):
            return await real(self, msg, session)
        seen.append(msg)
        return {"msg_id": msg.get("msg_id"), "ok": True, "result": {"n": 1}}

    monkeypatch.setattr(ws_server.WebSocketBridge, "_handle_message", handle)
    owner = ClientSession(local=True, authed=True, owner=True)
    resp = asyncio.run(bridge._handle_message(
        {"type": "probe_tools", "caller_id": "ext", "query": "q",
         "_note": "looking first"}, owner))
    assert resp["ok"] and len(seen) == 1
    entries = bridge._activity_log().read("ext")
    assert len(entries) == 1 and entries[0]["tool"] == "probe_tools"
    assert '"query": "q"' in entries[0]["input"]
    assert entries[0]["note"] == "looking first" and "_note" not in entries[0]["input"]
    assert [e.type.value for e in emitted] == [
        "agent.tool_use_start", "agent.tool_use_result"]

    # A native agent's call and the log read itself are not logged.
    asyncio.run(bridge._handle_message(
        {"type": "tool_reviews", "caller_id": "nat"}, owner))
    asyncio.run(bridge._handle_message(
        {"type": "agent_activity", "caller_id": "ext", "agent_id": "ext"}, owner))
    assert len(bridge._activity_log().read("ext")) == 1
    assert bridge._activity_log().read("nat") == []
