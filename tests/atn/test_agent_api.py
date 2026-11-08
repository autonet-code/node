"""The agent HTTP API (atn/agent_api.py) over a real socket, with a stub
bridge standing in for the daemon's message handler."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from atn.agent_api import AgentAPI


class _StubBridge:
    def __init__(self, data_dir):
        from atn import agent_tokens
        self.calls = []
        agents = {"ext": SimpleNamespace(provider="external", name="Ext",
                                         description="claude-code")}
        self.runtime = SimpleNamespace(
            registry=SimpleNamespace(_agents=agents,
                                     get_subtree_ids=lambda aid: {aid}),
            _config=SimpleNamespace(data_dir=data_dir))
        self.token = agent_tokens.issue(data_dir, "ext")
        self.auth = {"Authorization": f"Bearer {self.token}"}

    async def _handle_message(self, msg, session):
        self.calls.append((msg, session))
        t = msg["type"]
        if t == "probe_tools":
            return {"ok": True, "result": {"matches": [
                {"digest": "ab" * 32, "name": "csv_parse"}], "source": "local"}}
        if t == "create_agent":
            self.runtime.registry._agents[msg["id"]] = SimpleNamespace(
                provider=msg["provider"])
            return {"ok": True, "result": {"agent_id": msg["id"]}}
        if t == "use_tool" and not msg.get("name"):
            return {"ok": False, "error": "Missing 'name'"}
        return {"ok": True, "result": {"echo": t}}


async def _request(port, method, path, body=None, headers=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    h = {"Host": f"localhost:{port}", "Content-Length": str(len(data))}
    h.update(headers or {})
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(
        f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
    writer.write(head.encode() + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _, payload = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1])
    ctype = [l for l in head.split(b"\r\n") if l.lower().startswith(b"content-type")][0]
    if b"json" in ctype:
        return status, json.loads(payload)
    return status, payload.decode()


@pytest.fixture
def api_run(tmp_path):
    def run(coro_fn):
        async def main():
            bridge = _StubBridge(tmp_path)
            api = AgentAPI(bridge, "127.0.0.1", 0)
            api._server = await asyncio.start_server(api._serve, "127.0.0.1", 0)
            port = api._server.sockets[0].getsockname()[1]
            try:
                return await coro_fn(port, bridge)
            finally:
                await api.stop()
        return asyncio.run(main())
    return run


def test_index_guide_and_commands(api_run):
    async def go(port, bridge):
        s, index = await _request(port, "GET", "/")
        assert s == 200 and "POST /agents" in json.dumps(index)
        s, guide = await _request(port, "GET", "/guide")
        assert s == 200 and guide.startswith("#")
        s, cmds = await _request(port, "GET", "/commands")
        names = {c["name"] for g in cmds["groups"].values() for c in g}
        # Built from the daemon's tool definitions plus the network reads.
        assert {"adopt_tool", "attest_tools", "find_services",
                "autonet_status"} <= names
        # Running tools inside the daemon is for its own agents.
        assert "use_tool" not in names and "vet_tool" not in names
        assert "export_agent_key" not in names and "create_agent" not in names
        s, one = await _request(port, "GET", "/commands/use_tool")
        assert s == 200 and one["input_schema"]["type"] == "object"
        s, _ = await _request(port, "GET", "/commands/nope")
        assert s == 404
    api_run(go)


def test_agent_calls_are_scoped_to_the_agent(api_run):
    async def go(port, bridge):
        s, r = await _request(port, "POST", "/agents/ext/use_tool",
                              {"name": "csv_parse", "arguments": {}}, headers=bridge.auth)
        assert s == 200 and r == {"echo": "use_tool"}
        msg, session = bridge.calls[-1]
        assert msg["caller_id"] == "ext"
        assert session.root_agent_id == "ext" and session.scope_ids == {"ext"}
        assert not session.local and not session.owner

        s, r = await _request(port, "POST", "/agents/ext/use_tool", {}, headers=bridge.auth)
        assert s == 400 and r["help"] == "/commands/use_tool"

        s, r = await _request(port, "GET", "/agents/ext/tools?q=parse%20csv", headers=bridge.auth)
        assert r["matches"][0]["next"]["check_out"].startswith("POST /agents/ext/adopt_tool")

        s, r = await _request(port, "POST", "/agents/ext/export_agent_key", {}, headers=bridge.auth)
        assert s in (403, 404)
        s, r = await _request(port, "GET", "/agents/ghost")
        assert s == 404 and "create" in r
    api_run(go)


def test_create_agent_is_always_external(api_run):
    async def go(port, bridge):
        s, r = await _request(port, "POST", "/agents",
                              {"id": "cc1", "name": "CC", "provider": "anthropic",
                               "prompt": "x"})
        assert s == 200 and r["next"] == "/agents/cc1"
        msg, session = bridge.calls[-1]
        assert msg["provider"] == "external" and "prompt" not in msg
        assert session.owner
    api_run(go)


def test_refuses_browsers_and_foreign_hosts(api_run):
    async def go(port, bridge):
        s, _ = await _request(port, "GET", "/", headers={"Origin": "https://evil.example"})
        assert s == 403
        s, _ = await _request(port, "GET", "/", headers={"Host": "evil.example"})
        assert s == 403
        assert bridge.calls == []
    api_run(go)


def test_join_recommended_jurisdiction_is_logged_as_the_agent(api_run):
    async def go(port, bridge):
        bridge.runtime.jurisdiction = ""
        s, index = await _request(port, "GET", "/")
        assert "join_recommended_jurisdiction" in index["jurisdiction"]
        s, r = await _request(port, "POST",
                              "/agents/ext/join_recommended_jurisdiction", {}, headers=bridge.auth)
        assert s == 200
        msg, session = bridge.calls[-1]
        assert msg["type"] == "autonet_switch" and msg["recommended"] is True
        assert msg["caller_id"] == "ext" and session.owner
    api_run(go)


def test_recommended_jurisdiction_prefers_the_repo_then_the_release(monkeypatch):
    from atn import config
    doc = {"jurisdictions": {"autonet": {
        "name": "Repo DAO",
        "network": {"chain_id": 31337, "rpc_url": "http://x"},
        "contracts": {"dao": "0x" + "11" * 20, "dao_registry": "0x" + "22" * 20}}}}
    monkeypatch.setattr(config, "_fetch_registry", lambda url: doc)
    rec = config.recommended_jurisdiction()
    assert rec["name"] == "Repo DAO" and rec["source"] == config.RECOMMENDED_REGISTRY_URL
    assert "packaged" not in rec
    monkeypatch.setattr(config, "_fetch_registry", lambda url: None)
    rec = config.recommended_jurisdiction()
    assert rec is None or rec["source"] == "release"


def test_switch_to_recommended_pins_the_dao(tmp_path, monkeypatch):
    from atn import config, ws_server
    from atn.ws_auth import ClientSession
    rec = {"chain_id": 31337, "rpc_url": "http://x", "name": "Repo DAO",
           "dao_registry": "0x" + "22" * 20, "dao": "0x" + "11" * 20,
           "source": "repo"}
    monkeypatch.setattr(config, "recommended_jurisdiction", lambda: rec)
    monkeypatch.setattr(config, "check_jurisdiction", lambda reg, rpc: {
        "ok": True, "errors": [], "chain_id": 31337, "rpc_url": rpc,
        "dao_registry": reg, "dao": rec["dao"], "key": "31337:" + rec["dao"]})
    bridge = ws_server.WebSocketBridge.__new__(ws_server.WebSocketBridge)
    bridge.runtime = SimpleNamespace(
        jurisdiction="", registry=SimpleNamespace(_agents={}),
        _config=SimpleNamespace(data_dir=tmp_path))
    owner = ClientSession(local=True, authed=True, owner=True)
    resp = asyncio.run(bridge._handle_message(
        {"type": "autonet_switch", "recommended": True, "restart": False}, owner))
    assert resp["ok"] and resp["result"]["switched"]
    saved = json.loads((tmp_path / "jurisdiction.json").read_text())
    assert saved["dao"] == rec["dao"] and saved["name"] == "Repo DAO"
    assert "packaged" not in saved


def test_tokens_guard_every_agent_call(api_run):
    async def go(port, bridge):
        s, r = await _request(port, "POST", "/agents/ext/use_tool", {"name": "x"})
        assert s == 401 and "atn agent-token ext" in r["lost"]
        s, r = await _request(port, "POST", "/agents/ext/use_tool", {"name": "x"},
                              headers={"Authorization": "Bearer nope"})
        assert s == 401
        # A new agent gets its token once, at creation.
        s, r = await _request(port, "POST", "/agents", {"id": "cc2"})
        tok = r["token"]
        assert tok.startswith("atn_")
        s, r = await _request(port, "GET", "/agents/cc2/autonet_status",
                              headers={"Authorization": f"Bearer {tok}"})
        assert s == 200
        # An agent made elsewhere claims its token once; later it's the owner's.
        bridge.runtime.registry._agents["app_made"] = SimpleNamespace(
            provider="external", name="A", description="")
        s, r = await _request(port, "POST", "/agents/app_made/token", {})
        assert s == 200 and r["token"].startswith("atn_")
        s, r = await _request(port, "POST", "/agents/app_made/token", {})
        assert s == 403
    api_run(go)


def test_listing_notes_and_the_short_command_list(api_run):
    async def go(port, bridge):
        s, r = await _request(port, "GET", "/agents")
        assert [a["id"] for a in r["agents"]] == ["ext"]
        assert r["agents"][0]["has_token"] is True
        assert r["agents"][0]["harness"] == "claude-code"
        await _request(port, "POST", "/agents/ext/use_tool",
                       {"name": "x", "note": "checking before I write my own"},
                       headers=bridge.auth)
        msg, _ = bridge.calls[-1]
        assert msg["_note"] == "checking before I write my own" and "note" not in msg
        s, short = await _request(port, "GET", "/commands")
        s, full = await _request(port, "GET", "/commands?all=1")
        assert "other" not in short["groups"] and "more" in short
        assert "other" in full["groups"]
    api_run(go)


def test_every_agent_reply_carries_review_due(api_run):
    """Each reply to an agent names the tools it adopted and has not
    reviewed (Decision 2026-11-08), whatever the call was."""
    async def go(port, bridge):
        rec = SimpleNamespace(digest="cd" * 32, name="word_count")
        bridge.runtime.tool_store = SimpleNamespace(
            adopted_by=lambda aid: [rec] if aid == "ext" else [])
        s, r = await _request(port, "POST", "/agents/ext/autonet_status", {},
                              headers=bridge.auth)
        assert s == 200 and r == {"echo": "autonet_status",
                                  "review_due": ["word_count"]}
        # The owner's own calls are not reminded.
        s, index = await _request(port, "GET", "/")
        assert "review_due" not in index
    api_run(go)
