"""The agent HTTP API (atn/agent_api.py) over a real socket, with a stub
bridge standing in for the daemon's message handler."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from atn.agent_api import AgentAPI


class _StubBridge:
    def __init__(self):
        self.calls = []
        agents = {"ext": SimpleNamespace(provider="external")}
        self.runtime = SimpleNamespace(registry=SimpleNamespace(
            _agents=agents, get_subtree_ids=lambda aid: {aid}))

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
def api_run():
    def run(coro_fn):
        async def main():
            bridge = _StubBridge()
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
        assert {"use_tool", "attest_tools", "find_services",
                "autonet_status"} <= names
        assert "export_agent_key" not in names and "create_agent" not in names
        s, one = await _request(port, "GET", "/commands/use_tool")
        assert s == 200 and one["input_schema"]["type"] == "object"
        s, _ = await _request(port, "GET", "/commands/nope")
        assert s == 404
    api_run(go)


def test_agent_calls_are_scoped_to_the_agent(api_run):
    async def go(port, bridge):
        s, r = await _request(port, "POST", "/agents/ext/use_tool",
                              {"name": "csv_parse", "arguments": {}})
        assert s == 200 and r == {"echo": "use_tool"}
        msg, session = bridge.calls[-1]
        assert msg["caller_id"] == "ext"
        assert session.root_agent_id == "ext" and session.scope_ids == {"ext"}
        assert not session.local and not session.owner

        s, r = await _request(port, "POST", "/agents/ext/use_tool", {})
        assert s == 400 and r["help"] == "/commands/use_tool"

        s, r = await _request(port, "GET", "/agents/ext/tools?q=parse%20csv")
        assert r["matches"][0]["next"]["use"].startswith("POST /agents/ext/use_tool")

        s, r = await _request(port, "POST", "/agents/ext/export_agent_key", {})
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
