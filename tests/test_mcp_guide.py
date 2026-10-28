"""The MCP proxy tells a connecting agent how to use Autonet."""
import asyncio
import importlib

from mcp import types


def _server(monkeypatch, **env):
    for k in ("ATN_MCP_TOKEN", "ATN_MCP_AGENT", "ATN_MCP_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import atn.mcp_server as m
    return importlib.reload(m)


def test_instructions_point_to_the_guide(monkeypatch):
    m = _server(monkeypatch)
    opts = m.create_server().create_initialization_options()
    assert "atn_guide" in (opts.instructions or "")


def test_guide_tool_answers_without_a_daemon(monkeypatch):
    m = _server(monkeypatch)
    srv = m.create_server()
    handler = srv.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(method="tools/call",
                                params=types.CallToolRequestParams(name="atn_guide", arguments={}))
    res = asyncio.run(handler(req))
    text = res.root.content[0].text
    assert "novelty claim" in text and "breaks through" in text


def test_agent_env_names_the_caller(monkeypatch):
    m = _server(monkeypatch, ATN_MCP_AGENT="my-agent")
    sent = {}

    class _WS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def send(self, raw):
            import json
            sent.update(json.loads(raw))

        def __aiter__(self):
            async def gen():
                yield '{"msg_id": "%s", "ok": true, "result": {}}' % sent["msg_id"]
            return gen()

    monkeypatch.setattr(m, "_connect", lambda _ws: _WS())
    asyncio.run(m._call_daemon("probe_tools", {"query": "csv"}))
    assert sent["caller_id"] == "my-agent"
    sent.clear()
    asyncio.run(m._call_daemon("create_agent", {"id": "x"}))
    assert "caller_id" not in sent
