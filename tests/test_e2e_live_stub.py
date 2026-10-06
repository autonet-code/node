"""The live E2E harness stub (scripts/e2e_live/stub_server.py) speaks what the
daemon's OpenAICompatibleProvider expects, on both its paths, and serves the
local-only registry."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from pathlib import Path

import pytest

from atn.providers.base import ToolDefinition
from atn.providers.openai_compat import OpenAICompatibleProvider

_STUB = Path(__file__).resolve().parent.parent / "scripts" / "e2e_live" / "stub_server.py"
_spec = importlib.util.spec_from_file_location("e2e_live_stub", _STUB)
stub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stub)  # type: ignore[union-attr]


@pytest.fixture()
def server(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"jurisdictions": {"autonet": {
        "network": {"rpc_url": "http://127.0.0.1:18545", "chain_id": 1337}}}}))
    srv = stub.serve("127.0.0.1", 0, reg)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _provider(url: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(name="e2e_stub", base_url=f"{url}/v1",
                                    api_key="e2e", default_model="echo-1")


def test_plain_completion_echoes_marker(server):
    resp = asyncio.run(_provider(server).send(
        messages=[{"role": "user", "content": "hello there"}]))
    assert resp.text == f"{stub.MARKER} hello there"
    assert resp.stop_reason == "end_turn"
    assert not resp.tool_calls


def test_streamed_completion_matches_plain(server):
    chunks: list[str] = []

    async def on_chunk(c: str) -> None:
        chunks.append(c)

    resp = asyncio.run(_provider(server).send_stream(
        messages=[{"role": "user", "content": "stream me"}], on_chunk=on_chunk))
    assert resp.text == f"{stub.MARKER} stream me"
    assert "".join(chunks) == resp.text
    assert resp.usage.output_tokens > 0


def test_registry_is_served_verbatim(server):
    import httpx
    data = httpx.get(f"{server}/registry.json").json()
    assert data["jurisdictions"]["autonet"]["network"]["chain_id"] == 1337
    assert httpx.get(f"{server}/health").json()["ok"] is True


_TOOLS = [ToolDefinition(name="echo_upper", description="Upper-case text",
                         input_schema={"type": "object", "properties": {}})]


@pytest.mark.parametrize("streamed", [False, True])
def test_tool_trigger_calls_a_tool_then_finishes_with_text(server, streamed):
    prov = _provider(server)
    msgs = [{"role": "user", "content": "please E2E-TOOL now"}]

    async def on_chunk(_c: str) -> None:
        pass

    if streamed:
        resp = asyncio.run(prov.send_stream(messages=msgs, tools=_TOOLS,
                                            on_chunk=on_chunk))
    else:
        resp = asyncio.run(prov.send(messages=msgs, tools=_TOOLS))
    assert resp.stop_reason == "tool_use"
    assert [tc.name for tc in resp.tool_calls] == ["echo_upper"]
    # With the tool result as the newest message, the turn ends in text.
    body = {"messages": [
        {"role": "user", "content": "please E2E-TOOL now"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "x", "content": "OK"},
    ]}
    assert stub.mode_for(body) == "text"


def test_tool_trigger_names_a_specific_tool():
    body = {"messages": [{"role": "user", "content": "E2E-TOOL:atn_shell go"}],
            "tools": [{"type": "function", "function": {"name": "other"}}]}
    assert stub.tool_call_for(body)["function"]["name"] == "atn_shell"


def test_fail_trigger_surfaces_a_provider_error(server):
    with pytest.raises(Exception) as exc:
        asyncio.run(_provider(server).send(
            messages=[{"role": "user", "content": "E2E-FAIL please"}]))
    assert stub.FAIL_MESSAGE in str(exc.value)


def test_stall_trigger_holds_until_shutdown():
    import httpx
    srv = stub.serve("127.0.0.1", 0, None, stall_seconds=30)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    try:
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(url, json={"messages": [
                {"role": "user", "content": "E2E-STALL"}]}, timeout=1.0)
    finally:
        srv.shutdown()
        srv.server_close()
