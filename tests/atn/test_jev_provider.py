"""Jev provider: typed-question protocol, transport, usage/cost, and the
tool-binding refusal. The TypeSafe HTTP transport is mocked throughout."""
from __future__ import annotations

import json

import httpx
import pytest

from atn.agent_tools import _create_agent, _update_agent
from atn.config import ATNConfig, ProviderConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.provider_identity import routing_error, tool_binding_error
from atn.providers import jev as jev_mod
from atn.providers.base import ProviderError
from atn.providers.jev import JevProvider, build_request
from atn.runtime import Runtime

ROUTE = {"route": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "payments", "tech": "bugs"}}}
SYSTEM = "You route support tickets.\n\n```jev\n" + json.dumps(ROUTE) + "\n```\n"
ANSWER = {"route": {"choice": "billing",
                    "probabilities": {"billing": 0.9, "tech": 0.1},
                    "confidence": 0.8}}


def _mock_transport(monkeypatch, responses):
    """Patch httpx.AsyncClient in the jev module; returns the captured
    requests. ``responses`` is a list of httpx.Response, served in order."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(jev_mod.httpx, "AsyncClient", factory)

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(jev_mod.asyncio, "sleep", _no_sleep)
    return seen


def _ok(answers=ANSWER, input_tokens=120):
    return httpx.Response(200, json={
        "answers": answers, "model": "jev-1.13.0",
        "usage": {"input_tokens": input_tokens, "output_tokens": 8}})


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def test_system_block_declares_standing_questions():
    state, questions = build_request(
        [{"role": "user", "content": "My payout failed"}], SYSTEM)
    assert questions == ROUTE
    assert state == {"message": "My payout failed",
                     "instructions": "You route support tickets."}


def test_message_can_carry_a_full_request():
    req = {"state": {"ticket": "refund please"},
           "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}}
    state, questions = build_request(
        [{"role": "user", "content": json.dumps(req)}], SYSTEM)
    assert state == {"ticket": "refund please"}
    assert list(questions) == ["urgent"]


def test_no_questions_fails_loud():
    with pytest.raises(ProviderError, match="cannot generate text"):
        build_request([{"role": "user", "content": "hello"}], "be nice")


def test_bad_question_type_rejected():
    bad = "```jev\n" + json.dumps({"q": {"type": "essay", "instructions": "x"}}) + "\n```"
    with pytest.raises(ProviderError, match="type one of"):
        build_request([{"role": "user", "content": "x"}], bad)


# ---------------------------------------------------------------------------
# send / run_agent_loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_posts_typed_request_and_parses_answers(monkeypatch):
    seen = _mock_transport(monkeypatch, [_ok()])
    p = JevProvider(api_key="sk-test")
    r = await p.send(messages=[{"role": "user", "content": "payout failed"}],
                     system=SYSTEM, model="jev-latest")
    req = seen[0]
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "jev-latest"
    assert body["questions"] == ROUTE
    assert json.loads(r.text) == ANSWER
    assert r.model == "jev-1.13.0"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (120, 8)
    assert r.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_no_model_and_tools_are_refused(monkeypatch):
    _mock_transport(monkeypatch, [])
    p = JevProvider(api_key="k")
    with pytest.raises(ProviderError, match="No model specified"):
        await p.send(messages=[{"role": "user", "content": "x"}], system=SYSTEM)
    from atn.providers.base import ToolDefinition
    with pytest.raises(ProviderError, match="cannot call tools"):
        await p.send(messages=[{"role": "user", "content": "x"}], system=SYSTEM,
                     model="jev-latest",
                     tools=[ToolDefinition("t", "d", {"type": "object"})])


def test_missing_key_and_loop_flag():
    with pytest.raises(ProviderError, match="API key"):
        JevProvider(api_key="")
    assert JevProvider(api_key="k").supports_agent_loop is False


@pytest.mark.asyncio
async def test_retries_429_then_succeeds(monkeypatch):
    seen = _mock_transport(monkeypatch, [
        httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow"}),
        _ok()])
    p = JevProvider(api_key="k")
    r = await p.send(messages=[{"role": "user", "content": "x"}],
                     system=SYSTEM, model="jev-latest")
    assert len(seen) == 2 and json.loads(r.text) == ANSWER


@pytest.mark.asyncio
async def test_http_error_is_provider_error(monkeypatch):
    _mock_transport(monkeypatch, [httpx.Response(401, json={"error": {"message": "bad key"}})])
    p = JevProvider(api_key="k")
    with pytest.raises(ProviderError, match="401: bad key"):
        await p.send(messages=[{"role": "user", "content": "x"}],
                     system=SYSTEM, model="jev-latest")


@pytest.mark.asyncio
async def test_agent_loop_is_one_judgment_with_usage_and_cost(monkeypatch):
    seen = _mock_transport(monkeypatch, [_ok(input_tokens=1_000_000)])
    p = JevProvider(api_key="k", default_model="jev-latest")
    chunks: list[str] = []
    booked: list[tuple[int, str]] = []

    async def on_chunk(t):
        chunks.append(t)

    def recorder(tokens, model):
        booked.append((tokens, model))
        return True, None

    called_tools: list[str] = []

    async def tool_executor(name, _input):
        called_tools.append(name)
        return {}

    r = await p.run_agent_loop(
        message="payout failed", system=SYSTEM,
        tools=[{"name": "post_message", "description": "", "input_schema": {}}],
        tool_executor=tool_executor, on_chunk=on_chunk,
        usage_recorder=recorder, max_turns=50)
    assert len(seen) == 1 and not called_tools
    assert chunks == [r.text] and json.loads(r.text) == ANSWER
    assert booked == [(1_000_008, "jev-1.13.0")]
    stats = p.session_stats
    assert stats["num_turns"] == 1
    assert stats["cumulative_input_tokens"] == 1_000_000
    # $0.042 per million input tokens, output free.
    assert stats["total_cost_usd"] == pytest.approx(0.042)


# ---------------------------------------------------------------------------
# Daemon wiring: routing, tool binding, catalog
# ---------------------------------------------------------------------------

def _runtime(tmp_path, *, key="sk-test") -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    if key:
        config.providers["jev"] = ProviderConfig(name="jev", api_key=key)
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def test_tool_binding_rule():
    assert routing_error("jev", "jev-latest") is None
    assert routing_error("jev", "") is not None  # model still required
    assert tool_binding_error("jev", []) is None
    assert "cannot call tools" in tool_binding_error("jev", ["atn_core"])
    assert "cannot call tools" in tool_binding_error(["openai", "jev"], ["web"])
    assert tool_binding_error("openai", ["atn_core"]) is None


@pytest.mark.asyncio
async def test_create_defaults_jev_agent_to_no_tools(tmp_path):
    rt = _runtime(tmp_path)
    r = await _create_agent(rt, {"id": "router", "name": "router",
                                 "provider": "jev", "model": "jev-latest",
                                 "system_prompt": SYSTEM})
    assert r.get("status") == "registered", r
    assert rt.get_agent("router").tools == []

    r = await _create_agent(rt, {"id": "busy", "name": "busy",
                                 "provider": "jev", "model": "jev-latest",
                                 "tools": ["atn_core"]})
    assert "cannot call tools" in r["error"]
    assert rt.get_agent("busy") is None


@pytest.mark.asyncio
async def test_update_refuses_tools_on_jev(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "router", "name": "router",
                             "provider": "jev", "model": "jev-latest"})
    r = await _update_agent(rt, {"agent_id": "router", "tools": ["atn_core"]})
    assert "cannot call tools" in r["error"]


def test_resolve_builds_jev_and_refuses_tooled_agent(tmp_path):
    rt = _runtime(tmp_path)
    ok = AgentDefinition(id="r", name="r", mode=AgentMode.COGNITIVE,
                         provider="jev", cognitive_model="jev-latest", tools=[])
    prov = rt.providers.resolve_provider_with_fallback(ok)
    assert isinstance(prov, JevProvider)
    assert prov.provider_id == "jev" and prov._default_model == "jev-latest"
    bad = AgentDefinition(id="b", name="b", mode=AgentMode.COGNITIVE,
                          provider="jev", cognitive_model="jev-latest",
                          tools=["atn_core"])
    with pytest.raises(ProviderError, match="cannot call tools"):
        rt.providers.resolve_provider_with_fallback(bad)


def test_no_key_fails_loud_at_resolve(tmp_path):
    rt = _runtime(tmp_path, key="")
    defn = AgentDefinition(id="r", name="r", mode=AgentMode.COGNITIVE,
                           provider="jev", cognitive_model="jev-latest")
    with pytest.raises(ProviderError, match="API key"):
        rt.providers.resolve_provider_with_fallback(defn)


@pytest.mark.asyncio
async def test_provider_card_and_catalog(tmp_path):
    rt = _runtime(tmp_path)
    cards = {c["id"]: c for c in await rt.providers.provider_list(probe=False)}
    card = cards["jev"]
    assert card["auth_type"] == "api_key" and card["configured"] is True
    ids = [m["id"] for m in card["models"]]
    assert ids == ["jev-latest", "jev-1.13.0", "jev-preview"]
    assert all(m["capability_tier"] == 1 for m in card["models"])
