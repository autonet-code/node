"""Tests for the provider base abstraction — data types, interface, and orchestrate loop."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atn.providers.base import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestUsage:
    def test_fields_basic(self):
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.cache_read_tokens == 0
        assert u.cache_creation_tokens == 0

    def test_fields_with_cache(self):
        u = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=30, cache_creation_tokens=20)
        assert u.input_tokens == 100
        assert u.cache_read_tokens == 30
        assert u.cache_creation_tokens == 20

    def test_defaults_to_zero(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0


class TestToolDefinition:
    def test_fields(self):
        td = ToolDefinition(name="read_file", description="Read a file", input_schema={"type": "object"})
        assert td.name == "read_file"
        assert td.description == "Read a file"
        assert td.input_schema == {"type": "object"}


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(id="tc_1", name="write", input={"path": "/tmp/x"})
        assert tc.id == "tc_1"
        assert tc.name == "write"
        assert tc.input == {"path": "/tmp/x"}


class TestProviderResponse:
    def test_defaults(self):
        r = ProviderResponse()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.stop_reason == ""
        assert r.usage.input_tokens == 0
        assert r.usage.output_tokens == 0
        assert r.thinking == []

    def test_with_tool_calls(self):
        tc = ToolCall(id="1", name="bash", input={"command": "ls"})
        r = ProviderResponse(text="Running...", tool_calls=[tc], stop_reason="tool_use")
        assert len(r.tool_calls) == 1
        assert r.stop_reason == "tool_use"


class TestProviderError:
    def test_basic(self):
        err = ProviderError("bad request", status_code=400, provider="anthropic")
        assert str(err) == "bad request"
        assert err.status_code == 400
        assert err.provider == "anthropic"

    def test_defaults(self):
        err = ProviderError("fail")
        assert err.status_code is None
        assert err.provider == ""


# ---------------------------------------------------------------------------
# Abstract Provider interface
# ---------------------------------------------------------------------------

class ConcreteProvider(Provider):
    """Minimal concrete provider for testing the base class."""

    @property
    def name(self) -> str:
        return "test"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        return ProviderResponse(text="hello", usage=Usage(input_tokens=10, output_tokens=5))


class TestProviderInterface:
    @pytest.mark.asyncio
    async def test_send(self):
        p = ConcreteProvider()
        resp = await p.send(messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hello"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_name(self):
        p = ConcreteProvider()
        assert p.name == "test"

    def test_supports_orchestrate_default_true(self):
        p = ConcreteProvider()
        assert p.supports_orchestrate is True

    @pytest.mark.asyncio
    async def test_send_stream_falls_back_to_send(self):
        p = ConcreteProvider()
        chunks = []

        async def on_chunk(text: str):
            chunks.append(text)

        resp = await p.send_stream(
            messages=[{"role": "user", "content": "hi"}],
            on_chunk=on_chunk,
        )
        assert resp.text == "hello"
        assert chunks == ["hello"]


# ---------------------------------------------------------------------------
# send_orchestrate multi-turn loop
# ---------------------------------------------------------------------------

class OrchestrateProvider(Provider):
    """Provider that returns tool calls on first send, then end_turn."""

    def __init__(self, responses: list[ProviderResponse]):
        self._responses = list(responses)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "orchestrate_test"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class TestSendOrchestrate:
    @pytest.mark.asyncio
    async def test_single_turn_no_tools(self):
        """If no tool calls, returns immediately."""
        p = OrchestrateProvider([
            ProviderResponse(text="Done", stop_reason="end_turn",
                             usage=Usage(input_tokens=10, output_tokens=5)),
        ])
        resp = await p.send_orchestrate(
            message="hello", tools=[], max_turns=5,
        )
        assert resp.text == "Done"
        assert resp.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_multi_turn_with_tool_calls(self):
        """Tool calls trigger executor, results feed back in."""
        tc = ToolCall(id="tc1", name="read_file", input={"path": "/tmp"})
        p = OrchestrateProvider([
            # Turn 1: LLM asks to call a tool
            ProviderResponse(
                text="Let me read that.",
                tool_calls=[tc],
                stop_reason="tool_use",
                usage=Usage(input_tokens=20, output_tokens=10),
            ),
            # Turn 2: LLM produces final answer
            ProviderResponse(
                text="The file contains: hello",
                stop_reason="end_turn",
                usage=Usage(input_tokens=30, output_tokens=15),
            ),
        ])

        executor_calls = []

        async def executor(name, inp):
            executor_calls.append((name, inp))
            return {"content": "hello"}

        resp = await p.send_orchestrate(
            message="Read /tmp",
            tools=[{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}}],
            max_turns=5,
            tool_executor=executor,
        )

        assert resp.text == "The file contains: hello"
        assert len(executor_calls) == 1
        assert executor_calls[0] == ("read_file", {"path": "/tmp"})
        # Usage is accumulated
        assert resp.usage.input_tokens == 50
        assert resp.usage.output_tokens == 25

    @pytest.mark.asyncio
    async def test_tool_executor_error_is_caught(self):
        """If tool executor raises, error is returned to LLM."""
        tc = ToolCall(id="tc1", name="bad_tool", input={})
        p = OrchestrateProvider([
            ProviderResponse(tool_calls=[tc], stop_reason="tool_use",
                             usage=Usage(input_tokens=10, output_tokens=5)),
            ProviderResponse(text="Tool failed", stop_reason="end_turn",
                             usage=Usage(input_tokens=15, output_tokens=10)),
        ])

        async def failing_executor(name, inp):
            raise RuntimeError("boom")

        resp = await p.send_orchestrate(
            message="try",
            tools=[{"name": "bad_tool", "description": "Fails", "input_schema": {"type": "object"}}],
            tool_executor=failing_executor,
            max_turns=5,
        )
        assert resp.text == "Tool failed"

    @pytest.mark.asyncio
    async def test_no_executor_returns_tool_calls(self):
        """Without executor, tool calls are returned but not executed."""
        tc = ToolCall(id="tc1", name="bash", input={"cmd": "ls"})
        p = OrchestrateProvider([
            ProviderResponse(tool_calls=[tc], stop_reason="tool_use",
                             usage=Usage(input_tokens=10, output_tokens=5)),
        ])
        resp = await p.send_orchestrate(
            message="run ls",
            tools=[{"name": "bash", "description": "Run", "input_schema": {"type": "object"}}],
            tool_executor=None,
            max_turns=5,
        )
        assert len(resp.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_max_turns_exhausted(self):
        """If max_turns is hit, returns last response."""
        tc = ToolCall(id="tc1", name="loop", input={})
        # Every turn returns a tool call
        responses = [
            ProviderResponse(tool_calls=[tc], stop_reason="tool_use",
                             usage=Usage(input_tokens=5, output_tokens=5))
            for _ in range(3)
        ]
        p = OrchestrateProvider(responses)

        async def executor(name, inp):
            return {"ok": True}

        resp = await p.send_orchestrate(
            message="loop forever",
            tools=[{"name": "loop", "description": "Loop", "input_schema": {"type": "object"}}],
            tool_executor=executor,
            max_turns=3,
        )
        # Should have accumulated usage from all 3 turns
        assert resp.usage.input_tokens == 15
