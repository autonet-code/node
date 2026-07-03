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


# ---------------------------------------------------------------------------
# History-as-messages (cache-stable prompts)
# ---------------------------------------------------------------------------

class RecordingProvider(Provider):
    """Captures the messages list passed to each send() call."""

    def __init__(self, responses: list[ProviderResponse]):
        self._responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    @property
    def name(self) -> str:
        return "recording"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        # Deep-enough copy: the loop appends to the same list across turns.
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0)


class TestNormalizeHistory:
    def test_role_mapping_and_merge(self):
        from atn.providers.base import _normalize_history
        out = _normalize_history([
            {"role": "system", "content": "status briefing"},
            {"role": "user", "content": "do the thing"},
            {"role": "user", "content": "and another"},
            {"role": "assistant", "content": "done"},
        ])
        # system → user-tagged; consecutive user turns merged
        assert [m["role"] for m in out] == ["user", "assistant"]
        assert "[system] status briefing" in out[0]["content"]
        assert "and another" in out[0]["content"]

    def test_assistant_first_gets_resume_marker(self):
        from atn.providers.base import _normalize_history
        out = _normalize_history([
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "next"},
        ])
        assert out[0]["role"] == "user"
        assert out[1]["role"] == "assistant"

    def test_empty_turns_skipped(self):
        from atn.providers.base import _normalize_history
        out = _normalize_history([
            {"role": "user", "content": ""},
            {"role": "user", "content": "real"},
        ])
        assert len(out) == 1


class TestOrchestrateHistory:
    @pytest.mark.asyncio
    async def test_history_precedes_user_message(self):
        p = RecordingProvider([
            ProviderResponse(text="ok", stop_reason="end_turn",
                             usage=Usage(input_tokens=5, output_tokens=2)),
        ])
        await p.send_orchestrate(
            message="now do X",
            tools=[],
            history=[
                {"role": "user", "content": "earlier ask"},
                {"role": "assistant", "content": "earlier answer"},
            ],
        )
        msgs = p.seen_messages[0]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "earlier ask"
        assert msgs[-1]["content"] == "now do X"

    @pytest.mark.asyncio
    async def test_oversized_tool_result_truncated(self):
        from atn.providers.base import _TOOL_RESULT_CHAR_MAX
        tc = ToolCall(id="tc1", name="big", input={})
        p = RecordingProvider([
            ProviderResponse(tool_calls=[tc], stop_reason="tool_use",
                             usage=Usage(input_tokens=5, output_tokens=2)),
            ProviderResponse(text="done", stop_reason="end_turn",
                             usage=Usage(input_tokens=5, output_tokens=2)),
        ])

        async def executor(name, inp):
            return {"blob": "x" * (_TOOL_RESULT_CHAR_MAX * 2)}

        await p.send_orchestrate(
            message="go",
            tools=[{"name": "big", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=executor,
            max_turns=3,
        )
        # Second send's last message carries the tool_result block. §6 replaced
        # the tail-chop with head+tail elision.
        second = p.seen_messages[1]
        block = second[-1]["content"][0]
        assert block["type"] == "tool_result"
        assert len(block["content"]) <= _TOOL_RESULT_CHAR_MAX + 200
        assert "[elided" in block["content"]


class TestHeadTailTruncation:
    """§6: oversized tool results keep head + tail, elide the middle."""

    def test_under_cap_unchanged(self):
        from atn.providers.base import _truncate_tool_result, _TOOL_RESULT_CHAR_MAX
        s = "x" * (_TOOL_RESULT_CHAR_MAX - 10)
        assert _truncate_tool_result(s) == s

    def test_head_and_tail_preserved(self):
        from atn.providers.base import (
            _truncate_tool_result,
            _TOOL_RESULT_HEAD_CHARS,
            _TOOL_RESULT_TAIL_CHARS,
        )
        head = "H" * _TOOL_RESULT_HEAD_CHARS
        mid = "M" * 100_000
        tail = "T" * _TOOL_RESULT_TAIL_CHARS
        out = _truncate_tool_result(head + mid + tail)
        # Head kept verbatim, tail kept verbatim, middle elided with a marker.
        assert out.startswith("H" * 1000)
        assert out.endswith("T" * 1000)
        assert "[elided" in out
        assert "M" * 1000 not in out
        # Total is bounded well under the raw input.
        assert len(out) < _TOOL_RESULT_HEAD_CHARS + _TOOL_RESULT_TAIL_CHARS + 200

    @pytest.mark.asyncio
    async def test_loop_uses_head_tail(self):
        """The orchestrate loop truncates via head+tail (not tail-chop)."""
        from atn.providers.base import _TOOL_RESULT_HEAD_CHARS
        tc = ToolCall(id="tc1", name="big", input={})
        p = RecordingProvider([
            ProviderResponse(tool_calls=[tc], stop_reason="tool_use",
                             usage=Usage(input_tokens=5, output_tokens=2)),
            ProviderResponse(text="done", stop_reason="end_turn",
                             usage=Usage(input_tokens=5, output_tokens=2)),
        ])

        async def executor(name, inp):
            # A result whose head and tail carry distinct markers.
            return "HEAD_MARKER" + ("z" * 200_000) + "TAIL_MARKER"

        await p.send_orchestrate(
            message="go",
            tools=[{"name": "big", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=executor,
            max_turns=3,
        )
        block = p.seen_messages[1][-1]["content"][0]
        assert block["type"] == "tool_result"
        # Both ends survive; the old tail-chop dropped the tail.
        assert "HEAD_MARKER" in block["content"]
        assert "TAIL_MARKER" in block["content"]
        assert "[elided" in block["content"]


class TestMaxTokensFromSpec:
    """§7: per-request max_tokens = min(16384, spec.max_output_tokens)."""

    @pytest.mark.asyncio
    async def test_capped_at_16k_for_large_output_model(self):
        """A model with a huge output cap is bounded to 16k per request."""
        p = RecordingMaxTokensProvider([
            ProviderResponse(text="ok", stop_reason="end_turn",
                             usage=Usage(input_tokens=5, output_tokens=2)),
        ])
        # opus 4.8 has 128k output cap → should be clamped to 16384.
        await p.send_orchestrate(message="hi", tools=[], model="claude-opus-4-8")
        assert p.seen_max_tokens[0] == 16384

    @pytest.mark.asyncio
    async def test_small_output_cap_used_directly(self):
        """A model whose output cap < 16k uses that smaller cap."""
        p = RecordingMaxTokensProvider([
            ProviderResponse(text="ok", stop_reason="end_turn",
                             usage=Usage(input_tokens=5, output_tokens=2)),
        ])
        # haiku-4-5 has an 8192 output cap.
        await p.send_orchestrate(message="hi", tools=[], model="claude-haiku-4-5")
        assert p.seen_max_tokens[0] == 8192


class RecordingMaxTokensProvider(Provider):
    """Captures the max_tokens passed to send_stream each turn."""

    def __init__(self, responses: list[ProviderResponse]):
        self._responses = list(responses)
        self.seen_max_tokens: list[int] = []

    @property
    def name(self) -> str:
        return "recording_maxtok"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        self.seen_max_tokens.append(max_tokens)
        return self._responses.pop(0)


class TestAnthropicMovingBreakpoint:
    def test_string_content_marked(self):
        from atn.providers.anthropic import _with_moving_breakpoint
        msgs = [{"role": "user", "content": "hello"}]
        out = _with_moving_breakpoint(msgs)
        assert out[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Non-destructive: original untouched
        assert msgs[-1]["content"] == "hello"

    def test_block_content_marked_on_last_block_only(self):
        from atn.providers.anthropic import _with_moving_breakpoint
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "b", "content": "r2"},
            ]},
        ]
        out = _with_moving_breakpoint(msgs)
        blocks = out[-1]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}
        # Original blocks untouched
        assert "cache_control" not in msgs[-1]["content"][1]

    def test_empty_messages_passthrough(self):
        from atn.providers.anthropic import _with_moving_breakpoint
        assert _with_moving_breakpoint([]) == []


class TestStickyHistoryTrim:
    def test_trim_boundary_sticky_across_calls(self):
        from types import SimpleNamespace
        from atn.runtime.execution_engine import ExecutionEngine

        eng = ExecutionEngine.__new__(ExecutionEngine)

        def turn(role, content):
            return SimpleNamespace(role=role, content=content)

        # ~50k chars per turn; budget 400k, trim target 240k
        big = "y" * 50_000
        turns = [turn("user" if i % 2 == 0 else "assistant", big) for i in range(10)]
        out1 = eng._build_history_messages("agent1", turns)
        # 10 * 50k = 500k > 400k → trims to <= 240k (4 turns) + marker
        kept1 = [m for m in out1 if m["content"] != "[Earlier conversation trimmed]"]
        assert len(kept1) <= 5
        start_after_first = eng._history_trim_start["agent1"]
        assert start_after_first > 0

        # Two more turns: kept suffix still under budget → boundary unchanged
        turns += [turn("user", big), turn("assistant", big)]
        eng._build_history_messages("agent1", turns)
        assert eng._history_trim_start["agent1"] == start_after_first

    def test_heartbeat_boilerplate_compressed(self):
        from types import SimpleNamespace
        from atn.runtime.execution_engine import ExecutionEngine

        eng = ExecutionEngine.__new__(ExecutionEngine)
        turns = [
            SimpleNamespace(role="user", content="[HEARTBEAT] This is an automatic heartbeat wake-up from the runtime (interval: 300s). Check on active goals."),
            SimpleNamespace(role="assistant", content="Nothing to do."),
        ]
        out = eng._build_history_messages("a2", turns)
        assert out[0]["content"] == ExecutionEngine._HEARTBEAT_HISTORY_MARKER

    def test_reset_conversation_resets_boundary(self):
        from types import SimpleNamespace
        from atn.runtime.execution_engine import ExecutionEngine

        eng = ExecutionEngine.__new__(ExecutionEngine)
        eng._history_trim_start = {"a3": 50}
        turns = [SimpleNamespace(role="user", content="fresh start")]
        out = eng._build_history_messages("a3", turns)
        assert out == [{"role": "user", "content": "fresh start"}]
        assert eng._history_trim_start["a3"] == 0
