"""Tests for inner-loop budget enforcement in send_orchestrate (issue #21 / task #13)."""
from __future__ import annotations

import pytest

from atn.providers.base import (
    Provider,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)


class _StubProvider(Provider):
    """Minimal Provider where send_stream returns scripted responses."""

    def __init__(self, scripted: list[ProviderResponse]) -> None:
        self._scripted = list(scripted)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "stub"

    async def send(self, **_kwargs):  # pragma: no cover — unused in these tests
        raise NotImplementedError

    async def send_stream(self, **_kwargs):
        self._call_count += 1
        if not self._scripted:
            return ProviderResponse(text="", stop_reason="end_turn", usage=Usage())
        return self._scripted.pop(0)


@pytest.mark.asyncio
async def test_inner_loop_aborts_when_recorder_returns_not_ok():
    # Two scripted turns: first uses tools (so loop continues), second is terminal.
    # The recorder rejects the first turn → loop must stop before turn 2.
    turns = [
        ProviderResponse(
            text="working",
            tool_calls=[ToolCall(id="t1", name="noop", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=500, output_tokens=500),
        ),
        ProviderResponse(
            text="finish",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=10),
        ),
    ]
    provider = _StubProvider(turns)

    recorded: list[int] = []

    def _recorder(tokens: int) -> tuple[bool, str | None]:
        recorded.append(tokens)
        # Reject the very first turn.
        return False, "parent"

    async def _tool_executor(name, input):  # noqa: ARG001
        return {"ok": True}

    response = await provider.send_orchestrate(
        message="hi",
        tools=[],
        max_turns=10,
        tool_executor=_tool_executor,
        usage_recorder=_recorder,
    )

    assert response.stop_reason == "budget_exceeded"
    assert "parent" in response.text
    # Only one send_stream call before the recorder vetoed continuation.
    assert provider._call_count == 1
    assert recorded == [1000]


@pytest.mark.asyncio
async def test_inner_loop_lets_work_continue_when_recorder_ok():
    turns = [
        ProviderResponse(
            text="t1",
            tool_calls=[ToolCall(id="t1", name="noop", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=100, output_tokens=100),
        ),
        ProviderResponse(
            text="t2",
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=50),
        ),
    ]
    provider = _StubProvider(turns)

    async def _tool_executor(name, input):  # noqa: ARG001
        return {"ok": True}

    response = await provider.send_orchestrate(
        message="hi",
        tools=[],
        max_turns=10,
        tool_executor=_tool_executor,
        usage_recorder=lambda t: (True, None),
    )

    assert response.stop_reason == "end_turn"
    assert provider._call_count == 2


@pytest.mark.asyncio
async def test_no_recorder_runs_loop_unchanged():
    """Backward compat: omitting usage_recorder must not break orchestration."""
    turns = [
        ProviderResponse(
            text="done",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=10),
        ),
    ]
    provider = _StubProvider(turns)

    async def _tool_executor(name, input):  # noqa: ARG001
        return {"ok": True}

    response = await provider.send_orchestrate(
        message="hi",
        tools=[],
        max_turns=5,
        tool_executor=_tool_executor,
    )

    assert response.stop_reason == "end_turn"
    assert provider._call_count == 1


@pytest.mark.asyncio
async def test_recorder_exception_does_not_break_loop():
    """If usage_recorder throws, the loop should continue rather than crash."""
    turns = [
        ProviderResponse(
            text="t1",
            tool_calls=[ToolCall(id="t1", name="noop", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=100, output_tokens=100),
        ),
        ProviderResponse(
            text="t2",
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=50),
        ),
    ]
    provider = _StubProvider(turns)

    def _recorder(tokens: int):
        raise RuntimeError("recorder boom")

    async def _tool_executor(name, input):  # noqa: ARG001
        return {"ok": True}

    response = await provider.send_orchestrate(
        message="hi",
        tools=[],
        max_turns=5,
        tool_executor=_tool_executor,
        usage_recorder=_recorder,
    )

    # Loop completed normally — recorder failure must never kill the agent.
    assert response.stop_reason == "end_turn"
    assert provider._call_count == 2


@pytest.mark.asyncio
async def test_async_recorder_supported():
    """The recorder may be async."""
    turns = [
        ProviderResponse(
            text="t1",
            tool_calls=[ToolCall(id="t1", name="noop", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=100, output_tokens=100),
        ),
        ProviderResponse(
            text="t2",
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=50),
        ),
    ]
    provider = _StubProvider(turns)

    async def _async_recorder(tokens: int) -> tuple[bool, str | None]:
        return True, None

    async def _tool_executor(name, input):  # noqa: ARG001
        return {"ok": True}

    response = await provider.send_orchestrate(
        message="hi",
        tools=[],
        max_turns=5,
        tool_executor=_tool_executor,
        usage_recorder=_async_recorder,
    )
    assert response.stop_reason == "end_turn"
