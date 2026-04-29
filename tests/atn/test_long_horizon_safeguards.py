"""Tests for long-horizon harness safeguards.

Covers four guards added to keep autonomous agents from runaway shapes:

1. Spawn limit — too many direct children under one parent is refused.
2. Depth limit — registering an agent too deep below root is refused.
3. Repeat-call limit — the cognitive loop aborts after K consecutive
   identical tool calls.
4. Per-turn input ceiling — the cognitive loop refuses a turn whose
   estimated input exceeds the cap.

Each guard is permissive by default (defaults wide enough that normal
work doesn't hit them) and overridable per-agent via AgentDefinition.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.runtime.agent_registry import (
    DEFAULT_MAX_CHILDREN_PER_PARENT,
    DEFAULT_MAX_DEPTH_BELOW_ROOT,
)


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _bare_agent(agent_id: str, parent_id: str | None) -> AgentDefinition:
    """Build an agent with no system prompt or steps so register_agent
    skips identity/wallet generation (which requires a system_prompt).
    Enough for the spawn/depth checks which run first."""
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.PIPELINE,  # not cognitive — keeps registration cheap
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# Spawn limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_limit_blocks_at_default(tmp_path):
    """The 21st child under the same parent is refused at the default."""
    rt = _make_runtime(tmp_path)
    parent = _bare_agent("p", None)
    await rt.registry.register_agent(parent)

    # Register up to the default limit — all should succeed.
    for i in range(DEFAULT_MAX_CHILDREN_PER_PARENT):
        child = _bare_agent(f"p.{i}", "p")
        await rt.registry.register_agent(child)

    # The next one should hit the limit.
    overflow = _bare_agent("p.over", "p")
    with pytest.raises(ValueError, match="Spawn limit exceeded"):
        await rt.registry.register_agent(overflow)


@pytest.mark.asyncio
async def test_spawn_limit_respects_per_agent_override(tmp_path):
    """A parent with max_children=3 refuses the 4th child."""
    rt = _make_runtime(tmp_path)
    parent = _bare_agent("p", None)
    parent.max_children = 3
    await rt.registry.register_agent(parent)

    for i in range(3):
        await rt.registry.register_agent(_bare_agent(f"p.{i}", "p"))

    with pytest.raises(ValueError, match="Spawn limit exceeded"):
        await rt.registry.register_agent(_bare_agent("p.4th", "p"))


# ---------------------------------------------------------------------------
# Depth limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_depth_limit_blocks_at_default(tmp_path):
    """A chain root → c1 → c2 → ... → cN refuses to extend past the default depth."""
    rt = _make_runtime(tmp_path)
    root = _bare_agent("r", None)
    await rt.registry.register_agent(root)

    chain = ["r"]
    # Build a chain right up to the default depth
    for i in range(DEFAULT_MAX_DEPTH_BELOW_ROOT):
        child_id = f"r.{i}"
        await rt.registry.register_agent(_bare_agent(child_id, chain[-1]))
        chain.append(child_id)

    # Adding one more level should be refused.
    overflow = _bare_agent("r.too_deep", chain[-1])
    with pytest.raises(ValueError, match="Depth limit exceeded"):
        await rt.registry.register_agent(overflow)


@pytest.mark.asyncio
async def test_depth_limit_respects_per_agent_override(tmp_path):
    """A root with max_depth_below=2 refuses depth 3."""
    rt = _make_runtime(tmp_path)
    root = _bare_agent("r", None)
    root.max_depth_below = 2
    await rt.registry.register_agent(root)

    await rt.registry.register_agent(_bare_agent("r.a", "r"))   # depth 1
    await rt.registry.register_agent(_bare_agent("r.a.b", "r.a"))  # depth 2

    with pytest.raises(ValueError, match="Depth limit exceeded"):
        await rt.registry.register_agent(_bare_agent("r.a.b.c", "r.a.b"))  # depth 3


# ---------------------------------------------------------------------------
# Repeat-call limit
# ---------------------------------------------------------------------------

class _RepeatingProvider:
    """Minimal stub that satisfies the BaseProvider.send_orchestrate contract.

    Each send_stream() returns the same single tool call. If the loop
    keeps executing past the repeat-call limit, the test will time out
    or the cumulative_usage will grow forever — instead, the loop should
    abort with stop_reason="repeat_call_limit".
    """

    def __init__(self) -> None:
        self.calls = 0
        self._interrupted = False
        self.source_agent_id = "test"
        self.event_bus = None
        self._active_model = "test-model"
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_read = 0
        self._cumulative_cache_creation = 0
        self._cumulative_turns = 0
        self._last_input_tokens = 0
        self._COMPACTION_THRESHOLD = 0.9

    async def send_stream(self, **kwargs: Any):
        from atn.providers.base import ProviderResponse, ToolCall, Usage
        self.calls += 1
        return ProviderResponse(
            text="",
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
            tool_calls=[ToolCall(id=f"tc-{self.calls}", name="loop_tool", input={"x": 1})],
        )


@pytest.mark.asyncio
async def test_repeat_call_limit_aborts_after_default(tmp_path):
    """Five identical tool calls in a row trip the default repeat-call limit."""
    from atn.providers.base import Provider

    provider = _RepeatingProvider()

    async def tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    response = await Provider.send_orchestrate(
        provider,
        message="start",
        system="",
        tools=[{"name": "loop_tool", "description": "", "input_schema": {"type": "object"}}],
        max_turns=20,
        tool_executor=tool_executor,
    )

    assert response.stop_reason == "repeat_call_limit"
    # The default is 5 — so we expect 5 send_stream() calls before the abort.
    assert provider.calls == 5


@pytest.mark.asyncio
async def test_repeat_call_limit_respects_override(tmp_path):
    """Setting repeat_call_limit=3 trips after 3 identical calls."""
    from atn.providers.base import Provider

    provider = _RepeatingProvider()

    async def tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    response = await Provider.send_orchestrate(
        provider,
        message="start",
        system="",
        tools=[{"name": "loop_tool", "description": "", "input_schema": {"type": "object"}}],
        max_turns=20,
        tool_executor=tool_executor,
        repeat_call_limit=3,
    )

    assert response.stop_reason == "repeat_call_limit"
    assert provider.calls == 3


# ---------------------------------------------------------------------------
# Per-turn input ceiling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_turn_input_ceiling_refuses_oversized(tmp_path):
    """A message large enough to exceed the cap is refused before send_stream."""
    from atn.providers.base import Provider

    class _NeverCalledProvider(_RepeatingProvider):
        async def send_stream(self, **kwargs: Any):  # pragma: no cover
            raise AssertionError(
                "send_stream should not be called when per_turn_input_max is exceeded"
            )

    provider = _NeverCalledProvider()

    async def tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    # Build a message that, at chars/4, comfortably exceeds 1000 tokens.
    big_message = "x" * 8000  # ~2000 estimated tokens

    response = await Provider.send_orchestrate(
        provider,
        message=big_message,
        system="",
        tools=[],
        max_turns=5,
        tool_executor=tool_executor,
        per_turn_input_max=1000,
    )

    assert response.stop_reason == "per_turn_input_exceeded"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_per_turn_input_ceiling_passes_normal_size(tmp_path):
    """A normal-sized message passes the ceiling check."""
    from atn.providers.base import Provider, ProviderResponse, Usage

    class _OneShotProvider(_RepeatingProvider):
        async def send_stream(self, **kwargs: Any):
            self.calls += 1
            return ProviderResponse(
                text="done",
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
                model="test-model",
                tool_calls=[],
            )

    provider = _OneShotProvider()

    async def tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    response = await Provider.send_orchestrate(
        provider,
        message="hello",
        system="",
        tools=[],
        max_turns=5,
        tool_executor=tool_executor,
        per_turn_input_max=200_000,
    )

    assert response.stop_reason == "end_turn"
    assert provider.calls == 1
