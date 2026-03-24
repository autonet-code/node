"""Provider abstraction — canonical interface for LLM providers.

Every provider adapter implements `Provider.send()` with a unified message/tool
format.  The cognitive step executor calls this interface without caring which
LLM is behind it.

Canonical format (close to Anthropic's, adapters translate for other providers):

  Message:  {"role": "user"|"assistant", "content": str | list[ContentBlock]}
  Tool:     {"name": str, "description": str, "input_schema": dict}
  Response: ProviderResponse(text, tool_calls, usage, raw)

Tool calls returned by the LLM:
  ToolCall: {"id": str, "name": str, "input": dict}
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical data types
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A tool the LLM can call."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    """Token usage from a single request."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0       # tokens served from cache (cheap)
    cache_creation_tokens: int = 0   # tokens written to cache (one-time cost)


@dataclass
class ProviderResponse:
    """Unified response from any LLM provider."""
    text: str = ""
    thinking: list[str] = field(default_factory=list)  # reasoning blocks (provider-agnostic)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""          # "end_turn", "tool_use", "max_tokens"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None                # provider-specific raw response


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Abstract LLM provider interface."""

    # Attributes set by the runtime when the provider is used for cognitive agents.
    event_bus: Any = None
    source_agent_id: str = ""

    async def close(self) -> None:
        """Clean up resources.  Default is a no-op."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'anthropic', 'openai')."""

    @abstractmethod
    async def send(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Send a message to the LLM and return the response.

        Args:
            messages:   Conversation history in canonical format.
            system:     System prompt.
            model:      Model ID override (uses provider default if empty).
            max_tokens: Maximum tokens in the response.
            tools:      Tools the LLM can call.
            temperature: Sampling temperature.

        Returns:
            ProviderResponse with text and/or tool_calls.

        Raises:
            ProviderError for non-transient failures.
            Transient failures (429, 503) should be retried internally.
        """

    async def interrupt(self) -> None:
        """Interrupt any running generation.

        Default is a no-op.  BridgeProvider overrides this to signal
        the Claude SDK subprocess.  Other providers can override if
        they have a cancellation mechanism.
        """

    @property
    def supports_orchestrate(self) -> bool:
        """Whether this provider supports multi-turn orchestration.

        Returns True by default — the base class provides a generic
        implementation of ``send_orchestrate()`` that uses ``send_stream()``
        in a multi-turn loop.  Providers like BridgeProvider override this
        with a native implementation (e.g. Claude Agent SDK subprocess).
        """
        return True

    async def send_orchestrate(
        self,
        *,
        message: str,
        system: str = "",
        model: str = "",
        tools: list[dict[str, Any]],
        max_turns: int = 20,
        tool_executor: Any = None,
        on_chunk: Any = None,
        session_id: str = "",
    ) -> ProviderResponse:
        """Multi-turn orchestration with tool relay.

        Generic implementation that uses ``send_stream()`` in a loop.
        Each turn: call the LLM, execute any tool calls via ``tool_executor``,
        feed results back, and repeat until end_turn or max_turns.

        Providers with native multi-turn support (e.g. BridgeProvider) override
        this entirely.  Other providers (Anthropic direct, OpenAI, Ollama) use
        this base implementation — it works with any provider that has a working
        ``send_stream()`` and returns canonical ``ProviderResponse`` objects.

        Args:
            message:        User message text.
            system:         System prompt.
            model:          Model override.
            tools:          Tool definitions (name, description, input_schema).
            max_turns:      Max LLM turns for the multi-turn loop.
            tool_executor:  async (name, input) -> result dict.
            on_chunk:       Optional async callback for streaming text deltas.
            session_id:     Session ID to resume (ignored by generic impl).
        """
        # Convert tool dicts to ToolDefinition objects for send_stream()
        tool_defs: list[ToolDefinition] | None = None
        if tools:
            tool_defs = [
                ToolDefinition(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", {"type": "object", "properties": {}}),
                )
                for t in tools
            ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
        cumulative_usage = Usage()

        for turn in range(max_turns):
            response = await self.send_stream(
                messages=messages,
                system=system,
                model=model,
                max_tokens=16384,
                tools=tool_defs,
                temperature=0.0,
                on_chunk=on_chunk,
            )

            # Accumulate usage across turns
            cumulative_usage.input_tokens += response.usage.input_tokens
            cumulative_usage.output_tokens += response.usage.output_tokens
            cumulative_usage.cache_read_tokens += response.usage.cache_read_tokens
            cumulative_usage.cache_creation_tokens += response.usage.cache_creation_tokens

            # No tool calls or not a tool_use stop — we're done
            if not response.tool_calls or response.stop_reason != "tool_use":
                response.usage = cumulative_usage
                return response

            # No executor — return as-is (tool calls visible but not executed)
            if tool_executor is None:
                response.usage = cumulative_usage
                return response

            # Execute tool calls and build continuation messages
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_result_content: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                log.info("Orchestrate tool %s (turn %d/%d)", tc.name, turn + 1, max_turns)
                try:
                    result = await tool_executor(tc.name, tc.input)
                except Exception as exc:
                    log.warning("Tool %s failed: %s", tc.name, exc)
                    result = {"error": str(exc)}
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_result_content})

        # Exhausted max_turns — return last response with accumulated usage
        response.usage = cumulative_usage
        return response

    async def send_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        on_chunk: Any = None,
        on_thinking: Any = None,
    ) -> ProviderResponse:
        """Send with streaming — calls on_chunk(text) for each text delta.

        Default implementation falls back to non-streaming send().
        Providers override this to implement real SSE/streaming.

        Args:
            on_chunk: async callable(str) invoked with each text chunk.
            on_thinking: async callable(str) invoked with each thinking block.
            (other args: same as send())

        Returns:
            Final ProviderResponse with complete text and usage.
        """
        response = await self.send(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
        )
        # Emit the full text as a single chunk for non-streaming providers
        if on_chunk and response.text:
            await on_chunk(response.text)
        return response


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Non-transient provider error (bad request, auth failure, etc)."""

    def __init__(self, message: str, status_code: int | None = None, provider: str = "") -> None:
        self.status_code = status_code
        self.provider = provider
        super().__init__(message)
