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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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

    @property
    def supports_orchestrate(self) -> bool:
        """Whether this provider supports multi-turn orchestration.

        Providers that implement ``send_orchestrate()`` (a bidirectional
        multi-turn tool loop managed by the provider itself) should override
        this to return True.  The cognitive step executor uses this flag to
        decide whether to delegate orchestration to the provider or run its
        own generic multi-turn loop.
        """
        return False

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

        Providers that set ``supports_orchestrate = True`` must implement
        this method.  The default raises NotImplementedError.

        The method handles a bidirectional conversation: the provider calls
        the LLM, relays tool calls to ``tool_executor``, feeds results back,
        and loops until end_turn or max_turns.

        Args:
            message:        User message text.
            system:         System prompt.
            model:          Model override.
            tools:          Tool definitions (name, description, input_schema).
            max_turns:      Max LLM turns for the multi-turn loop.
            tool_executor:  async (name, input) -> result dict.
            on_chunk:       Optional async callback for streaming text deltas.
            session_id:     Session ID to resume (enables prompt caching).
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not implement send_orchestrate(). "
            f"Set supports_orchestrate = True only if this method is implemented."
        )

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
