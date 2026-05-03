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

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await coroutine/awaitable, otherwise pass through.

    Lets callers pass either a sync function (returning a tuple) or an async
    function (returning a coroutine) as a callback.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _call_recorder(recorder: Any, turn_tokens: int, model: str = "") -> Any:
    """Call usage_recorder with the right arity.

    Backward-compatible: accepts recorders that take just turn_tokens, or
    take (turn_tokens, model). Inspect the signature once at call site.
    """
    try:
        sig = inspect.signature(recorder)
        positional = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) >= 2:
            return recorder(turn_tokens, model)
    except (TypeError, ValueError):
        pass
    return recorder(turn_tokens)


# ---------------------------------------------------------------------------
# Context window registry
# ---------------------------------------------------------------------------

CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "o3": 200_000,
    "o4-mini": 200_000,
    "o4": 200_000,
    # Google
    "gemini-3-pro": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    # Defaults
    "default": 128_000,
}


def get_context_window(model: str) -> int:
    """Resolve context window size for a model identifier."""
    for prefix, size in CONTEXT_WINDOWS.items():
        if prefix != "default" and model.startswith(prefix):
            return size
    return CONTEXT_WINDOWS.get("default", 128_000)


def classify_model(model: str) -> str:
    """Bucket a model identifier into 'haiku', 'sonnet', 'opus', or 'other'.

    Routes through ``model_specs.model_class`` so the model store is the
    single source of truth.
    """
    from ..model_specs import model_class
    return model_class(model)


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
    """Token usage from a single request.

    Convention (Anthropic-style, enforced at every provider boundary):
    the four counters are **disjoint** — a given token is counted in exactly
    one bucket.  Total billable tokens = input + output + cache_read + cache_creation.

    Providers that use OpenAI's shape (``prompt_tokens`` is the total and
    ``prompt_tokens_details.cached_tokens`` is a subset of that total) MUST
    normalize before constructing a ``Usage``: subtract cached from input,
    then populate ``cache_read_tokens`` separately.  Failing to normalize
    double-counts the cached portion in every downstream roll-up (budgets,
    EIT, snapshots).
    """
    input_tokens: int = 0            # fresh input only, NOT including cache reads
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

    # Session tracking (works for ALL providers, not just bridge)
    _cumulative_input_tokens: int = 0
    _cumulative_output_tokens: int = 0
    _cumulative_cache_read: int = 0
    _cumulative_cache_creation: int = 0
    _cumulative_turns: int = 0
    _last_input_tokens: int = 0
    _active_model: str = ""

    @property
    def session_stats(self) -> dict[str, Any]:
        """Session statistics — base implementation for non-bridge providers.

        BridgeProvider overrides this with richer stats from the Claude SDK.
        """
        ctx_window = get_context_window(self._active_model) if self._active_model else 0
        pct = round(100 * self._last_input_tokens / ctx_window, 1) if ctx_window > 0 and self._last_input_tokens > 0 else None
        return {
            "session_id": "",
            "active_model": self._active_model,
            "num_turns": self._cumulative_turns,
            "total_cost_usd": 0,
            "context_window": ctx_window,
            "max_output_tokens": 0,
            "last_input_tokens": self._last_input_tokens,
            "cumulative_input_tokens": self._cumulative_input_tokens,
            "cumulative_output_tokens": self._cumulative_output_tokens,
            "cumulative_cache_read": self._cumulative_cache_read,
            "cumulative_cache_creation": self._cumulative_cache_creation,
            "context_used_pct": pct,
            "compaction_count": self._compaction_count,
        }

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

    # Context compaction: when input tokens exceed this fraction of the
    # context window, summarize the conversation to free space.
    _COMPACTION_THRESHOLD = 0.80
    _compaction_count: int = 0

    # Interrupt flag — set via interrupt() to stop the orchestration loop.
    _interrupted: bool = False

    async def interrupt(self) -> None:
        """Signal the orchestration loop to stop after the current turn."""
        self._interrupted = True

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
        per_turn_input_max: int | None = None,
        repeat_call_limit: int | None = None,
        usage_recorder: Any = None,
        **_unused: Any,
    ) -> ProviderResponse:
        """Multi-turn orchestration with tool relay.

        Generic implementation that uses ``send_stream()`` in a loop.
        Each turn: call the LLM, execute any tool calls via ``tool_executor``,
        feed results back, and repeat until end_turn or max_turns.

        Includes context compaction: when input tokens approach the context
        window limit, the conversation history is summarized into a single
        message to free space, matching the BridgeProvider's SDK compaction.

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
        self._interrupted = False

        # Track active model for session stats
        if model:
            self._active_model = model

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

        # Long-horizon safeguards (None = use defaults defined here).
        # Estimate input tokens for the next turn as chars/4 across all messages
        # — rough but sufficient for runaway prevention. Refuse when the
        # estimate exceeds per_turn_input_max.
        effective_per_turn_max = per_turn_input_max if per_turn_input_max is not None else 200_000
        effective_repeat_limit = repeat_call_limit if repeat_call_limit is not None else 5
        recent_call_signatures: list[str] = []  # last N tool-call fingerprints

        for turn in range(max_turns):
            if self._interrupted:
                return ProviderResponse(
                    text="",
                    stop_reason="interrupted",
                    usage=cumulative_usage,
                    model=model or self._active_model,
                )

            # Pre-flight per-turn input ceiling — refuse a turn whose estimated
            # input would exceed the configured cap. Estimate via chars/4 across
            # the message stack plus the system prompt.
            estimated_input = (len(system) + sum(
                len(m["content"]) if isinstance(m.get("content"), str)
                else len(json.dumps(m.get("content"), default=str))
                for m in messages
            )) // 4
            if effective_per_turn_max > 0 and estimated_input > effective_per_turn_max:
                log.warning(
                    "Per-turn input ceiling hit (estimated %d > %d) — aborting cognitive loop "
                    "for agent %s. Raise AgentDefinition.per_turn_input_max if intended.",
                    estimated_input, effective_per_turn_max, self.source_agent_id or "?",
                )
                cumulative_usage = cumulative_usage  # no-op for clarity
                return ProviderResponse(
                    text=(
                        f"Aborted: estimated input {estimated_input} tokens exceeds "
                        f"per_turn_input_max ({effective_per_turn_max}). Increase the "
                        f"limit on this agent if it legitimately needs to process "
                        f"larger contexts."
                    ),
                    stop_reason="per_turn_input_exceeded",
                    usage=cumulative_usage,
                    model=model or self._active_model,
                )

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

            # Update session tracking
            self._cumulative_input_tokens += response.usage.input_tokens
            self._cumulative_output_tokens += response.usage.output_tokens
            self._cumulative_cache_read += response.usage.cache_read_tokens
            self._cumulative_cache_creation += response.usage.cache_creation_tokens
            self._cumulative_turns += 1
            # Total input = uncached + cache_read + cache_creation (full context sent)
            self._last_input_tokens = (
                response.usage.input_tokens
                + response.usage.cache_read_tokens
                + response.usage.cache_creation_tokens
            )
            if response.model:
                self._active_model = response.model

            # Inner-loop budget enforcement. usage_recorder is supplied by the
            # execution engine and rolls per-turn tokens into the cascading
            # _budget_used dict; if any ancestor's cap is now exceeded, abort
            # the loop early instead of running all max_turns.
            if usage_recorder is not None:
                try:
                    turn_total = (
                        response.usage.input_tokens
                        + response.usage.output_tokens
                        + response.usage.cache_read_tokens
                        + response.usage.cache_creation_tokens
                    )
                    if turn_total > 0:
                        rec_model = response.model or model or self._active_model
                        ok, blocker = await _maybe_await(
                            _call_recorder(usage_recorder, turn_total, rec_model)
                        )
                        if not ok:
                            log.warning(
                                "Inner-loop budget exceeded mid-orchestration "
                                "(blocker=%s) — aborting cognitive loop for agent %s",
                                blocker, self.source_agent_id or "?",
                            )
                            return ProviderResponse(
                                text=(
                                    f"Aborted: budget exceeded "
                                    f"(blocked by '{blocker}'). Adjust the budget on "
                                    f"that agent or wait for the next period."
                                ),
                                stop_reason="budget_exceeded",
                                usage=cumulative_usage,
                                model=model or self._active_model,
                            )
                except Exception:
                    log.exception("usage_recorder failed; continuing")

            # Emit per-turn usage event (same format as BridgeProvider)
            if self.event_bus and self.source_agent_id:
                from ..events import Event, EventType
                await self.event_bus.emit(Event(
                    type=EventType.STEP_OUTPUT,
                    source=self.source_agent_id,
                    data={
                        "agent_id": self.source_agent_id,
                        "channel": "usage",
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "cache_read_tokens": response.usage.cache_read_tokens,
                            "cache_creation_tokens": response.usage.cache_creation_tokens,
                        },
                        "cumulative": {
                            "input_tokens": self._cumulative_input_tokens,
                            "output_tokens": self._cumulative_output_tokens,
                            "cache_read_tokens": self._cumulative_cache_read,
                            "cache_creation_tokens": self._cumulative_cache_creation,
                            "total_cost_usd": 0,
                        },
                    },
                ))

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

            # Loop-detection: track tool-call fingerprints across turns. If
            # the last K tool calls (across this and prior turns) are byte-for-byte
            # identical, the agent is stuck — abort with a structured error.
            if effective_repeat_limit > 0 and response.tool_calls:
                for tc in response.tool_calls:
                    sig = tc.name + ":" + json.dumps(tc.input, sort_keys=True, default=str)
                    recent_call_signatures.append(sig)
                # Trim to the window we care about
                if len(recent_call_signatures) > effective_repeat_limit:
                    recent_call_signatures = recent_call_signatures[-effective_repeat_limit:]
                if (
                    len(recent_call_signatures) >= effective_repeat_limit
                    and len(set(recent_call_signatures)) == 1
                ):
                    log.warning(
                        "Repeat-call limit hit (%d consecutive identical tool calls) — "
                        "aborting cognitive loop for agent %s. Last call: %s",
                        effective_repeat_limit, self.source_agent_id or "?",
                        recent_call_signatures[-1][:200],
                    )
                    return ProviderResponse(
                        text=(
                            f"Aborted: {effective_repeat_limit} consecutive identical tool "
                            f"calls detected (likely stuck in a loop). Last call: "
                            f"{recent_call_signatures[-1][:200]}"
                        ),
                        stop_reason="repeat_call_limit",
                        usage=cumulative_usage,
                        model=model or self._active_model,
                    )

            tool_result_content: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                log.info("Orchestrate tool %s (turn %d/%d)", tc.name, turn + 1, max_turns)

                # Emit tool use start event
                if self.event_bus and self.source_agent_id:
                    from ..events import Event, EventType
                    await self.event_bus.emit(Event(
                        type=EventType.AGENT_TOOL_USE_START,
                        source=self.source_agent_id,
                        data={
                            "agent_id": self.source_agent_id,
                            "tool_use_id": tc.id,
                            "tool_name": tc.name,
                            "input": tc.input,
                        },
                    ))
                    await self.event_bus.emit(Event(
                        type=EventType.STEP_OUTPUT,
                        source=self.source_agent_id,
                        data={
                            "agent_id": self.source_agent_id,
                            "channel": "tool_call",
                            "tool_name": tc.name,
                            "tool_input": tc.input,
                        },
                    ))

                is_error = False
                try:
                    result = await tool_executor(tc.name, tc.input)
                except Exception as exc:
                    log.warning("Tool %s failed: %s", tc.name, exc)
                    result = {"error": str(exc)}
                    is_error = True

                # Emit tool use result event
                if self.event_bus and self.source_agent_id:
                    from ..events import Event, EventType
                    result_str = str(result)
                    await self.event_bus.emit(Event(
                        type=EventType.AGENT_TOOL_USE_RESULT,
                        source=self.source_agent_id,
                        data={
                            "agent_id": self.source_agent_id,
                            "tool_use_id": tc.id,
                            "tool_name": tc.name,
                            "is_error": is_error,
                            "result_preview": result_str[:500],
                        },
                    ))

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_result_content})

            # Context compaction: if input tokens approach the context window,
            # summarize the conversation to free space.
            ctx_window = get_context_window(self._active_model) if self._active_model else 0
            if ctx_window > 0 and self._last_input_tokens > ctx_window * self._COMPACTION_THRESHOLD:
                messages = await self._compact_messages(messages, system, model)

        # Exhausted max_turns — return last response with accumulated usage
        response.usage = cumulative_usage
        return response

    async def _compact_messages(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
    ) -> list[dict[str, Any]]:
        """Summarize conversation history to free context space.

        Asks the LLM to produce a concise summary of the conversation so far,
        then replaces the message history with the summary as a single user
        message.  The original user request is preserved at the end.
        """
        self._compaction_count += 1
        log.info("Context compaction #%d triggered (last_input=%d tokens)",
                 self._compaction_count, self._last_input_tokens)

        # Emit compaction event
        if self.event_bus and self.source_agent_id:
            from ..events import Event, EventType
            await self.event_bus.emit(Event(
                type=EventType.CONTEXT_COMPACTION,
                source=self.source_agent_id,
                data={
                    "agent_id": self.source_agent_id,
                    "compaction_count": self._compaction_count,
                    "pre_tokens": self._last_input_tokens,
                    "status": "in_progress",
                },
            ))

        # Build a summary request using just the conversation (no tools)
        summary_prompt = (
            "Summarize the conversation so far in a concise format that preserves "
            "all critical context: what was requested, what actions were taken, "
            "what results were observed, what decisions were made, and what remains "
            "to be done. Be specific about file paths, code changes, error messages, "
            "and tool results. This summary will replace the conversation history."
        )
        summary_messages = messages + [{"role": "user", "content": summary_prompt}]

        try:
            summary_resp = await self.send(
                messages=summary_messages,
                system="You are a conversation summarizer. Produce a dense, accurate summary.",
                model=model,
                max_tokens=4096,
            )
            summary_text = summary_resp.text
        except Exception as exc:
            log.warning("Compaction summary failed: %s — continuing without compaction", exc)
            return messages

        # Extract the original user request (first message)
        original_request = messages[0]["content"] if messages else ""

        # Rebuild messages: summary + original request
        compacted: list[dict[str, Any]] = [
            {"role": "user", "content": (
                f"[Context compaction — conversation summary follows]\n\n"
                f"{summary_text}\n\n"
                f"---\n\n"
                f"[Original request]\n{original_request}\n\n"
                f"Continue from where you left off. Do not repeat completed work."
            )},
        ]

        log.info("Compaction #%d complete: %d messages → 1 summary message",
                 self._compaction_count, len(messages))

        if self.event_bus and self.source_agent_id:
            from ..events import Event, EventType
            await self.event_bus.emit(Event(
                type=EventType.CONTEXT_COMPACTION,
                source=self.source_agent_id,
                data={
                    "agent_id": self.source_agent_id,
                    "compaction_count": self._compaction_count,
                    "pre_tokens": self._last_input_tokens,
                    "status": "completed",
                },
            ))

        return compacted

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
