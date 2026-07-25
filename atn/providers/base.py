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

import asyncio
import inspect
import json
import logging
import random
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
    """Resolve context window size for a model identifier.

    ``model_specs`` is the single source of truth (§7). We delegate there
    first; the legacy ``CONTEXT_WINDOWS`` dict remains only as a fallback for
    model strings the store doesn't recognise (``resolve`` returns the
    default spec).
    """
    from ..model_specs import resolve
    spec = resolve(model)
    if spec.id != "default":
        return spec.context_window
    # Unknown to the store — try the legacy prefix table before the default.
    for prefix, size in CONTEXT_WINDOWS.items():
        if prefix != "default" and model.startswith(prefix):
            return size
    return spec.context_window


def classify_model(model: str) -> str:
    """Bucket a model identifier into 'haiku', 'sonnet', 'opus', or 'other'.

    Routes through ``model_specs.model_class`` so the model store is the
    single source of truth.
    """
    from ..model_specs import model_class
    return model_class(model)


# Structured compaction summary template (§2/§15). Goal / Progress / Key
# decisions / Critical context / Next steps — matches the harness compaction
# shape. Factored out so both the running-loop compactor (_compact_messages)
# and the idle-path manual compactor (compact_agent) use ONE template.
COMPACTION_SUMMARY_PROMPT = (
    "Summarize the conversation so far under EXACTLY these headings, "
    "preserving all critical context (file paths, code changes, error "
    "messages, tool results). This summary replaces the older history.\n\n"
    "## Goal\n## Progress\n## Key decisions\n## Critical context\n## Next steps"
)

# System prompt for the summarizer send() call.
COMPACTION_SUMMARIZER_SYSTEM = (
    "You are a conversation summarizer. Produce a dense, accurate summary."
)


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

    def total_tokens(self) -> int:
        """All four buckets summed — the raw context/EIT figure. This is
        context SIZE, not spend: cache_read can dominate it while costing ~10%.
        Use for snapshots/telemetry, NOT for budget enforcement."""
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_creation_tokens)

    def budget_tokens(self) -> int:
        """Tokens to meter against a budget cap — REAL NEW tokens, not context
        size.

        The defect this fixes: a cached system+tools prefix is re-sent on every
        turn as ``cache_read`` (Anthropic bills it at ~10% and it is no new
        work), yet the old meter summed all four buckets at full weight. That
        made an agent which spent $0.009 trip a 5,000-token cap, because its
        47k of cache_read dwarfed the ~3.3k of tokens it actually generated.

        We exclude cache_read and count the rest at 1:1 — i.e. tokens the agent
        genuinely produced/consumed this run:
            input          (fresh prompt tokens)
            cache_creation (one-time prefix write)
            output         (generated tokens)
        This keeps a "tokens" budget meaning *token count* (not a cost-weighted
        unit, which would silently change the meaning of every existing cap);
        it simply stops charging the cheap cached re-read. For test-simple this
        is 489 + 2840 + 63 = 3392, comfortably under a 5,000 cap."""
        return (self.input_tokens
                + self.cache_creation_tokens
                + self.output_tokens)


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
    # Mid-turn steering messages (§5) that arrived but weren't drained before
    # the run ended. The caller MUST re-post these to the agent's inbox so they
    # aren't lost — never silently dropped.
    undelivered_steering: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Abstract LLM provider interface."""

    # Attributes set by the runtime when the provider is used for cognitive agents.
    event_bus: Any = None
    source_agent_id: str = ""

    # v3 review step: True when this provider's own orchestrate loop
    # injects the closing attest turn (the generic base loop does). A
    # provider that runs an external loop it can't inject into (the SDK
    # bridge) overrides this to False, and the CALLER runs the review
    # step as a follow-up session turn (delegate_prompts.
    # needs_review_reinvoke).
    handles_review_step: bool = True

    # Session tracking (works for ALL providers, not just bridge)
    _cumulative_input_tokens: int = 0
    _cumulative_output_tokens: int = 0
    _cumulative_cache_read: int = 0
    _cumulative_cache_creation: int = 0
    _cumulative_turns: int = 0
    _last_input_tokens: int = 0
    _active_model: str = ""

    # Live context snapshot (set by send_orchestrate for the duration of a
    # run). The message list is otherwise LOCAL to the loop; these references
    # let an out-of-band inspector (context_inspect / the get_context worker
    # cmd) measure the exact context the loop is working with. _live_messages
    # is the same list object the loop mutates, so compactions and appends are
    # visible to readers. None = no run has populated a snapshot (bridge
    # providers, which override send_orchestrate, never set one).
    _live_system: str = ""
    _live_tools: Any = None
    _live_messages: Any = None
    _live_max_tokens: int = 0

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
    # Compactions within the current send_orchestrate run (spiral guard, §2).
    # Reset at the start of each run.
    _compactions_this_run: int = 0

    # Interrupt flag — set via interrupt() to stop the orchestration loop.
    _interrupted: bool = False

    # Manual-compaction flag (§15). Set via request_compaction() by the
    # compact_agent tool while an orchestration is running; the loop honors it
    # at the next iteration boundary (next to the steering drain) by forcing a
    # context reduction, then clears it. requested_by is threaded onto the
    # emitted CONTEXT_COMPACTION event so the owner sees who asked.
    _compact_requested: bool = False
    _compact_requested_by: str = ""

    # Mid-turn steering queue (§5). Lazily created; only non-None while an
    # orchestration is active. send_user_message() enqueues onto it; the loop
    # drains it at each iteration boundary and appends each item as a user turn.
    _steering_queue: asyncio.Queue | None = None

    async def interrupt(self) -> None:
        """Signal the orchestration loop to stop after the current turn."""
        self._interrupted = True

    def request_compaction(self, requested_by: str = "") -> bool:
        """Request a manual compaction at the next loop iteration (§15).

        Sets a flag honored by ``send_orchestrate`` at its iteration boundary
        (right where steering is drained): it forces a ``_reduce_context`` pass
        even if the size estimate is under budget. Returns True if an
        orchestration is active (``_steering_queue`` is open) to consume the
        request, False otherwise so the caller can fall back to the idle path.

        BridgeProvider has no honoring loop of its own (its orchestration lives
        in the SDK subprocess); it does not override this, so a running bridge
        agent reports no active generic loop here and the tool answers
        ``unsupported_while_running`` — see the tool handler.
        """
        if self._steering_queue is None:
            return False
        self._compact_requested = True
        self._compact_requested_by = requested_by
        return True

    async def send_user_message(self, content: str) -> bool:
        """Inject a user message mid-orchestration (§5).

        Backed by ``_steering_queue`` (created for the duration of a
        ``send_orchestrate`` run). Returns True if an orchestration is active
        to consume the message (queued), False otherwise so the caller can
        fall back to the inbox.

        Providers with native mid-turn injection (BridgeProvider) override this.
        """
        if self._steering_queue is None:
            return False
        await self._steering_queue.put(content)
        return True

    def _drain_steering(self) -> list[str]:
        """Pop all queued steering messages (§5). Never blocks."""
        return _drain_queue(getattr(self, "_steering_queue", None))

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
        history: list[dict[str, Any]] | None = None,
        review_tools: bool = True,
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
        # Open the steering queue for the duration of this run (§5). Any
        # send_user_message() call now enqueues instead of returning False.
        self._steering_queue = asyncio.Queue()
        self._compactions_this_run = 0

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

        # Prior conversation as real messages (NOT baked into the system
        # prompt). Keeping the system prompt byte-stable across executions
        # and the history a growing message prefix is what lets provider
        # prompt caches hit: the static prefix is cached forever, and only
        # the appended tail is fresh input.
        messages: list[dict[str, Any]] = _normalize_history(history) if history else []
        messages.append({"role": "user", "content": message})
        cumulative_usage = Usage()

        # Long-horizon safeguards (None = use defaults defined here).
        # Estimate input tokens for the next turn as chars/4 across all messages
        # — rough but sufficient for runaway prevention. Refuse when the
        # estimate exceeds per_turn_input_max.
        effective_per_turn_max = per_turn_input_max if per_turn_input_max is not None else 200_000
        # §4: exact-repeat threshold drops 5 → 3 (opencode DOOM_LOOP_THRESHOLD).
        effective_repeat_limit = repeat_call_limit if repeat_call_limit is not None else 3
        recent_call_signatures: list[str] = []  # last N tool-call fingerprints
        # §4 oscillation guard: fingerprints of the last few tool calls, plus a
        # one-shot "already warned" flag so the model gets one chance to self-
        # correct before we abort.
        osc_fingerprints: list[str] = []
        loop_warning_issued = False
        # v3 review step (docs/tool_substrate.md, Decision 2026-07-08): the
        # post-use review is CORE — per-axis scores are the only signal that
        # ranks library retrieval and drifts tool positions. Track which
        # framework tools the run called; if it used registered tools
        # (use_tool / register_tool) and never attested, inject ONE closing
        # review turn at the natural end instead of finalizing. Self-gating:
        # runs that never touched registered tools are unaffected.
        called_tool_names: set[str] = set()
        review_injected = False

        # Per-request output cap comes from the model spec (§7), bounded at 16k
        # so a huge output ceiling doesn't shrink the usable input budget.
        from ..model_specs import max_output_tokens as _spec_max_output
        max_tokens = min(16_384, _spec_max_output(model or self._active_model))
        if max_tokens <= 0:
            max_tokens = 16_384
        # Small-window models (locals at 16k): the output cap must scale with
        # the window, or the pre-send input budget (window − max_tokens −
        # buffer) goes negative and context reduction can never succeed
        # (observed live: 16k window − 16k max_tokens − 8k buffer → instant
        # "compaction spiral" abort on turn 1).
        _ctx_for_cap = get_context_window(model or self._active_model)
        if _ctx_for_cap > 0:
            max_tokens = min(max_tokens, max(1_024, _ctx_for_cap // 4))

        # Publish the live context snapshot for out-of-band inspection
        # (context_inspect.breakdown_from_provider). Same list object as the
        # loop below mutates — kept after the run ends so the last-known
        # context stays inspectable until the next run replaces it.
        self._live_system = system
        self._live_tools = list(tools) if tools else []
        self._live_messages = messages
        self._live_max_tokens = max_tokens

        for turn in range(max_turns):
            if self._interrupted:
                return Provider._finalize_orchestrate(self, 
                    ProviderResponse(
                        text="",
                        stop_reason="interrupted",
                        model=model or self._active_model,
                    ),
                    messages, cumulative_usage, abort_reason="interrupted",
                )

            # §5: drain any mid-turn steering into the message stack at the
            # iteration boundary (after prior tool results were appended, before
            # the next send). Each item becomes a user turn.
            for steer in Provider._drain_steering(self):
                log.info("Injecting steering message (%d chars) for agent %s",
                         len(steer), self.source_agent_id or "?")
                messages.append({"role": "user", "content": steer})

            # §15: honor a manual compaction request at the same iteration
            # boundary. Force a reduction pass regardless of the size estimate
            # (the owner asked for it explicitly). Consume the flag once so it
            # runs a single time per request. An overflow that reduction can't
            # resolve aborts the run like any other context_overflow.
            if self._compact_requested:
                self._compact_requested = False
                requested_by = self._compact_requested_by
                self._compact_requested_by = ""
                log.info("Manual compaction requested by %s for agent %s",
                         requested_by or "?", self.source_agent_id or "?")
                try:
                    await Provider._reduce_context(
                        self, messages, system, model, message, max_tokens,
                        force=True, manual=True, requested_by=requested_by,
                    )
                except ContextOverflowError as exc:
                    log.warning("Manual compaction reduction gave up: %s", exc)
                    return Provider._finalize_orchestrate(self,
                        ProviderResponse(
                            text=f"Aborted: {exc}",
                            stop_reason="context_overflow",
                            model=model or self._active_model,
                        ),
                        messages, cumulative_usage, abort_reason="context_overflow",
                    )

            # Pre-flight per-turn input ceiling — refuse a turn whose estimated
            # input would exceed the configured cap. Estimate via chars/4 across
            # the message stack plus the system prompt.
            estimated_input = _estimate_chars(system, messages) // 4
            if effective_per_turn_max > 0 and estimated_input > effective_per_turn_max:
                log.warning(
                    "Per-turn input ceiling hit (estimated %d > %d) — aborting cognitive loop "
                    "for agent %s. Raise AgentDefinition.per_turn_input_max if intended.",
                    estimated_input, effective_per_turn_max, self.source_agent_id or "?",
                )
                return Provider._finalize_orchestrate(self, 
                    ProviderResponse(
                        text=(
                            f"Aborted: estimated input {estimated_input} tokens exceeds "
                            f"per_turn_input_max ({effective_per_turn_max}). Increase the "
                            f"limit on this agent if it legitimately needs to process "
                            f"larger contexts."
                        ),
                        stop_reason="per_turn_input_exceeded",
                        model=model or self._active_model,
                    ),
                    messages, cumulative_usage, abort_reason="per_turn_input_exceeded",
                )

            # §2 pre-send context reduction: if the estimated next-request size
            # exceeds ctx_window − max_tokens − 8k buffer, prune then compact
            # BEFORE sending (cheaper than an overflow round-trip).
            ctx_window_pre = get_context_window(self._active_model or model)
            if ctx_window_pre > 0 and estimated_input > (
                ctx_window_pre - max_tokens - _reduction_buffer(ctx_window_pre)
            ):
                try:
                    await Provider._reduce_context(
                        self, messages, system, model, message, max_tokens,
                    )
                except ContextOverflowError as exc:
                    log.warning("Pre-send context reduction gave up: %s", exc)
                    return Provider._finalize_orchestrate(self, 
                        ProviderResponse(
                            text=f"Aborted: {exc}",
                            stop_reason="context_overflow",
                            model=model or self._active_model,
                        ),
                        messages, cumulative_usage, abort_reason="context_overflow",
                    )

            async def _reduce() -> None:
                # Reactive path (overflow 400) — force compaction even if our
                # estimate looks under budget; the provider disagrees.
                await self._reduce_context(
                    messages, system, model, message, max_tokens, force=True,
                )

            try:
                response = await Provider._send_stream_with_retry(self, 
                    messages=messages,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                    tool_defs=tool_defs,
                    on_chunk=on_chunk,
                    reduce_context=_reduce,
                )
            except ContextOverflowError as exc:
                # §1/§2: overflow that survived reduction → abort, don't retry.
                log.warning("Context overflow abort for agent %s: %s",
                            self.source_agent_id or "?", exc)
                return Provider._finalize_orchestrate(self, 
                    ProviderResponse(
                        text=f"Aborted: context overflow could not be reduced ({exc}).",
                        stop_reason="context_overflow",
                        model=model or self._active_model,
                    ),
                    messages, cumulative_usage, abort_reason="context_overflow",
                )
            except Exception as exc:
                # §1: fatal (or transient that exhausted retries) → structured
                # provider_error abort with orphan repair.
                log.warning("Provider error abort for agent %s: %s",
                            self.source_agent_id or "?", exc)
                return Provider._finalize_orchestrate(self, 
                    ProviderResponse(
                        text=f"Aborted: provider error ({exc}).",
                        stop_reason="provider_error",
                        model=model or self._active_model,
                    ),
                    messages, cumulative_usage, abort_reason="provider_error",
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
                    # Meter REAL SPEND, not context size: a large cached prefix
                    # (cache_read) is re-sent every turn but costs ~10% and is
                    # no new work — counting it at full weight false-fails sane
                    # caps. See Usage.budget_tokens().
                    turn_total = response.usage.budget_tokens()
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
                            return Provider._finalize_orchestrate(self, 
                                ProviderResponse(
                                    text=(
                                        f"Aborted: budget exceeded "
                                        f"(blocked by '{blocker}'). Adjust the budget on "
                                        f"that agent or wait for the next period."
                                    ),
                                    stop_reason="budget_exceeded",
                                    model=model or self._active_model,
                                ),
                                messages, cumulative_usage, abort_reason="budget_exceeded",
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
                # v3 review step: one closing turn before finalize when the
                # run used registered tools without attesting. Mirrors the
                # loop-warning injection pattern (synthetic user turn +
                # continue) and spends an iteration of the SAME max_turns
                # budget — never extends it. One-shot: if the agent still
                # doesn't attest on the extra turn, the next natural end
                # finalizes normally.
                # Import the canonical trigger set rather than repeating it:
                # this condition and needs_review_reinvoke (the bridge path)
                # must agree, and a duplicated literal here silently drifted
                # from the real set once already.
                from ..delegate_prompts import _REVIEW_TRIGGER_TOOLS
                if (review_tools and not review_injected
                        and tool_executor is not None
                        and (_REVIEW_TRIGGER_TOOLS & called_tool_names)
                        and "attest_tools" not in called_tool_names):
                    review_injected = True
                    log.info(
                        "Review step: injecting closing attest turn for "
                        "agent %s", self.source_agent_id or "?",
                    )
                    if response.text:
                        # Keep the agent's final answer in history so the
                        # review turn doesn't erase it.
                        messages.append(
                            {"role": "assistant", "content": response.text})
                    from ..delegate_prompts import REVIEW_STEP_PROMPT
                    messages.append(
                        {"role": "user", "content": REVIEW_STEP_PROMPT})
                    continue
                return Provider._finalize_orchestrate(self,
                    response, messages, cumulative_usage,
                )

            # No executor — return as-is (tool calls visible but not executed)
            if tool_executor is None:
                return Provider._finalize_orchestrate(self, 
                    response, messages, cumulative_usage,
                )

            # Execute tool calls and build continuation messages
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                called_tool_names.add(tc.name)
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Loop-detection (§4): two signals, both over tool-call fingerprints.
            #   1. exact-repeat: last K identical calls in a row (K default 3).
            #   2. oscillation: over the last 6 calls, ≤2 distinct fingerprints
            #      each seen ≥2 times (A/B/A/B/... shapes the exact check misses).
            # Before aborting for EITHER, inject one warning turn and give the
            # model a chance to self-correct; abort only if it repeats again.
            if effective_repeat_limit > 0 and response.tool_calls:
                for tc in response.tool_calls:
                    sig = tc.name + ":" + json.dumps(tc.input, sort_keys=True, default=str)
                    recent_call_signatures.append(sig)
                    osc_fingerprints.append(sig)
                # Trim each window to what it needs.
                if len(recent_call_signatures) > effective_repeat_limit:
                    recent_call_signatures = recent_call_signatures[-effective_repeat_limit:]
                if len(osc_fingerprints) > _OSCILLATION_WINDOW:
                    osc_fingerprints = osc_fingerprints[-_OSCILLATION_WINDOW:]

                exact_repeat = (
                    len(recent_call_signatures) >= effective_repeat_limit
                    and len(set(recent_call_signatures)) == 1
                )
                oscillating = _is_oscillating(osc_fingerprints)

                if exact_repeat or oscillating:
                    kind = "repeat" if exact_repeat else "oscillation"
                    if not loop_warning_issued:
                        # One warning turn — the model gets a chance to change
                        # approach or conclude before we abort.
                        loop_warning_issued = True
                        log.info(
                            "Loop suspected (%s) for agent %s — injecting one "
                            "warning turn before abort", kind, self.source_agent_id or "?",
                        )
                        # We already appended the assistant turn above; the loop
                        # normally appends tool_results next. Append them, then
                        # the warning, so history stays valid (no orphan).
                        _tr = [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": json.dumps(
                                    {"note": "not executed — loop guard warning"}
                                ),
                            }
                            for tc in response.tool_calls
                        ]
                        messages.append({"role": "user", "content": _tr})
                        messages.append({"role": "user", "content": (
                            "You appear to be looping — repeating the same tool "
                            "call(s) without progress. Change your approach or "
                            "conclude with a final answer. Do not repeat the last "
                            "call as-is."
                        )})
                        # Reset the windows so a genuinely-corrected next turn
                        # isn't instantly re-flagged by stale fingerprints.
                        recent_call_signatures.clear()
                        osc_fingerprints.clear()
                        continue

                    log.warning(
                        "Loop guard abort (%s) for agent %s after warning. Last call: %s",
                        kind, self.source_agent_id or "?",
                        (recent_call_signatures or osc_fingerprints)[-1][:200],
                    )
                    return Provider._finalize_orchestrate(self, 
                        ProviderResponse(
                            text=(
                                f"Aborted: {kind} loop detected (agent kept repeating "
                                f"tool calls after a warning to change approach)."
                            ),
                            stop_reason="loop_detected" if oscillating and not exact_repeat
                            else "repeat_call_limit",
                            model=model or self._active_model,
                        ),
                        messages, cumulative_usage,
                        abort_reason="loop_detected",
                    )

            tool_result_content: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                # §8: malformed tool arguments. When an adapter can't parse the
                # tool-call args as JSON it degrades them to {"raw": <text>}.
                # Do NOT execute — synthesize an error tool_result telling the
                # model to re-emit, and continue the loop.
                if _is_malformed_args(tc.input):
                    log.warning(
                        "Malformed tool args for %s (agent %s) — not executing",
                        tc.name, self.source_agent_id or "?",
                    )
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps({
                            "error": "arguments were not valid JSON — re-emit the "
                                     "tool call with well-formed JSON arguments",
                        }),
                        "is_error": True,
                    })
                    continue

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

                result_json = json.dumps(result, default=str)
                # Cap oversized tool results (§6): head+tail, keeping the
                # command echo and the error tail while eliding the middle.
                # On small windows the cap scales down so one result can never
                # exceed a third of the input budget (chars).
                _ctx_tokens = get_context_window(self._active_model or model)
                _result_cap = 0
                if 0 < _ctx_tokens <= 64_000:
                    _budget_chars = max(
                        0,
                        _ctx_tokens - max_tokens - _reduction_buffer(_ctx_tokens),
                    ) * 4
                    _result_cap = max(4_000, _budget_chars // 3)
                result_json = _truncate_tool_result(result_json, cap=_result_cap)
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_json,
                })
            messages.append({"role": "user", "content": tool_result_content})

            # §2 note: context reduction runs PRE-SEND at the top of the next
            # iteration (estimate-based) and reactively on an overflow 400, so
            # the old post-tool compaction is no longer needed here.

        # Exhausted max_turns — return last response with accumulated usage
        return Provider._finalize_orchestrate(self, response, messages, cumulative_usage)

    def _finalize_orchestrate(
        self,
        response: ProviderResponse,
        messages: list[dict[str, Any]],
        cumulative_usage: Usage,
        abort_reason: str = "",
    ) -> ProviderResponse:
        """Shared exit path for send_orchestrate (§3, §5).

        - Repairs orphaned ``tool_use`` blocks in ``messages`` in place so the
          history replays cleanly against Anthropic-shape APIs.
        - Drains any undrained steering messages into
          ``response.undelivered_steering`` so the caller can re-post them to
          the inbox (never silently dropped).
        - Closes the steering queue.

        Every ``return`` from send_orchestrate goes through here.
        """
        _repair_orphan_tool_uses(messages, abort_reason)
        undelivered = _drain_queue(getattr(self, "_steering_queue", None))
        self._steering_queue = None
        if undelivered:
            log.info(
                "Surfacing %d undelivered steering message(s) for agent %s "
                "(re-post to inbox)", len(undelivered),
                getattr(self, "source_agent_id", "") or "?",
            )
            response.undelivered_steering = undelivered
        response.usage = cumulative_usage
        return response

    async def _send_stream_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        max_tokens: int,
        tool_defs: list[ToolDefinition] | None,
        on_chunk: Any,
        reduce_context: Any,
    ) -> ProviderResponse:
        """Per-turn send_stream wrapped in the §1 retry gate.

        Routes raised errors by class (see ``classify_provider_error``):
          - transient → up to 3 loop-level retries, exp backoff + jitter,
            honoring Retry-After; the request is re-sent unchanged.
          - overflow → run §2 context reduction (``reduce_context``), then
            re-send once; a second overflow raises ContextOverflowError.
          - fatal → re-raise for the loop to abort as provider_error.
        """
        backoff_schedule = (2.0, 8.0, 30.0)
        overflow_reduced = False
        transient_attempts = 0
        while True:
            try:
                return await self.send_stream(
                    messages=messages,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                    tools=tool_defs,
                    temperature=0.0,
                    on_chunk=on_chunk,
                )
            except Exception as exc:
                route = classify_provider_error(exc)
                if route == "overflow":
                    if overflow_reduced:
                        # Already reduced once and still overflowing — give up.
                        log.warning(
                            "Context overflow persists after reduction for agent %s",
                            self.source_agent_id or "?",
                        )
                        raise ContextOverflowError(
                            f"context overflow after reduction: {exc}",
                            status_code=getattr(exc, "status_code", 400),
                            provider=self.name,
                        )
                    log.info(
                        "Context overflow — running reduction for agent %s: %s",
                        self.source_agent_id or "?", exc,
                    )
                    await reduce_context()
                    overflow_reduced = True
                    continue
                if route == "transient":
                    if transient_attempts >= len(backoff_schedule):
                        log.warning(
                            "Transient error exhausted %d loop-level retries for "
                            "agent %s: %s", len(backoff_schedule),
                            self.source_agent_id or "?", exc,
                        )
                        raise
                    wait = backoff_schedule[transient_attempts]
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except (TypeError, ValueError):
                            pass
                    wait += random.uniform(0, wait * 0.25)  # jitter
                    transient_attempts += 1
                    log.warning(
                        "Transient error (loop retry %d/%d, sleeping %.1fs) for "
                        "agent %s: %s", transient_attempts, len(backoff_schedule),
                        wait, self.source_agent_id or "?", exc,
                    )
                    await asyncio.sleep(wait)
                    continue
                # fatal — let the loop abort as provider_error
                raise

    async def _reduce_context(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        original_request: str,
        max_tokens: int,
        force: bool = False,
        manual: bool = False,
        requested_by: str = "",
    ) -> None:
        """Two-tier context reduction: prune, then compact (§2).

        Operates on ``messages`` IN PLACE. Tier 1 prunes old tool_result
        bodies (free). Tier 2 compacts via the LLM summarizer with a spiral
        guard. Called both pre-send (estimate over threshold) and on an
        overflow 400 from the provider.

        ``force`` is set on the reactive (overflow-400) path: the provider has
        just told us we're over budget, so our chars/4 estimate is wrong low —
        run compaction even if the estimate looks under budget.

        ``manual`` / ``requested_by`` (§15): set on the manual compaction path
        so the emitted CONTEXT_COMPACTION events carry ``manual: true`` and the
        requester id. They ride through to ``_compact_messages``.
        """
        ctx_window = get_context_window(self._active_model or model)
        budget_chars = max(
            0, (ctx_window - max_tokens - _reduction_buffer(ctx_window))
        ) * 4

        # Tier 1 — prune old tool_result bodies (no LLM call).
        est_before = _estimate_chars(system, messages)
        if force or est_before > budget_chars:
            saved = _prune_tool_results(messages)
            if saved >= _PRUNE_MIN_SAVINGS:
                log.info(
                    "Context prune freed ~%d chars for agent %s",
                    saved, self.source_agent_id or "?",
                )

        # Tier 2 — compact (only if still over budget, or forced by an overflow).
        est_after_prune = _estimate_chars(system, messages)
        if not force and est_after_prune <= budget_chars:
            return

        if self._compactions_this_run >= _MAX_COMPACTIONS_PER_RUN:
            log.warning(
                "Compaction spiral guard: %d compactions already this run for "
                "agent %s — refusing further compaction",
                self._compactions_this_run, self.source_agent_id or "?",
            )
            if _trim_tail_tool_results(messages, budget_chars) and \
                    _estimate_chars(system, messages) <= budget_chars:
                return
            raise ContextOverflowError(
                "context reduction exhausted (max compactions reached)",
                status_code=400, provider=self.name,
            )

        self._compactions_this_run += 1
        compacted = await self._compact_messages(
            messages, system, model, original_request=original_request,
            manual=manual, requested_by=requested_by,
        )
        est_after_compact = _estimate_chars(system, compacted)

        # Spiral guard: a compaction that barely moves the needle means the
        # summarizer isn't helping — abort rather than loop.
        reduction = 1.0 - (est_after_compact / est_before) if est_before > 0 else 0.0
        if reduction < _COMPACTION_MIN_REDUCTION:
            # Last resort before giving up: the usual reason compaction can't
            # help on a small window is an oversized tool_result inside the
            # retained tail (which prune protects and compact keeps verbatim).
            # Trim those in place; if that lands us under budget, carry on.
            if _trim_tail_tool_results(compacted, budget_chars) and \
                    _estimate_chars(system, compacted) <= budget_chars:
                messages[:] = compacted
                return
            log.warning(
                "Compaction reduced estimate by only %.1f%% (<%.0f%%) for agent "
                "%s — aborting to avoid a spiral", reduction * 100,
                _COMPACTION_MIN_REDUCTION * 100, self.source_agent_id or "?",
            )
            raise ContextOverflowError(
                "context reduction ineffective (compaction spiral)",
                status_code=400, provider=self.name,
            )

        # Apply the compacted list in place.
        messages[:] = compacted

    async def _compact_messages(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        original_request: str = "",
        manual: bool = False,
        requested_by: str = "",
    ) -> list[dict[str, Any]]:
        """Summarize conversation history to free context space.

        Asks the LLM to produce a concise summary of the conversation so far,
        then replaces the message history with the summary as a single user
        message.  The original user request is preserved at the end.

        ``manual`` / ``requested_by`` (§15): when set, the emitted
        CONTEXT_COMPACTION events carry ``manual: true`` and the requester id.
        """
        # §15: extra event fields shared by all three status emissions below.
        _manual_fields = {"manual": True, "requested_by": requested_by} if manual else {}

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
                    **_manual_fields,
                },
            ))

        # Retention set (§2): the tail we NEVER summarize away —
        #   (a) the original task message (messages[0] under history-as-messages)
        #   (b) the last 2 turns
        #   (c) the last ~20k chars of user messages
        # Everything before that boundary is folded into one structured summary.
        if not original_request:
            original_request = messages[0]["content"] if messages else ""

        retained_tail = _compaction_retained_tail(messages)
        to_summarize = messages[: len(messages) - len(retained_tail)]

        # Nothing to fold — the retained tail already IS the whole thing.
        if not to_summarize:
            return messages

        # Structured summary template (Goal / Progress / Key decisions /
        # Critical context / Next steps) — factored out (§15) so the idle-path
        # manual compactor reuses the exact same prompt.
        summary_messages = to_summarize + [
            {"role": "user", "content": COMPACTION_SUMMARY_PROMPT}
        ]

        try:
            summary_resp = await self.send(
                messages=summary_messages,
                system=COMPACTION_SUMMARIZER_SYSTEM,
                model=model,
                max_tokens=4096,
            )
            summary_text = summary_resp.text
            if not summary_text.strip():
                raise ValueError("summarizer returned empty text")
        except Exception as exc:
            # Fallback hard truncation (§2): NEVER no-op into a guaranteed
            # overflow. Drop the oldest turns, keeping the retained tail.
            log.warning(
                "Compaction summary failed (%s) — falling back to hard "
                "truncation for agent %s", exc, self.source_agent_id or "?",
            )
            compacted = _hard_truncate(messages, original_request, retained_tail)
            if self.event_bus and self.source_agent_id:
                from ..events import Event, EventType
                await self.event_bus.emit(Event(
                    type=EventType.CONTEXT_COMPACTION,
                    source=self.source_agent_id,
                    data={
                        "agent_id": self.source_agent_id,
                        "compaction_count": self._compaction_count,
                        "pre_tokens": self._last_input_tokens,
                        "status": "hard_truncated",
                        **_manual_fields,
                    },
                ))
            return compacted

        # Rebuild: summary message + retained tail (original task rides in the
        # summary header so it survives even if it fell outside the tail).
        compacted: list[dict[str, Any]] = [
            {"role": "user", "content": (
                f"[Context compaction — conversation summary follows]\n\n"
                f"{summary_text}\n\n"
                f"---\n\n"
                f"[Original request]\n{original_request}\n\n"
                f"Continue from where you left off. Do not repeat completed work."
            )},
        ]
        compacted.extend(retained_tail)

        log.info("Compaction #%d complete: %d messages → 1 summary + %d retained",
                 self._compaction_count, len(messages), len(retained_tail))

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
                    **_manual_fields,
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
# Module helpers
# ---------------------------------------------------------------------------

# Hard cap on a single tool result fed back into the loop (~10k tokens).
_TOOL_RESULT_CHAR_MAX = 40_000

# Head+tail split for an oversized tool result (§6). Keeps the command echo
# (head) and the error tail — the two parts that carry signal — and elides the
# middle. head + tail must stay under _TOOL_RESULT_CHAR_MAX.
_TOOL_RESULT_HEAD_CHARS = 24_000
_TOOL_RESULT_TAIL_CHARS = 12_000


def _drain_queue(queue: "asyncio.Queue | None") -> list[str]:
    """Pop everything currently in a steering queue (§5). Never blocks."""
    if queue is None:
        return []
    drained: list[str] = []
    while True:
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return drained


def _truncate_tool_result(result_json: str, cap: int = 0) -> str:
    """Head+tail truncation of an oversized tool result (§6).

    Replaces the old tail-chop: keep the first _TOOL_RESULT_HEAD_CHARS and the
    last _TOOL_RESULT_TAIL_CHARS, eliding the middle with a marker. An
    untruncated multi-MB result would not only blow this turn's input — it gets
    re-sent on EVERY subsequent turn of the loop.

    ``cap``: optional per-request override below _TOOL_RESULT_CHAR_MAX.
    Small-window (local) models must cap results well under the window or a
    single tool result lands in the protected last-2-turns tail at a size no
    reduction tier is allowed to touch (observed live: one get_snapshot on a
    16k window → unrecoverable "compaction spiral" on the next turn).
    """
    limit = min(_TOOL_RESULT_CHAR_MAX, cap) if cap > 0 else _TOOL_RESULT_CHAR_MAX
    total = len(result_json)
    if total <= limit:
        return result_json
    head = min(_TOOL_RESULT_HEAD_CHARS, int(limit * 0.65))
    tail = min(_TOOL_RESULT_TAIL_CHARS, limit - head)
    elided = total - head - tail
    return (
        result_json[:head]
        + f"\n…[elided {elided} chars of {total}]…\n"
        + result_json[-tail:]
    )


# ---------------------------------------------------------------------------
# Context reduction helpers (§2)
# ---------------------------------------------------------------------------

# Prune (tier 1): protect the most recent N chars of tool output and only bother
# pruning if it saves at least M chars.
_PRUNE_PROTECT_RECENT_CHARS = 40_000
_PRUNE_MIN_SAVINGS = 20_000

# Compact (tier 2): retain the last ~20k chars of user messages, keep the last
# 2 turns verbatim, and cap compactions per run (spiral guard).
_COMPACT_RETAIN_USER_CHARS = 20_000
_COMPACT_RETAIN_TURNS = 2
_MAX_COMPACTIONS_PER_RUN = 2
_COMPACTION_MIN_REDUCTION = 0.20


def _estimate_chars(system: str, messages: list[dict[str, Any]]) -> int:
    """Chars across the system prompt + message stack (the chars/4 basis)."""
    return len(system) + sum(
        len(m["content"]) if isinstance(m.get("content"), str)
        else len(json.dumps(m.get("content"), default=str))
        for m in messages
    )


def _trim_tail_tool_results(messages: list[dict[str, Any]], budget_chars: int) -> bool:
    """Last-resort reduction: truncate oversized tool_result bodies ANYWHERE
    in the message stack, including the tail that prune/compact protect.

    Prune skips the last 2 turns and compact retains them verbatim — correct
    for big windows, but on a small one a single oversized tool_result in
    that tail makes reduction structurally impossible. Trimming a result is
    strictly better than aborting the run. Returns True if anything shrank.
    """
    per_result_cap = max(2_000, budget_chars // 4)
    trimmed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, str) and len(body) > per_result_cap:
                block["content"] = _truncate_tool_result(body, cap=per_result_cap)
                trimmed = True
    return trimmed


def _reduction_buffer(ctx_window: int) -> int:
    """Headroom (tokens) reserved on top of max_tokens when budgeting the
    input side of a request. 8k on large windows; scales down on small
    (local-model) windows so window − max_tokens − buffer stays positive."""
    if ctx_window <= 0:
        return 8_000
    return min(8_000, max(1_024, ctx_window // 4))


def _prune_tool_results(messages: list[dict[str, Any]]) -> int:
    """Replace old tool_result bodies with a stub, in place (§2 tier 1).

    Protects the most recent ``_PRUNE_PROTECT_RECENT_CHARS`` of tool output
    (walking newest→oldest) and everything in the last 2 assistant turns.
    Returns the estimated chars freed.
    """
    # Index of the second-to-last assistant turn: results after it are recent
    # and must not be pruned.
    assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    protect_from = assistant_idxs[-_COMPACT_RETAIN_TURNS] if len(assistant_idxs) >= _COMPACT_RETAIN_TURNS else 0

    saved = 0
    recent_budget = _PRUNE_PROTECT_RECENT_CHARS
    # Walk newest→oldest so the char budget protects the freshest output.
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            body = block.get("content", "")
            body_len = len(body) if isinstance(body, str) else len(json.dumps(body, default=str))
            # Protect recent turns and the recent-char budget.
            if i >= protect_from or recent_budget > 0:
                recent_budget -= body_len
                continue
            if not isinstance(body, str) or body.startswith("[pruned:"):
                continue
            name = ""
            # tool_result carries no name; leave name empty (looked up by id
            # upstream if needed).
            content[j] = {
                **block,
                "content": f"[pruned: {body_len} chars, tool={name or block.get('tool_use_id', '')}]",
            }
            saved += body_len - len(content[j]["content"])
    return saved


def _compaction_retained_tail(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The suffix of ``messages`` that compaction must keep verbatim (§2).

    Keeps the last ``_COMPACT_RETAIN_TURNS`` messages, extended backward to
    also cover the last ~``_COMPACT_RETAIN_USER_CHARS`` chars of user messages.
    Never includes messages[0] (the original task — it rides in the summary
    header instead).
    """
    if len(messages) <= 1:
        return []
    n = len(messages)
    keep_from = max(1, n - _COMPACT_RETAIN_TURNS)
    # Walk further back while we're still within the user-char budget.
    user_chars = 0
    i = keep_from
    while i > 1:
        m = messages[i - 1]
        if m.get("role") == "user":
            c = m.get("content")
            clen = len(c) if isinstance(c, str) else len(json.dumps(c, default=str))
            if user_chars + clen > _COMPACT_RETAIN_USER_CHARS:
                break
            user_chars += clen
        i -= 1
    return messages[i:]


def _hard_truncate(
    messages: list[dict[str, Any]],
    original_request: str,
    retained_tail: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback when the summarizer fails (§2): drop oldest, keep the tail.

    Never a no-op — always returns a strictly smaller list than the input when
    there is anything to drop, so an overflow can't recur unchanged.
    """
    head = {"role": "user", "content": (
        f"[Context truncated — older history dropped after a summarizer failure]\n\n"
        f"[Original request]\n{original_request}\n\n"
        f"Continue from where you left off. Do not repeat completed work."
    )}
    return [head, *retained_tail]


# ---------------------------------------------------------------------------
# Orphan repair (§3)
# ---------------------------------------------------------------------------

def _repair_orphan_tool_uses(messages: list[dict[str, Any]], reason: str) -> None:
    """Append synthetic tool_result blocks for any dangling tool_use (§3).

    If the trailing assistant message carries ``tool_use`` blocks with no
    matching ``tool_result`` (an abort/interrupt mid-turn), append a user
    message with cancelled tool_result blocks so the persisted history replays
    cleanly against Anthropic-shape APIs. In place.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return
    content = messages[-1].get("content")
    if not isinstance(content, list):
        return
    open_ids = [
        b["id"] for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
    ]
    if not open_ids:
        return
    reason = reason or "aborted"
    results = [
        {
            "type": "tool_result",
            "tool_use_id": tid,
            "content": json.dumps({"cancelled": True, "reason": reason}),
            "is_error": True,
        }
        for tid in open_ids
    ]
    messages.append({"role": "user", "content": results})


# ---------------------------------------------------------------------------
# Loop detection (§4) + malformed args (§8)
# ---------------------------------------------------------------------------

# Oscillation guard window: over the last N tool calls, if there are ≤2 distinct
# fingerprints and each appeared ≥2 times, the agent is oscillating.
_OSCILLATION_WINDOW = 6


def _is_oscillating(fingerprints: list[str]) -> bool:
    """True if the recent fingerprints show a low-diversity oscillation (§4)."""
    if len(fingerprints) < _OSCILLATION_WINDOW:
        return False
    window = fingerprints[-_OSCILLATION_WINDOW:]
    counts: dict[str, int] = {}
    for fp in window:
        counts[fp] = counts.get(fp, 0) + 1
    return len(counts) <= 2 and all(c >= 2 for c in counts.values())


def _is_malformed_args(args: Any) -> bool:
    """True if tool args are the adapter's {"raw": ...} degradation shape (§8).

    An adapter that couldn't parse the tool-call arguments as JSON emits
    ``{"raw": <original text>}``. That single-key shape is unambiguous — a
    real tool taking exactly one string field named ``raw`` doesn't exist in
    the surface, and the loss of structure means we must not execute.
    """
    return (
        isinstance(args, dict)
        and set(args.keys()) == {"raw"}
        and isinstance(args.get("raw"), str)
    )


def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape prior-conversation dicts into valid Messages-API history.

    - roles other than user/assistant (e.g. "system" turns) become user
      turns tagged ``[system]``;
    - consecutive same-role turns are merged (the API requires alternation);
    - the list must start with a user turn — a leading assistant turn gets
      a synthetic resume marker in front of it.
    """
    out: list[dict[str, Any]] = []
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if not content:
            continue
        if role not in ("user", "assistant"):
            content = f"[system] {content}"
            role = "user"
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n\n" + content}
        else:
            out.append({"role": role, "content": content})
    if out and out[0]["role"] == "assistant":
        out.insert(0, {"role": "user", "content": "[Conversation resumed]"})
    return out


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Non-transient provider error (bad request, auth failure, etc)."""

    def __init__(self, message: str, status_code: int | None = None, provider: str = "") -> None:
        self.status_code = status_code
        self.provider = provider
        super().__init__(message)


class ContextOverflowError(ProviderError):
    """The request exceeded the model's context / token limit.

    Adapters map their provider's overflow shape (a 400 whose message matches
    a context/token-limit pattern) onto this shared class so the loop can route
    it to context reduction (§2) rather than a blind retry. It is NOT a
    transient failure — retrying the same request will fail identically.
    """


# Substrings that mark a 400 as a context-window overflow rather than a
# generic bad request. Case-insensitive match against the error message.
_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context length",
    "context window",
    "maximum context",
    "context_length_exceeded",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "reduce the length",
    "maximum number of tokens",
    "exceeds the maximum",
    "token limit",
)


def is_overflow_message(message: str) -> bool:
    """True if a provider error message looks like a context-limit overflow."""
    low = (message or "").lower()
    return any(p in low for p in _OVERFLOW_PATTERNS)


# HTTP statuses that are transient at the loop level (escaped the adapter's
# own bounded retry): rate limits, overload, gateway/service errors.
_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def classify_provider_error(exc: Exception) -> str:
    """Bucket a raised provider exception into a loop route (§1).

    Returns one of ``"overflow"``, ``"transient"``, ``"fatal"``. The
    adapters already translate their provider-specific overflow shape into
    ``ContextOverflowError``; this also sniffs a plain ``ProviderError`` whose
    400 message matches an overflow pattern, so an adapter that hasn't been
    upgraded still routes correctly.
    """
    if isinstance(exc, ContextOverflowError):
        return "overflow"
    # Timeouts / dropped streams surface as their own transport exceptions;
    # match by name to avoid importing httpx here.
    exc_name = type(exc).__name__
    if exc_name in ("TimeoutException", "ReadTimeout", "ConnectTimeout",
                    "ConnectError", "ReadError", "RemoteProtocolError",
                    "StreamError", "PoolTimeout"):
        return "transient"
    if isinstance(exc, ProviderError):
        status = exc.status_code
        if status == 400 and is_overflow_message(str(exc)):
            return "overflow"
        if status in _TRANSIENT_STATUSES:
            return "transient"
        return "fatal"
    return "fatal"
