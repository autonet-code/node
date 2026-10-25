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

# System prompt for the IDLE-path summarizer (agent_tools.compact_agent),
# which runs outside any loop and has no cache prefix to preserve. The running
# loop does not use it: see COMPACTION_INLINE_PROMPT (§20).
COMPACTION_SUMMARIZER_SYSTEM = (
    "You are a conversation summarizer. Produce a dense, accurate summary."
)

# In-loop compaction instruction (§20). Appended as the final user message of
# a request that otherwise matches the loop's own requests byte for byte
# (system prompt, tools, history prefix), so the prompt cache still hits.
COMPACTION_INLINE_PROMPT = (
    "[Context checkpoint] Stop working on the task for this one reply and do "
    "not call any tools. The history above is about to be replaced by what "
    "you write now, followed by only the most recent messages. Write a "
    "handoff note that lets you continue seamlessly from it alone: keep "
    "exact file paths, commands, identifiers, error messages, results you "
    "still need, and anything left unfinished. Use exactly these headings:\n\n"
    "## Goal\n## Progress\n## Key decisions\n## Critical context\n## Next steps"
)


# §22 Per-run plan tool. Handled by the generic loop itself: the plan is stored
# in run state (never executed by tool_executor) and carried through
# compaction and hard resets so it survives context reduction.
_PLAN_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed")
_PLAN_MAX_STEPS = 50

UPDATE_PLAN_TOOL: dict[str, Any] = {
    "name": "update_plan",
    "description": (
        "Record or revise your step-by-step plan for the current task. Send "
        "the complete list every time (it replaces the previous one). Each "
        "step has a short description and a status: pending, in_progress or "
        "completed; keep at most one step in_progress. Use it for multi-step "
        "work so progress stays visible and survives context compaction; "
        "skip it for trivial one-step tasks. It does not run anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Optional note on why the plan changed.",
            },
            "plan": {
                "type": "array",
                "description": "The full ordered list of steps.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {"type": "string", "enum": list(_PLAN_STATUSES)},
                    },
                    "required": ["step", "status"],
                },
            },
        },
        "required": ["plan"],
    },
}


def format_plan(plan: list[dict[str, str]] | None, explanation: str = "") -> str:
    """Render the run plan as text for compaction / reset messages (§22).
    Empty string when there is no plan."""
    if not plan:
        return ""
    lines = ["[Current plan]"]
    if explanation:
        lines.append(f"Note: {explanation}")
    for i, item in enumerate(plan, 1):
        lines.append(f"{i}. [{item.get('status', '')}] {item.get('step', '')}")
    return "\n".join(lines)


# §4 loop guard warning turn (one per run, before an abort).
LOOP_WARNING_PROMPT = (
    "You appear to be looping — repeating the same tool "
    "call(s) without progress. Change your approach or "
    "conclude with a final answer. Do not repeat the last "
    "call as-is."
)

# Loop-guard defaults when neither the agent definition nor its distro
# policy sets them (atn/harness_policy.py reads these for the reference).
_DEFAULT_PER_TURN_INPUT_MAX = 200_000
# §4: exact-repeat threshold drops 5 → 3 (opencode DOOM_LOOP_THRESHOLD).
_DEFAULT_REPEAT_CALL_LIMIT = 3


def _policy_of(obj: Any = None) -> Any:
    """The HarnessPolicy in force for ``obj`` (a provider mid-run), else the
    reference policy (docs/tool_substrate.md, Decision 2026-10-12)."""
    pol = getattr(obj, "_run_policy", None) if obj is not None else None
    if pol is not None:
        return pol
    from ..harness_policy import reference_policy
    return reference_policy()


# §24: sent when a reply hit the output token cap in the middle of a tool call.
TRUNCATED_TOOL_CALL_PROMPT = (
    "Your last reply reached the output token limit before its tool call was "
    "complete, so it was discarded and nothing was run. Issue the call again "
    "with smaller arguments (for example, split a large file write into "
    "several smaller writes or edits)."
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

    # v3 review step: True when this provider's own agent loop
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
    # Session-cumulative dollar spend, priced from atn.metering's per-model
    # table. Stays 0 for unpriced models (local/ollama, subscription), which
    # is what the UI cost readouts gate on. Cumulative like the _cumulative_*
    # counters above — NOT reset per run.
    _total_cost_usd: float = 0.0

    # Live context snapshot (set by run_agent_loop for the duration of a
    # run). The message list is otherwise LOCAL to the loop; these references
    # let an out-of-band inspector (context_inspect / the get_context worker
    # cmd) measure the exact context the loop is working with. _live_messages
    # is the same list object the loop mutates, so compactions and appends are
    # visible to readers. None = no run has populated a snapshot (bridge
    # providers, which override run_agent_loop, never set one).
    _live_system: str = ""
    _live_tools: Any = None
    _live_messages: Any = None
    _live_max_tokens: int = 0
    # Tool definitions of the current run (§20: reused by the compactor).
    _run_tool_defs: Any = None
    # Per-run plan (§22), set by the update_plan tool. Reset each run.
    _run_plan: list = []
    _run_plan_explanation: str = ""
    # Distro loop policy for the current run (atn/harness_policy.py). None =
    # reference. Set at the top of run_agent_loop.
    _run_policy: Any = None

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
            "total_cost_usd": round(self._total_cost_usd, 6),
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

    @property
    def provider_id(self) -> str:
        """Canonical provider id: the key budgets and usage are booked under.
        Defaults to ``name``; per-agent instances whose name carries the agent
        id override it."""
        return self.name

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
    def supports_agent_loop(self) -> bool:
        """Whether this provider supports multi-turn agent loop.

        Returns True by default — the base class provides a generic
        implementation of ``run_agent_loop()`` that uses ``send_stream()``
        in a multi-turn loop.  Providers like BridgeProvider override this
        with a native implementation (e.g. Claude Agent SDK subprocess).
        """
        return True

    # Context compaction: the real trigger is the pre-send budget check in
    # run_agent_loop (window - max_tokens - _reduction_buffer(window)); see
    # compaction_trigger_tokens() below for the single source of truth.
    _compaction_count: int = 0
    # Compactions within the current run_agent_loop run (spiral guard, §2).
    # Reset at the start of each run.
    _compactions_this_run: int = 0
    # Tier-3 hard context resets within the current run (§2). When prune +
    # compact + tail-trim can't get under budget, the stack is rebuilt from
    # the original task + a tool-call digest instead of aborting the run.
    _hard_resets_this_run: int = 0

    # Interrupt flag — set via interrupt() to stop the agent loop.
    _interrupted: bool = False

    # Manual-compaction flag (§15). Set via request_compaction() by the
    # compact_agent tool while an agent-loop run is running; the loop honors it
    # at the next iteration boundary (next to the steering drain) by forcing a
    # context reduction, then clears it. requested_by is threaded onto the
    # emitted CONTEXT_COMPACTION event so the owner sees who asked.
    _compact_requested: bool = False
    _compact_requested_by: str = ""

    # Mid-turn steering queue (§5). Lazily created; only non-None while an
    # agent-loop run is active. send_user_message() enqueues onto it; the loop
    # drains it at each iteration boundary and appends each item as a user turn.
    _steering_queue: asyncio.Queue | None = None

    async def interrupt(self) -> None:
        """Signal the agent loop to stop after the current turn."""
        self._interrupted = True

    def request_compaction(self, requested_by: str = "") -> bool:
        """Request a manual compaction at the next loop iteration (§15).

        Sets a flag honored by ``run_agent_loop`` at its iteration boundary
        (right where steering is drained): it forces a ``_reduce_context`` pass
        even if the size estimate is under budget. Returns True if an
        agent-loop run is active (``_steering_queue`` is open) to consume the
        request, False otherwise so the caller can fall back to the idle path.

        BridgeProvider has no honoring loop of its own (its agent-loop run lives
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
        """Inject a user message mid-run (§5).

        Backed by ``_steering_queue`` (created for the duration of a
        ``run_agent_loop`` run). Returns True if an agent-loop run is active
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

    async def run_agent_loop(
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
        harness_policy: Any = None,
        **_unused: Any,
    ) -> ProviderResponse:
        """Multi-turn agent-loop run with tool relay.

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
            harness_policy: The agent's distro loop policy (HarnessPolicy or
                            its dict overlay). None = the reference policy,
                            byte-identical to the pre-policy loop.
        """
        from ..harness_policy import as_policy
        pol = as_policy(harness_policy)
        # Read by _reduce_context / _compact_messages / helpers via _policy_of.
        self._run_policy = pol
        self._interrupted = False
        # Open the steering queue for the duration of this run (§5). Any
        # send_user_message() call now enqueues instead of returning False.
        self._steering_queue = asyncio.Queue()
        self._compactions_this_run = 0
        self._hard_resets_this_run = 0

        # Track active model for session stats
        if model:
            self._active_model = model

        # §22: the per-run plan tool rides on every generic-loop tool surface.
        # It is handled by the loop itself (never reaches tool_executor).
        self._run_plan = []
        self._run_plan_explanation = ""
        if tools and tool_executor is not None and not any(
            isinstance(t, dict) and t.get("name") == UPDATE_PLAN_TOOL["name"]
            for t in tools
        ):
            tools = [*tools, UPDATE_PLAN_TOOL]

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
        # §20: the in-loop compactor re-sends these so its request shares the
        # loop's prompt-cache prefix.
        self._run_tool_defs = tool_defs
        # §21: tools that may run concurrently within one turn.
        parallel_names = parallel_safe_tool_names(tools or [], policy=pol)

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
        # Precedence: agent definition > distro policy > reference.
        effective_per_turn_max = (per_turn_input_max if per_turn_input_max is not None
                                  else pol.per_turn_input_max)
        effective_repeat_limit = (repeat_call_limit if repeat_call_limit is not None
                                  else pol.repeat_call_limit)
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
        # (outer_name, args) for every call, so the review gate can resolve
        # what `use_tool` actually DISPATCHED to — only REGISTERED tools
        # enter consensus, and only those owe a review.
        called_tool_calls: list[tuple[str, dict]] = []
        review_injected = False
        # The agent's answer from before the review turn: the review is
        # bookkeeping, never the answer (same rule as the bridge path), so
        # every exit returns this text when it was set.
        self._answer_before_review = None
        # Verification step (§16): code files the run wrote/edited. If any,
        # inject ONE closing verify turn at the natural end (same pattern as
        # the review step) — agents that never touched code are unaffected.
        verify_injected = False
        modified_code_files: set[str] = set()

        # Per-request output cap comes from the model spec (§7), bounded at 16k
        # so a huge output ceiling doesn't shrink the usable input budget, then
        # clamped to a quarter of the window. Small-window models (locals at
        # 16k) need that clamp or the pre-send input budget (window − max_tokens
        # − buffer) goes negative and context reduction can never succeed
        # (observed live: 16k window − 16k max_tokens − 8k buffer → instant
        # "compaction spiral" abort on turn 1). See output_reserve_tokens().
        max_tokens = output_reserve_tokens(model or self._active_model)

        # Publish the live context snapshot for out-of-band inspection
        # (context_inspect.breakdown_from_provider). Same list object as the
        # loop below mutates — kept after the run ends so the last-known
        # context stays inspectable until the next run replaces it.
        self._live_system = system
        self._live_tools = list(tools) if tools else []
        self._live_messages = messages
        self._live_max_tokens = max_tokens

        # §19: (provider-reported input tokens of the last response, length of
        # the message stack that request carried). None = no usable usage yet.
        usage_anchor: tuple[int, int] | None = None
        # §23: highest context-headroom level already announced this run.
        headroom_noted = 0
        # §24: one re-issue request per output-cap-truncated tool turn.
        truncated_retry_used = False

        for turn in range(max_turns):
            if self._interrupted:
                return Provider._finalize_agent_loop(self, 
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
                usage_anchor = None
                try:
                    await Provider._reduce_context(
                        self, messages, system, model, message, max_tokens,
                        force=True, manual=True, requested_by=requested_by,
                    )
                except ContextOverflowError as exc:
                    log.warning("Manual compaction reduction gave up: %s", exc)
                    return Provider._finalize_agent_loop(self,
                        ProviderResponse(
                            text=f"Aborted: {exc}",
                            stop_reason="context_overflow",
                            model=model or self._active_model,
                        ),
                        messages, cumulative_usage, abort_reason="context_overflow",
                    )

            # Pre-flight per-turn input ceiling — refuse a turn whose estimated
            # input would exceed the configured cap. §19: the estimate is the
            # provider-reported input of the last response (system + tools +
            # the stack as it was then) plus chars/4 of what was appended
            # since; chars/4 over the whole stack only when no usage is known
            # yet (first turn, or right after a reduction rewrote the stack).
            estimated_input = _next_input_estimate(system, messages, usage_anchor)
            if effective_per_turn_max > 0 and estimated_input > effective_per_turn_max:
                log.warning(
                    "Per-turn input ceiling hit (estimated %d > %d) — aborting cognitive loop "
                    "for agent %s. Raise AgentDefinition.per_turn_input_max if intended.",
                    estimated_input, effective_per_turn_max, self.source_agent_id or "?",
                )
                return Provider._finalize_agent_loop(self, 
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
                        est_tokens=estimated_input,
                    )
                    usage_anchor = None  # stack rewritten; usage no longer maps
                except ContextOverflowError as exc:
                    log.warning("Pre-send context reduction gave up: %s", exc)
                    return Provider._finalize_agent_loop(self, 
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
                nonlocal usage_anchor
                usage_anchor = None
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
                return Provider._finalize_agent_loop(self, 
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
                return Provider._finalize_agent_loop(self, 
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
            # §19 anchor: that input covered exactly messages[:len(messages)]
            # (nothing has been appended for this response yet).
            usage_anchor = (
                (self._last_input_tokens, len(messages))
                if self._last_input_tokens > 0 else None
            )
            if response.model:
                self._active_model = response.model

            # Dollar spend for this turn, priced off the metering table. Returns
            # None for unpriced models (local/ollama, subscription), so those
            # keep reporting 0 and the UI cost chips stay hidden.
            try:
                from ..metering import cost_usd as _cost_usd
                _usd = _cost_usd(
                    response.model or model or self._active_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_read_tokens=response.usage.cache_read_tokens,
                    cache_write_tokens=response.usage.cache_creation_tokens,
                )
                if _usd:
                    self._total_cost_usd += _usd
            except Exception:
                log.debug("cost accumulation failed; continuing", exc_info=True)

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
                                "Inner-loop budget exceeded mid-run "
                                "(blocker=%s) — aborting cognitive loop for agent %s",
                                blocker, self.source_agent_id or "?",
                            )
                            return Provider._finalize_agent_loop(self, 
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
                            "total_cost_usd": round(self._total_cost_usd, 6),
                        },
                    },
                ))

            # §24: the reply hit the output cap while emitting tool calls. The
            # last call is cut off (its args are partial or degraded to
            # {"raw": ...}), so the turn is discarded rather than finalized as
            # if the agent had finished: ask once for a shorter re-issue.
            if (response.stop_reason == "max_tokens" and response.tool_calls
                    and tool_executor is not None):
                if not truncated_retry_used:
                    truncated_retry_used = True
                    log.info(
                        "Output cap hit mid tool call for agent %s; asking "
                        "for a shorter re-issue", self.source_agent_id or "?",
                    )
                    if response.text:
                        messages.append(
                            {"role": "assistant", "content": response.text})
                    messages.append({"role": "user",
                                     "content": pol.truncated_tool_call_prompt})
                    continue
            elif response.tool_calls and response.stop_reason == "tool_use":
                truncated_retry_used = False

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
                # §16 verify step: one closing turn before finalize when the
                # run modified code files. Same one-shot injection pattern as
                # the review step below; spends a turn of the same budget.
                if (modified_code_files and not verify_injected
                        and pol.verify_enabled
                        and tool_executor is not None):
                    verify_injected = True
                    log.info(
                        "Verify step: injecting closing verification turn "
                        "for agent %s (%d code files)",
                        self.source_agent_id or "?", len(modified_code_files),
                    )
                    if response.text:
                        messages.append(
                            {"role": "assistant", "content": response.text})
                    messages.append({
                        "role": "user",
                        "content": pol.verify_step_prompt.format(
                            files=", ".join(sorted(modified_code_files)[:20])),
                    })
                    continue
                from ..delegate_prompts import review_owed_tools
                _owed = (
                    review_owed_tools(called_tool_calls)
                    if (review_tools and pol.review_enabled and not review_injected
                        and tool_executor is not None
                        and "attest_tools" not in called_tool_names)
                    else []
                )
                if _owed:
                    review_injected = True
                    self._answer_before_review = response.text or None
                    log.info(
                        "Review step: injecting closing attest turn for "
                        "agent %s (%s)", self.source_agent_id or "?",
                        ", ".join(_owed[:5]),
                    )
                    if response.text:
                        # Keep the agent's final answer in history so the
                        # review turn doesn't erase it.
                        messages.append(
                            {"role": "assistant", "content": response.text})
                    from ..delegate_prompts import format_review_prompt
                    messages.append(
                        {"role": "user",
                         "content": format_review_prompt(_owed, policy=pol)})
                    continue
                return Provider._finalize_agent_loop(self,
                    response, messages, cumulative_usage,
                )

            # No executor — return as-is (tool calls visible but not executed)
            if tool_executor is None:
                return Provider._finalize_agent_loop(self, 
                    response, messages, cumulative_usage,
                )

            # Execute tool calls and build continuation messages
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                called_tool_names.add(tc.name)
                called_tool_calls.append((tc.name, tc.input or {}))
                if tc.name in ("write_file", "edit_file"):
                    _p = str((tc.input or {}).get("path", ""))
                    if _p.endswith(pol.code_file_suffixes):
                        modified_code_files.add(_p)
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
                if len(osc_fingerprints) > pol.oscillation_window:
                    osc_fingerprints = osc_fingerprints[-pol.oscillation_window:]

                exact_repeat = (
                    len(recent_call_signatures) >= effective_repeat_limit
                    and len(set(recent_call_signatures)) == 1
                )
                oscillating = _is_oscillating(osc_fingerprints,
                                              window=pol.oscillation_window)

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
                        messages.append({"role": "user",
                                         "content": pol.loop_warning_prompt})
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
                    return Provider._finalize_agent_loop(self, 
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

            async def _run_one(tc: ToolCall) -> dict[str, Any]:
                """Execute one tool call; always returns its tool_result block."""
                if tc.name == UPDATE_PLAN_TOOL["name"] and not _is_malformed_args(tc.input):
                    # §22: plan lives in run state; nothing is executed.
                    err = Provider._apply_plan_update(self, tc.input or {})
                    if not err and self.event_bus and self.source_agent_id:
                        from ..events import Event, EventType
                        await self.event_bus.emit(Event(
                            type=EventType.STEP_OUTPUT,
                            source=self.source_agent_id,
                            data={
                                "agent_id": self.source_agent_id,
                                "channel": "plan",
                                "plan": list(self._run_plan),
                                "explanation": self._run_plan_explanation,
                            },
                        ))
                    block = {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Error: {err}" if err else "Plan updated.",
                    }
                    if err:
                        block["is_error"] = True
                    return block
                # §8: malformed tool arguments. When an adapter can't parse the
                # tool-call args as JSON it degrades them to {"raw": <text>}.
                # Do NOT execute — synthesize an error tool_result telling the
                # model to re-emit, and continue the loop.
                if _is_malformed_args(tc.input):
                    log.warning(
                        "Malformed tool args for %s (agent %s) — not executing",
                        tc.name, self.source_agent_id or "?",
                    )
                    return {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps({
                            "error": "arguments were not valid JSON — re-emit the "
                                     "tool call with well-formed JSON arguments",
                        }),
                        "is_error": True,
                    }

                log.info("Agent loop tool %s (turn %d/%d)", tc.name, turn + 1, max_turns)

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

                # §18: readable text for string / shell-shaped results, JSON
                # only for genuinely structured ones.
                result_json = format_tool_result(result)
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
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_json,
                }
                if is_error:
                    block["is_error"] = True
                return block

            # §21: consecutive parallel-safe (read-only) calls run concurrently;
            # anything else runs alone, in order, after the batch before it has
            # finished (read/write-lock semantics). Results keep call order.
            tool_result_content = await _execute_tool_batches(
                response.tool_calls, _run_one, parallel_names,
                max_parallel=pol.max_parallel_tool_calls,
            )
            messages.append({"role": "user", "content": tool_result_content})

            # §23 headroom signal: once the next request crosses 70/80/90% of
            # the usable input budget, tell the model with these tool results
            # (never in the system prompt, which must stay byte-stable for the
            # prompt cache). Each level is announced once; dropping back under
            # the first level (after a reduction) re-arms it.
            _trigger = compaction_trigger_tokens(
                self._active_model or model,
                get_context_window(self._active_model or model),
            )
            if _trigger > 0 and tool_result_content:
                _est = _next_input_estimate(system, messages, usage_anchor)
                _level = _headroom_level(_est, _trigger,
                                         levels=pol.headroom_levels)
                if _level > headroom_noted:
                    last = tool_result_content[-1]
                    if isinstance(last.get("content"), str):
                        last["content"] += _headroom_note(
                            _est, _trigger, template=pol.headroom_note_template)
                    headroom_noted = _level
                elif _level == 0:
                    headroom_noted = 0

            # §2 note: context reduction runs PRE-SEND at the top of the next
            # iteration (estimate-based) and reactively on an overflow 400, so
            # the old post-tool compaction is no longer needed here.

        # Exhausted max_turns — return last response with accumulated usage
        return Provider._finalize_agent_loop(self, response, messages, cumulative_usage)

    def _apply_plan_update(self, args: dict[str, Any]) -> str:
        """Validate and store an ``update_plan`` call (§22). Returns an error
        string, or "" on success."""
        plan = args.get("plan")
        if not isinstance(plan, list):
            return "'plan' must be a list of {step, status} objects"
        clean: list[dict[str, str]] = []
        in_progress = 0
        for item in plan:
            if not isinstance(item, dict):
                return "each plan item must be an object with 'step' and 'status'"
            step = str(item.get("step") or "").strip()
            status = str(item.get("status") or "").strip()
            if not step:
                return "each plan item needs a non-empty 'step'"
            if status not in _PLAN_STATUSES:
                return (f"invalid status {status!r}; use one of "
                        + ", ".join(_PLAN_STATUSES))
            in_progress += status == "in_progress"
            clean.append({"step": step[:500], "status": status})
        if in_progress > 1:
            return "at most one step may be in_progress"
        if len(clean) > _PLAN_MAX_STEPS:
            return f"plan too long (max {_PLAN_MAX_STEPS} steps)"
        self._run_plan = clean
        expl = args.get("explanation")
        if isinstance(expl, str):
            self._run_plan_explanation = expl.strip()[:1000]
        return ""

    def _finalize_agent_loop(
        self,
        response: ProviderResponse,
        messages: list[dict[str, Any]],
        cumulative_usage: Usage,
        abort_reason: str = "",
    ) -> ProviderResponse:
        """Shared exit path for run_agent_loop (§3, §5).

        - Repairs orphaned ``tool_use`` blocks in ``messages`` in place so the
          history replays cleanly against Anthropic-shape APIs.
        - Drains any undrained steering messages into
          ``response.undelivered_steering`` so the caller can re-post them to
          the inbox (never silently dropped).
        - Closes the steering queue.

        - Keeps the agent's answer from before a review turn as the text.

        Every ``return`` from run_agent_loop goes through here.
        """
        _repair_orphan_tool_uses(messages, abort_reason)
        answer = getattr(self, "_answer_before_review", None)
        if answer:
            response.text = answer
        self._answer_before_review = None
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
        connection_attempts = 0
        while True:
            # §24: whole-stack tool_use/tool_result pairing before every send
            # attempt (a reactive reduction may have just rewritten the stack).
            _fixed = _repair_history(messages)
            if _fixed:
                log.warning("Repaired %d unpaired tool call/result block(s) in "
                            "history for agent %s", _fixed,
                            self.source_agent_id or "?")
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
                if route == "transient" and _is_connection_error(exc):
                    # §24: a dropped/refused connection is usually a short
                    # network blip, unlike a 429/5xx that needs the server to
                    # cool off. Separate, quicker and longer schedule so it
                    # neither waits 30s for a blip nor eats the 429 budget.
                    if connection_attempts >= len(_CONNECTION_BACKOFF):
                        log.warning(
                            "Connection error exhausted %d loop-level retries "
                            "for agent %s: %s", len(_CONNECTION_BACKOFF),
                            self.source_agent_id or "?", exc,
                        )
                        raise
                    wait = _CONNECTION_BACKOFF[connection_attempts]
                    wait += random.uniform(0, wait * 0.25)
                    connection_attempts += 1
                    log.warning(
                        "Connection error (retry %d/%d, sleeping %.1fs) for "
                        "agent %s: %s", connection_attempts,
                        len(_CONNECTION_BACKOFF), wait,
                        self.source_agent_id or "?", exc,
                    )
                    await asyncio.sleep(wait)
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
                            # Capped: a huge Retry-After would otherwise park
                            # the run silently for hours.
                            wait = max(wait, min(float(retry_after),
                                                 _RETRY_AFTER_CAP_S))
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
        est_tokens: int = 0,
    ) -> None:
        """Two-tier context reduction: prune, then compact (§2).

        ``est_tokens`` (§19): the caller's usage-calibrated estimate of the
        next request. When it exceeds the raw chars/4 figure (tool schemas,
        tokenizer density), the gap is carried as a fixed char offset on every
        budget comparison below, so reduction targets the real size and not
        the optimistic chars/4 one.

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
        # §19 calibration: shrink the char budget by the gap between the real
        # (usage-derived) size and chars/4, so the comparisons below stay in
        # raw chars but account for what chars/4 cannot see.
        if est_tokens > 0:
            gap = est_tokens * 4 - _estimate_chars(system, messages)
            if gap > 0:
                budget_chars = max(0, budget_chars - gap)

        pol = _policy_of(self)
        # Tier 1 — prune old tool_result bodies (no LLM call).
        est_before = _estimate_chars(system, messages)
        if force or est_before > budget_chars:
            saved = _prune_tool_results(messages, policy=pol)
            if saved >= pol.prune_min_savings:
                log.info(
                    "Context prune freed ~%d chars for agent %s",
                    saved, self.source_agent_id or "?",
                )

        # Tier 2 — compact (only if still over budget, or forced by an overflow).
        est_after_prune = _estimate_chars(system, messages)
        if not force and est_after_prune <= budget_chars:
            return

        if self._compactions_this_run >= pol.max_compactions_per_run:
            log.warning(
                "Compaction spiral guard: %d compactions already this run for "
                "agent %s — refusing further compaction",
                self._compactions_this_run, self.source_agent_id or "?",
            )
            if _trim_tail_tool_results(messages, budget_chars) and \
                    _estimate_chars(system, messages) <= budget_chars:
                return
            if self._try_hard_reset(messages, original_request):
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
        if reduction < pol.compaction_min_reduction:
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
                "%s", reduction * 100,
                pol.compaction_min_reduction * 100, self.source_agent_id or "?",
            )
            if self._try_hard_reset(messages, original_request):
                return
            raise ContextOverflowError(
                "context reduction ineffective (compaction spiral)",
                status_code=400, provider=self.name,
            )

        # Apply the compacted list in place.
        messages[:] = compacted

    def _try_hard_reset(
        self, messages: list[dict[str, Any]], original_request: str,
    ) -> bool:
        """Tier-3 reduction (§2): rebuild the stack instead of aborting.

        When prune + compact + tail-trim can't get under budget (the normal
        end state on small local-model windows), the run used to abort and
        throw away all work in progress. A hard reset keeps the run alive:
        the stack becomes the original task + a non-LLM digest of every tool
        call already made, so the model resumes with fresh context, on-disk
        state intact, without repeating exploration. Capped per run; returns
        False when the cap is spent (caller aborts as before).
        """
        pol = _policy_of(self)
        if self._hard_resets_this_run >= pol.max_hard_resets_per_run:
            return False
        self._hard_resets_this_run += 1
        # The fresh stack may legitimately need compaction again much later.
        self._compactions_this_run = 0
        digest = _context_reset_digest(
            messages, cap_chars=pol.reset_digest_cap_chars)
        original = original_request
        if not original and messages:
            first = messages[0].get("content")
            original = first if isinstance(first, str) else json.dumps(first, default=str)
        notice = (
            "[CONTEXT RESET] The conversation exceeded the context window and "
            "was cleared. Everything you already did is preserved ON DISK — "
            "do not repeat prior exploration. Record of your actions so far:\n"
            f"{digest}\n\n"
            + (f"{_plan}\n\n" if (_plan := format_plan(
                self._run_plan, self._run_plan_explanation)) else "")
            + "Resume from that state and finish the task. Re-read only what "
            "you need, keep tool output small, and delegate any remaining "
            "broad exploration to child agents."
        )
        messages[:] = [{"role": "user", "content": f"{original}\n\n{notice}"}]
        log.warning(
            "Hard context reset %d/%d for agent %s (stack rebuilt, ~%d chars)",
            self._hard_resets_this_run, pol.max_hard_resets_per_run,
            self.source_agent_id or "?", _estimate_chars("", messages),
        )
        return True

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

        retained_tail = _compaction_retained_tail(messages, policy=_policy_of(self))
        to_summarize = messages[: len(messages) - len(retained_tail)]

        # Nothing to fold — the retained tail already IS the whole thing.
        if not to_summarize:
            return messages

        # §20 cache-friendly summary: same system prompt, same tools, same
        # history prefix as the loop's own requests, with the summarization
        # instruction appended as the final user message. A separate
        # summarizer system prompt guaranteed a full prompt-cache miss on the
        # most expensive request of the run.
        try:
            summary_text = await self._summarize_for_compaction(
                messages, system, model)
        except Exception as exc:
            # Fallback hard truncation (§2): NEVER no-op into a guaranteed
            # overflow. Drop the oldest turns, keeping the retained tail.
            log.warning(
                "Compaction summary failed (%s) — falling back to hard "
                "truncation for agent %s", exc, self.source_agent_id or "?",
            )
            compacted = _hard_truncate(
                messages, original_request, retained_tail,
                plan_text=format_plan(self._run_plan, self._run_plan_explanation),
            )
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
        # §22: the run plan is carried verbatim, not left to the summarizer.
        _plan_text = format_plan(self._run_plan, self._run_plan_explanation)
        compacted: list[dict[str, Any]] = [
            {"role": "user", "content": (
                f"[Context compaction — conversation summary follows]\n\n"
                f"{summary_text}\n\n"
                f"---\n\n"
                + (f"{_plan_text}\n\n" if _plan_text else "")
                + f"[Original request]\n{original_request}\n\n"
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

    async def _summarize_for_compaction(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
    ) -> str:
        """Ask the agent's own model for a handoff summary of ``messages`` (§20).

        The request is the loop's own request (same system prompt, same tool
        schemas, same message prefix) plus one appended user instruction, so
        the provider's prompt cache covers everything but the instruction.

        If the request itself overflows the window, the oldest history items
        are dropped (in valid units: the remainder always starts on a plain
        user message) and the request retried, up to
        ``_COMPACT_OVERFLOW_RETRIES`` times. If the model answers with a tool
        call instead of text, retry once without tools. Raises on final
        failure; the caller falls back to hard truncation.
        """
        tools = getattr(self, "_run_tool_defs", None) or None
        pol = _policy_of(self)
        instruction = {"role": "user",
                       "content": _compaction_instruction(self)}
        history = list(messages)
        dropped = 0
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = await self.send(
                    messages=history + [instruction],
                    system=system,
                    model=model,
                    max_tokens=pol.compact_summary_max_tokens,
                    tools=tools,
                )
            except Exception as exc:
                if (classify_provider_error(exc) == "overflow"
                        and attempts <= pol.compact_overflow_retries):
                    before = len(history)
                    history = _drop_oldest_history(
                        history, fraction=pol.compact_drop_fraction)
                    dropped += before - len(history)
                    if not history:
                        raise
                    log.info(
                        "Compaction request overflowed; dropped %d oldest "
                        "item(s) and retrying for agent %s", dropped,
                        self.source_agent_id or "?",
                    )
                    continue
                raise
            text = (resp.text or "").strip()
            if text:
                return text
            if tools and attempts <= pol.compact_overflow_retries:
                # The model reached for a tool instead of summarizing.
                tools = None
                continue
            raise ValueError("summarizer returned empty text")

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


# §21 Parallel read-only tools. Calls to these tools have no side effects the
# model could observe through another call in the same turn, so a turn's
# consecutive run of them executes concurrently. Everything not listed (and
# not flagged ``parallel_safe: True`` on its tool dict) is exclusive: it runs
# alone, in call order. A tool dict may also opt OUT with
# ``parallel_safe: False``.
PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset({
    # shell bundle (read side)
    "read_file", "list_directory", "search_files",
    # web bundle
    "web_fetch", "web_search",
    # framework tools that only read registries / stores
    "probe_tools", "list_tools", "list_agents", "get_agent", "get_execution",
    "get_output", "get_snapshot", "list_connectors", "get_history",
    "get_goals", "get_projects", "get_credit_budget", "get_my_budget_status",
    "get_usage", "list_tasks", "get_user_profile", "get_latest_thought",
    "get_children_status", "find_services",
})

# Upper bound on concurrently running calls inside one parallel batch.
_MAX_PARALLEL_TOOL_CALLS = 8


def parallel_safe_tool_names(tools: list[dict[str, Any]],
                             policy: Any = None) -> frozenset[str]:
    """The run's parallel-safe tool names: the allowlist (the distro policy's,
    else PARALLEL_SAFE_TOOLS) plus per-tool ``parallel_safe: True`` flags,
    minus explicit ``parallel_safe: False``."""
    names = set(policy.parallel_safe_tools if policy is not None
                else PARALLEL_SAFE_TOOLS)
    for t in tools:
        if not isinstance(t, dict):
            continue
        flag = t.get("parallel_safe")
        if flag is True:
            names.add(t.get("name", ""))
        elif flag is False:
            names.discard(t.get("name", ""))
    names.discard("")
    return frozenset(names)


async def _execute_tool_batches(
    calls: list[ToolCall],
    run_one: Any,
    parallel_names: frozenset[str],
    max_parallel: int = 0,
) -> list[dict[str, Any]]:
    """Run a turn's tool calls with read/write-lock semantics (§21).

    Walks the calls in order. A maximal run of consecutive parallel-safe calls
    (readers) executes concurrently; any other call (a writer) waits for the
    batch before it and runs alone. Equivalent to every call taking a fair
    RW lock in call order. ``run_one`` never raises; results come back in the
    original call order whatever the completion order.
    """
    results: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max_parallel if max_parallel > 0
                            else _MAX_PARALLEL_TOOL_CALLS)

    async def _bounded(tc: ToolCall) -> dict[str, Any]:
        async with sem:
            return await run_one(tc)

    i = 0
    n = len(calls)
    while i < n:
        tc = calls[i]
        if tc.name in parallel_names and not _is_malformed_args(tc.input):
            j = i
            while (j < n and calls[j].name in parallel_names
                   and not _is_malformed_args(calls[j].input)):
                j += 1
            if j - i == 1:
                results.append(await run_one(tc))
            else:
                results.extend(await asyncio.gather(
                    *(_bounded(c) for c in calls[i:j])))
            i = j
        else:
            results.append(await run_one(tc))
            i += 1
    return results


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


# Single-string result keys whose value IS the tool output (read_file's
# "content", search_files' "matches", ...). A dict carrying only one of these
# is sent as the bare text.
_TEXT_RESULT_KEYS: tuple[str, ...] = ("content", "matches", "output", "text", "result")


def format_tool_result(result: Any) -> str:
    """Render a tool result as the text the model reads (§18).

    JSON-encoding every result wraps shell output in quotes and escaped
    newlines, which models read worse and which costs tokens. So:
      - a string goes through as-is;
      - a shell-shaped dict (``output`` and/or ``exit_code``, optional
        ``error``) becomes ``Exit code: N`` / ``Error: ...`` + ``Output:``;
      - a dict with a single text field (``content``, ``matches``, ...)
        becomes that text; a dict with only ``error`` becomes ``Error: ...``;
      - anything else (genuinely structured) stays JSON.
    Always returns a str, so every adapter receives valid tool content.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        keys = set(result.keys())
        if keys and keys <= {"output", "exit_code", "error"} and (
            "exit_code" in keys or "output" in keys
        ) and isinstance(result.get("output", ""), str):
            lines: list[str] = []
            if "exit_code" in result:
                lines.append(f"Exit code: {result.get('exit_code')}")
            if result.get("error"):
                lines.append(f"Error: {result['error']}")
            lines.append("Output:")
            lines.append(result.get("output") or "")
            return "\n".join(lines)
        if len(keys) == 1:
            (k, v), = result.items()
            if k in _TEXT_RESULT_KEYS and isinstance(v, str):
                return v
            if k == "error" and isinstance(v, str):
                return f"Error: {v}"
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


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

# Verify step (§16): file suffixes that count as "code" for the closing
# verification turn.
_CODE_FILE_SUFFIXES = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".rb", ".php", ".cs", ".swift", ".kt", ".sol",
)

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

# §20: output cap for the compaction summary, and how many times a summary
# request that itself overflows is retried with older history dropped.
_COMPACT_SUMMARY_MAX_TOKENS = 4096
_COMPACT_OVERFLOW_RETRIES = 4
# Fraction of the history dropped (from the front) per overflow retry.
_COMPACT_DROP_FRACTION = 0.2


def _is_plain_user(msg: dict[str, Any]) -> bool:
    """A user message that does not answer an earlier tool call."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        return not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return True


def _drop_oldest_history(history: list[dict[str, Any]],
                         fraction: float = 0.0) -> list[dict[str, Any]]:
    """Drop the oldest ~20% of ``history`` (at least one item) in valid units.

    A leading tool_result message whose tool_use was dropped goes too (it
    would be an orphan), and if the remainder then starts on an assistant
    turn a short user marker is put in front so roles still start on user.
    """
    n = max(1, int(len(history) * (fraction or _COMPACT_DROP_FRACTION)))
    if history and history[0].get("content") == _HISTORY_OMITTED_MARKER:
        n += 1  # never count our own marker as history
    rest = history[n:]
    while rest and rest[0].get("role") == "user" and not _is_plain_user(rest[0]):
        rest = rest[1:]
    if rest and rest[0].get("role") != "user":
        rest = [{"role": "user", "content": _HISTORY_OMITTED_MARKER}, *rest]
    return rest


_HISTORY_OMITTED_MARKER = "[Earlier history omitted to fit the context window]"


def _compaction_instruction(provider: Any) -> str:
    """The final user message of an in-loop compaction request (§20): the
    distro policy's instruction, the reference COMPACTION_INLINE_PROMPT by
    default."""
    return _policy_of(provider).compaction_inline_prompt


# Hard reset (tier 3): stack rebuilds per run, and the size cap on the
# tool-call digest carried into the fresh stack.
_MAX_HARD_RESETS_PER_RUN = 2
_RESET_DIGEST_CAP_CHARS = 6_000


def _context_reset_digest(messages: list[dict[str, Any]],
                          cap_chars: int = 0) -> str:
    """Non-LLM digest of a message stack for a tier-3 hard reset.

    One line per tool call (name + truncated input), newest kept when over
    the cap, plus the last assistant prose. Cheap by construction — this
    path only runs when the LLM summarizer already failed to help.
    """
    entries: list[str] = []
    last_text = ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            if content.strip():
                last_text = content
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and (block.get("text") or "").strip():
                last_text = block["text"]
            elif block.get("type") == "tool_use":
                try:
                    arg = json.dumps(block.get("input", {}), default=str)
                except Exception:
                    arg = "{}"
                entries.append(f"- {block.get('name', '?')} {arg[:160]}")
    digest = "\n".join(entries)
    cap = cap_chars or _RESET_DIGEST_CAP_CHARS
    if len(digest) > cap:
        digest = "…(older calls elided)…\n" + digest[-cap:]
    if last_text:
        digest += "\n\nYour last note before the reset: " + last_text[:500]
    return digest or "(no tool calls recorded)"


def _estimate_chars(system: str, messages: list[dict[str, Any]]) -> int:
    """Chars across the system prompt + message stack (the chars/4 basis)."""
    return len(system) + sum(
        len(m["content"]) if isinstance(m.get("content"), str)
        else len(json.dumps(m.get("content"), default=str))
        for m in messages
    )


def _next_input_estimate(
    system: str,
    messages: list[dict[str, Any]],
    usage_anchor: "tuple[int, int] | None",
) -> int:
    """Tokens the next request will carry (§19).

    With an anchor ``(reported_input_tokens, stack_len_at_that_send)``: the
    provider's own count for that request (system, tool schemas and the stack
    as it was, which chars/4 cannot see) plus chars/4 of the messages appended
    since. Without one (first turn, or the stack was rewritten by a
    reduction): chars/4 over system + the whole stack.
    """
    if usage_anchor is not None:
        reported, n = usage_anchor
        if reported > 0 and 0 <= n <= len(messages):
            return reported + _estimate_chars("", messages[n:]) // _CHARS_PER_TOKEN
    return _estimate_chars(system, messages) // _CHARS_PER_TOKEN


# §23: fractions of the usable input budget (compaction trigger) at which the
# model is told how much room is left. Each is announced once per run.
_HEADROOM_LEVELS: tuple[float, ...] = (0.70, 0.80, 0.90)


def _headroom_level(est_tokens: int, trigger_tokens: int,
                    levels: "tuple[float, ...] | None" = None) -> int:
    """1-based index of the highest headroom level reached, 0 if none."""
    if trigger_tokens <= 0:
        return 0
    frac = est_tokens / trigger_tokens
    level = 0
    for i, threshold in enumerate(
            _HEADROOM_LEVELS if levels is None else levels, 1):
        if frac >= threshold:
            level = i
    return level


HEADROOM_NOTE_TEMPLATE = (
    "\n\n[Context: about {pct}% of the usable context is in use; roughly "
    "{remaining} tokens remain before older history is summarized. Keep "
    "tool output small and prefer finishing or delegating over broad "
    "new exploration.]"
)


def _headroom_note(est_tokens: int, trigger_tokens: int,
                   template: str = "") -> str:
    """Short context-headroom note appended to a tool result (§23)."""
    remaining = max(0, trigger_tokens - est_tokens)
    pct = min(100, round(100 * est_tokens / trigger_tokens)) if trigger_tokens else 0
    return (template or HEADROOM_NOTE_TEMPLATE).format(
        pct=pct, remaining=remaining)


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


# Codebase-wide estimation convention (matches atn/context_inspect.py).
_CHARS_PER_TOKEN = 4
# History ceiling used when the model's window can't be resolved. Also the
# upper bound on the scaled budget, so large-window models keep the historical
# behaviour instead of suddenly injecting far more history.
_DEFAULT_HISTORY_CHAR_BUDGET = 400_000


def output_reserve_tokens(model: str, ctx_window: int = 0) -> int:
    """Per-request output cap the agent loop reserves for ``model`` (§7).

    Bounded at 16k so a huge output ceiling doesn't shrink the usable input
    budget, then clamped to a quarter of the window so small-window (local)
    models keep a positive input budget. Single source of truth for the cap
    applied in run_agent_loop.
    """
    from ..model_specs import max_output_tokens as _spec_max_output
    reserve = min(16_384, _spec_max_output(model))
    if reserve <= 0:
        reserve = 16_384
    ctx = ctx_window if ctx_window > 0 else get_context_window(model)
    if ctx > 0:
        reserve = min(reserve, max(1_024, ctx // 4))
    return reserve


def compaction_trigger_tokens(model: str, ctx_window: int = 0) -> int:
    """Estimated input size (tokens) at which pre-send context reduction fires.

    Mirrors the budget check in run_agent_loop: an estimated input above
    ``window - max_tokens - _reduction_buffer(window)`` triggers prune, then
    compaction. Returns 0 when the window is unknown, so consumers (the
    context inspector, the UI's threshold line) can hide the marker rather
    than draw a made-up one.
    """
    if not model and ctx_window <= 0:
        return 0
    ctx = ctx_window if ctx_window > 0 else get_context_window(model)
    if ctx <= 0:
        return 0
    return max(0, ctx - output_reserve_tokens(model, ctx) - _reduction_buffer(ctx))


def history_char_budget(model: str) -> int:
    """Chars of prior-turn history the execution engine may inject for ``model``.

    Half the usable input budget (``compaction_trigger_tokens`` in chars),
    leaving room for the system prompt, tool definitions and the live tool
    loop. Falls back to the historical constant when the window is unknown.
    A 200k-window model lands above the old 400k ceiling, so API providers
    are unchanged; a 16k local model gets ~16k chars and no longer arrives
    over budget on its first send.
    """
    trigger = compaction_trigger_tokens(model)
    if trigger <= 0:
        return _DEFAULT_HISTORY_CHAR_BUDGET
    return max(20_000, min(_DEFAULT_HISTORY_CHAR_BUDGET,
                           trigger * _CHARS_PER_TOKEN // 2))


def _prune_tool_results(messages: list[dict[str, Any]], policy: Any = None) -> int:
    """Replace old tool_result bodies with a stub, in place (§2 tier 1).

    Protects the most recent ``_PRUNE_PROTECT_RECENT_CHARS`` of tool output
    (walking newest→oldest) and everything in the last 2 assistant turns.
    Returns the estimated chars freed.
    """
    # Index of the second-to-last assistant turn: results after it are recent
    # and must not be pruned.
    retain_turns = (policy.compact_retain_turns if policy is not None
                    else _COMPACT_RETAIN_TURNS)
    assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    protect_from = assistant_idxs[-retain_turns] if len(assistant_idxs) >= retain_turns else 0

    saved = 0
    recent_budget = (policy.prune_protect_recent_chars if policy is not None
                     else _PRUNE_PROTECT_RECENT_CHARS)
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


def _compaction_retained_tail(messages: list[dict[str, Any]],
                              policy: Any = None) -> list[dict[str, Any]]:
    """The suffix of ``messages`` that compaction must keep verbatim (§2).

    Keeps the last ``_COMPACT_RETAIN_TURNS`` messages, extended backward to
    also cover the last ~``_COMPACT_RETAIN_USER_CHARS`` chars of user messages.
    Never includes messages[0] (the original task — it rides in the summary
    header instead).
    """
    if len(messages) <= 1:
        return []
    n = len(messages)
    retain_turns = (policy.compact_retain_turns if policy is not None
                    else _COMPACT_RETAIN_TURNS)
    retain_chars = (policy.compact_retain_user_chars if policy is not None
                    else _COMPACT_RETAIN_USER_CHARS)
    keep_from = max(1, n - retain_turns)
    # Walk further back while we're still within the user-char budget.
    user_chars = 0
    i = keep_from
    while i > 1:
        m = messages[i - 1]
        if m.get("role") == "user":
            c = m.get("content")
            clen = len(c) if isinstance(c, str) else len(json.dumps(c, default=str))
            if user_chars + clen > retain_chars:
                break
            user_chars += clen
        i -= 1
    return messages[i:]


def _hard_truncate(
    messages: list[dict[str, Any]],
    original_request: str,
    retained_tail: list[dict[str, Any]],
    plan_text: str = "",
) -> list[dict[str, Any]]:
    """Fallback when the summarizer fails (§2): drop oldest, keep the tail.

    Never a no-op — always returns a strictly smaller list than the input when
    there is anything to drop, so an overflow can't recur unchanged.
    ``plan_text`` (§22) carries the run plan across the truncation.
    """
    head = {"role": "user", "content": (
        f"[Context truncated — older history dropped after a summarizer failure]\n\n"
        + (f"{plan_text}\n\n" if plan_text else "")
        + f"[Original request]\n{original_request}\n\n"
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


def _tool_use_ids(msg: dict[str, Any]) -> list[str]:
    content = msg.get("content")
    if msg.get("role") != "assistant" or not isinstance(content, list):
        return []
    return [
        b["id"] for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
    ]


def _repair_history(messages: list[dict[str, Any]]) -> int:
    """Pair every tool_use with a tool_result across the WHOLE stack (§24).

    Run before each send. The tail-only §3 repair covers exit paths; this
    covers anything a reduction, steering or history splice left behind in
    the middle of the stack (e.g. a compaction tail that starts on a
    tool_result whose tool_use was summarized away):
      - a tool_use with no result in the next message gets a synthetic
        ``missing`` error result (added to the next tool-result message, or
        a new user message inserted right after the assistant turn);
      - a tool_result whose id is not a tool_use of the immediately
        preceding assistant turn is turned into a short text block.
    The trailing assistant turn is left alone: its results are what the loop
    is about to append. In place; returns the number of repairs (0 = the
    stack was already valid, nothing touched).
    """
    repairs = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            ids = _tool_use_ids(msg)
            if ids and i + 1 < len(messages):
                nxt = messages[i + 1]
                nxt_content = nxt.get("content")
                have: set[str] = set()
                if nxt.get("role") == "user" and isinstance(nxt_content, list):
                    have = {
                        b.get("tool_use_id") for b in nxt_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                missing = [tid for tid in ids if tid not in have]
                if missing:
                    synthetic = [
                        {"type": "tool_result", "tool_use_id": tid,
                         "content": "Error: no result was recorded for this call.",
                         "is_error": True}
                        for tid in missing
                    ]
                    if have:
                        # Results must lead the message, before any text.
                        nxt["content"] = [*nxt_content, *synthetic]
                        _results_first(nxt)
                    else:
                        messages.insert(i + 1, {"role": "user", "content": synthetic})
                    repairs += len(missing)
        elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
            valid = set(_tool_use_ids(messages[i - 1])) if i > 0 else set()
            new_blocks = []
            changed = False
            for b in msg["content"]:
                if (isinstance(b, dict) and b.get("type") == "tool_result"
                        and b.get("tool_use_id") not in valid):
                    new_blocks.append({
                        "type": "text",
                        "text": "[result of an earlier tool call whose request "
                                "is no longer in context]",
                    })
                    changed = True
                    repairs += 1
                else:
                    new_blocks.append(b)
            if changed:
                msg["content"] = new_blocks
        i += 1
    return repairs


def _results_first(msg: dict[str, Any]) -> None:
    """Order a user message's blocks tool_results first (API requirement)."""
    content = msg.get("content")
    if isinstance(content, list):
        msg["content"] = sorted(
            content,
            key=lambda b: 0 if isinstance(b, dict) and b.get("type") == "tool_result" else 1,
        )


# ---------------------------------------------------------------------------
# Loop detection (§4) + malformed args (§8)
# ---------------------------------------------------------------------------

# Oscillation guard window: over the last N tool calls, if there are ≤2 distinct
# fingerprints and each appeared ≥2 times, the agent is oscillating.
_OSCILLATION_WINDOW = 6


def _is_oscillating(fingerprints: list[str], window: int = 0) -> bool:
    """True if the recent fingerprints show a low-diversity oscillation (§4)."""
    size = window or _OSCILLATION_WINDOW
    if len(fingerprints) < size:
        return False
    window = fingerprints[-size:]
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


# §24: loop-level backoff for connection-level failures (refused / reset /
# dropped stream), separate from the 429/5xx schedule.
_CONNECTION_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
_CONNECTION_ERROR_NAMES: frozenset[str] = frozenset({
    "ConnectError", "ConnectTimeout", "ReadError", "WriteError",
    "RemoteProtocolError", "StreamError",
})
# Longest Retry-After the loop will honor before retrying anyway.
_RETRY_AFTER_CAP_S = 300.0


def _is_connection_error(exc: BaseException) -> bool:
    """True for connection-level failures (also when wrapped as the cause)."""
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 4:
        if (type(cur).__name__ in _CONNECTION_ERROR_NAMES
                or isinstance(cur, ConnectionError)):
            return True
        cur = cur.__cause__
        seen += 1
    return False


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
    if _is_connection_error(exc) and getattr(exc, "status_code", None) is None:
        return "transient"
    if isinstance(exc, ProviderError):
        status = exc.status_code
        if status == 400 and is_overflow_message(str(exc)):
            return "overflow"
        if status in _TRANSIENT_STATUSES:
            return "transient"
        return "fatal"
    return "fatal"
