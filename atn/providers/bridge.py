"""Claude Max bridge provider -- runs the Claude Agent SDK via subprocess.

Spawns a TypeScript bridge process (bridge/claude-bridge.ts) that wraps the
Claude Agent SDK.  Communication uses NDJSON over stdin/stdout pipes.

Supports two request types:
  - create:      single-turn LLM call (maxTurns=1, no tools).
  - orchestrate: multi-turn with ATN tools relayed as in-process MCP server.

Config keys (in ~/.atn/config.yaml under providers.claude_max):
  model       (str)   "sonnet" | "opus" | "haiku".  Default: "sonnet".

The bridge subprocess is started lazily on first request and kept alive
for subsequent requests.  It shuts down when the provider is closed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)
from ..events import Event, EventBus, EventType

log = logging.getLogger(__name__)

# Locate the bridge directory: try importlib first (pip install), then relative path (dev)
def _find_bridge_dir() -> Path:
    try:
        import bridge as _bridge_pkg
        return Path(_bridge_pkg.__file__).resolve().parent
    except ImportError:
        pass
    return Path(__file__).resolve().parent.parent.parent / "bridge"

_BRIDGE_DIR = _find_bridge_dir()
_BRIDGE_SCRIPT = _BRIDGE_DIR / "claude-bridge.ts"


_EVENT_PREFIX = "@@EVENT@@"

# Wedge guard for the orchestrate stdout loop. The SDK can legitimately go
# a long time without stdout traffic (its built-in tools run inside the
# subprocess), but stderr events keep flowing while it works. "Wedged" is
# therefore TOTAL silence — no stdout line AND no stderr event — for this
# long. Overridable for unusual deployments.
_ORCH_IDLE_CEILING = float(os.environ.get("ATN_BRIDGE_IDLE_CEILING", "1800"))
# How often the guarded readline wakes up to re-check liveness/activity.
_ORCH_PROBE_INTERVAL = 120.0
# Timeout for stdin write flushes (pipe-deadlock guard).
_STDIN_DRAIN_TIMEOUT = 60.0
# Cap on a single tool result relayed to the SDK (~10k tokens). Oversized
# results inflate every subsequent SDK turn and can wedge the stdin pipe.
_TOOL_RESULT_CHAR_MAX = 40_000

# §11 first-event watchdog: the bridge must produce its FIRST stream event
# within this many seconds of an orchestrate request being sent. A model the
# SDK loop rejects (haiku on the orchestrate path) can otherwise hot-spin at
# 100% CPU with zero output until the 1800s idle ceiling. On timeout we kill
# the subprocess and raise so the execution fails loudly. Overridable.
_FIRST_EVENT_TIMEOUT = float(os.environ.get("ATN_BRIDGE_FIRST_EVENT_TIMEOUT", "120"))
# §11 kill ladder: grace window between a cooperative interrupt and a verified
# hard kill of the tracked bun PID. Polled, then taskkill /F (win) / SIGKILL.
_KILL_GRACE_SECONDS = float(os.environ.get("ATN_BRIDGE_KILL_GRACE", "5"))


def _model_is_orchestrator_capable(model: str) -> tuple[bool, str]:
    """§11 model-tier guard (READ-ONLY over model_specs).

    The SDK orchestrate loop is known to misbehave on non-orchestrator-capable
    models — a haiku-class model made the bun bridge hot-spin at 100% CPU with
    zero output for 20+ min. Reject orchestrate on those models with a clear
    error BEFORE spawning the SDK loop.

    Uses ``model_specs.get_model_tier`` when present (the §14 flag another
    agent may add), else falls back to the class bucket: haiku is not
    orchestrator-capable. Returns (ok, reason).
    """
    try:
        from .. import model_specs as _ms
    except Exception:
        return True, ""  # can't resolve — don't block
    # Prefer an explicit tier/capability helper if model_specs exposes one.
    getter = getattr(_ms, "get_model_tier", None)
    if callable(getter):
        try:
            tier = getter(model)
        except Exception:
            tier = None
        if isinstance(tier, str) and tier.lower() in ("haiku", "fast", "small"):
            return False, (
                f"model '{model}' (tier '{tier}') is not orchestrator-capable; "
                "the SDK orchestrate loop hot-spins on it — pick a "
                "sonnet/opus-class model or set the model explicitly"
            )
        if isinstance(tier, str):
            return True, ""
    # Fallback: class bucket. Haiku is the known-bad orchestrate tier.
    try:
        klass = _ms.model_class(model)
    except Exception:
        klass = "other"
    # Also catch the bare "haiku" family alias the bridge accepts
    # (mapModelToClaudeModel maps any string containing "haiku" → haiku),
    # which resolves to the default spec (klass "other") in the store.
    if klass == "haiku" or "haiku" in (model or "").lower():
        return False, (
            f"model '{model}' (haiku-class) is not orchestrator-capable; "
            "the SDK orchestrate loop hot-spins on it — pick a "
            "sonnet/opus-class model or set the model explicitly"
        )
    return True, ""


class BridgeProvider(Provider):
    """Claude Max provider via the TypeScript bridge subprocess.

    Implements the standard Provider interface (maxTurns=1).  Each call to
    send() creates a fresh bridge session, collects the response, and
    cleans up.
    """

    # The SDK owns the agentic loop — this provider cannot inject the v3
    # closing review turn mid-loop. The caller (execution engine / worker
    # loop) runs the review step as a follow-up turn on the same session.
    handles_review_step: bool = False

    def __init__(
        self,
        model: str = "sonnet",
        bridge_script: str | Path | None = None,
    ) -> None:
        self._model = model
        self._bridge_script = Path(bridge_script) if bridge_script else _BRIDGE_SCRIPT
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        # Bounded: a slow/stuck consumer must not let the queue grow without
        # limit in a long-lived daemon. The producer drops (with a log) when
        # full rather than blocking — blocking would stop the stderr drain
        # and stall the SDK subprocess on its next stderr write.
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)
        # Monotonic timestamp of the last line seen from the subprocess
        # (stdout or stderr). The orchestrate wedge guard keys off this.
        self._last_subprocess_activity: float = time.monotonic()

        # Optional EventBus for emitting tool use events.
        # Set by the caller (e.g. runtime) after construction.
        self.event_bus: EventBus | None = None
        self.source_agent_id: str = ""

        # §5 delivery-ack (finding 8): injected user_messages, keyed by id.
        # value = content. An entry is present from the moment send_user_message
        # writes it until the bridge acks it (user_message_consumed → removed)
        # or the orchestration ends (unconsumed → re-posted to the inbox by
        # execution_control). ``_injection_seq`` mints monotonic ids.
        self._injection_seq: int = 0
        self._pending_injections: dict[str, str] = {}
        self._first_event_timeout: float = _FIRST_EVENT_TIMEOUT
        # Armed per-orchestration; set on the first bridge event (stdout line
        # or stderr @@EVENT@@) so the watchdog can tell "wedged from the start"
        # from "busy". None between orchestrations.
        self._first_event: asyncio.Event | None = None

        # Cumulative session stats — updated after each orchestrate response.
        # These track the SDK's view of the session, not our own bookkeeping.
        self._session_id: str = ""
        self._sdk_num_turns: int = 0
        self._cumulative_turns: int = 0
        self._total_cost_usd: float = 0.0
        self._context_window: int = 0
        self._max_output_tokens: int = 0
        self._cumulative_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._cumulative_cache_read: int = 0
        self._cumulative_cache_creation: int = 0
        self._last_input_tokens: int = 0
        self._compaction_count: int = 0
        self._last_compaction_pre_tokens: int = 0

        # Claude Max subscription-quota snapshot.  Keyed by rateLimitType
        # (five_hour, seven_day, seven_day_sonnet, seven_day_opus, overage).
        # Populated from SDKRateLimitEvent stream messages; surfaced via
        # session_stats["rate_limits"].
        self._rate_limits: dict[str, dict[str, Any]] = {}

        # Per-model-class estimator for the 5h subscription window. We track
        # cumulative tokens by class (haiku / sonnet / opus / other) and pair
        # each refresh_usage() reading with a per-class breakdown of the
        # tokens consumed since the previous reading. Anthropic charges
        # different rates per class against the 5h window; the EMA per class
        # gives us a tokens-per-1%-of-window factor we can apply when
        # resolving pct_subscription_5h budgets.
        self._cum_tokens_by_class: dict[str, int] = {
            "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
        }
        self._last_snapshot: dict[str, Any] | None = None
        # Bootstrap relative-cost multipliers (Anthropic's published rough
        # ratios). Used until the EMA has at least one observation per class.
        # Normalized so sonnet=1.0 — when we see a Sonnet-only delta we can
        # set tokens_per_pct[sonnet] from data and derive others.
        self._bootstrap_relative = {
            "haiku": 5.0,    # Haiku is ~5x cheaper per token, so ~5x more tokens per 1%
            "sonnet": 1.0,
            "opus": 0.2,     # Opus is ~5x more expensive per token, so ~5x fewer tokens per 1%
            "other": 1.0,    # Treat unknown as Sonnet-equivalent.
        }
        # Bootstrap absolute rate (tokens per 1% of 5h window for Sonnet).
        # Loose — refined as soon as real data arrives. ~6M tokens/5h →
        # ~60k tokens/pct for Sonnet.
        self._bootstrap_sonnet_rate = 60_000.0
        self._tokens_per_pct_by_class: dict[str, float] = {}
        self._estimator_obs_count: dict[str, int] = {
            "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
        }
        # Reconciliation: log running discrepancy between our predicted
        # subscription-pct burn (computed turn-by-turn from the estimator)
        # and the actual delta on the 5h header. Big discrepancies trigger
        # a recalibration nudge on the most-active class for the period.
        self._predicted_pct_since_refresh: float = 0.0
        self._predicted_tokens_by_class_since_refresh: dict[str, int] = {
            "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
        }
        self._last_reconciliation: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "claude_max"

    @property
    def supports_orchestrate(self) -> bool:
        return True

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
        # Extract user message text from the last message
        user_text = ""
        if messages:
            last = messages[-1]
            content = last.get("content", "")
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                user_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

        if not user_text:
            raise ProviderError(
                "No user message text found",
                provider="claude_max",
            )

        request = {
            "type": "create",
            "message": user_text,
            "system": system,
            "model": model or self._model,
        }

        resp = await self._send_request(request)

        if not resp.get("ok"):
            raise ProviderError(
                f"Bridge error: {resp.get('error', 'unknown')}",
                provider="claude_max",
            )

        # Parse response into canonical format
        text = resp.get("text", "")
        stop_reason = resp.get("stop_reason", "end_turn")

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        for tc in resp.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                input=tc.get("input", {}),
            ))

        # Parse usage
        usage_data = resp.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage_data.get("cache_creation_input_tokens", 0),
        )

        return ProviderResponse(
            text=text,
            thinking=resp.get("thinking", []),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=resp.get("model", model or self._model),
            raw=resp,
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

        The bridge emits @@EVENT@@ lines on stderr as the SDK generates
        content.  We drain the event queue and call on_chunk for each
        text_delta and on_thinking for each thinking block, while
        concurrently waiting for the final JSON response on stdout.
        """
        # Extract user message text (same as send())
        user_text = ""
        if messages:
            last = messages[-1]
            content = last.get("content", "")
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                user_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

        if not user_text:
            raise ProviderError("No user message text found", provider="claude_max")

        request = {
            "type": "create",
            "message": user_text,
            "system": system,
            "model": model or self._model,
        }

        # Drain any stale events from previous requests
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send request and stream events concurrently
        async def _stream_events() -> None:
            """Read events from the queue and call on_chunk/on_thinking."""
            while True:
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=120)
                except asyncio.TimeoutError:
                    break
                try:
                    if event.get("type") == "done":
                        break
                    if event.get("type") == "text_delta" and on_chunk:
                        text = event.get("text", "")
                        if text:
                            await on_chunk(text)
                    elif event.get("type") == "thinking" and on_thinking:
                        text = event.get("text", "")
                        if text:
                            await on_thinking(text)
                except Exception:
                    log.exception("Error processing stream event: %s", event)

        async def _get_response() -> dict[str, Any]:
            """Send request and get the final JSON response."""
            return await self._send_request(request)

        # Run both concurrently: stream events while waiting for response
        event_task = asyncio.create_task(_stream_events())
        try:
            resp = await _get_response()
        finally:
            # Ensure event streaming finishes (response arrives after done event)
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

        if not resp.get("ok"):
            raise ProviderError(
                f"Bridge error: {resp.get('error', 'unknown')}",
                provider="claude_max",
            )

        # Parse response (same as send())
        text = resp.get("text", "")
        stop_reason = resp.get("stop_reason", "end_turn")

        tool_calls: list[ToolCall] = []
        for tc in resp.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                input=tc.get("input", {}),
            ))

        usage_data = resp.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage_data.get("cache_creation_input_tokens", 0),
        )

        return ProviderResponse(
            text=text,
            thinking=resp.get("thinking", []),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=resp.get("model", model or self._model),
            raw=resp,
        )

    # ------------------------------------------------------------------
    # Orchestrate — multi-turn with tool relay
    # ------------------------------------------------------------------

    async def send_orchestrate(
        self,
        *,
        message: str,
        system: str = "",
        model: str = "",
        tools: list[dict[str, Any]],
        max_turns: int = 20,
        tool_executor: Callable[..., Any],
        on_chunk: Callable[..., Any] | None = None,
        session_id: str = "",
        usage_recorder: Any = None,
        **kwargs,
    ) -> ProviderResponse:
        """Multi-turn orchestrator call through the bridge.

        Unlike send()/send_stream() which do one request → one response, this
        method handles a bidirectional conversation on stdout:

        1. Sends an 'orchestrate' request to the bridge.
        2. Reads stdout lines in a loop:
           - tool_call → executes via tool_executor, writes tool_result to stdin
           - Final response JSON (has 'id' field) → parse and return
        3. Concurrently drains stderr for @@EVENT@@ streaming events.

        Args:
            message:        User message text.
            system:         System prompt (passed as SDK systemPrompt for caching).
            model:          Model override (sonnet/opus/haiku).
            tools:          Tool definitions in bridge format (name, description, input_schema).
            max_turns:      Max LLM turns for the multi-turn loop.
            tool_executor:  async (name, input, runtime) -> result dict.
            on_chunk:       Optional async callback for streaming text deltas.
            session_id:     SDK session ID to resume (enables prompt caching).
        """
        # This is a long-running operation that interleaves reads and writes.
        # It CANNOT use _send_request/_send_raw (which assume one-shot).
        # We acquire the lock to prevent concurrent bridge use, but hold it
        # for the entire orchestration session.
        # §11 model-tier guard: reject non-orchestrator-capable models (haiku)
        # BEFORE acquiring the lock or spawning the SDK loop, since the loop
        # hot-spins on them. Cheap, pure model check.
        effective_model = model or self._model
        _ok, _reason = _model_is_orchestrator_capable(effective_model)
        if not _ok:
            raise ProviderError(_reason, provider="claude_max")

        async with self._lock:
            await self._ensure_process()
            assert self._process and self._process.stdin and self._process.stdout

            self._request_id += 1
            request_id = f"req-{self._request_id}"

            if model:
                self._model = model  # Keep session_stats in sync
            request: dict[str, Any] = {
                "id": request_id,
                "type": "orchestrate",
                "message": message,
                "system_prompt": system,
                "model": effective_model,
                "tools": tools,
                "max_turns": max_turns,
                "native_tools": kwargs.get("native_tools", False),
            }
            _allowlist = kwargs.get("native_tool_allowlist")
            if _allowlist:
                request["native_tool_allowlist"] = list(_allowlist)
            _thinking = kwargs.get("thinking")
            if _thinking is not None:
                request["thinking"] = _thinking
            _budget = kwargs.get("thinking_budget_tokens")
            if _budget is not None:
                request["thinking_budget_tokens"] = int(_budget)
            if session_id:
                request["session_id"] = session_id

            # Drain stale events
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Write the orchestrate request
            line = json.dumps(request) + "\n"
            try:
                self._process.stdin.write(line.encode())
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                raise ProviderError(
                    f"Bridge stdin broken during orchestrate: {exc}",
                    provider="claude_max",
                ) from exc
            # Reset the wedge clock: the process may have sat idle since the
            # previous orchestration, and a stale timestamp would get a
            # healthy-but-thinking bridge killed on the first probe.
            self._last_subprocess_activity = time.monotonic()

            # §11 first-event watchdog: arm a signal that the stderr drain and
            # the stdout loop set on the FIRST piece of bridge traffic. If it
            # stays unset past _first_event_timeout the SDK loop is wedged
            # (haiku hot-spin) — kill the subprocess and fail loudly instead of
            # waiting out the 1800s idle ceiling.
            self._first_event = asyncio.Event()
            watchdog_task = asyncio.create_task(self._first_event_watchdog())

            log.info("Orchestrate request sent (tools=%d, max_turns=%d)", len(tools), max_turns)

            # Shared state between the stream-events task and the outer
            # orchestrate loop. budget_blocker carries the agent_id whose cap
            # was crossed; if set, we treat the orchestration outcome as
            # budget-exceeded rather than a normal end_turn.
            budget_state = {"blocker": None}  # type: dict[str, Any]

            from .base import _call_recorder, _maybe_await

            # Start streaming events to on_chunk concurrently
            stream_task: asyncio.Task | None = None

            async def _stream_events() -> None:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            self._event_queue.get(), timeout=300,
                        )
                    except asyncio.TimeoutError:
                        break
                    try:
                        if event.get("type") == "done":
                            break
                        if event.get("type") == "text_delta":
                            text = event.get("text", "")
                            if text and on_chunk:
                                await on_chunk(text)
                        elif event.get("type") == "tool_use_start" and self.event_bus:
                            tool_name = event.get("tool_name", "")
                            tool_input = event.get("input", {})
                            log.debug("AGENT_TOOL_USE_START: tool=%s agent=%s",
                                      tool_name, self.source_agent_id)
                            await self.event_bus.emit(Event(
                                type=EventType.AGENT_TOOL_USE_START,
                                source=self.source_agent_id,
                                data={
                                    "agent_id": self.source_agent_id,
                                    "tool_use_id": event.get("tool_use_id", ""),
                                    "tool_name": tool_name,
                                    "input": tool_input,
                                },
                            ))
                            # Also emit STEP_OUTPUT with channel=tool_call
                            # so the voice service can play tones and narrate.
                            await self.event_bus.emit(Event(
                                type=EventType.STEP_OUTPUT,
                                source=self.source_agent_id,
                                data={
                                    "agent_id": self.source_agent_id,
                                    "channel": "tool_call",
                                    "tool_name": tool_name,
                                    "tool_input": tool_input,
                                },
                            ))
                        elif event.get("type") == "thinking" and self.event_bus:
                            text = event.get("text", "")
                            if text:
                                await self.event_bus.emit(Event(
                                    type=EventType.STEP_OUTPUT,
                                    source=self.source_agent_id,
                                    data={
                                        "agent_id": self.source_agent_id,
                                        "channel": "thinking",
                                        "content": text[:2000],
                                    },
                                ))
                        elif event.get("type") == "tool_use_result":
                            # For ATN MCP-relayed tools, the enriched result is emitted
                            # from the Python tool executor below.  For SDK built-in tools
                            # (Read, Write, Bash, etc.), we emit from the stderr event.
                            tool_use_id = event.get("tool_use_id", "")
                            if tool_use_id and self.event_bus:
                                await self.event_bus.emit(Event(
                                    type=EventType.AGENT_TOOL_USE_RESULT,
                                    source=self.source_agent_id,
                                    data={
                                        "agent_id": self.source_agent_id,
                                        "tool_use_id": tool_use_id,
                                        "tool_name": event.get("tool_name", ""),
                                        "is_error": event.get("is_error", False),
                                        "result_preview": event.get("result_preview", ""),
                                    },
                                ))
                        elif event.get("type") == "compaction":
                            self._compaction_count += 1
                            self._last_compaction_pre_tokens = event.get("pre_tokens", 0)
                            log.info(
                                "Context compaction #%d (trigger=%s, pre_tokens=%d)",
                                self._compaction_count,
                                event.get("trigger", "auto"),
                                self._last_compaction_pre_tokens,
                            )
                            if self.event_bus:
                                await self.event_bus.emit(Event(
                                    type=EventType.CONTEXT_COMPACTION,
                                    source=self.source_agent_id,
                                    data={
                                        "agent_id": self.source_agent_id,
                                        "compaction_count": self._compaction_count,
                                        "trigger": event.get("trigger", "auto"),
                                        "pre_tokens": self._last_compaction_pre_tokens,
                                        "status": "completed",
                                    },
                                ))
                        elif event.get("type") == "status":
                            status = event.get("status", "idle")
                            if status == "compacting" and self.event_bus:
                                await self.event_bus.emit(Event(
                                    type=EventType.CONTEXT_COMPACTION,
                                    source=self.source_agent_id,
                                    data={
                                        "agent_id": self.source_agent_id,
                                        "compaction_count": self._compaction_count,
                                        "status": "in_progress",
                                    },
                                ))
                        elif event.get("type") == "usage":
                            # Per-turn usage from the TS bridge. Roll into the
                            # cascading budget; if the recorder vetoes, signal
                            # the SDK to stop via interrupt().
                            _u_in = int(event.get("input_tokens", 0))
                            _u_out = int(event.get("output_tokens", 0))
                            _u_cr = int(event.get("cache_read_input_tokens", 0))
                            _u_cc = int(event.get("cache_creation_input_tokens", 0))
                            # Context size (all buckets) — for the context-%
                            # estimator only.
                            turn_total = _u_in + _u_out + _u_cr + _u_cc
                            # Real new tokens — for budget enforcement. Exclude
                            # cache_read: it's the cached prefix re-sent every
                            # turn (bills ~10%, no new work) and full-weighting
                            # it false-fails sane caps. Mirrors
                            # Usage.budget_tokens().
                            turn_spend = _u_in + _u_cc + _u_out
                            turn_model = event.get("model", "")
                            if turn_total > 0:
                                # Update per-class cumulative counter for the estimator.
                                self.record_turn_tokens(turn_total, turn_model)
                                if usage_recorder is not None:
                                    try:
                                        ok, blocker = await _maybe_await(
                                            _call_recorder(
                                                usage_recorder, turn_spend, turn_model
                                            )
                                        )
                                        if not ok:
                                            budget_state["blocker"] = blocker
                                            log.warning(
                                                "Bridge inner-loop budget exceeded "
                                                "(blocker=%s) — interrupting SDK",
                                                blocker,
                                            )
                                            await self.interrupt()
                                    except Exception:
                                        log.exception("usage_recorder failed; continuing")
                    except Exception:
                        log.exception("Error processing stream event: %s", event)

            stream_task = asyncio.create_task(_stream_events())

            # Read stdout lines: may be tool_call or final response
            try:
                final_resp: dict[str, Any] | None = None

                while True:
                    raw = await self._read_stdout_guarded()
                    if not raw:
                        raise ProviderError(
                            "Bridge stdout closed during orchestrate",
                            provider="claude_max",
                        )
                    self._last_subprocess_activity = time.monotonic()
                    if self._first_event is not None:
                        self._first_event.set()

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("bridge: non-JSON stdout line: %s", raw[:200])
                        continue

                    # Tool call relay: execute and send result back
                    if msg.get("type") == "tool_call":
                        call_id = msg["call_id"]
                        tool_name = msg["name"]
                        tool_input = msg.get("input", {})
                        log.info("Orchestrate tool_call: %s (call_id=%s)", tool_name, call_id)

                        is_error = False
                        try:
                            result = await tool_executor(tool_name, tool_input)
                        except Exception as exc:
                            log.exception("Tool execution error: %s", tool_name)
                            result = {"error": f"Tool '{tool_name}' failed: {exc}"}
                            is_error = True

                        # Emit enriched tool result event with preview
                        if self.event_bus:
                            result_str = str(result)
                            await self.event_bus.emit(Event(
                                type=EventType.AGENT_TOOL_USE_RESULT,
                                source=self.source_agent_id,
                                data={
                                    "agent_id": self.source_agent_id,
                                    "tool_use_id": call_id,
                                    "tool_name": tool_name,
                                    "is_error": is_error,
                                    "result_preview": result_str[:500],
                                },
                            ))

                        # Send tool_result back to bridge. Cap oversized
                        # results: they inflate every subsequent SDK turn,
                        # and a multi-MB line can wedge the stdin pipe
                        # against an SDK that is mid-write on stdout.
                        result_payload: Any = result
                        _probe = json.dumps(result, default=str)
                        if len(_probe) > _TOOL_RESULT_CHAR_MAX:
                            result_payload = (
                                _probe[:_TOOL_RESULT_CHAR_MAX]
                                + f"... [truncated: result was {len(_probe)} chars; "
                                "re-invoke the tool with a narrower query if more is needed]"
                            )
                        result_msg = json.dumps({
                            "type": "tool_result",
                            "call_id": call_id,
                            "result": result_payload,
                        }, default=str) + "\n"
                        try:
                            self._process.stdin.write(result_msg.encode())
                            await asyncio.wait_for(
                                self._process.stdin.drain(),
                                timeout=_STDIN_DRAIN_TIMEOUT,
                            )
                        except asyncio.TimeoutError as exc:
                            raise ProviderError(
                                f"Bridge stdin flush stalled >{int(_STDIN_DRAIN_TIMEOUT)}s "
                                "sending tool_result (pipe deadlock?)",
                                provider="claude_max",
                            ) from exc
                        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                            raise ProviderError(
                                f"Bridge stdin broken sending tool_result: {exc}",
                                provider="claude_max",
                            ) from exc
                        # Our own write counts as activity — a tool that ran
                        # longer than the idle ceiling must not leave a stale
                        # timestamp that gets a healthy bridge killed.
                        self._last_subprocess_activity = time.monotonic()
                        continue

                    # Final response (has 'id' field matching our request)
                    if "id" in msg:
                        final_resp = msg
                        break

                    log.debug("bridge: unexpected stdout message: %s", str(msg)[:200])

            finally:
                if stream_task:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
                self._first_event = None
                # Re-post any injection the bridge never consumed to the inbox
                # so a delegate_message that raced turn completion is not lost.
                # execution_control reads/clears _pending_injections on the
                # delegate_message path; here we just log a warning for any
                # that remain (the caller-side fallback owns the re-post).
                if self._pending_injections:
                    log.warning(
                        "orchestration ended with %d unconsumed injection(s): %s",
                        len(self._pending_injections),
                        list(self._pending_injections.keys()),
                    )

            if final_resp is None:
                raise ProviderError(
                    "No final response from bridge orchestrate",
                    provider="claude_max",
                )

            if not final_resp.get("ok"):
                raise ProviderError(
                    f"Bridge orchestrate error: {final_resp.get('error', 'unknown')}",
                    provider="claude_max",
                )

            # Parse into ProviderResponse
            usage_data = final_resp.get("usage", {})
            usage = Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage_data.get("cache_creation_input_tokens", 0),
            )

            # Update cumulative session stats from bridge context data
            ctx = final_resp.get("context", {})
            if ctx:
                self._sdk_num_turns = ctx.get("num_turns", self._sdk_num_turns)
                self._cumulative_turns += 1
                self._total_cost_usd = ctx.get("total_cost_usd", self._total_cost_usd)
                self._context_window = ctx.get("context_window", self._context_window)
                self._max_output_tokens = ctx.get("max_output_tokens", self._max_output_tokens)
            if final_resp.get("session_id"):
                self._session_id = final_resp["session_id"]
            # Context occupancy: prefer per-response usage from the last assistant
            # message (accurate for multi-round tool-use turns).  Fall back to
            # the aggregate modelUsage sum if the bridge didn't report it.
            last_round = usage_data.get("last_round_input_tokens", 0)
            if last_round > 0:
                self._last_input_tokens = last_round
            else:
                self._last_input_tokens = (
                    usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
                )
            self._cumulative_input_tokens += usage.input_tokens
            self._cumulative_output_tokens += usage.output_tokens
            self._cumulative_cache_read += usage.cache_read_tokens
            self._cumulative_cache_creation += usage.cache_creation_tokens

            # Emit per-turn usage event for live frontend display
            if self.event_bus:
                await self.event_bus.emit(Event(
                    type=EventType.STEP_OUTPUT,
                    source=self.source_agent_id,
                    data={
                        "agent_id": self.source_agent_id,
                        "channel": "usage",
                        "usage": {
                            # Total context = uncached + cache_read + cache_creation
                            "input_tokens": self._last_input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_read_tokens": usage.cache_read_tokens,
                            "cache_creation_tokens": usage.cache_creation_tokens,
                        },
                        "cumulative": {
                            "input_tokens": self._cumulative_input_tokens,
                            "output_tokens": self._cumulative_output_tokens,
                            "cache_read_tokens": self._cumulative_cache_read,
                            "cache_creation_tokens": self._cumulative_cache_creation,
                            "total_cost_usd": self._total_cost_usd,
                        },
                    },
                ))

            # Inner-loop budget veto wins over the bridge's own stop_reason —
            # the bridge sees an SDK interrupt and reports interrupted/end_turn,
            # but the cause was a budget cap. Surface it explicitly.
            stop_reason = final_resp.get("stop_reason", "end_turn")
            if budget_state["blocker"]:
                stop_reason = "budget_exceeded"
                blocker = budget_state["blocker"]
                bridge_text = final_resp.get("text", "")
                final_text = (
                    f"Aborted: bridge orchestration budget exceeded "
                    f"(blocked by '{blocker}'). Adjust the budget on that "
                    f"agent or wait for the next period."
                )
                return ProviderResponse(
                    text=bridge_text or final_text,
                    thinking=final_resp.get("thinking", []),
                    tool_calls=[],
                    stop_reason=stop_reason,
                    usage=usage,
                    model=final_resp.get("model", model or self._model),
                    raw=final_resp,
                )

            return ProviderResponse(
                text=final_resp.get("text", ""),
                thinking=final_resp.get("thinking", []),
                tool_calls=[],  # Tools are handled inside the bridge
                stop_reason=stop_reason,
                usage=usage,
                model=final_resp.get("model", model or self._model),
                raw=final_resp,
            )

    # ------------------------------------------------------------------
    # Mid-session control — bypass the lock (used during orchestrate)
    # ------------------------------------------------------------------

    async def interrupt(self) -> None:
        """Interrupt the active orchestration session.

        Sends an interrupt message to the bridge, which calls q.interrupt()
        on the running SDK query.  The query winds down gracefully and emits
        its final response.  Safe to call while send_orchestrate() is running.

        Does NOT acquire the lock — the bridge's readline stdin parser
        handles interleaved single-line writes correctly.
        """
        if not self._process or self._process.returncode is not None:
            return
        assert self._process.stdin
        try:
            line = json.dumps({"type": "interrupt"}) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            log.info("Interrupt sent to bridge")
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.warning("Failed to send interrupt (bridge stdin broken)")

    async def refresh_usage(self) -> dict[str, Any]:
        """Fetch subscription-window utilization for Claude Max.

        Anthropic's `/v1/messages` endpoint returns `anthropic-ratelimit-unified-*`
        response headers on every successful call. These headers carry the same
        five-hour / seven-day utilization that Claude Code's `/usage` slash
        command displays. The SDK's stream-event API does NOT emit these
        consistently, so we make a minimal API call ourselves and read the
        headers directly.

        Stores the parsed result on ``self._rate_limits`` and returns it.
        Reads the OAuth token from ``~/.claude/.credentials.json``.
        """
        import time
        from pathlib import Path
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed — cannot refresh usage")
            return dict(self._rate_limits)

        creds_path = Path.home() / ".claude" / ".credentials.json"
        if not creds_path.exists():
            log.warning("Claude credentials file not found at %s", creds_path)
            return dict(self._rate_limits)
        try:
            creds = json.loads(creds_path.read_text(encoding="utf-8"))
            token = creds.get("claudeAiOauth", {}).get("accessToken", "")
        except Exception as exc:
            log.warning("Failed to read Claude credentials: %s", exc)
            return dict(self._rate_limits)
        if not token:
            log.warning("No OAuth token found in credentials")
            return dict(self._rate_limits)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "oauth-2025-04-20",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        except Exception as exc:
            log.warning("Usage probe request failed: %s", exc)
            return dict(self._rate_limits)

        h = resp.headers
        # Parse the anthropic-ratelimit-unified-* headers into our existing
        # _rate_limits shape (keyed by rateLimitType).
        def _parse_window(prefix: str, key: str) -> dict[str, Any] | None:
            status = h.get(f"anthropic-ratelimit-unified-{prefix}-status")
            util = h.get(f"anthropic-ratelimit-unified-{prefix}-utilization")
            reset = h.get(f"anthropic-ratelimit-unified-{prefix}-reset")
            if status is None and util is None and reset is None:
                return None
            return {
                "status": status or "unknown",
                "rateLimitType": key,
                "utilization": float(util) if util is not None else None,
                "resetsAt": int(reset) if reset is not None else None,
                "updatedAt": int(time.time() * 1000),
            }

        windows = {
            "five_hour": _parse_window("5h", "five_hour"),
            "seven_day": _parse_window("7d", "seven_day"),
            "overage": _parse_window("overage", "overage"),
        }
        for key, entry in windows.items():
            if entry is not None:
                self._rate_limits[key] = entry
        log.info(
            "refresh_usage: 5h=%s, 7d=%s, overage=%s",
            windows.get("five_hour", {}).get("utilization") if windows.get("five_hour") else "?",
            windows.get("seven_day", {}).get("utilization") if windows.get("seven_day") else "?",
            windows.get("overage", {}).get("utilization") if windows.get("overage") else "?",
        )

        # Estimator snapshot: pair the 5h utilization with the cumulative-tokens
        # counter. Two consecutive non-reset snapshots give us a rolling
        # tokens-per-percent factor.
        five_h = self._rate_limits.get("five_hour")
        if five_h and five_h.get("utilization") is not None:
            self._record_usage_snapshot(float(five_h["utilization"]))
        return dict(self._rate_limits)

    def _record_usage_snapshot(self, utilization: float) -> None:
        """Pair the 5h utilization reading with the per-class cumulative-tokens
        counters. Diffs across successive snapshots refine the per-class
        tokens-per-percent rates via an EMA.

        Anthropic only exposes one aggregate 5h utilization number, so when
        the snapshot delta straddles multiple model classes we allocate the
        utilization delta proportionally using the bootstrap multipliers (or,
        once the EMA has data, the current rates themselves).

        utilization is the fraction (0.0..1.0) of the 5h window consumed.
        """
        snap = {
            "utilization": utilization,
            "tokens_by_class": dict(self._cum_tokens_by_class),
        }
        prev = self._last_snapshot
        if prev is not None:
            util_delta = utilization - prev["utilization"]
            class_deltas = {
                k: self._cum_tokens_by_class[k] - prev["tokens_by_class"].get(k, 0)
                for k in self._cum_tokens_by_class
            }
            total_delta = sum(class_deltas.values())
            if util_delta > 0.005 and total_delta > 0:
                # Allocate the util_delta across classes by their relative
                # cost (tokens-per-pct ratio). Convention: lower
                # tokens_per_pct = more expensive per pct = more pct burned
                # for the same token count. So each class contributes:
                #   pct_contributed = (tokens_in_class / rate_for_class)
                # and we scale the contributions to match the observed util_delta.
                per_class_rate = self._effective_rates()
                contributions: dict[str, float] = {}
                for cls, tokens in class_deltas.items():
                    if tokens <= 0:
                        contributions[cls] = 0.0
                    else:
                        rate = per_class_rate.get(cls, self._bootstrap_sonnet_rate)
                        contributions[cls] = tokens / rate  # in pct units
                contrib_sum = sum(contributions.values())
                if contrib_sum > 0:
                    # Scale so the contributions sum equals the observed util_delta * 100.
                    scale = (util_delta * 100.0) / contrib_sum
                    # Each class's actual rate this period = tokens / (contributed_pct * scale)
                    for cls, tokens in class_deltas.items():
                        if tokens > 0 and contributions[cls] > 0:
                            actual_pct = contributions[cls] * scale
                            if actual_pct > 0.05:  # skip near-zero attributions
                                new_rate = tokens / actual_pct
                                self._update_class_rate(cls, new_rate)
            elif util_delta < -0.05:
                # 5h window rolled over — wipe history so we start fresh.
                self._last_snapshot = snap
                return
        self._last_snapshot = snap

    def _effective_rates(self) -> dict[str, float]:
        """Return the rates we'd use right now: EMA values where present,
        bootstrap derivations elsewhere. Always returns a value per class.
        """
        out: dict[str, float] = {}
        sonnet_rate = self._tokens_per_pct_by_class.get(
            "sonnet", self._bootstrap_sonnet_rate
        )
        for cls in ("haiku", "sonnet", "opus", "other"):
            if cls in self._tokens_per_pct_by_class:
                out[cls] = self._tokens_per_pct_by_class[cls]
            else:
                out[cls] = sonnet_rate * self._bootstrap_relative[cls]
        return out

    def _update_class_rate(self, cls: str, new_rate: float) -> None:
        """EMA update for the rate of a given class."""
        if new_rate <= 0:
            return
        prior = self._tokens_per_pct_by_class.get(cls)
        if prior is None:
            self._tokens_per_pct_by_class[cls] = new_rate
        else:
            self._tokens_per_pct_by_class[cls] = 0.7 * prior + 0.3 * new_rate
        self._estimator_obs_count[cls] = self._estimator_obs_count.get(cls, 0) + 1

    def record_turn_tokens(self, tokens: int, model: str) -> None:
        """Account a turn's tokens against the cumulative-by-class counter.

        Called by the bridge stream-event handler on each per-turn `usage`
        event so the estimator's snapshot has accurate per-class deltas.
        Also accumulates a predicted pct burn for reconciliation.
        """
        if tokens <= 0:
            return
        from .base import classify_model
        cls = classify_model(model)
        self._cum_tokens_by_class[cls] = self._cum_tokens_by_class.get(cls, 0) + tokens
        # Predicted pct burn at our current rate estimate (or bootstrap).
        rates = self._effective_rates()
        rate = rates.get(cls)
        if rate and rate > 0:
            self._predicted_pct_since_refresh += tokens / rate
        self._predicted_tokens_by_class_since_refresh[cls] = (
            self._predicted_tokens_by_class_since_refresh.get(cls, 0) + tokens
        )

    async def reconcile_after_orchestration(self) -> dict[str, Any]:
        """Reconcile predicted vs actual subscription burn after an orchestration.

        Refreshes subscription headers (which gives us the *actual* util_5h
        delta from Anthropic), compares to ``_predicted_pct_since_refresh``
        which we accumulated turn-by-turn at our current rate estimate.

        If predicted is too high → our rates are too low (we're undercharging
        per token; need higher tokens/pct... wait, that's reversed).
        Predicted_pct = tokens / rate. So if predicted > actual, we
        overestimated burn for given tokens, meaning rate is too low (we
        thought each token burned more pct than it really did) → bump rate up.
        If predicted < actual, we underestimated, rate too high → bump down.

        Adjustment is applied to the class that consumed the most tokens in
        this orchestration. Conservative: 30% of the discrepancy, blended via
        EMA so a single odd reading can't whipsaw the estimator.
        """
        prev_5h = (self._rate_limits.get("five_hour") or {}).get("utilization")
        await self.refresh_usage()  # populates self._rate_limits and runs snapshot
        new_5h = (self._rate_limits.get("five_hour") or {}).get("utilization")

        actual_pct_delta = None
        if prev_5h is not None and new_5h is not None:
            actual_pct_delta = (new_5h - prev_5h) * 100.0

        predicted_pct = self._predicted_pct_since_refresh
        report: dict[str, Any] = {
            "predicted_pct": predicted_pct,
            "actual_pct": actual_pct_delta,
            "tokens_by_class": dict(self._predicted_tokens_by_class_since_refresh),
            "adjusted_class": None,
            "rate_before": None,
            "rate_after": None,
        }

        if (
            actual_pct_delta is not None
            and actual_pct_delta > 0.5
            and predicted_pct > 0
        ):
            ratio = actual_pct_delta / predicted_pct
            if abs(ratio - 1.0) > 0.10:
                # Pick the class with the most tokens this period.
                tokens_by_class = self._predicted_tokens_by_class_since_refresh
                dominant = max(tokens_by_class, key=lambda k: tokens_by_class[k])
                if tokens_by_class.get(dominant, 0) > 0:
                    rate_before = self._effective_rates().get(dominant, 0)
                    # actual > predicted → ratio > 1 → our rate was too high
                    # (we thought it took more tokens to burn 1pp than it did).
                    # New rate = old rate / ratio.
                    target_rate = rate_before / ratio
                    # 30% blend via EMA.
                    adjusted = 0.7 * rate_before + 0.3 * target_rate
                    self._tokens_per_pct_by_class[dominant] = adjusted
                    self._estimator_obs_count[dominant] = (
                        self._estimator_obs_count.get(dominant, 0) + 1
                    )
                    report["adjusted_class"] = dominant
                    report["rate_before"] = rate_before
                    report["rate_after"] = adjusted
                    log.info(
                        "Reconciliation: %s rate %.1f → %.1f (predicted %.2fpp, "
                        "actual %.2fpp, ratio %.3f)",
                        dominant, rate_before, adjusted,
                        predicted_pct, actual_pct_delta, ratio,
                    )

        # Reset per-orchestration counters.
        self._predicted_pct_since_refresh = 0.0
        self._predicted_tokens_by_class_since_refresh = {
            k: 0 for k in self._predicted_tokens_by_class_since_refresh
        }
        self._last_reconciliation = report
        return report

    @property
    def tokens_per_pct_by_class(self) -> dict[str, float]:
        """Effective tokens-per-1%-of-5h rate for each model class.

        Includes bootstrap fallbacks for classes that haven't been observed
        yet, so callers can always look up a rate. Use ``estimator_obs_count``
        to gauge confidence.
        """
        return self._effective_rates()

    def tokens_per_pct_for_model(self, model_id: str) -> float:
        """Resolve a tokens-per-pct rate for a specific model id.

        Routes through the model store to find the class, then returns that
        class's effective rate. All versions in a class share a rate because
        Anthropic doesn't break out subscription utilization by version.
        """
        from ..model_specs import resolve as _resolve_model
        spec = _resolve_model(model_id)
        rates = self._effective_rates()
        return rates.get(spec.klass, self._bootstrap_sonnet_rate)

    def tokens_per_pct_by_model(self, model_ids: list[str]) -> dict[str, float]:
        """Bulk version: rate per model id for the supplied list."""
        return {m: self.tokens_per_pct_for_model(m) for m in model_ids}

    @property
    def tokens_per_percent(self) -> float | None:
        """Backward-compat single rate (Sonnet-equivalent).

        Returns the Sonnet rate from the EMA if observed, else None.
        """
        return self._tokens_per_pct_by_class.get("sonnet")

    async def send_user_message(self, content: str) -> str | None:
        """Inject a user message into the active orchestration session.

        Assigns a monotonic injection id, records it in ``_pending_injections``,
        and writes it to the bridge tagged with that id. The bridge acks with
        ``user_message_consumed`` the moment the SDK loop dequeues it (which
        clears the pending entry), or ``user_message_dropped`` if it raced turn
        completion (leaving it pending for the inbox fallback).

        Returns the injection id on a successful write (so the caller can check
        whether it was consumed via ``injection_consumed``), or None if there
        was no active process / the write failed.

        Does NOT acquire the lock — same rationale as interrupt().
        """
        if not self._process or self._process.returncode is not None:
            log.warning("send_user_message: no active bridge process")
            return None
        assert self._process.stdin
        self._injection_seq += 1
        inj_id = f"inj-{self._injection_seq}"
        # Track BEFORE writing so a consumed-ack that races back can't find the
        # entry missing.
        self._pending_injections[inj_id] = content
        try:
            line = json.dumps({
                "type": "user_message", "content": content, "id": inj_id,
            }) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            log.info("User message injected (%d chars, id=%s)", len(content), inj_id)
            return inj_id
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.warning("Failed to send user_message (bridge stdin broken)")
            self._pending_injections.pop(inj_id, None)
            return None

    def injection_consumed(self, inj_id: str) -> bool:
        """True once the bridge has acked ``inj_id`` as consumed by the SDK loop
        (i.e. it is no longer pending). An unknown id reads as consumed."""
        return inj_id not in self._pending_injections

    def take_unconsumed_injection(self, inj_id: str) -> str | None:
        """Pop and return the content of an injection that was never consumed,
        so the caller can re-post it to the inbox (§5 fallback). Returns None if
        the injection was already consumed (or unknown)."""
        return self._pending_injections.pop(inj_id, None)

    # ------------------------------------------------------------------
    # Session context inspection
    # ------------------------------------------------------------------

    @property
    def session_stats(self) -> dict[str, Any]:
        """Return cumulative session stats for the current bridge session.

        These are updated after each orchestrate response.  Includes token
        counts, context window size, and a utilisation percentage that tells
        the UI how full the context is.
        """
        pct: float | None = None
        if self._context_window > 0 and self._last_input_tokens > 0:
            pct = round(100 * self._last_input_tokens / self._context_window, 1)

        return {
            "session_id": self._session_id,
            "active_model": self._model,
            "num_turns": self._cumulative_turns,
            "total_cost_usd": self._total_cost_usd,
            "context_window": self._context_window,
            "max_output_tokens": self._max_output_tokens,
            "last_input_tokens": self._last_input_tokens,
            "cumulative_input_tokens": self._cumulative_input_tokens,
            "cumulative_output_tokens": self._cumulative_output_tokens,
            "cumulative_cache_read": self._cumulative_cache_read,
            "cumulative_cache_creation": self._cumulative_cache_creation,
            "context_used_pct": pct,
            "compaction_count": self._compaction_count,
            "rate_limits": dict(self._rate_limits),
        }

    async def get_session_context(self) -> dict[str, Any]:
        """Fetch the real conversation from the bridge via getSessionMessages().

        Returns the messages the SDK is working with — the source of truth
        for what the model sees.  Includes session stats alongside messages.

        Returns {"messages": [...], "stats": {...}} on success,
        or {"error": "..."} on failure.
        """
        if not self._session_id:
            return {"error": "No session ID available (no orchestration has completed yet)"}

        try:
            resp = await self._send_request({
                "type": "session_context",
                "session_id": self._session_id,
            })
        except ProviderError as exc:
            return {"error": str(exc)}

        if not resp.get("ok"):
            return {"error": resp.get("error", "Unknown bridge error")}

        return {
            "messages": resp.get("messages", []),
            "stats": {
                **self.session_stats,
                "total_messages": resp.get("stats", {}).get("total_messages", 0),
            },
        }

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    async def _ensure_process(self) -> None:
        """Spawn bridge subprocess if not running."""
        if self._process and self._process.returncode is None:
            return

        if not self._bridge_script.exists():
            raise ProviderError(
                f"Bridge script not found: {self._bridge_script}",
                provider="claude_max",
            )

        # Check that node_modules exist
        node_modules = self._bridge_script.parent / "node_modules"
        if not node_modules.exists():
            raise ProviderError(
                f"Bridge dependencies not installed. "
                f"Run: cd {self._bridge_script.parent} && bun install  (or npm install)",
                provider="claude_max",
            )

        # Fresh queue for new process (bounded — see __init__)
        self._event_queue = asyncio.Queue(maxsize=10_000)

        # Resolve JS runtime: prefer bun, fall back to node
        import shutil
        runtime = shutil.which("bun")
        runtime_args = ["run", str(self._bridge_script)]
        if not runtime:
            runtime = shutil.which("node")
            if runtime:
                # node needs the compiled .ts → use tsx or run .ts directly
                # bun handles .ts natively; for node we need tsx or pre-compiled JS
                tsx = shutil.which("tsx") or shutil.which("npx")
                if tsx and "npx" in str(tsx):
                    runtime_args = ["tsx", str(self._bridge_script)]
                elif tsx:
                    runtime = tsx
                    runtime_args = [str(self._bridge_script)]
                else:
                    runtime_args = ["--import", "tsx", str(self._bridge_script)]
        if not runtime:
            raise ProviderError(
                "Neither bun nor node found. Install bun: https://bun.sh/docs/installation",
                provider="claude_max",
            )

        log.info("Spawning bridge: %s %s", runtime, " ".join(runtime_args))

        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        env = {
            **os.environ,
            "ENABLE_TOOL_SEARCH": "1",
            # Extend prompt cache TTL from 5 min default to 1 hour (sliding
            # window — refreshes on every cache hit).  Lets heartbeat agents
            # with intervals < 1h keep their cached prefix warm across wakes,
            # paying ~10% of input price instead of full price on rehydration.
            "ENABLE_PROMPT_CACHING_1H": "1",
        }
        # Clear CLAUDECODE to avoid nesting guard in the SDK
        env.pop("CLAUDECODE", None)

        self._process = await asyncio.create_subprocess_exec(
            runtime, *runtime_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            **kwargs,
        )

        # Background task to drain stderr (debug logs)
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Wait for the "bridge ready" log line on stderr
        # (the bridge logs it immediately on startup)
        # Give it a generous timeout for first cold start
        log.info("Bridge process started (pid=%s)", self._process.pid)

    async def _read_stdout_guarded(self) -> bytes:
        """``stdout.readline()`` with a wedge guard.

        Waits in short slices so it can notice (a) the subprocess exiting
        without closing stdout, and (b) total silence — no stdout line and
        no stderr event — exceeding ``_ORCH_IDLE_CEILING``. Long quiet
        stretches on stdout alone are NORMAL (the SDK's built-in tools run
        in-process and only stderr events flow), so silence is measured
        against ``_last_subprocess_activity``, which both pipes update.

        On a wedge: kill the subprocess (the next call respawns it) and
        raise ProviderError instead of blocking the agent — and the
        provider lock — forever.
        """
        assert self._process and self._process.stdout
        while True:
            try:
                return await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=_ORCH_PROBE_INTERVAL,
                )
            except asyncio.TimeoutError:
                if self._process.returncode is not None:
                    raise ProviderError(
                        f"Bridge process exited ({self._process.returncode}) "
                        "during orchestrate",
                        provider="claude_max",
                    )
                silent_for = time.monotonic() - self._last_subprocess_activity
                if silent_for < _ORCH_IDLE_CEILING:
                    continue        # stderr is still flowing — SDK is busy, not wedged
                log.error(
                    "Bridge wedged: no stdout/stderr activity for %.0fs — killing "
                    "subprocess (pid=%s)", silent_for, self._process.pid,
                )
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
                raise ProviderError(
                    f"Bridge unresponsive (no activity for {int(silent_for)}s) — "
                    "subprocess killed; it will respawn on the next request",
                    provider="claude_max",
                )

    async def _first_event_watchdog(self) -> None:
        """§11: kill the subprocess if the FIRST bridge event never arrives.

        Armed on each orchestrate request. If ``_first_event`` is not set
        within ``_first_event_timeout`` the SDK loop is wedged from the start
        (the haiku hot-spin) — hard-kill the tracked PID so the blocked
        ``_read_stdout_guarded`` sees the process exit and raises, failing the
        execution loudly instead of waiting out the 1800s idle ceiling.
        """
        ev = self._first_event
        if ev is None:
            return
        try:
            await asyncio.wait_for(ev.wait(), timeout=self._first_event_timeout)
            return  # first event arrived — healthy start
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            return
        proc = self._process
        pid = proc.pid if proc else None
        log.error(
            "Bridge first-event watchdog fired: no event within %.0fs "
            "(pid=%s) — killing subprocess (wedged SDK loop)",
            self._first_event_timeout, pid,
        )
        await self.terminate_subprocess(reason="first_event_timeout")

    async def terminate_subprocess(self, reason: str = "") -> bool:
        """§11 kill ladder — verified hard kill of the tracked bun PID.

        A cooperative interrupt cannot stop a spinning/wedged SDK loop, so a
        "killed" agent must not leave the subprocess running. Try graceful
        ``kill()`` (SIGTERM/terminate), poll for exit within
        ``_KILL_GRACE_SECONDS``, then escalate to a hard kill of the OS PID
        (``taskkill /F`` on Windows, SIGKILL elsewhere). Returns True once the
        process is confirmed gone (or was already gone).
        """
        proc = self._process
        if proc is None:
            return True
        pid = proc.pid
        if proc.returncode is not None:
            return True

        # Step 1: graceful terminate + short poll.
        try:
            proc.terminate()
        except (ProcessLookupError, Exception):
            pass
        deadline = time.monotonic() + _KILL_GRACE_SECONDS
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                log.info("Bridge subprocess pid=%s exited gracefully (reason=%s)", pid, reason)
                return True
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=0.25)
                return True
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

        # Step 2: escalate to a hard kill of the tracked PID, verified.
        log.warning(
            "Bridge subprocess pid=%s did not exit within %.0fs — hard-killing (reason=%s)",
            pid, _KILL_GRACE_SECONDS, reason,
        )
        try:
            proc.kill()
        except (ProcessLookupError, Exception):
            pass
        # OS-level fallback: proc.kill() can be a no-op if the transport is
        # confused; taskkill/SIGKILL by PID is the backstop.
        try:
            if sys.platform == "win32":
                await asyncio.create_subprocess_exec(
                    "taskkill", "/F", "/T", "/PID", str(pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                import signal as _signal
                try:
                    os.kill(pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except Exception:
            log.debug("OS-level kill fallback failed for pid=%s", pid, exc_info=True)

        # Verify exit by polling one more grace window.
        deadline = time.monotonic() + _KILL_GRACE_SECONDS
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                log.info("Bridge subprocess pid=%s confirmed killed (reason=%s)", pid, reason)
                return True
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=0.25)
                return True
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
        log.error("Bridge subprocess pid=%s could NOT be confirmed dead", pid)
        return False

    async def _drain_stderr(self) -> None:
        """Read bridge stderr, routing @@EVENT@@ lines to the event queue.

        Lines prefixed with @@EVENT@@ are streaming events from the bridge
        (text_delta, done, etc).  Everything else is debug logging.
        """
        assert self._process and self._process.stderr
        while True:
            raw = await self._process.stderr.readline()
            if not raw:
                break
            self._last_subprocess_activity = time.monotonic()
            text = raw.decode(errors="replace").rstrip()
            if not text:
                continue
            if text.startswith(_EVENT_PREFIX):
                # Any bridge event proves the SDK loop is alive — satisfy the
                # first-event watchdog (§11).
                if self._first_event is not None:
                    self._first_event.set()
                payload = text[len(_EVENT_PREFIX):]
                try:
                    event = json.loads(payload)
                    etype = event.get("type", "")
                    if etype == "rate_limit":
                        # Subscription-quota update — store directly and skip the queue.
                        key = event.get("rateLimitType", "unknown")
                        self._rate_limits[key] = {
                            k: v for k, v in event.items() if k != "type"
                        }
                        log.info(
                            "rate_limit %s: utilization=%s status=%s",
                            key, event.get("utilization"), event.get("status"),
                        )
                        continue
                    if etype in ("user_message_consumed", "user_message_dropped"):
                        # §5 delivery-ack. Consumed → the SDK loop read the
                        # injection; drop it from pending. Dropped → the bridge
                        # couldn't queue it (raced turn end); leave it pending so
                        # the caller's inbox fallback re-posts it.
                        inj_id = event.get("id", "")
                        if etype == "user_message_consumed":
                            self._pending_injections.pop(inj_id, None)
                            log.info("injection %s consumed by SDK loop", inj_id)
                        else:
                            log.warning("injection %s dropped by bridge (raced turn end)", inj_id)
                        continue
                    if etype not in ("text_delta",):
                        log.info("bridge stderr event: type=%s", etype)
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        # Drop rather than block: blocking here stops the
                        # stderr drain and stalls the SDK on its next write.
                        log.warning("bridge event queue full — dropping %s event", etype)
                except json.JSONDecodeError:
                    log.debug("bridge: bad event JSON: %s", payload[:200])
            else:
                log.debug("bridge: %s", text)

    async def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request to the bridge and read the JSON response.

        Serializes access through a lock.  If the bridge dies mid-request,
        respawns and retries once.
        """
        async with self._lock:
            return await self._send_raw(request, _retry=True)

    async def _send_raw(
        self, request: dict[str, Any], _retry: bool = True,
    ) -> dict[str, Any]:
        await self._ensure_process()
        assert self._process and self._process.stdin and self._process.stdout

        self._request_id += 1
        request["id"] = f"req-{self._request_id}"

        line = json.dumps(request) + "\n"

        # Write request
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if not _retry:
                raise ProviderError(
                    f"Bridge stdin broken: {exc}",
                    provider="claude_max",
                ) from exc
            log.warning("Bridge stdin broken, respawning and retrying")
            await self._kill_process()
            return await self._send_raw(request, _retry=False)

        # Read response
        resp_line = await self._process.stdout.readline()
        if not resp_line:
            if not _retry:
                raise ProviderError(
                    "Bridge process closed stdout unexpectedly",
                    provider="claude_max",
                )
            log.warning("Bridge stdout EOF, respawning and retrying")
            await self._kill_process()
            return await self._send_raw(request, _retry=False)

        try:
            return json.loads(resp_line)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Bridge returned invalid JSON: {resp_line[:200]}",
                provider="claude_max",
            ) from exc

    async def _kill_process(self) -> None:
        """Kill the bridge process and clean up."""
        if self._process:
            try:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass
            self._process = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def close(self) -> None:
        """Shut down the bridge process gracefully."""
        proc = self._process
        self._process = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        if proc is None or proc.returncode is not None:
            return
        # Ask the bridge to exit cleanly
        try:
            async with self._lock:
                await self._send_raw({"type": "shutdown"}, _retry=False)
        except Exception:
            pass
        # Wait for graceful exit, then force-kill
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except (asyncio.TimeoutError, Exception):
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
        # Close pipes explicitly so no transport lingers for GC
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        # Close the underlying transport to prevent __del__ warnings
        transport = getattr(proc, '_transport', None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
