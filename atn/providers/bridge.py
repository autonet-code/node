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

# Path to the bridge script relative to this file
_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent / "bridge"
_BRIDGE_SCRIPT = _BRIDGE_DIR / "claude-bridge.ts"


_EVENT_PREFIX = "@@EVENT@@"


class BridgeProvider(Provider):
    """Claude Max provider via the TypeScript bridge subprocess.

    Implements the standard Provider interface (maxTurns=1).  Each call to
    send() creates a fresh bridge session, collects the response, and
    cleans up.
    """

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
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Optional EventBus for emitting tool use events.
        # Set by the caller (e.g. runtime) after construction.
        self.event_bus: EventBus | None = None
        self.source_agent_id: str = ""

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
        async with self._lock:
            await self._ensure_process()
            assert self._process and self._process.stdin and self._process.stdout

            self._request_id += 1
            request_id = f"req-{self._request_id}"

            effective_model = model or self._model
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
            }
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

            log.info("Orchestrate request sent (tools=%d, max_turns=%d)", len(tools), max_turns)

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
                    except Exception:
                        log.exception("Error processing stream event: %s", event)

            stream_task = asyncio.create_task(_stream_events())

            # Read stdout lines: may be tool_call or final response
            try:
                final_resp: dict[str, Any] | None = None

                while True:
                    raw = await self._process.stdout.readline()
                    if not raw:
                        raise ProviderError(
                            "Bridge stdout closed during orchestrate",
                            provider="claude_max",
                        )

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

                        # Send tool_result back to bridge
                        result_msg = json.dumps({
                            "type": "tool_result",
                            "call_id": call_id,
                            "result": result,
                        }, default=str) + "\n"
                        try:
                            self._process.stdin.write(result_msg.encode())
                            await self._process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                            raise ProviderError(
                                f"Bridge stdin broken sending tool_result: {exc}",
                                provider="claude_max",
                            ) from exc
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

            return ProviderResponse(
                text=final_resp.get("text", ""),
                thinking=final_resp.get("thinking", []),
                tool_calls=[],  # Tools are handled inside the bridge
                stop_reason=final_resp.get("stop_reason", "end_turn"),
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

    async def send_user_message(self, content: str) -> None:
        """Inject a user message into the active orchestration session.

        The message is queued and picked up by the SDK on the next turn.
        Safe to call while send_orchestrate() is running.

        Does NOT acquire the lock — same rationale as interrupt().
        """
        if not self._process or self._process.returncode is not None:
            log.warning("send_user_message: no active bridge process")
            return
        assert self._process.stdin
        try:
            line = json.dumps({"type": "user_message", "content": content}) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            log.info("User message injected (%d chars)", len(content))
        except (BrokenPipeError, ConnectionResetError, OSError):
            log.warning("Failed to send user_message (bridge stdin broken)")

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
                f"Bridge dependencies not installed. Run: cd {self._bridge_script.parent} && bun install",
                provider="claude_max",
            )

        # Fresh queue for new process
        self._event_queue = asyncio.Queue()

        log.info("Spawning bridge: bun run %s", self._bridge_script)

        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        env = {**os.environ, "ENABLE_TOOL_SEARCH": "1"}
        # Clear CLAUDECODE to avoid nesting guard in the SDK
        env.pop("CLAUDECODE", None)

        self._process = await asyncio.create_subprocess_exec(
            "bun", "run", str(self._bridge_script),
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
            text = raw.decode(errors="replace").rstrip()
            if not text:
                continue
            if text.startswith(_EVENT_PREFIX):
                payload = text[len(_EVENT_PREFIX):]
                try:
                    event = json.loads(payload)
                    etype = event.get("type", "")
                    if etype not in ("text_delta",):
                        log.info("bridge stderr event: type=%s", etype)
                    await self._event_queue.put(event)
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
