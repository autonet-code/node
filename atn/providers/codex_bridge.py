"""Codex bridge provider -- runs the OpenAI Codex SDK via subprocess.

Spawns a TypeScript bridge process (bridge/codex-bridge.ts) that wraps the
OpenAI Codex SDK (@openai/codex-sdk).  Communication uses NDJSON over
stdin/stdout pipes — same protocol as the Claude Max bridge.

Supports two request types:
  - create:      single-turn LLM call (one prompt → one response).
  - orchestrate: multi-turn with Codex's built-in tools + ATN tool relay.

Config keys (in ~/.atn/config.yaml under providers.codex_max):
  model       (str)   "o4-mini" | "o3" | "gpt-4.1".  Default: "o4-mini".

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

# Locate the bridge directory
def _find_bridge_dir() -> Path:
    try:
        import bridge as _bridge_pkg
        return Path(_bridge_pkg.__file__).resolve().parent
    except ImportError:
        pass
    return Path(__file__).resolve().parent.parent.parent / "bridge"

_BRIDGE_DIR = _find_bridge_dir()
_BRIDGE_SCRIPT = _BRIDGE_DIR / "codex-bridge.ts"

_EVENT_PREFIX = "@@EVENT@@"


class CodexBridgeProvider(Provider):
    """OpenAI Codex provider via the TypeScript bridge subprocess.

    Mirrors the BridgeProvider interface but uses the Codex SDK instead
    of the Claude Agent SDK.  Supports single-turn and multi-turn
    orchestration with ATN tool relay.
    """

    def __init__(
        self,
        model: str = "o4-mini",
        bridge_script: str | Path | None = None,
        api_key: str = "",
    ) -> None:
        self._model = model
        self._bridge_script = Path(bridge_script) if bridge_script else _BRIDGE_SCRIPT
        self._api_key = api_key
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Optional EventBus for emitting tool use events.
        self.event_bus: EventBus | None = None
        self.source_agent_id: str = ""

        # Cumulative session stats
        self._session_id: str = ""
        self._cumulative_turns: int = 0
        self._total_cost_usd: float = 0.0
        self._context_window: int = 200_000  # o4-mini default
        self._max_output_tokens: int = 100_000
        self._cumulative_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._cumulative_cache_read: int = 0
        self._cumulative_cache_creation: int = 0
        self._last_input_tokens: int = 0
        self._compaction_count: int = 0

    @property
    def name(self) -> str:
        return "codex_max"

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
                provider="codex_max",
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
                f"Codex bridge error: {resp.get('error', 'unknown')}",
                provider="codex_max",
            )

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
        """Send with streaming — calls on_chunk(text) for each text delta."""
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
            raise ProviderError("No user message text found", provider="codex_max")

        request = {
            "type": "create",
            "message": user_text,
            "system": system,
            "model": model or self._model,
        }

        # Drain stale events
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        async def _stream_events() -> None:
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
            return await self._send_request(request)

        event_task = asyncio.create_task(_stream_events())
        try:
            resp = await _get_response()
        finally:
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

        if not resp.get("ok"):
            raise ProviderError(
                f"Codex bridge error: {resp.get('error', 'unknown')}",
                provider="codex_max",
            )

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
        """Multi-turn orchestrator call through the Codex bridge.

        Handles the bidirectional tool relay protocol:
        1. Sends an 'orchestrate' request to the bridge.
        2. Reads stdout lines for tool_call or final response.
        3. Concurrently drains stderr for streaming events.
        """
        async with self._lock:
            await self._ensure_process()
            assert self._process and self._process.stdin and self._process.stdout

            self._request_id += 1
            request_id = f"req-{self._request_id}"

            effective_model = model or self._model
            if model:
                self._model = model

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
                    f"Codex bridge stdin broken during orchestrate: {exc}",
                    provider="codex_max",
                ) from exc

            log.info("Codex orchestrate request sent (tools=%d, max_turns=%d)", len(tools), max_turns)

            # Start streaming events
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
                    except Exception:
                        log.exception("Error processing stream event: %s", event)

            stream_task = asyncio.create_task(_stream_events())

            # Read stdout lines: tool_call or final response
            try:
                final_resp: dict[str, Any] | None = None

                while True:
                    raw = await self._process.stdout.readline()
                    if not raw:
                        raise ProviderError(
                            "Codex bridge stdout closed during orchestrate",
                            provider="codex_max",
                        )

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("codex-bridge: non-JSON stdout line: %s", raw[:200])
                        continue

                    # Tool call relay: execute and send result back
                    if msg.get("type") == "tool_call":
                        call_id = msg["call_id"]
                        tool_name = msg["name"]
                        tool_input = msg.get("input", {})
                        log.info("Codex orchestrate tool_call: %s (call_id=%s)", tool_name, call_id)

                        is_error = False
                        try:
                            result = await tool_executor(tool_name, tool_input)
                        except Exception as exc:
                            log.exception("Tool execution error: %s", tool_name)
                            result = {"error": f"Tool '{tool_name}' failed: {exc}"}
                            is_error = True

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
                                f"Codex bridge stdin broken sending tool_result: {exc}",
                                provider="codex_max",
                            ) from exc
                        continue

                    # Final response (has 'id' field)
                    if "id" in msg:
                        final_resp = msg
                        break

                    log.debug("codex-bridge: unexpected stdout message: %s", str(msg)[:200])

            finally:
                if stream_task:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

            if final_resp is None:
                raise ProviderError(
                    "No final response from Codex bridge orchestrate",
                    provider="codex_max",
                )

            if not final_resp.get("ok"):
                raise ProviderError(
                    f"Codex bridge orchestrate error: {final_resp.get('error', 'unknown')}",
                    provider="codex_max",
                )

            # Parse into ProviderResponse
            usage_data = final_resp.get("usage", {})
            usage = Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage_data.get("cache_creation_input_tokens", 0),
            )

            # Update cumulative session stats
            ctx = final_resp.get("context", {})
            if ctx:
                self._cumulative_turns += ctx.get("num_turns", 0)
                self._context_window = ctx.get("context_window", self._context_window)
                self._max_output_tokens = ctx.get("max_output_tokens", self._max_output_tokens)
            if final_resp.get("session_id"):
                self._session_id = final_resp["session_id"]
            self._last_input_tokens = usage.input_tokens + usage.cache_read_tokens
            self._cumulative_input_tokens += usage.input_tokens
            self._cumulative_output_tokens += usage.output_tokens
            self._cumulative_cache_read += usage.cache_read_tokens
            self._cumulative_cache_creation += usage.cache_creation_tokens

            # Emit per-turn usage event
            if self.event_bus:
                await self.event_bus.emit(Event(
                    type=EventType.STEP_OUTPUT,
                    source=self.source_agent_id,
                    data={
                        "agent_id": self.source_agent_id,
                        "channel": "usage",
                        "usage": {
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
                tool_calls=[],
                stop_reason=final_resp.get("stop_reason", "end_turn"),
                usage=usage,
                model=final_resp.get("model", model or self._model),
                raw=final_resp,
            )

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    @property
    def session_stats(self) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    async def _ensure_process(self) -> None:
        """Spawn bridge subprocess if not running."""
        if self._process and self._process.returncode is None:
            return

        if not self._bridge_script.exists():
            raise ProviderError(
                f"Codex bridge script not found: {self._bridge_script}",
                provider="codex_max",
            )

        # Check that node_modules exist
        node_modules = self._bridge_script.parent / "node_modules"
        if not node_modules.exists():
            raise ProviderError(
                f"Bridge dependencies not installed. "
                f"Run: cd {self._bridge_script.parent} && npm install",
                provider="codex_max",
            )

        self._event_queue = asyncio.Queue()

        # Resolve JS runtime: prefer bun, fall back to node
        import shutil
        runtime = shutil.which("bun")
        runtime_args = ["run", str(self._bridge_script)]
        if not runtime:
            runtime = shutil.which("node")
            if runtime:
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
                provider="codex_max",
            )

        log.info("Spawning Codex bridge: %s %s", runtime, " ".join(runtime_args))

        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        env = {**os.environ}
        # Pass API key to the bridge
        if self._api_key:
            env["CODEX_API_KEY"] = self._api_key
            env["OPENAI_API_KEY"] = self._api_key

        self._process = await asyncio.create_subprocess_exec(
            runtime, *runtime_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            **kwargs,
        )

        self._stderr_task = asyncio.create_task(self._drain_stderr())
        log.info("Codex bridge process started (pid=%s)", self._process.pid)

    async def _drain_stderr(self) -> None:
        """Read bridge stderr, routing @@EVENT@@ lines to the event queue."""
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
                        log.info("codex-bridge stderr event: type=%s", etype)
                    await self._event_queue.put(event)
                except json.JSONDecodeError:
                    log.debug("codex-bridge: bad event JSON: %s", payload[:200])
            else:
                log.debug("codex-bridge: %s", text)

    async def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request and read the JSON response."""
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

        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if not _retry:
                raise ProviderError(
                    f"Codex bridge stdin broken: {exc}",
                    provider="codex_max",
                ) from exc
            log.warning("Codex bridge stdin broken, respawning and retrying")
            await self._kill_process()
            return await self._send_raw(request, _retry=False)

        resp_line = await self._process.stdout.readline()
        if not resp_line:
            if not _retry:
                raise ProviderError(
                    "Codex bridge process closed stdout unexpectedly",
                    provider="codex_max",
                )
            log.warning("Codex bridge stdout EOF, respawning and retrying")
            await self._kill_process()
            return await self._send_raw(request, _retry=False)

        try:
            return json.loads(resp_line)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Codex bridge returned invalid JSON: {resp_line[:200]}",
                provider="codex_max",
            ) from exc

    async def _kill_process(self) -> None:
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
        try:
            async with self._lock:
                await self._send_raw({"type": "shutdown"}, _retry=False)
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except (asyncio.TimeoutError, Exception):
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        transport = getattr(proc, '_transport', None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
