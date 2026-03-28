"""Anthropic provider adapter — calls the Claude Messages API via httpx.

Handles:
  - Message format translation (canonical -> Anthropic API)
  - Tool definition translation
  - Response parsing (text blocks, tool_use blocks)
  - Retry with exponential backoff for transient errors (429, 503, 529)

Provider Unification Analysis: AnthropicProvider vs BridgeProvider
===================================================================
Two paths exist for running Claude models:

1. BridgeProvider (bridge.py) — Claude Agent SDK via TypeScript subprocess
   - Spawns a bun subprocess running bridge/claude-bridge.ts
   - Uses Claude Agent SDK's native multi-turn orchestration (send_orchestrate)
   - The SDK handles its own agentic loop, context compaction, session resumption
   - Built-in shell tools (Bash, Read, Write, Glob, Grep) live in the SDK process
   - ATN framework tools are relayed as MCP tool_call/tool_result over stdin/stdout
   - Streaming events arrive on stderr as @@EVENT@@ NDJSON lines
   - Supports session_id for cross-turn prompt caching and context persistence
   - Requires bun + node_modules installed; heavier subprocess lifecycle

2. AnthropicProvider (this file) — Python REST client via httpx
   - Direct HTTP calls to the Anthropic Messages API
   - Multi-turn orchestration uses the generic loop in base.py (send_orchestrate)
     which calls send_stream() repeatedly, executing tool calls in Python
   - Shell tools come from atn/shell_tools.py (Python implementations)
   - No subprocess, no SDK dependency — pure Python
   - No native session resumption or context compaction
   - Lighter weight, easier to test, works without bun/node

Key differences:
  - BridgeProvider has native context compaction (SDK auto-summarises on overflow)
  - BridgeProvider has session resumption via session_id (persistent conversations)
  - BridgeProvider emits richer events (tool_use_start/result, compaction, thinking)
  - AnthropicProvider uses the base class generic agentic loop (simpler but less capable)
  - AnthropicProvider could be enhanced to use the Anthropic Python SDK's agent loop
    for parity with BridgeProvider without the TypeScript subprocess overhead

Path forward:
  - Short-term: Both paths coexist. BridgeProvider is used for the orchestrator
    (needs session persistence + compaction). AnthropicProvider is used for
    delegates and one-shot calls (lighter weight, no subprocess overhead).
  - Medium-term: Consider migrating AnthropicProvider to use the Anthropic Python
    SDK's agent loop (anthropic.Agent) instead of the generic base.py loop.
    This would give parity with BridgeProvider (compaction, session management)
    without the TypeScript subprocess, and could eventually replace BridgeProvider.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .base import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)

log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Retry config for transient errors
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503, 529}
_INITIAL_BACKOFF = 1.0  # seconds
_BACKOFF_MULTIPLIER = 2.0


class AnthropicProvider(Provider):

    def __init__(
        self,
        api_key: str,
        default_model: str = "",
        base_url: str = "",
    ) -> None:
        if not api_key:
            raise ProviderError("Anthropic API key is required", provider="anthropic")
        self._api_key = api_key
        self._default_model = default_model or _DEFAULT_MODEL
        self._base_url = (base_url.rstrip("/") + "/v1/messages") if base_url else _API_URL

    @property
    def name(self) -> str:
        return "anthropic"

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
        model = model or self._default_model

        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            # Structure system prompt as a content block with cache_control.
            # Anthropic caches the system prompt across requests with identical
            # prefix — subsequent calls with the same system prompt pay only
            # for cache reads (~90% cheaper than re-processing).
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            body["tools"] = [_tool_to_api(t) for t in tools]

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        # Send with retry
        data = await self._send_with_retry(headers, body)

        return _parse_response(data, model)

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
        model = model or self._default_model

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            body["tools"] = [_tool_to_api(t) for t in tools]

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        return await self._send_stream_impl(headers, body, model, on_chunk, on_thinking)

    async def _send_stream_impl(
        self, headers: dict, body: dict, model: str, on_chunk: Any,
        on_thinking: Any = None,
    ) -> ProviderResponse:
        """SSE streaming request to Anthropic API."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
        stop_reason = ""
        response_model = model

        # Track current content block type and tool_use being built
        current_block_type: str = ""
        current_tool: dict[str, Any] | None = None
        tool_json_acc = ""
        thinking_acc: list[str] = []

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", self._base_url, headers=headers, json=body,
            ) as resp:
                if resp.status_code != 200:
                    # Read full error body
                    await resp.aread()
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ProviderError(
                        f"Anthropic API error {resp.status_code}: {err_msg}",
                        status_code=resp.status_code,
                        provider="anthropic",
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    import json as _json
                    event = _json.loads(data_str)
                    event_type = event.get("type", "")

                    if event_type == "message_start":
                        msg = event.get("message", {})
                        response_model = msg.get("model", model)
                        u = msg.get("usage", {})
                        usage.input_tokens = u.get("input_tokens", 0)
                        usage.cache_read_tokens = u.get("cache_read_input_tokens", 0)
                        usage.cache_creation_tokens = u.get("cache_creation_input_tokens", 0)

                    elif event_type == "content_block_start":
                        block = event.get("content_block", {})
                        current_block_type = block.get("type", "")
                        if current_block_type == "tool_use":
                            current_tool = {"id": block.get("id", ""), "name": block.get("name", "")}
                            tool_json_acc = ""
                        elif current_block_type == "thinking":
                            thinking_acc = []

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type", "")
                        if delta_type == "text_delta":
                            chunk = delta.get("text", "")
                            if chunk:
                                text_parts.append(chunk)
                                if on_chunk:
                                    await on_chunk(chunk)
                        elif delta_type == "input_json_delta":
                            tool_json_acc += delta.get("partial_json", "")
                        elif delta_type == "thinking_delta":
                            chunk = delta.get("thinking", "")
                            if chunk:
                                thinking_acc.append(chunk)

                    elif event_type == "content_block_stop":
                        if current_block_type == "thinking" and thinking_acc:
                            full_thinking = "".join(thinking_acc)
                            thinking_parts.append(full_thinking)
                            if on_thinking:
                                await on_thinking(full_thinking)
                            thinking_acc = []
                        elif current_tool is not None:
                            try:
                                tool_input = _json.loads(tool_json_acc) if tool_json_acc else {}
                            except _json.JSONDecodeError:
                                tool_input = {"raw": tool_json_acc}
                            tool_calls.append(ToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                input=tool_input,
                            ))
                            current_tool = None
                            tool_json_acc = ""
                        current_block_type = ""

                    elif event_type == "message_delta":
                        delta = event.get("delta", {})
                        stop_reason = delta.get("stop_reason", stop_reason)
                        u = event.get("usage", {})
                        usage.output_tokens = u.get("output_tokens", usage.output_tokens)

        return ProviderResponse(
            text="".join(text_parts),
            thinking=thinking_parts,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=response_model,
        )

    async def _send_with_retry(
        self, headers: dict, body: dict,
    ) -> dict:
        """Send request with exponential backoff on transient errors."""
        backoff = _INITIAL_BACKOFF

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    resp = await client.post(self._base_url, headers=headers, json=body)

                    if resp.status_code == 200:
                        return resp.json()

                    if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                        # Check for Retry-After header
                        retry_after = resp.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else backoff
                        log.warning(
                            "Anthropic %d (attempt %d/%d), retrying in %.1fs",
                            resp.status_code, attempt + 1, _MAX_RETRIES + 1, wait,
                        )
                        await asyncio.sleep(wait)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue

                    # Non-retryable error
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ProviderError(
                        f"Anthropic API error {resp.status_code}: {err_msg}",
                        status_code=resp.status_code,
                        provider="anthropic",
                    )

                except httpx.TimeoutException:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "Anthropic timeout (attempt %d/%d), retrying in %.1fs",
                            attempt + 1, _MAX_RETRIES + 1, backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        "Anthropic API timeout after retries",
                        provider="anthropic",
                    )

                except httpx.ConnectError as exc:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "Anthropic connection error (attempt %d/%d): %s",
                            attempt + 1, _MAX_RETRIES + 1, exc,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        f"Anthropic connection failed after retries: {exc}",
                        provider="anthropic",
                    )

        # Should never reach here
        raise ProviderError("Exhausted retries", provider="anthropic")


# ---------------------------------------------------------------------------
# Format translation
# ---------------------------------------------------------------------------

def _tool_to_api(tool: ToolDefinition) -> dict:
    """Convert canonical ToolDefinition to Anthropic API format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _parse_response(data: dict, model: str) -> ProviderResponse:
    """Parse Anthropic API response into canonical ProviderResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in data.get("content", []):
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                input=block["input"],
            ))

    usage_data = data.get("usage", {})

    return ProviderResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=data.get("stop_reason", ""),
        usage=Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage_data.get("cache_creation_input_tokens", 0),
        ),
        model=data.get("model", model),
        raw=data,
    )
