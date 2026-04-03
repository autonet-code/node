"""Ollama provider adapter — uses the native /api/chat endpoint.

The native Ollama API (as opposed to the OpenAI-compatible /v1 endpoint)
supports the `think` parameter, which lets us disable "thinking mode" on
models like qwen3 and deepseek-r1 that otherwise burn all tokens on
internal reasoning.

Handles:
  - Native Ollama message format (same as OpenAI, mostly)
  - Thinking mode control via `think: false`
  - Retry with exponential backoff for transient errors
"""
from __future__ import annotations

import asyncio
import json
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

_DEFAULT_BASE = "http://localhost:11434"
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503, 502}
_INITIAL_BACKOFF = 1.0
_BACKOFF_MULTIPLIER = 2.0


def _normalize_content(content: Any) -> str:
    """Convert Anthropic-style content blocks to plain string.

    The base.py send_orchestrate() loop builds messages with list-of-dict
    content blocks (Anthropic format).  Ollama expects plain strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(block.get("content", str(block)))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


class OllamaProvider(Provider):
    """Provider for local Ollama models via the native /api/chat endpoint."""

    def __init__(
        self,
        base_url: str = "",
        default_model: str = "",
    ) -> None:
        self._default_model = default_model
        base_url = (base_url or _DEFAULT_BASE).rstrip("/")
        self._url = f"{base_url}/api/chat"

    @property
    def name(self) -> str:
        return "ollama"

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

        # Build message list with system prompt
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": _normalize_content(msg["content"])})

        body: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            body["tools"] = [_tool_to_ollama(t) for t in tools]

        data = await self._send_with_retry(body)
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

        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": _normalize_content(msg["content"])})

        body: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,      # Enable streaming (NDJSON)
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            body["tools"] = [_tool_to_ollama(t) for t in tools]

        return await self._recv_stream(body, model, on_chunk)

    async def _recv_stream(
        self, body: dict, model: str, on_chunk: Any,
    ) -> ProviderResponse:
        """NDJSON streaming from Ollama native API."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
        response_model = model
        done_reason = ""

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", self._url, json=body,
                headers={"content-type": "application/json"},
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise ProviderError(
                        f"Ollama API error {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        provider="ollama",
                    )

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        text_parts.append(content)
                        if on_chunk:
                            await on_chunk(content)

                    # Tool calls (appear in final chunk)
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {"raw": args}
                        tool_calls.append(ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            input=args,
                        ))

                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason", "")
                        usage.input_tokens = chunk.get("prompt_eval_count", 0)
                        usage.output_tokens = chunk.get("eval_count", 0)
                        response_model = chunk.get("model", model)

        if tool_calls:
            stop_reason = "tool_use"
        elif done_reason == "stop":
            stop_reason = "end_turn"
        elif done_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = done_reason or "end_turn"

        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=response_model,
        )

    async def _send_with_retry(self, body: dict) -> dict:
        backoff = _INITIAL_BACKOFF

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._url,
                        json=body,
                        headers={"content-type": "application/json"},
                    )

                    if resp.status_code == 200:
                        return resp.json()

                    if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                        log.warning(
                            "Ollama %d (attempt %d/%d), retrying in %.1fs",
                            resp.status_code, attempt + 1, _MAX_RETRIES + 1, backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue

                    raise ProviderError(
                        f"Ollama API error {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        provider="ollama",
                    )

                except httpx.TimeoutException:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "Ollama timeout (attempt %d/%d), retrying",
                            attempt + 1, _MAX_RETRIES + 1,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        "Ollama API timeout after retries",
                        provider="ollama",
                    )

                except httpx.ConnectError as exc:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "Ollama connection error (attempt %d/%d): %s",
                            attempt + 1, _MAX_RETRIES + 1, exc,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        f"Ollama connection failed: {exc}. Is Ollama running?",
                        provider="ollama",
                    )

        raise ProviderError("Exhausted retries", provider="ollama")


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _tool_to_ollama(tool: ToolDefinition) -> dict:
    """Convert to Ollama's tool format (same as OpenAI function calling)."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _parse_response(data: dict, model: str) -> ProviderResponse:
    """Parse native Ollama response into canonical ProviderResponse."""
    msg = data.get("message", {})
    text = msg.get("content", "")

    # Parse tool calls
    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"raw": args}
        tool_calls.append(ToolCall(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            input=args,
        ))

    # Determine stop reason
    done_reason = data.get("done_reason", "")
    if tool_calls:
        stop_reason = "tool_use"
    elif done_reason == "stop":
        stop_reason = "end_turn"
    elif done_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = done_reason or "end_turn"

    # Usage — Ollama reports tokens differently
    usage = Usage(
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )

    return ProviderResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
        model=data.get("model", model),
        raw=data,
    )
