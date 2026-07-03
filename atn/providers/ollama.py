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
    ContextOverflowError,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
    is_overflow_message,
)

log = logging.getLogger(__name__)

_DEFAULT_BASE = "http://localhost:11434"
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503, 502}
_INITIAL_BACKOFF = 1.0
_BACKOFF_MULTIPLIER = 2.0
# Fallback context window when model_specs can't be imported or a model can't
# be sized — matches the local-model default (§7).
_DEFAULT_NUM_CTX = 16_384


def _num_ctx_for(model: str) -> int:
    """Context window for ``options.num_ctx`` from model_specs (§9). Unknown
    local models resolve to 16384; falls back to that if the lookup fails."""
    try:
        from ..model_specs import get_context_window
        ctx = get_context_window(model)
        return ctx if ctx > 0 else _DEFAULT_NUM_CTX
    except Exception:
        return _DEFAULT_NUM_CTX


def _normalize_content(content: Any) -> str:
    """Convert Anthropic-style content blocks to plain string.

    The base.py send_orchestrate() loop builds messages with list-of-dict
    content blocks (Anthropic format).  Ollama expects plain strings.

    Used only for plain-text blocks. Structured ``tool_use`` / ``tool_result``
    linkage is preserved natively by :func:`_translate_history` instead of being
    flattened here (agentic_loop.md §9 / live finding 5).
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


def _translate_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate canonical (Anthropic-shape) history into ollama /api/chat native.

    The base.py send_orchestrate() loop builds:
      - assistant turns as ``{"role":"assistant","content":[{type:text}, {type:tool_use, id, name, input}]}``
      - tool results as ``{"role":"user","content":[{type:tool_result, tool_use_id, content}]}``

    Ollama's native format wants:
      - assistant tool calls as ``{"role":"assistant","content":<text>,"tool_calls":[{"function":{"name","arguments"}}]}``
      - each tool result as its own ``{"role":"tool","content":<result>,"tool_name":<name>}`` message

    Preserving this linkage (rather than flattening to text via
    ``_normalize_content``) is what lets a local model follow multi-turn tool
    use past turn 1. ``tool_name`` is recovered from the tool_use that produced
    each result (ollama accepts ``tool_name`` on tool messages).
    """
    # Map tool_use_id -> tool name so tool_result messages can carry tool_name.
    id_to_name: dict[str, str] = {}
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Plain-text content (string, or list without structured blocks) —
        # keep the existing normalization.
        if not isinstance(content, list):
            out.append({"role": role, "content": _normalize_content(content)})
            continue

        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )

        if role == "assistant" and has_tool_use:
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "")
                    tuid = block.get("id", "")
                    if tuid:
                        id_to_name[tuid] = name
                    tool_calls.append({
                        "function": {
                            "name": name,
                            "arguments": block.get("input", {}) or {},
                        },
                    })
            out.append({
                "role": "assistant",
                "content": "\n".join(p for p in text_parts if p),
                "tool_calls": tool_calls,
            })
        elif has_tool_result:
            # One ollama "tool" message per tool_result block, preserving the
            # per-result linkage (name recovered from the originating tool_use).
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tuid = block.get("tool_use_id", "")
                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "content": block.get("content", ""),
                    }
                    name = id_to_name.get(tuid, "")
                    if name:
                        tool_msg["tool_name"] = name
                    out.append(tool_msg)
        else:
            out.append({"role": role, "content": _normalize_content(content)})

    return out


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

        # Build message list with system prompt. Structured tool history is
        # translated natively (§9) so multi-turn tool use round-trips.
        ollama_messages: list[dict[str, Any]] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(_translate_history(messages))

        body: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # §9: never rely on ollama's 4096 default — it silently
                # truncates larger prompts. Size from model_specs.
                "num_ctx": _num_ctx_for(model),
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
        ollama_messages.extend(_translate_history(messages))

        body: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,      # Enable streaming (NDJSON)
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # §9: size the context window from model_specs, never ollama's
                # 4096 default.
                "num_ctx": _num_ctx_for(model),
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
                    # §1: map a context-overflow error to the shared class so
                    # the loop routes it to reduction, not a blind retry.
                    if resp.status_code == 400 and is_overflow_message(resp.text):
                        raise ContextOverflowError(
                            f"Ollama context overflow: {resp.text[:200]}",
                            status_code=400, provider="ollama",
                        )
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

                    # §1: context-overflow → shared class (route to reduction).
                    if resp.status_code == 400 and is_overflow_message(resp.text):
                        raise ContextOverflowError(
                            f"Ollama context overflow: {resp.text[:200]}",
                            status_code=400, provider="ollama",
                        )
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
