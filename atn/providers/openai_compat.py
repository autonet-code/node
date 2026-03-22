"""OpenAI-compatible provider adapter.

Covers any provider that exposes an OpenAI-compatible chat completions
endpoint: Ollama (/v1), Google Gemini (via OpenAI-compat endpoint), and
any OpenAI API-compatible service.

Handles:
  - Message format translation (canonical Anthropic-style -> OpenAI API)
  - Tool definition translation (Anthropic input_schema -> OpenAI function parameters)
  - Response parsing (choices, tool_calls)
  - Retry with exponential backoff for transient errors (429, 503)
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

# Retry config
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503, 502}
_INITIAL_BACKOFF = 1.0
_BACKOFF_MULTIPLIER = 2.0


class OpenAICompatibleProvider(Provider):
    """Provider for any OpenAI-compatible chat completions API.

    Works with: Ollama, Google Gemini (via OpenAI endpoint), OpenAI, etc.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
    ) -> None:
        self._name = name
        self._api_key = api_key
        self._default_model = default_model

        # Normalise base URL to end with /chat/completions
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            if base_url.endswith("/v1"):
                base_url += "/chat/completions"
            elif base_url.endswith("/openai"):
                base_url += "/chat/completions"
            else:
                base_url += "/v1/chat/completions"
        self._url = base_url

    @property
    def name(self) -> str:
        return self._name

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

        # Translate messages — prepend system as a system message
        oai_messages = _to_openai_messages(messages, system)

        body: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [_tool_to_openai(t) for t in tools]

        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

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
        oai_messages = _to_openai_messages(messages, system)

        body: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = [_tool_to_openai(t) for t in tools]

        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        return await self._recv_stream(headers, body, model, on_chunk)

    async def _recv_stream(
        self, headers: dict, body: dict, model: str, on_chunk: Any,
    ) -> ProviderResponse:
        """SSE streaming for OpenAI-compatible endpoints."""
        import json as _json

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
        finish_reason = ""
        response_model = model
        usage = Usage()

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", self._url, headers=headers, json=body,
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {})
                        if isinstance(err_msg, dict):
                            err_msg = err_msg.get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ProviderError(
                        f"{self._name} API error {resp.status_code}: {err_msg}",
                        status_code=resp.status_code,
                        provider=self._name,
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    event = _json.loads(data_str)
                    response_model = event.get("model", response_model)

                    # Usage (some providers include it in the final chunk)
                    u = event.get("usage")
                    if u:
                        usage.input_tokens = u.get("prompt_tokens", usage.input_tokens)
                        usage.output_tokens = u.get("completion_tokens", usage.output_tokens)

                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr

                        # Text content
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                            if on_chunk:
                                await on_chunk(content)

                        # Tool calls (streamed incrementally)
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "name": tc_delta.get("function", {}).get("name", ""),
                                    "arguments": "",
                                }
                            acc = tool_calls_acc[idx]
                            if tc_delta.get("id"):
                                acc["id"] = tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                acc["name"] = func["name"]
                            if func.get("arguments"):
                                acc["arguments"] += func["arguments"]

        # Build tool calls
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[idx]
            args = acc["arguments"]
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except (_json.JSONDecodeError, ValueError):
                    args = {"raw": args}
            tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], input=args))

        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }

        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason_map.get(finish_reason, finish_reason),
            usage=usage,
            model=response_model,
        )

    async def _send_with_retry(self, headers: dict, body: dict) -> dict:
        backoff = _INITIAL_BACKOFF

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    resp = await client.post(self._url, headers=headers, json=body)

                    if resp.status_code == 200:
                        return resp.json()

                    if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                        retry_after = resp.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else backoff
                        log.warning(
                            "%s %d (attempt %d/%d), retrying in %.1fs",
                            self._name, resp.status_code, attempt + 1, _MAX_RETRIES + 1, wait,
                        )
                        await asyncio.sleep(wait)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue

                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {})
                        if isinstance(err_msg, dict):
                            err_msg = err_msg.get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ProviderError(
                        f"{self._name} API error {resp.status_code}: {err_msg}",
                        status_code=resp.status_code,
                        provider=self._name,
                    )

                except httpx.TimeoutException:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "%s timeout (attempt %d/%d), retrying in %.1fs",
                            self._name, attempt + 1, _MAX_RETRIES + 1, backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        f"{self._name} API timeout after retries",
                        provider=self._name,
                    )

                except httpx.ConnectError as exc:
                    if attempt < _MAX_RETRIES:
                        log.warning(
                            "%s connection error (attempt %d/%d): %s",
                            self._name, attempt + 1, _MAX_RETRIES + 1, exc,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _BACKOFF_MULTIPLIER
                        continue
                    raise ProviderError(
                        f"{self._name} connection failed after retries: {exc}",
                        provider=self._name,
                    )

        raise ProviderError("Exhausted retries", provider=self._name)


# ---------------------------------------------------------------------------
# Format translation: Canonical (Anthropic-style) -> OpenAI
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    """Convert canonical messages to OpenAI format."""
    oai: list[dict[str, Any]] = []

    if system:
        oai.append({"role": "system", "content": system})

    for msg in messages:
        oai.append({"role": msg["role"], "content": msg["content"]})

    return oai


def _tool_to_openai(tool: ToolDefinition) -> dict:
    """Convert canonical ToolDefinition to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _parse_response(data: dict, model: str) -> ProviderResponse:
    """Parse OpenAI-format response into canonical ProviderResponse."""
    choices = data.get("choices", [])
    if not choices:
        return ProviderResponse(
            text="",
            stop_reason="error",
            model=data.get("model", model),
        )

    msg = choices[0].get("message", {})
    text = msg.get("content") or ""
    finish_reason = choices[0].get("finish_reason", "")

    # Map OpenAI finish reasons to our canonical ones
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }
    stop_reason = stop_reason_map.get(finish_reason, finish_reason)

    # Parse tool calls
    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        # Parse arguments — may be a JSON string
        args = func.get("arguments", {})
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"raw": args}
        tool_calls.append(ToolCall(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            input=args,
        ))

    # Usage
    usage_data = data.get("usage", {})

    return ProviderResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        ),
        model=data.get("model", model),
        raw=data,
    )
