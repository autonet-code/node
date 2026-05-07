"""Substrate-backed inference handler for AutonetHost.

Phase 7.3 of native world-model integration.

Wires AutonetHost's ``/rpb/inference/1.0.0`` request handler to a
``SubstrateProvider``: when a peer requests inference, this handler
runs the substrate locate + LLM render pipeline and returns the
response.

Trio ↔ asyncio bridge
---------------------

AutonetHost runs in trio (a libp2p requirement). ``SubstrateProvider``
inherits from the ATN ``Provider`` base which is asyncio-shaped. We
bridge by running the provider's ``send`` coroutine inside a fresh
asyncio loop on a worker thread — same pattern ATN uses to start the
autonet service (atn/autonet_service.py:685).

Wire format
-----------

Request dict (what AutonetHost.request_inference sends):
  {
    "messages": [...],     # canonical message list
    "system": "...",       # optional system prompt
    "model": "qwen3:4b",   # renderer model
    "max_tokens": 1024,
    "temperature": 0.0,
    "tools": [...]         # optional
  }

Response dict (what AutonetHost expects back):
  {
    "text": "...",         # the renderer's text response
    "tool_calls": [...],   # if the renderer emitted any
    "stop_reason": "...",
    "usage": {...},        # input/output tokens etc.
    "model": "qwen3:4b"    # what model was actually used
  }

Errors are surfaced as ``{"error": "..."}`` — AutonetHost.request_inference
raises RuntimeError on those.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional


logger = logging.getLogger(__name__)


def make_substrate_inference_handler(
    substrate_provider: Any,
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Build the handler coroutine that AutonetHost will invoke.

    Args:
        substrate_provider: a ``SubstrateProvider`` instance bound to
            this daemon's WorldService and renderer.

    Returns:
        An async function that takes the wire-format request dict
        and returns the wire-format response dict.
    """
    if substrate_provider is None:
        raise ValueError("substrate_provider is required")

    async def _handler(request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract the call shape from the wire format.
            messages = request.get("messages") or []
            system = request.get("system", "") or ""
            model = request.get("model", "") or ""
            max_tokens = int(request.get("max_tokens", 1024))
            temperature = float(request.get("temperature", 0.0))
            tools = request.get("tools")

            # The handler runs in whatever event loop AutonetHost calls
            # us from (trio). SubstrateProvider.send is asyncio-shaped.
            # We bridge by running send() in a worker thread with a
            # dedicated asyncio loop.
            response = await _run_provider_in_thread(
                substrate_provider,
                messages=messages,
                system=system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )

            return _provider_response_to_wire(response, model)

        except Exception as e:
            logger.warning("substrate inference handler failed: %s", e, exc_info=True)
            return {"error": str(e)}

    return _handler


async def _run_provider_in_thread(
    provider: Any,
    *,
    messages: list,
    system: str,
    model: str,
    max_tokens: int,
    temperature: float,
    tools: Optional[list],
) -> Any:
    """Run ``provider.send(...)`` in a worker thread that has its own
    asyncio loop.

    Uses ``trio.to_thread.run_sync`` if we're in trio (the
    AutonetHost case), or ``asyncio.to_thread`` if we're in asyncio
    (test case where the handler is called directly without going
    through libp2p).
    """
    def _sync_call() -> Any:
        return asyncio.run(provider.send(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
        ))

    # Detect runtime. Prefer trio if it's running; fall back to
    # asyncio (handler invoked from a non-libp2p context, e.g. tests).
    try:
        import trio
        # If we're inside trio, this won't raise.
        trio.lowlevel.current_task()  # type: ignore[attr-defined]
        return await trio.to_thread.run_sync(_sync_call)
    except (ImportError, RuntimeError):
        pass

    # asyncio path
    return await asyncio.to_thread(_sync_call)


def _provider_response_to_wire(response: Any, model: str) -> Dict[str, Any]:
    """Convert a ``ProviderResponse`` to the inference wire-format dict."""
    usage = getattr(response, "usage", None)
    usage_dict: Dict[str, Any] = {}
    if usage is not None:
        for attr in (
            "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_creation_tokens",
        ):
            v = getattr(usage, attr, None)
            if v is not None:
                usage_dict[attr] = v

    tool_calls_out = []
    for tc in getattr(response, "tool_calls", None) or []:
        tool_calls_out.append({
            "id": getattr(tc, "id", ""),
            "name": getattr(tc, "name", ""),
            "input": getattr(tc, "input", {}),
        })

    return {
        "text": getattr(response, "text", "") or "",
        "tool_calls": tool_calls_out,
        "stop_reason": getattr(response, "stop_reason", "") or "",
        "usage": usage_dict,
        "model": model,
    }
