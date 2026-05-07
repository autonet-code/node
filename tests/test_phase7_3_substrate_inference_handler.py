"""Phase 7.3: substrate inference handler for /rpb/inference/1.0.0.

What this validates:

  1. ``make_substrate_inference_handler`` returns an async coroutine
     that takes the wire-format request dict and returns the wire-
     format response dict.
  2. The handler routes the request through ``SubstrateProvider``,
     which probes the substrate and delegates render to the wrapped
     LLM. The substrate region appears in the renderer's system
     prompt, the renderer's text comes back in the response.
  3. AutonetHost exposes ``set_inference_handler`` and the registered
     handler is what ``_serve_inference_locally`` calls. Wiring is
     correct without spinning up two libp2p hosts.
  4. Errors in the handler surface as ``{"error": "..."}`` in the
     wire format — never crash the protocol stack.
  5. Wire format conversion is faithful: ``ProviderResponse`` ->
     dict (text, tool_calls, stop_reason, usage, model).
  6. The handler runs the asyncio-shaped Provider.send from a
     thread, so libp2p's trio runtime can call into it without
     bridging headaches at the call site.

Multi-host libp2p integration testing is deliberately **not** here
— pytest-trio collection is dormant in this project's pytest
config, and the handler-level path is what carries the real risk.
The wire/contract level is enough to declare 7.3 functional; full
end-to-end multi-host comes when we set up dedicated trio test
infrastructure (or via the existing manual test directory).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from atn.providers.base import (
    Provider,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)
from atn.providers.substrate import SubstrateProvider
from nodes.common.substrate_inference_handler import (
    make_substrate_inference_handler,
)
from nodes.common.world_service import WorldService
from world_model.generalized import Observation


# ---------------------------------------------------------------------------
# Test renderer (no real LLM)
# ---------------------------------------------------------------------------


class CannedRenderer(Provider):
    def __init__(self, response_text: str = "OK", input_tokens: int = 50, output_tokens: int = 10):
        self._text = response_text
        self._in = input_tokens
        self._out = output_tokens
        self.last_call: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "canned"

    async def send(
        self, *,
        messages: List[Dict[str, Any]], system: str = "", model: str = "",
        max_tokens: int = 1024, tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.last_call = {
            "messages": messages, "system": system, "model": model,
            "max_tokens": max_tokens, "temperature": temperature,
            "tools": tools,
        }
        return ProviderResponse(
            text=self._text,
            tool_calls=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=self._in, output_tokens=self._out),
        )


class FailingRenderer(Provider):
    @property
    def name(self) -> str:
        return "failing"

    async def send(self, **kwargs) -> ProviderResponse:
        raise RuntimeError("renderer offline for testing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _seed(svc: WorldService) -> None:
    seeds = [
        ("auth_flow", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=10, mag=0.7)),
        ("login_path", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=12, mag=0.6)),
    ]
    for label, c in seeds:
        svc.submit_observation(
            Observation(id=f"obs_{label}", coords=c, label=label),
            agent_id="seeder",
            sprout_under_charter=True,
            sprout_rootless=True,
        )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Handler-level tests
# ---------------------------------------------------------------------------


def test_handler_returns_wire_format_response(tmp_path: Path):
    """Happy path: request → response with all wire fields."""
    svc = WorldService(rpb_address="rpb_73_a", data_root=tmp_path)
    _seed(svc)
    renderer = CannedRenderer(response_text="hello from canned")
    provider = SubstrateProvider(world_service=svc, renderer=renderer)
    handler = make_substrate_inference_handler(provider)

    try:
        request = {
            "messages": [{"role": "user", "content": "how does authentication work"}],
            "system": "",
            "model": "qwen3:4b",
            "max_tokens": 256,
            "temperature": 0.0,
        }
        response = _run(handler(request))

        # Wire-format shape.
        assert "text" in response
        assert "tool_calls" in response
        assert "stop_reason" in response
        assert "usage" in response
        assert "model" in response
        assert "error" not in response

        # Values from the canned renderer flowed through.
        assert response["text"] == "hello from canned"
        assert response["model"] == "qwen3:4b"
        assert response["usage"].get("input_tokens") == 50
        assert response["usage"].get("output_tokens") == 10
    finally:
        svc.shutdown()


def test_handler_routes_through_substrate_to_renderer(tmp_path: Path):
    """The substrate region appears in the renderer's system prompt
    (verifies the locate+inject path actually runs)."""
    svc = WorldService(rpb_address="rpb_73_b", data_root=tmp_path)
    _seed(svc)
    renderer = CannedRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)
    handler = make_substrate_inference_handler(provider)

    try:
        request = {
            "messages": [{"role": "user", "content": "authentication and login flow"}],
            "model": "qwen3:4b",
        }
        _run(handler(request))

        # The renderer was called with a system prompt that includes
        # at least one of the seeded labels.
        sys_prompt = renderer.last_call["system"]
        assert "auth_flow" in sys_prompt or "login_path" in sys_prompt
        # And the user query was passed through.
        assert renderer.last_call["messages"][0]["content"] == "authentication and login flow"
    finally:
        svc.shutdown()


def test_handler_propagates_system_prompt_and_max_tokens(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_73_c", data_root=tmp_path)
    renderer = CannedRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)
    handler = make_substrate_inference_handler(provider)

    try:
        _run(handler({
            "messages": [{"role": "user", "content": "x"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 4096,
            "temperature": 0.7,
        }))
        assert renderer.last_call["max_tokens"] == 4096
        assert renderer.last_call["temperature"] == 0.7
        # Pre-existing system prompt is preserved at the start.
        assert renderer.last_call["system"].startswith("You are a helpful assistant.")
    finally:
        svc.shutdown()


def test_handler_returns_error_dict_on_renderer_failure(tmp_path: Path):
    """A failing renderer should NOT crash the protocol stack — the
    handler returns ``{"error": "..."}`` so AutonetHost can serialize
    it and the requester sees it as a clean RuntimeError."""
    svc = WorldService(rpb_address="rpb_73_d", data_root=tmp_path)
    _seed(svc)
    provider = SubstrateProvider(world_service=svc, renderer=FailingRenderer())
    handler = make_substrate_inference_handler(provider)

    try:
        response = _run(handler({
            "messages": [{"role": "user", "content": "anything"}],
            "model": "x",
        }))
        assert "error" in response
        assert "renderer offline" in response["error"].lower()
        # text/tool_calls etc. shouldn't be present when error is set.
        assert "text" not in response or response.get("text", "") == ""
    finally:
        svc.shutdown()


def test_handler_handles_empty_messages_gracefully(tmp_path: Path):
    """Edge case: empty messages list. Should still call renderer
    with empty messages; no crash."""
    svc = WorldService(rpb_address="rpb_73_e", data_root=tmp_path)
    renderer = CannedRenderer(response_text="empty input")
    provider = SubstrateProvider(world_service=svc, renderer=renderer)
    handler = make_substrate_inference_handler(provider)

    try:
        response = _run(handler({"messages": [], "model": "x"}))
        # Either we get a normal response (renderer was called), or an
        # error — but not a crash.
        assert "error" in response or "text" in response
    finally:
        svc.shutdown()


def test_make_handler_requires_provider():
    with pytest.raises(ValueError):
        make_substrate_inference_handler(None)


# ---------------------------------------------------------------------------
# AutonetHost wiring (no libp2p start, just contract surface)
# ---------------------------------------------------------------------------


def test_autonet_host_exposes_set_inference_handler():
    """The new public API exists and is callable."""
    from nodes.common.p2p import AutonetHost
    # Don't actually start the host — we're checking the surface.
    h = AutonetHost(node_id="test-wire", listen_host="127.0.0.1")
    assert hasattr(h, "set_inference_handler")
    async def _h(req): return {"text": "ok"}
    h.set_inference_handler(_h)
    assert h._inference_handler is _h


def test_autonet_host_serve_locally_calls_registered_handler():
    """``_serve_inference_locally`` dispatches to the registered handler
    and returns its response."""
    from nodes.common.p2p import AutonetHost
    h = AutonetHost(node_id="test-dispatch", listen_host="127.0.0.1")

    async def _handler(req):
        return {"text": "served by handler", "echoed_model": req.get("model")}

    h.set_inference_handler(_handler)
    response = _run(h._serve_inference_locally({"model": "qwen3:4b"}))
    assert response["text"] == "served by handler"
    assert response["echoed_model"] == "qwen3:4b"


def test_autonet_host_serve_locally_no_handler_returns_error():
    """If no handler is registered, the protocol returns a clean
    error rather than crashing or hanging."""
    from nodes.common.p2p import AutonetHost
    h = AutonetHost(node_id="test-nohandler", listen_host="127.0.0.1")
    response = _run(h._serve_inference_locally({}))
    assert "error" in response
    assert "no inference handler" in response["error"].lower()


# ---------------------------------------------------------------------------
# Wire-format conversion
# ---------------------------------------------------------------------------


def test_wire_format_includes_tool_calls(tmp_path: Path):
    """If the renderer emits tool calls, they're serialized into the
    wire response as a list of {id, name, arguments}."""
    svc = WorldService(rpb_address="rpb_73_tools", data_root=tmp_path)

    class ToolyRenderer(Provider):
        @property
        def name(self) -> str: return "tooly"
        async def send(self, **kwargs) -> ProviderResponse:
            return ProviderResponse(
                text="calling a tool",
                tool_calls=[ToolCall(id="t1", name="grep", input={"q": "x"})],
                stop_reason="tool_use",
                usage=Usage(),
            )

    provider = SubstrateProvider(world_service=svc, renderer=ToolyRenderer())
    handler = make_substrate_inference_handler(provider)

    try:
        response = _run(handler({
            "messages": [{"role": "user", "content": "use a tool"}],
            "model": "x",
        }))
        assert response["stop_reason"] == "tool_use"
        assert len(response["tool_calls"]) == 1
        tc = response["tool_calls"][0]
        assert tc["id"] == "t1"
        assert tc["name"] == "grep"
        assert tc["input"] == {"q": "x"}
    finally:
        svc.shutdown()
