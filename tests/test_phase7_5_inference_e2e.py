"""Phase 7.5: end-to-end inference path between two daemons.

Two AutonetHost instances (daemons A and B). Daemon B hosts a serving
agent: it registers the substrate handler on its libp2p surface and
advertises inference capability. Daemon A's runtime discovers the
serving agent and dispatches a request through to B's handler.

The on-chain identity of the serving entity is the **agent**
(`agent_address` in the inference advertisement) — the daemon is the
wire transport, not the consensus actor. Phase 7.5b will tighten the
discovery surface to surface agent matches rather than peer matches.
For now a daemon hosts at most one inference-serving agent, so the
ambiguity is harmless.

The byte-on-the-wire libp2p layer is deliberately stubbed out — the
actual /rpb/inference/1.0.0 framing is tested at unit level by
tests/test_p2p.py and the trio runtime is not collected by this
project's pytest config. What this test covers is the **integration
glue**: 7.3 handler + 7.4 advertisement + 7.4 discovery all line up.

Tested:

  1. B's advertisement appears in A's discovery results after the
     capability is shared (simulated via direct cache injection,
     mimicking what _handle_capability_stream does on real bytes).
  2. A picks the right peer when multiple inference providers exist.
  3. The full request shape that A constructs is what B's handler
     accepts and processes.
  4. End-to-end: A sends "how does authentication work" → B's
     substrate handler runs (substrate locate + canned-LLM render) →
     A receives the wire-format response with text + region citations.
  5. If B's handler fails, A receives a clean error dict (not a
     crashed connection).
  6. A serving daemon that lost its substrate (advertising stops)
     drops out of A's discovery.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from atn.providers.base import (
    Provider,
    ProviderResponse,
    ToolDefinition,
    Usage,
)
from atn.providers.substrate import SubstrateProvider
from nodes.common.p2p import AutonetHost, NodeCapability
from nodes.common.substrate_inference_handler import (
    make_substrate_inference_handler,
)
from nodes.common.world_service import WorldService
from world_model.generalized import Observation


# ---------------------------------------------------------------------------
# Test renderer
# ---------------------------------------------------------------------------


class CannedRenderer(Provider):
    def __init__(self, response_text: str = "ok"):
        self._text = response_text
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
        }
        return ProviderResponse(
            text=self._text,
            tool_calls=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=42, output_tokens=8),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _seed_substrate(svc: WorldService) -> None:
    seeds = [
        ("auth_flow",   _coord((0.0, 0.0, 0.6, 0.0), embed_idx=10, mag=0.7)),
        ("login_path",  _coord((0.0, 0.0, 0.6, 0.0), embed_idx=12, mag=0.6)),
        ("billing_calc", _coord((0.0, 0.0, 0.0, 0.6), embed_idx=400, mag=0.7)),
    ]
    for label, c in seeds:
        svc.submit_observation(
            Observation(id=f"obs_{label}", coords=c, label=label),
            agent_id="seeder",
            sprout_under_charter=True,
            sprout_rootless=True,
        )


def _push_capability_to_peer(source: AutonetHost, target: AutonetHost) -> None:
    """Mimic what _handle_capability_stream does on the wire: source's
    capability bytes arrive at target and get cached."""
    cap_bytes = source._capability.to_bytes()
    received = NodeCapability.from_bytes(cap_bytes)
    target._known_capabilities[received.peer_id or source._capability.peer_id or "src"] = received


def _build_serving_daemon(
    tmp_path: Path,
    *,
    rpb: str,
    peer_id: str,
    renderer_text: str = "served by daemon B",
    price_atn: int = 1000,
    agent_address: str = "0xB0",
):
    """Construct (host, world_service, renderer) for a daemon that
    serves inference. Returns a tuple — caller is responsible for
    cleanup (svc.shutdown())."""
    host = AutonetHost(node_id=f"serving-{peer_id}", listen_host="127.0.0.1")
    # Spoof the peer_id so discovery has something to match on
    # (the real peer_id is set when libp2p starts up).
    host._capability.peer_id = peer_id

    svc = WorldService(rpb_address=rpb, data_root=tmp_path / peer_id)
    _seed_substrate(svc)
    renderer = CannedRenderer(response_text=renderer_text)
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    handler = make_substrate_inference_handler(provider)
    host.set_inference_handler(handler)
    host.set_inference_advertisement(
        renderer_model="qwen3:4b",
        price_atn=price_atn,
        agent_address=agent_address,
    )
    return host, svc, renderer


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_a_discovers_b_after_capability_shared(tmp_path: Path):
    """Daemon B advertises; A receives the capability; A's discovery
    surfaces B."""
    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b, svc_b, _renderer_b = _build_serving_daemon(
        tmp_path, rpb="rpb_75_a", peer_id="peer-B",
    )
    try:
        # Pre-condition: A has nothing in its capability cache.
        assert a.discover_inference_providers() == []

        _push_capability_to_peer(b, a)

        providers = a.discover_inference_providers()
        assert len(providers) == 1
        assert providers[0].peer_id == "peer-B"
        assert providers[0].inference["renderer_model"] == "qwen3:4b"
        assert providers[0].inference["agent_address"] == "0xB0"
    finally:
        svc_b.shutdown()


def test_a_picks_cheaper_provider_among_options(tmp_path: Path):
    """Multiple providers in cache → discover_inference_providers
    returns them with the cheaper one ranking higher (no measured
    latencies in this test)."""
    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b, svc_b, _ = _build_serving_daemon(
        tmp_path, rpb="rpb_75_b1", peer_id="peer-B",
        price_atn=2000,
    )
    c, svc_c, _ = _build_serving_daemon(
        tmp_path, rpb="rpb_75_b2", peer_id="peer-C",
        price_atn=500,
    )
    try:
        _push_capability_to_peer(b, a)
        _push_capability_to_peer(c, a)
        providers = a.discover_inference_providers()
        peer_ids = [p.peer_id for p in providers]
        # peer-C cheaper -> ranks first.
        assert peer_ids[0] == "peer-C"
        assert peer_ids[1] == "peer-B"
    finally:
        svc_b.shutdown()
        svc_c.shutdown()


def test_end_to_end_request_flows_through_substrate_to_response(tmp_path: Path):
    """The substantive integration test: A's request shape lands on
    B's substrate handler, which runs the substrate locate + render
    pipeline, and returns a wire-format response that A can
    consume."""
    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b, svc_b, renderer_b = _build_serving_daemon(
        tmp_path, rpb="rpb_75_e2e", peer_id="peer-B",
        renderer_text="auth uses the auth_flow node",
    )
    try:
        _push_capability_to_peer(b, a)

        # A picks B.
        providers = a.discover_inference_providers()
        assert len(providers) == 1
        chosen = providers[0]

        # A constructs the request shape that request_inference would
        # have serialized to bytes. Same dict shape that the substrate
        # handler accepts.
        request = {
            "messages": [{"role": "user", "content": "how does authentication work"}],
            "system": "",
            "model": chosen.inference["renderer_model"],
            "max_tokens": 256,
            "temperature": 0.0,
        }

        # In the real libp2p path: A.request_inference(b_peer_id, request)
        # → B._handle_inference_stream → B._serve_inference_locally(request).
        # We invoke the last leg directly.
        response = _run(b._serve_inference_locally(request))

        # No error path triggered.
        assert "error" not in response
        # Wire-format response.
        assert response["model"] == "qwen3:4b"
        assert response["text"] == "auth uses the auth_flow node"
        assert response["usage"]["input_tokens"] == 42

        # B's renderer received an augmented system prompt with the
        # substrate region (auth_flow / login_path labels).
        sys_prompt = renderer_b.last_call["system"]
        assert "auth_flow" in sys_prompt or "login_path" in sys_prompt
    finally:
        svc_b.shutdown()


def test_b_handler_failure_returns_error_to_a(tmp_path: Path):
    """If B's substrate handler crashes, A gets {"error": ...} back
    rather than a connection failure."""

    class FailingRenderer(Provider):
        @property
        def name(self) -> str: return "failing"
        async def send(self, **kwargs) -> ProviderResponse:
            raise RuntimeError("renderer offline")

    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b = AutonetHost(node_id="B", listen_host="127.0.0.1")
    b._capability.peer_id = "peer-B-broken"
    svc_b = WorldService(rpb_address="rpb_75_fail", data_root=tmp_path / "B")
    provider = SubstrateProvider(world_service=svc_b, renderer=FailingRenderer())
    b.set_inference_handler(make_substrate_inference_handler(provider))
    b.set_inference_advertisement(
        renderer_model="qwen3:4b", price_atn=0, agent_address="0xB0",
    )

    try:
        _push_capability_to_peer(b, a)
        providers = a.discover_inference_providers()
        assert len(providers) == 1

        request = {
            "messages": [{"role": "user", "content": "anything"}],
            "model": "qwen3:4b",
        }
        response = _run(b._serve_inference_locally(request))
        assert "error" in response
        assert "renderer offline" in response["error"].lower()
    finally:
        svc_b.shutdown()


def test_b_drops_out_of_discovery_after_clearing_advertisement(tmp_path: Path):
    """Advertising daemon stops serving → it's removed from A's
    discovery results after the next capability advertisement."""
    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b, svc_b, _ = _build_serving_daemon(
        tmp_path, rpb="rpb_75_drop", peer_id="peer-B",
    )
    try:
        _push_capability_to_peer(b, a)
        assert len(a.discover_inference_providers()) == 1

        # B stops advertising.
        b.clear_inference_advertisement()
        # The new (cleared) capability propagates to A.
        _push_capability_to_peer(b, a)

        assert a.discover_inference_providers() == []
    finally:
        svc_b.shutdown()


def test_request_response_round_trip_is_json_serializable(tmp_path: Path):
    """The wire format is JSON. A constructs a dict, B returns a dict.
    Both sides must be json.dumps/loads-clean for the bytes-level
    libp2p layer to carry them."""
    a = AutonetHost(node_id="A", listen_host="127.0.0.1")
    b, svc_b, _ = _build_serving_daemon(
        tmp_path, rpb="rpb_75_json", peer_id="peer-B",
    )
    try:
        _push_capability_to_peer(b, a)

        request = {
            "messages": [{"role": "user", "content": "json round trip"}],
            "model": "qwen3:4b",
            "max_tokens": 100,
        }
        # Round-trip the request through json.
        request_bytes = json.dumps(request).encode("utf-8")
        request_decoded = json.loads(request_bytes.decode("utf-8"))

        response = _run(b._serve_inference_locally(request_decoded))

        # Round-trip the response too.
        response_bytes = json.dumps(response).encode("utf-8")
        response_decoded = json.loads(response_bytes.decode("utf-8"))
        assert response_decoded["text"] == response["text"]
        assert response_decoded["usage"] == response["usage"]
    finally:
        svc_b.shutdown()


def test_capability_advertisement_to_bytes_carries_everything(tmp_path: Path):
    """The full capability — including the inference dict and the
    role list — survives a to_bytes/from_bytes round-trip, which
    is what the real libp2p CAPABILITY_PROTOCOL transport relies
    on."""
    b, svc_b, _ = _build_serving_daemon(
        tmp_path, rpb="rpb_75_bytes", peer_id="peer-B-bytes",
        price_atn=12345,
        agent_address="0xDEADBEEF",
    )
    try:
        cap_bytes = b._capability.to_bytes()
        rebuilt = NodeCapability.from_bytes(cap_bytes)
        assert "inference-provider" in rebuilt.roles
        assert rebuilt.inference["price_atn"] == 12345
        assert rebuilt.inference["agent_address"] == "0xDEADBEEF"
        assert rebuilt.inference["renderer_model"] == "qwen3:4b"
        assert rebuilt.inference["schema"] == 1
    finally:
        svc_b.shutdown()
