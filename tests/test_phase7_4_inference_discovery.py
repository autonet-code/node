"""Phase 7.4: inference capability advertisement + discovery.

What's tested at the contract/method level:

  1. set_inference_advertisement marks the host as an inference
     provider, fills the inference dict, is idempotent on roles.
  2. clear_inference_advertisement reverses it cleanly.
  3. discover_inference_providers filters known_capabilities by
     role, by max_price_atn, by renderer_model, and rejects
     advertisements without an agent_address by default.
  4. Result ordering: peers with measured latency come before
     peers without, and within each group cheaper price wins.
  5. The advertise_capability transport carries the new inference
     dict in the wire form (NodeCapability.to_bytes / from_bytes
     round-trips).
  6. The deprecated advertise_models() shim wires the old signature
     into the new path without breaking callers.

Multi-host libp2p is intentionally out of scope (pytest-trio
collection is dormant in this project's pytest config). The
discovery method operates on the local known_capabilities cache,
so we populate it directly to test filtering — same observable
behavior as a real peer publishing its capability over the
CAPABILITY_PROTOCOL.
"""

from __future__ import annotations

from typing import List

import pytest

from nodes.common.p2p import AutonetHost, NodeCapability


def _populate_capability(
    host: AutonetHost,
    *,
    peer_id: str,
    roles: list,
    inference: dict,
    latency_ms: float = None,
) -> None:
    """Drop a NodeCapability into the host's known_capabilities cache
    as if it had arrived over the CAPABILITY_PROTOCOL. Optionally seed
    a latency reading.

    The host's _latency_tracker is normally created inside run(); for
    tests we create a tracker on first use and inject readings
    directly into its peer dict to bypass the trio-based ping path."""
    cap = NodeCapability(
        peer_id=peer_id,
        node_id=f"node-{peer_id[:8]}",
        roles=list(roles),
        inference=dict(inference),
    )
    host._known_capabilities[peer_id] = cap
    if latency_ms is not None:
        from nodes.common.p2p import PeerLatency, PeerLatencyTracker
        if host._latency_tracker is None:
            host._latency_tracker = PeerLatencyTracker(host=None)
        host._latency_tracker._peers[peer_id] = PeerLatency(
            peer_id=peer_id,
            ema_rtt_ms=float(latency_ms),
            samples=1,
            reachable=True,
        )


# ---------------------------------------------------------------------------
# set_inference_advertisement
# ---------------------------------------------------------------------------


def test_set_inference_advertisement_adds_role_and_dict():
    h = AutonetHost(node_id="serve-1", listen_host="127.0.0.1")
    assert "inference-provider" not in h._capability.roles
    assert h._capability.inference == {}

    h.set_inference_advertisement(
        renderer_model="qwen3:4b",
        price_atn=1000,
        agent_address="0xa0",
    )

    assert "inference-provider" in h._capability.roles
    inf = h._capability.inference
    assert inf["renderer_model"] == "qwen3:4b"
    assert inf["price_atn"] == 1000
    assert inf["agent_address"] == "0xa0"
    assert inf["schema"] == 1


def test_set_inference_advertisement_is_idempotent_on_roles():
    """Calling twice doesn't duplicate the role."""
    h = AutonetHost(node_id="serve-2", listen_host="127.0.0.1")
    h.set_inference_advertisement(
        renderer_model="m", price_atn=10, agent_address="0xa",
    )
    h.set_inference_advertisement(
        renderer_model="m2", price_atn=20, agent_address="0xb",
    )
    assert h._capability.roles.count("inference-provider") == 1
    # Second call replaces the dict.
    assert h._capability.inference["renderer_model"] == "m2"
    assert h._capability.inference["price_atn"] == 20


def test_set_inference_advertisement_extras_are_merged():
    h = AutonetHost(node_id="serve-3", listen_host="127.0.0.1")
    h.set_inference_advertisement(
        renderer_model="m", price_atn=5, agent_address="0xc",
        extras={"region": "us-east", "tier": "free"},
    )
    inf = h._capability.inference
    assert inf["region"] == "us-east"
    assert inf["tier"] == "free"
    # Standard fields not clobbered.
    assert inf["renderer_model"] == "m"


def test_clear_inference_advertisement_removes_role_and_dict():
    h = AutonetHost(node_id="serve-4", listen_host="127.0.0.1")
    h.set_inference_advertisement(
        renderer_model="m", price_atn=1, agent_address="0xd",
    )
    h.clear_inference_advertisement()
    assert "inference-provider" not in h._capability.roles
    assert h._capability.inference == {}


def test_clear_inference_advertisement_preserves_other_roles():
    h = AutonetHost(node_id="serve-5", listen_host="127.0.0.1")
    h._capability.roles.append("solver")
    h.set_inference_advertisement(
        renderer_model="m", price_atn=1, agent_address="0xe",
    )
    h.clear_inference_advertisement()
    assert h._capability.roles == ["solver"]


# ---------------------------------------------------------------------------
# discover_inference_providers
# ---------------------------------------------------------------------------


def test_discover_returns_only_inference_providers():
    h = AutonetHost(node_id="seek-1", listen_host="127.0.0.1")
    _populate_capability(
        h, peer_id="peer-A",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 100, "agent_address": "0xa"},
    )
    _populate_capability(
        h, peer_id="peer-B",
        roles=["solver"],  # not an inference provider
        inference={"renderer_model": "m", "price_atn": 50, "agent_address": "0xb"},
    )
    _populate_capability(
        h, peer_id="peer-C",
        roles=["inference-provider"],
        inference={},  # no inference dict — should be filtered
    )

    result = h.discover_inference_providers()
    peer_ids = [c.peer_id for c in result]
    assert "peer-A" in peer_ids
    assert "peer-B" not in peer_ids
    assert "peer-C" not in peer_ids


def test_discover_filters_by_max_price():
    h = AutonetHost(node_id="seek-2", listen_host="127.0.0.1")
    _populate_capability(
        h, peer_id="cheap",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 50, "agent_address": "0xa"},
    )
    _populate_capability(
        h, peer_id="expensive",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 5_000, "agent_address": "0xb"},
    )

    result = h.discover_inference_providers(max_price_atn=1_000)
    peer_ids = [c.peer_id for c in result]
    assert "cheap" in peer_ids
    assert "expensive" not in peer_ids


def test_discover_filters_by_renderer_model():
    h = AutonetHost(node_id="seek-3", listen_host="127.0.0.1")
    _populate_capability(
        h, peer_id="qwen-host",
        roles=["inference-provider"],
        inference={"renderer_model": "qwen3:4b", "price_atn": 100, "agent_address": "0xa"},
    )
    _populate_capability(
        h, peer_id="phi-host",
        roles=["inference-provider"],
        inference={"renderer_model": "phi-3-mini", "price_atn": 100, "agent_address": "0xb"},
    )

    result = h.discover_inference_providers(renderer_model="qwen3:4b")
    peer_ids = [c.peer_id for c in result]
    assert peer_ids == ["qwen-host"]


def test_discover_requires_agent_address_by_default():
    h = AutonetHost(node_id="seek-4", listen_host="127.0.0.1")
    _populate_capability(
        h, peer_id="anon-provider",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 50, "agent_address": ""},
    )
    # Default: filtered out.
    assert h.discover_inference_providers() == []
    # Opt-in to allow: included.
    result = h.discover_inference_providers(require_agent_address=False)
    assert len(result) == 1


def test_discover_orders_by_latency_then_price():
    h = AutonetHost(node_id="seek-5", listen_host="127.0.0.1")
    # Three providers: same renderer, different latencies and prices.
    _populate_capability(
        h, peer_id="fast-cheap",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 20, "agent_address": "0x1"},
        latency_ms=10.0,
    )
    _populate_capability(
        h, peer_id="fast-expensive",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 200, "agent_address": "0x2"},
        latency_ms=15.0,
    )
    _populate_capability(
        h, peer_id="unmeasured-cheap",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 5, "agent_address": "0x3"},
        # no latency reading
    )
    _populate_capability(
        h, peer_id="unmeasured-expensive",
        roles=["inference-provider"],
        inference={"renderer_model": "m", "price_atn": 500, "agent_address": "0x4"},
    )

    result = h.discover_inference_providers()
    peer_ids = [c.peer_id for c in result]
    # Measured peers come first, ordered by latency.
    assert peer_ids[0] == "fast-cheap"
    assert peer_ids[1] == "fast-expensive"
    # Unmeasured peers come after, ordered by price.
    assert peer_ids[2] == "unmeasured-cheap"
    assert peer_ids[3] == "unmeasured-expensive"


def test_discover_empty_when_no_providers():
    h = AutonetHost(node_id="seek-6", listen_host="127.0.0.1")
    assert h.discover_inference_providers() == []


# ---------------------------------------------------------------------------
# Wire-format round-trip
# ---------------------------------------------------------------------------


def test_inference_field_round_trips_via_capability_bytes():
    """NodeCapability.to_bytes / from_bytes preserves the inference
    dict — required for the existing CAPABILITY_PROTOCOL transport
    to carry the new field unchanged."""
    cap = NodeCapability(
        peer_id="p1", node_id="n1",
        roles=["inference-provider"],
        inference={
            "schema": 1,
            "renderer_model": "qwen3:4b",
            "price_atn": 12345,
            "agent_address": "0xCAFE",
        },
    )
    rebuilt = NodeCapability.from_bytes(cap.to_bytes())
    assert rebuilt.peer_id == "p1"
    assert rebuilt.roles == ["inference-provider"]
    assert rebuilt.inference["renderer_model"] == "qwen3:4b"
    assert rebuilt.inference["price_atn"] == 12345
    assert rebuilt.inference["agent_address"] == "0xCAFE"


# ---------------------------------------------------------------------------
# Deprecated shim
# ---------------------------------------------------------------------------


def test_advertise_models_shim_wires_into_new_path():
    """The deprecated advertise_models(models, agent_address) API
    routes into set_inference_advertisement so old callers continue
    to work."""
    import asyncio
    h = AutonetHost(node_id="shim", listen_host="127.0.0.1")
    # advertise_capability tries to push to peers; when _host is None
    # it returns early without raising. So we can run the coroutine.
    asyncio.get_event_loop().run_until_complete(
        h.advertise_models(models=["qwen3:4b", "phi-3-mini"], agent_address="0xabc")
    )
    assert "inference-provider" in h._capability.roles
    inf = h._capability.inference
    assert inf["renderer_model"] == "qwen3:4b"
    assert inf["agent_address"] == "0xabc"
    # Old shim doesn't carry price; defaults to 0.
    assert inf["price_atn"] == 0
