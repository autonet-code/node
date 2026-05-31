"""Phase 8: sponsor → dependent inference, end-to-end across the provider seam.

The libp2p transport hop (/rpb/inference/1.0.0) is already covered by the
Phase 7.5 e2e test. This test exercises everything Phase 8 added *around* that
hop, with a fake p2p host standing in for the wire:

  dependent RPBNetworkProvider.send()
    → builds request carrying the dependent's own agent_address
    → (fake transport) → real sponsor handler
    → handler authorizes against the binding store, serves via a fake LLM
       provider, meters tokens, returns remaining budget
    → provider parses response, tracks remaining budget

Covers: identity on the wire, sponsor targeting by address, authorize/reject,
metering, exhaustion, and the single-thread (sub-agent shares grant) model.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atn.sponsor_bindings import SponsorBindingStore
from atn.providers.base import ProviderResponse, Usage
from atn.providers.rpb import RPBNetworkProvider
from atn.autonet_service import AutonetBridge
from atn.config import RPBConfig


SPONSOR_ADDRESS = "0x1111111111111111111111111111111111111111"
DEPENDENT_ADDRESS = "0x2222222222222222222222222222222222222222"


class _FakeLLM:
    """Stands in for the sponsor's real provider (Anthropic/etc)."""
    def __init__(self, in_tok=20, out_tok=10):
        self._in, self._out = in_tok, out_tok

    async def send(self, **kwargs):
        return ProviderResponse(
            text="sponsored answer",
            tool_calls=None,
            usage=Usage(input_tokens=self._in, output_tokens=self._out),
            model=kwargs.get("model", "test-model"),
            stop_reason="end_turn",
        )


class _Runtime:
    def __init__(self, bindings):
        self.sponsor_bindings = bindings


class _FakeAd(dict):
    """Agent advertisement; dict so both attr and .get access work like the
    real per-agent ad dicts in capability gossip."""


class _FakeCapability:
    def __init__(self, agents):
        self.agents = agents


class _FakeP2PHost:
    """Routes request_inference straight into the sponsor handler, and
    advertises the sponsor in _known_capabilities for discovery."""
    def __init__(self, sponsor_handler, *, address, model):
        self._handler = sponsor_handler
        self._latency_tracker = None
        self._known_capabilities = {
            "peerSPONSOR": _FakeCapability(agents=[
                _FakeAd(address=address, model=model, is_sponsor=True)
            ])
        }

    async def request_inference(self, peer_id: str, request: dict) -> dict:
        # Mirror the real AutonetHost.request_inference contract: it raises
        # RuntimeError when the serving peer returns an error dict.
        assert peer_id == "peerSPONSOR"
        response = await self._handler(request)
        if "error" in response:
            raise RuntimeError(f"Provider error: {response['error']}")
        return response


def _build_sponsor(tmp_path: Path, llm) -> tuple[AutonetBridge, SponsorBindingStore]:
    store = SponsorBindingStore(tmp_path)
    bridge = AutonetBridge(RPBConfig())
    bridge._runtime = _Runtime(store)
    bridge._resolve_sponsor_provider = lambda rpb_cfg, model: llm
    bridge.config.sponsor_model = "test-model"
    return bridge, store


def _build_dependent(host) -> RPBNetworkProvider:
    return RPBNetworkProvider(
        agent_id="dep01",
        agent_address=DEPENDENT_ADDRESS,
        p2p_host=host,
        model="test-model",
        sponsor_address=SPONSOR_ADDRESS,
    )


def test_bound_dependent_routes_serves_meters(tmp_path: Path):
    bridge, store = _build_sponsor(tmp_path, _FakeLLM(in_tok=20, out_tok=10))
    store.add(DEPENDENT_ADDRESS, budget_tokens=1000, label="employee alice")
    handler = bridge._create_sponsor_handler(bridge.config)
    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="test-model")
    dep = _build_dependent(host)

    resp = asyncio.run(dep.send(messages=[{"role": "user", "content": "hi"}], model="test-model"))

    assert resp.text == "sponsored answer"
    # 20 + 10 metered → 970 remaining
    assert store.remaining(DEPENDENT_ADDRESS) == 970
    assert dep._remaining_budget_tokens == 970


def test_unbound_dependent_rejected(tmp_path: Path):
    bridge, store = _build_sponsor(tmp_path, _FakeLLM())
    # No binding added for DEPENDENT_ADDRESS.
    handler = bridge._create_sponsor_handler(bridge.config)
    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="test-model")
    dep = _build_dependent(host)

    with pytest.raises(RuntimeError, match="not an authorized dependent"):
        asyncio.run(dep.send(messages=[{"role": "user", "content": "hi"}], model="test-model"))


def test_budget_exhaustion_across_calls(tmp_path: Path):
    bridge, store = _build_sponsor(tmp_path, _FakeLLM(in_tok=20, out_tok=10))  # 30/call
    store.add(DEPENDENT_ADDRESS, budget_tokens=50)  # enough for exactly one call
    handler = bridge._create_sponsor_handler(bridge.config)
    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="test-model")
    dep = _build_dependent(host)

    # First call: 30 spent, 20 remaining — succeeds.
    resp1 = asyncio.run(dep.send(messages=[{"role": "user", "content": "1"}], model="test-model"))
    assert resp1.text == "sponsored answer"
    assert store.remaining(DEPENDENT_ADDRESS) == 20

    # Second call: 20 remaining > 0, so it is admitted and served (60 spent
    # total, clamps to 0). The grant is now exhausted.
    resp2 = asyncio.run(dep.send(messages=[{"role": "user", "content": "2"}], model="test-model"))
    assert resp2.text == "sponsored answer"
    assert store.remaining(DEPENDENT_ADDRESS) == 0

    # Third call: remaining == 0 → rejected before serving.
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(dep.send(messages=[{"role": "user", "content": "3"}], model="test-model"))


def test_sponsor_targeting_ignores_other_sponsors(tmp_path: Path):
    """A dependent that names SPONSOR_ADDRESS must not route to a different
    is_sponsor peer advertising the same model."""
    bridge, store = _build_sponsor(tmp_path, _FakeLLM())
    store.add(DEPENDENT_ADDRESS, budget_tokens=1000)
    handler = bridge._create_sponsor_handler(bridge.config)

    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="test-model")
    # Add a DIFFERENT sponsor advertising the same model.
    host._known_capabilities["peerOTHER"] = _FakeCapability(agents=[
        _FakeAd(address="0x9999999999999999999999999999999999999999",
                model="test-model", is_sponsor=True)
    ])
    dep = _build_dependent(host)

    providers = asyncio.run(dep.discover_providers("test-model"))
    # Only the named sponsor qualifies.
    assert len(providers) == 1
    assert providers[0]["address"] == SPONSOR_ADDRESS.lower()


def test_named_sponsor_matches_regardless_of_model(tmp_path: Path):
    """When a dependent names a sponsor, the employer's model wins — the
    sponsor qualifies even if the dependent requests a different model."""
    bridge, store = _build_sponsor(tmp_path, _FakeLLM())
    store.add(DEPENDENT_ADDRESS, budget_tokens=1000)
    handler = bridge._create_sponsor_handler(bridge.config)
    # Sponsor advertises a different model than the dependent asks for.
    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="sponsor-only-model")
    dep = _build_dependent(host)  # dep._model == "test-model"

    providers = asyncio.run(dep.discover_providers("test-model"))
    assert len(providers) == 1
    assert providers[0]["address"] == SPONSOR_ADDRESS.lower()


def test_single_thread_subagent_shares_grant(tmp_path: Path):
    """A sub-agent of the dependent presents the SAME dependent address and
    draws on the same grant — the sponsor sees one thread, not a tree."""
    bridge, store = _build_sponsor(tmp_path, _FakeLLM(in_tok=20, out_tok=10))
    store.add(DEPENDENT_ADDRESS, budget_tokens=1000)
    handler = bridge._create_sponsor_handler(bridge.config)
    host = _FakeP2PHost(handler, address=SPONSOR_ADDRESS, model="test-model")

    # Two providers — the root dependent and a "sub-agent" — both carry the
    # same agent_address (the dependent root's), per the single-thread model.
    root = _build_dependent(host)
    sub = RPBNetworkProvider(
        agent_id="dep01.sub",
        agent_address=DEPENDENT_ADDRESS,  # same identity
        p2p_host=host,
        model="test-model",
        sponsor_address=SPONSOR_ADDRESS,
    )

    asyncio.run(root.send(messages=[{"role": "user", "content": "root"}], model="test-model"))
    asyncio.run(sub.send(messages=[{"role": "user", "content": "sub"}], model="test-model"))

    # Both calls drew on the one grant: 2 * 30 = 60 spent → 940 remaining.
    assert store.remaining(DEPENDENT_ADDRESS) == 940
