"""Phase 6.4: end-to-end substrate provider via the ProviderManager
with ollama as the renderer (mocked HTTP so no live ollama is needed).

Validates:
  1. ProviderManager registers "substrate" as a known provider.
  2. _build_substrate_provider returns a SubstrateProvider whose
     renderer is an OllamaProvider.
  3. The lazy world_service_resolver path works — set the resolver,
     build the provider, the resolver pulls the current world_service.
  4. End-to-end via mocked ollama: pre-seed the substrate, send a
     query through the SubstrateProvider, the mock ollama receives
     a system prompt containing the substrate region, and returns a
     response that the substrate provider passes back unchanged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest

from atn.providers.substrate import SubstrateProvider
from nodes.common.world_service import WorldService
from world_model.generalized import Observation


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _seed(svc: WorldService) -> None:
    seeds = [
        ("auth_flow", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=10, mag=0.7)),
        ("login_flow", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=12, mag=0.6)),
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
# ProviderManager wiring
# ---------------------------------------------------------------------------


def test_substrate_is_known_provider():
    from atn.runtime.provider_manager import ProviderManager
    assert "substrate" in ProviderManager._KNOWN_PROVIDERS
    info = ProviderManager._KNOWN_PROVIDERS["substrate"]
    assert info["auth_type"] == "local"
    # Substrate isn't yet orchestrator-capable (Phase 7+).
    assert info["orchestrator_capable"] is False


def test_provider_manager_builds_substrate_provider(tmp_path: Path):
    """Build SubstrateProvider directly via the manager's _resolve
    path with a stubbed config and direct world_service reference."""
    from atn.runtime.provider_manager import ProviderManager

    svc = WorldService(rpb_address="rpb_64_a", data_root=tmp_path)
    try:
        # Minimal stub for the manager: it needs ATNConfig.providers
        # (a dict-shaped attribute), credential_store, executors, events.
        # We build the bare minimum the substrate path uses.
        config = MagicMock()
        config.providers = {}
        manager = ProviderManager(
            config=config,
            credential_store=MagicMock(),
            executors={},
            events=MagicMock(),
        )
        manager._world_service = svc

        provider = manager._resolve_provider_by_name(
            "substrate", model="qwen3:4b", agent_id="test-agent",
        )
        assert isinstance(provider, SubstrateProvider)
        assert provider.name == "substrate"
        # Renderer is an OllamaProvider (named "ollama").
        assert provider.renderer.name == "ollama"
    finally:
        svc.shutdown()


def test_world_service_resolver_takes_precedence(tmp_path: Path):
    """The resolver wins over the direct reference — important
    because autonet's WorldService isn't constructed until inside
    AutonetService.start()."""
    from atn.runtime.provider_manager import ProviderManager

    svc1 = WorldService(rpb_address="rpb_64_b1", data_root=tmp_path / "a")
    svc2 = WorldService(rpb_address="rpb_64_b2", data_root=tmp_path / "b")
    try:
        config = MagicMock()
        config.providers = {}
        manager = ProviderManager(
            config=config,
            credential_store=MagicMock(),
            executors={},
            events=MagicMock(),
        )
        # Set both: direct points at svc1, resolver at svc2.
        # Resolver should win.
        manager._world_service = svc1
        manager._world_service_resolver = lambda: svc2

        provider = manager._resolve_provider_by_name(
            "substrate", model="x", agent_id="t",
        )
        # Provider's WorldService is svc2 (resolver), not svc1.
        assert provider._world_service is svc2
    finally:
        svc1.shutdown()
        svc2.shutdown()


def test_substrate_provider_raises_if_no_world_service():
    """If neither direct nor resolver is set, building substrate
    must fail with a clear error rather than crashing later."""
    from atn.providers.base import ProviderError
    from atn.runtime.provider_manager import ProviderManager

    config = MagicMock()
    config.providers = {}
    manager = ProviderManager(
        config=config,
        credential_store=MagicMock(),
        executors={},
        events=MagicMock(),
    )
    with pytest.raises(ProviderError, match="WorldService"):
        manager._resolve_provider_by_name(
            "substrate", model="x", agent_id="t",
        )


# ---------------------------------------------------------------------------
# End-to-end via mocked ollama
# ---------------------------------------------------------------------------


def _make_mock_ollama_handler(captured: Dict[str, Any]):
    """Returns an httpx MockTransport handler that records the request
    and returns a canned ollama response."""
    def handler(request: httpx.Request) -> httpx.Response:
        # Capture the body so the test can inspect what the ollama
        # provider sent.
        body = json.loads(request.content.decode("utf-8"))
        captured["request_body"] = body
        # Return a canned ollama-shaped response.
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "Authentication uses the auth_flow node we located.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        )
    return handler


def test_substrate_via_ollama_end_to_end(tmp_path: Path, monkeypatch):
    """Pre-seed substrate → substrate provider → mocked ollama →
    response. Verifies the integration path without a live ollama."""
    svc = WorldService(rpb_address="rpb_64_e2e", data_root=tmp_path)
    _seed(svc)

    captured: Dict[str, Any] = {}

    # Patch the ollama provider's httpx.AsyncClient to use a mock
    # transport. We do this by replacing the AsyncClient constructor
    # at the module level for the duration of this test.
    original_client = httpx.AsyncClient

    def mock_client(*args, **kwargs):
        return original_client(
            transport=httpx.MockTransport(_make_mock_ollama_handler(captured)),
            timeout=kwargs.get("timeout", 30),
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)

    try:
        from atn.providers.ollama import OllamaProvider
        renderer = OllamaProvider(
            base_url="http://localhost:11434",
            default_model="qwen3:4b",
        )
        provider = SubstrateProvider(
            world_service=svc,
            renderer=renderer,
            renderer_model="qwen3:4b",
        )

        response = _run(provider.send(
            messages=[{"role": "user", "content": "how does authentication work"}],
        ))

        # The mock ollama returned the canned response.
        assert "auth_flow" in response.text or "authentication" in response.text.lower()
        # The mock ollama received a request whose system prompt
        # contains substrate region content.
        body = captured["request_body"]
        sys_msg = next((m for m in body["messages"] if m["role"] == "system"), None)
        assert sys_msg is not None
        assert "auth_flow" in sys_msg["content"] or "login_flow" in sys_msg["content"]
        # And the user's message was passed through.
        user_msg = next(m for m in body["messages"] if m["role"] == "user")
        assert "authentication" in user_msg["content"]
        # Model id passed through to ollama.
        assert body["model"] == "qwen3:4b"
    finally:
        svc.shutdown()
