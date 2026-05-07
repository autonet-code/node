"""Phase 6.3: SubstrateProvider — composite substrate locate + LLM render.

Tests use a mock renderer Provider so the substrate orchestration
logic is exercised without an actual LLM call. Phase 6.4 swaps in
the real ollama renderer.

What's tested:

  1. The provider has the correct ``name`` ("substrate") and
     conforms to the ``Provider`` ABC.
  2. send() extracts the most recent user message, probes the
     substrate, calls the renderer with an augmented system prompt
     containing the region.
  3. The substrate region appears in the renderer's system prompt
     in a format the LLM can consume.
  4. An empty world produces an empty region; the renderer is still
     called (with a "(no relevant nodes found)" placeholder) and the
     query falls back to general LLM knowledge.
  5. A pre-existing system prompt wraps (precedes) the substrate
     context block.
  6. The renderer's response is returned unchanged.
  7. Anthropic-style content blocks (list of {type: text, text: ...})
     are correctly extracted as the query.
"""

from __future__ import annotations

import asyncio
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
from nodes.common.world_service import WorldService
from world_model.generalized import Observation


# ---------------------------------------------------------------------------
# Mock renderer
# ---------------------------------------------------------------------------


class MockRenderer(Provider):
    """Records calls; returns a canned response."""

    def __init__(self, response_text: str = "ok"):
        self._response_text = response_text
        self.last_call: Optional[Dict[str, Any]] = None
        self.call_count: int = 0

    @property
    def name(self) -> str:
        return "mock"

    async def send(
        self, *,
        messages: List[Dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.call_count += 1
        self.last_call = {
            "messages": messages,
            "system": system,
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "temperature": temperature,
        }
        return ProviderResponse(
            text=self._response_text,
            tool_calls=[],
            stop_reason="end_turn",
            usage=Usage(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _seed(svc: WorldService) -> None:
    """Seed the world with a couple of usefulness regions."""
    seeds = [
        ("auth_flow", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=10, mag=0.7)),
        ("login_path", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=12, mag=0.6)),
        ("billing_calc", _coord((0.0, 0.0, 0.0, 0.6), embed_idx=400, mag=0.7)),
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
# Construction & ABC conformance
# ---------------------------------------------------------------------------


def test_provider_has_correct_name(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_sp_a", data_root=tmp_path)
    try:
        provider = SubstrateProvider(
            world_service=svc, renderer=MockRenderer(),
        )
        assert provider.name == "substrate"
    finally:
        svc.shutdown()


def test_construction_requires_world_service():
    with pytest.raises(ValueError):
        SubstrateProvider(world_service=None, renderer=MockRenderer())


def test_construction_requires_renderer(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_sp_b", data_root=tmp_path)
    try:
        with pytest.raises(ValueError):
            SubstrateProvider(world_service=svc, renderer=None)
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# send() orchestration
# ---------------------------------------------------------------------------


def test_send_probes_substrate_and_calls_renderer(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_sp_c", data_root=tmp_path)
    _seed(svc)
    renderer = MockRenderer(response_text="hello from mock LLM")
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        response = _run(provider.send(
            messages=[{"role": "user", "content": "how does authentication work"}],
        ))
        assert response.text == "hello from mock LLM"
        assert renderer.call_count == 1
        assert renderer.last_call is not None
        # The renderer was given a system prompt that includes
        # something from the substrate region.
        sys_prompt = renderer.last_call["system"]
        assert "substrate" in sys_prompt.lower()
        # The query string is in the system prompt's # Query block.
        assert "how does authentication work" in sys_prompt
        # The user's original message went through unchanged.
        assert renderer.last_call["messages"][0]["content"] == "how does authentication work"
    finally:
        svc.shutdown()


def test_send_includes_region_labels_in_system_prompt(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_sp_d", data_root=tmp_path)
    _seed(svc)
    renderer = MockRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        # A query close to the auth seeds should pull them up.
        _run(provider.send(
            messages=[{"role": "user", "content": "authentication and login flow"}],
        ))
        sys_prompt = renderer.last_call["system"]
        # At least one of the auth-related labels should appear.
        # (We don't assert on which ranks first — that's the
        # locator's job and tested in 6.1.)
        labels_in_prompt = sum(
            1 for label in ("auth_flow", "login_path", "billing_calc")
            if label in sys_prompt
        )
        assert labels_in_prompt > 0, sys_prompt
    finally:
        svc.shutdown()


def test_send_with_empty_world_falls_back_gracefully(tmp_path: Path):
    """Probing a charter-only world produces only charter root nodes.
    The renderer is still called; the system prompt indicates the
    region is sparse but valid."""
    svc = WorldService(rpb_address="rpb_sp_e", data_root=tmp_path)
    renderer = MockRenderer(response_text="no substrate context, falling back")
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        response = _run(provider.send(
            messages=[{"role": "user", "content": "tell me a joke"}],
        ))
        assert response.text == "no substrate context, falling back"
        assert renderer.call_count == 1
        # Renderer was called even with an empty world.
    finally:
        svc.shutdown()


def test_send_preserves_pre_existing_system_prompt(tmp_path: Path):
    """Caller-supplied system prompt comes first; substrate context
    appended after."""
    svc = WorldService(rpb_address="rpb_sp_f", data_root=tmp_path)
    _seed(svc)
    renderer = MockRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        _run(provider.send(
            messages=[{"role": "user", "content": "hi"}],
            system="You are a helpful assistant.",
        ))
        sys_prompt = renderer.last_call["system"]
        # The pre-existing system prompt is preserved at the start.
        assert sys_prompt.startswith("You are a helpful assistant.")
        # And the substrate block is appended.
        assert "substrate" in sys_prompt.lower()
    finally:
        svc.shutdown()


def test_send_passes_through_temperature_and_max_tokens(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_sp_g", data_root=tmp_path)
    renderer = MockRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        _run(provider.send(
            messages=[{"role": "user", "content": "x"}],
            max_tokens=2048,
            temperature=0.7,
        ))
        assert renderer.last_call["max_tokens"] == 2048
        assert renderer.last_call["temperature"] == 0.7
    finally:
        svc.shutdown()


def test_renderer_model_override(tmp_path: Path):
    """Constructor-level renderer_model is used when send() doesn't
    pass an override."""
    svc = WorldService(rpb_address="rpb_sp_h", data_root=tmp_path)
    renderer = MockRenderer()
    provider = SubstrateProvider(
        world_service=svc, renderer=renderer,
        renderer_model="qwen3:4b",
    )

    try:
        _run(provider.send(
            messages=[{"role": "user", "content": "x"}],
        ))
        assert renderer.last_call["model"] == "qwen3:4b"
    finally:
        svc.shutdown()


def test_send_handles_anthropic_content_blocks(tmp_path: Path):
    """ATN's runtime sometimes uses list-of-dict content blocks
    (Anthropic format). The substrate provider should extract the
    text from those for query purposes."""
    svc = WorldService(rpb_address="rpb_sp_i", data_root=tmp_path)
    _seed(svc)
    renderer = MockRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        _run(provider.send(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "explain authentication"},
                ],
            }],
        ))
        sys_prompt = renderer.last_call["system"]
        assert "explain authentication" in sys_prompt
    finally:
        svc.shutdown()


def test_send_uses_most_recent_user_message_as_query(tmp_path: Path):
    """In a multi-turn conversation, only the most recent user
    message becomes the substrate query — earlier messages are
    history, not the active question."""
    svc = WorldService(rpb_address="rpb_sp_j", data_root=tmp_path)
    _seed(svc)
    renderer = MockRenderer()
    provider = SubstrateProvider(world_service=svc, renderer=renderer)

    try:
        _run(provider.send(
            messages=[
                {"role": "user", "content": "earlier irrelevant question"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "now: how does login work"},
            ],
        ))
        sys_prompt = renderer.last_call["system"]
        # The latest query is what got embedded.
        assert "now: how does login work" in sys_prompt
        # The earlier query did NOT become the embedded query.
        # We can't assert it's "absent" from the prompt because the
        # system prompt format does include the query verbatim;
        # the assertion above is sufficient — only ONE query block
        # exists, and it's the latest.
        assert sys_prompt.count("# Query") == 1
    finally:
        svc.shutdown()
