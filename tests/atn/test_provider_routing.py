"""§10 Provider routing (fail-loud) + update_agent eviction.

- Routing is explicit (decision 2026-10-06): an agent names its provider and
  model, a keyless gemini/openai pin raises instead of falling back onto the
  bridge, and an agent that names no provider (or a model id as its
  provider) raises rather than being routed by model prefix.
- update_agent must evict + close the cached provider when provider/model changes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atn.providers.base import ProviderError
from atn.providers.ollama import OllamaProvider
from atn.runtime.provider_manager import ProviderManager


def _make_manager() -> ProviderManager:
    config = MagicMock()
    config.providers = {}
    mgr = ProviderManager(
        config=config,
        credential_store=MagicMock(),
        executors={},
        events=MagicMock(),
    )
    # No API keys configured by default.
    mgr._resolve_api_key = MagicMock(return_value="")
    return mgr


# ---------------------------------------------------------------------------
# Fail-loud routing
# ---------------------------------------------------------------------------

def _defn(provider, model):
    from atn.models import AgentDefinition, AgentMode
    return AgentDefinition(id="agent-1", name="A", mode=AgentMode.COGNITIVE,
                           provider=provider, cognitive_model=model)


class TestFailLoudRouting:
    def test_keyless_gemini_raises(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="Gemini API key"):
            mgr.resolve_provider_with_fallback(_defn("gemini", "gemini-3-pro"))

    def test_keyless_openai_raises(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="OpenAI API key"):
            mgr.resolve_provider_with_fallback(_defn("openai", "gpt-5.5"))

    def test_no_provider_raises_instead_of_routing_by_model(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="provider is required"):
            mgr.resolve_provider_with_fallback(_defn("", "qwen3.5:4b"))

    def test_model_id_as_provider_raises(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="unknown provider"):
            mgr.resolve_provider_with_fallback(
                _defn("claude-fable-5", "claude-haiku-4-5"))

    def test_pinned_ollama_routes_to_ollama(self):
        mgr = _make_manager()
        prov = mgr.resolve_provider_with_fallback(_defn("ollama", "qwen3.5:4b"))
        assert isinstance(prov, OllamaProvider)


# ---------------------------------------------------------------------------
# update_agent eviction
# ---------------------------------------------------------------------------

class _FakeProvider:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
class TestUpdateAgentEviction:
    async def _make_runtime_with_agent(self, agent_id="a1"):
        from atn.models import AgentDefinition, AgentMode

        runtime = MagicMock()
        defn = AgentDefinition(
            id=agent_id, name="A", mode=AgentMode.COGNITIVE,
            provider="claude_max", cognitive_model="claude-sonnet-4-6",
        )
        runtime.get_agent = MagicMock(return_value=defn)
        runtime._config.agents_dir = "/tmp/agents"
        # providers registry with a cached active provider
        pmgr = MagicMock()
        pmgr._active_providers = {}
        pmgr._cached_session_stats = {}
        runtime.providers = pmgr
        return runtime, defn, pmgr

    async def test_model_change_evicts_and_closes(self, monkeypatch):
        from atn import agent_tools as tools

        monkeypatch.setattr(tools, "save_agent", lambda *a, **k: None)
        runtime, defn, pmgr = await self._make_runtime_with_agent()
        fake = _FakeProvider()
        pmgr._active_providers["a1"] = fake
        pmgr._cached_session_stats["a1"] = {"x": 1}

        res = await tools._update_agent(runtime, {"agent_id": "a1", "model": "gpt-5.5"})

        assert res["status"] == "updated"
        assert "model" in res["changed"]
        assert fake.closed is True
        assert "a1" not in pmgr._active_providers
        assert "a1" not in pmgr._cached_session_stats

    async def test_provider_change_evicts(self, monkeypatch):
        from atn import agent_tools as tools

        monkeypatch.setattr(tools, "save_agent", lambda *a, **k: None)
        runtime, defn, pmgr = await self._make_runtime_with_agent()
        fake = _FakeProvider()
        pmgr._active_providers["a1"] = fake

        res = await tools._update_agent(runtime, {"agent_id": "a1", "provider": "ollama"})

        assert "provider" in res["changed"]
        assert fake.closed is True
        assert "a1" not in pmgr._active_providers

    async def test_non_provider_change_does_not_evict(self, monkeypatch):
        from atn import agent_tools as tools

        monkeypatch.setattr(tools, "save_agent", lambda *a, **k: None)
        runtime, defn, pmgr = await self._make_runtime_with_agent()
        fake = _FakeProvider()
        pmgr._active_providers["a1"] = fake

        res = await tools._update_agent(runtime, {"agent_id": "a1", "name": "renamed"})

        assert "name" in res["changed"]
        # Untouched — a name change must not tear down the live provider.
        assert fake.closed is False
        assert pmgr._active_providers.get("a1") is fake


# ---------------------------------------------------------------------------
# create_agent provider field
# ---------------------------------------------------------------------------

class TestCreateAgentProviderSchema:
    def test_schema_provider_is_open_string(self):
        """No enum: custom provider ids from config.yaml are valid too
        (fbaa84a). The built-ins are named in the description."""
        from atn.agent_tools import _TOOLS

        create = next(t for t in _TOOLS if t.name == "create_agent")
        props = create.input_schema["properties"]
        assert "provider" in props
        assert props["provider"]["type"] == "string"
        assert "enum" not in props["provider"]
        assert "ollama" in props["provider"]["description"]
        assert "rpb" in props["provider"]["description"]


# ---------------------------------------------------------------------------
# Sponsored inference: the dependent identity is the OWNER WALLET
# (ratified 2026-07-25, docs/sponsored_inference.md)
# ---------------------------------------------------------------------------

def _rpb_manager(*, owner_wallet: str, sponsor_address: str) -> ProviderManager:
    """A manager whose daemon config carries the sponsorship settings."""
    config = MagicMock()
    config.providers = {}
    config.autonet.owner_wallet = owner_wallet
    config.autonet.sponsor_address = sponsor_address
    mgr = ProviderManager(
        config=config,
        credential_store=MagicMock(),
        executors={},
        events=MagicMock(),
    )
    mgr._resolve_api_key = MagicMock(return_value="")
    return mgr


def _rpb_defn(agent_address: str = ""):
    """An agent definition pinned to the rpb provider.

    ``identity.address`` is set deliberately: it must be IGNORED for
    sponsorship, which keys on the daemon's owner wallet instead.
    """
    defn = MagicMock()
    defn.provider = "rpb"
    defn.cognitive_model = "some-model"
    defn.id = "dependent-agent"
    defn.identity.address = agent_address
    return defn


def test_dependent_presents_owner_wallet_not_agent_address():
    wallet = "0x" + "11" * 20
    sponsor = "0x" + "22" * 20
    mgr = _rpb_manager(owner_wallet=wallet, sponsor_address=sponsor)
    # The agent has its own keypair address; sponsorship must not use it.
    provider = mgr.resolve_provider_with_fallback(_rpb_defn("0x" + "99" * 20))

    assert provider._agent_address == wallet
    assert provider._sponsor_address == sponsor


def test_unregistered_agent_still_sponsorable_via_wallet():
    """An agent with no on-chain identity is sponsorable: the daemon's wallet
    is the dependent, so per-agent registration is irrelevant."""
    wallet = "0x" + "33" * 20
    mgr = _rpb_manager(owner_wallet=wallet, sponsor_address="")
    provider = mgr.resolve_provider_with_fallback(_rpb_defn(""))

    assert provider._agent_address == wallet


def test_no_owner_wallet_means_no_dependent_identity():
    """No wallet => nothing for a sponsor to bind. The provider still builds
    (open discovery is legal); it simply presents no identity, and any real
    sponsor refuses it with 'Missing agent_address'."""
    mgr = _rpb_manager(owner_wallet="", sponsor_address="")
    provider = mgr.resolve_provider_with_fallback(_rpb_defn("0x" + "44" * 20))

    assert provider._agent_address == ""
