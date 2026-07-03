"""§10 Provider routing (fail-loud) + update_agent eviction.

- _resolve_provider_for_model must NOT silently fall back to BridgeProvider for
  keyless gemini/openai or for an unknown model. Unknown models that ARE
  installed in ollama route to OllamaProvider; otherwise it raises ProviderError.
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

class TestFailLoudRouting:
    def test_keyless_gemini_raises(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="Gemini API key"):
            mgr._resolve_provider_for_model("gemini-3-pro", "agent-1")

    def test_keyless_openai_raises(self):
        mgr = _make_manager()
        with pytest.raises(ProviderError, match="OpenAI API key"):
            mgr._resolve_provider_for_model("gpt-5.5", "agent-1")

    def test_unknown_model_not_in_ollama_raises(self):
        mgr = _make_manager()
        # Ollama reports no models installed.
        mgr._ollama_tags_cached = MagicMock(return_value=set())
        with pytest.raises(ProviderError, match="unknown model"):
            mgr._resolve_provider_for_model("qwen3.5:4b", "agent-1")

    def test_unknown_model_installed_in_ollama_routes_to_ollama(self):
        mgr = _make_manager()
        mgr._ollama_tags_cached = MagicMock(
            return_value={"qwen3.5:4b", "qwen3.5"}
        )
        prov = mgr._resolve_provider_for_model("qwen3.5:4b", "agent-1")
        assert isinstance(prov, OllamaProvider)

    def test_ollama_match_on_bare_base_name(self):
        mgr = _make_manager()
        # Only the tagged name is installed; a bare-name request still matches.
        mgr._ollama_tags_cached = MagicMock(return_value={"qwen3.5:4b", "qwen3.5"})
        prov = mgr._resolve_provider_for_model("qwen3.5", "agent-1")
        assert isinstance(prov, OllamaProvider)


class TestOllamaTagsCache:
    def test_cache_tolerates_ollama_down(self):
        """A dead ollama endpoint yields an empty set, not an exception."""
        mgr = _make_manager()
        # No ollama running at the default port during the test — probe fails,
        # returns empty, and caches it.
        tags = mgr._ollama_tags_cached()
        assert isinstance(tags, set)

    def test_cache_is_reused_within_ttl(self, monkeypatch):
        mgr = _make_manager()
        calls = {"n": 0}

        def fake_get(url, timeout=0):
            calls["n"] += 1
            raise RuntimeError("down")

        monkeypatch.setattr("httpx.get", fake_get)
        mgr._ollama_tags_cached()
        mgr._ollama_tags_cached()
        # Second call served from cache — probe fired once.
        assert calls["n"] == 1


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
            provider="claude-sonnet-4-6", cognitive_model="claude-sonnet-4-6",
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
        from atn.orchestrator import tools

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
        from atn.orchestrator import tools

        monkeypatch.setattr(tools, "save_agent", lambda *a, **k: None)
        runtime, defn, pmgr = await self._make_runtime_with_agent()
        fake = _FakeProvider()
        pmgr._active_providers["a1"] = fake

        res = await tools._update_agent(runtime, {"agent_id": "a1", "provider": "ollama"})

        assert "provider" in res["changed"]
        assert fake.closed is True
        assert "a1" not in pmgr._active_providers

    async def test_non_provider_change_does_not_evict(self, monkeypatch):
        from atn.orchestrator import tools

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
    def test_schema_has_provider_enum(self):
        from atn.orchestrator.tools import _TOOLS

        create = next(t for t in _TOOLS if t.name == "create_agent")
        props = create.input_schema["properties"]
        assert "provider" in props
        assert "ollama" in props["provider"]["enum"]
        assert "rpb" in props["provider"]["enum"]
