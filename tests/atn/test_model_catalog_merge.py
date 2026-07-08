"""Live model-catalog fetch + merge into the curated per-provider lists.

Covers ProviderManager._fetch_provider_catalog / _merged_provider_models /
get_available_models:
  - union by id (curated entries keep order/metadata; fetched-only appended)
  - fallback to curated on fetch failure (no key, network error, non-200)
  - OpenAI noise filtering (embeddings/whisper/tts/... dropped)
  - new curated additions (claude-sonnet-5) present in both catalogs
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from atn.runtime.provider_manager import ProviderManager, _PROVIDER_MODELS


def _make_manager() -> ProviderManager:
    config = MagicMock()
    config.providers = {}
    mgr = ProviderManager(
        config=config,
        credential_store=MagicMock(),
        executors={},
        events=MagicMock(),
    )
    mgr._resolve_api_key = MagicMock(return_value="")
    return mgr


# ---------------------------------------------------------------------------
# Curated additions
# ---------------------------------------------------------------------------

def test_sonnet5_in_both_catalogs():
    for pid in ("claude_max", "anthropic"):
        ids = [m["id"] for m in _PROVIDER_MODELS[pid]]
        assert "claude-sonnet-5" in ids


# ---------------------------------------------------------------------------
# OpenAI noise filter
# ---------------------------------------------------------------------------

def test_openai_chat_capable_filter():
    keep = ["gpt-5.5", "gpt-4o", "o3", "o4-mini", "o1"]
    drop = [
        "text-embedding-3-large", "whisper-1", "tts-1", "dall-e-3",
        "text-moderation-latest", "gpt-4o-audio-preview", "gpt-4o-realtime",
        "davinci-002", "babbage-002", "omni-moderation-latest",
    ]
    for m in keep:
        assert ProviderManager._openai_chat_capable(m), m
    for m in drop:
        assert not ProviderManager._openai_chat_capable(m), m


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------

def test_merge_union_appends_fetched_only():
    mgr = _make_manager()
    curated_ids = [m["id"] for m in _PROVIDER_MODELS["anthropic"]]
    # Cache a fetch result: one already-curated id + one brand-new id.
    mgr._catalog_cache["anthropic"] = [
        {"id": curated_ids[0], "name": "SHOULD NOT OVERRIDE"},
        {"id": "claude-future-9", "name": "claude-future-9"},
    ]
    merged = mgr._merged_provider_models("anthropic")
    merged_ids = [m["id"] for m in merged]
    # Curated entries keep their order + metadata (name not overridden).
    assert merged_ids[: len(curated_ids)] == curated_ids
    assert merged[0]["name"] != "SHOULD NOT OVERRIDE"
    # Fetched-only id is appended once.
    assert merged_ids.count("claude-future-9") == 1
    assert merged_ids[-1] == "claude-future-9"


def test_merge_no_cache_returns_curated():
    mgr = _make_manager()
    merged = mgr._merged_provider_models("anthropic")
    assert [m["id"] for m in merged] == [m["id"] for m in _PROVIDER_MODELS["anthropic"]]


def test_get_available_models_includes_fetched():
    mgr = _make_manager()
    mgr._catalog_cache["anthropic"] = [{"id": "claude-future-9", "name": "claude-future-9"}]
    models = mgr.get_available_models("anthropic")
    ids = [m["id"] for m in models]
    assert "claude-future-9" in ids
    # Fetched-only id still gets a sane tier + context window (default path).
    fut = next(m for m in models if m["id"] == "claude-future-9")
    assert fut["capability_tier"] >= 1
    assert fut["context_window"] > 0


# ---------------------------------------------------------------------------
# Fetch fallback behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_no_key_returns_empty():
    mgr = _make_manager()  # _resolve_api_key returns ""
    out = await mgr._fetch_provider_catalog("anthropic")
    assert out == []


@pytest.mark.asyncio
async def test_fetch_non_catalog_provider_returns_empty():
    mgr = _make_manager()
    mgr._resolve_api_key = MagicMock(return_value="sk-test")
    out = await mgr._fetch_provider_catalog("claude_max")
    assert out == []


@pytest.mark.asyncio
async def test_fetch_network_failure_returns_empty(monkeypatch):
    mgr = _make_manager()
    mgr._resolve_api_key = MagicMock(return_value="sk-test")

    import httpx

    class _BoomClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)
    out = await mgr._fetch_provider_catalog("anthropic")
    assert out == []  # no stale cache → curated-only (empty fetch list)


@pytest.mark.asyncio
async def test_fetch_success_caches_and_merges(monkeypatch):
    mgr = _make_manager()
    mgr._resolve_api_key = MagicMock(return_value="sk-test")

    import httpx

    class _Resp:
        status_code = 200
        def json(self):
            return {"data": [
                {"id": "claude-opus-4-8"},        # already curated
                {"id": "claude-brandnew-1"},      # new
            ]}

    class _OkClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _OkClient)
    out = await mgr._fetch_provider_catalog("anthropic")
    ids = [m["id"] for m in out]
    assert "claude-brandnew-1" in ids
    # Now the merge picks it up.
    merged_ids = [m["id"] for m in mgr._merged_provider_models("anthropic")]
    assert "claude-brandnew-1" in merged_ids


@pytest.mark.asyncio
async def test_fetch_openai_filters_noise(monkeypatch):
    mgr = _make_manager()
    mgr._resolve_api_key = MagicMock(return_value="sk-test")

    import httpx

    class _Resp:
        status_code = 200
        def json(self):
            return {"data": [
                {"id": "gpt-5.5"},
                {"id": "text-embedding-3-large"},
                {"id": "whisper-1"},
                {"id": "o5-mini"},
            ]}

    class _OkClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _OkClient)
    out = await mgr._fetch_provider_catalog("openai")
    ids = [m["id"] for m in out]
    assert "gpt-5.5" in ids
    assert "o5-mini" in ids
    assert "text-embedding-3-large" not in ids
    assert "whisper-1" not in ids
