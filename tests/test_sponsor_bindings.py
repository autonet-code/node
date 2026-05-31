"""Tests for sponsor/dependent inference ("work AI").

Two layers:
  1. SponsorBindingStore — add / spend / remaining / exhaust / persistence.
  2. The sponsor inference handler — an unbound dependent is rejected, a bound
     one is served and metered, and an exhausted grant is rejected.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atn.sponsor_bindings import SponsorBindingStore
from atn.providers.base import ProviderResponse, Usage


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_add_authorize_and_case_insensitive(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    s.add("0xABCdef", budget_tokens=1000, label="alice")
    assert s.is_authorized("0xabcdef")      # lookup is case-insensitive
    assert s.is_authorized("0xABCDEF")
    assert not s.is_authorized("0xDEAD")


def test_spend_and_remaining(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    s.add("0xA", budget_tokens=1000)
    s.record_spend("0xA", 600)
    assert s.remaining("0xA") == 400
    s.record_spend("0xA", 600)               # over-spend clamps at 0
    assert s.remaining("0xA") == 0
    assert not s.is_authorized("0xA")        # exhausted → unauthorized


def test_unlimited_budget(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    s.add("0xUNL", budget_tokens=0)          # 0 = unlimited
    assert s.is_authorized("0xunl")
    assert s.remaining("0xunl") == -1
    s.record_spend("0xunl", 10_000_000)
    assert s.is_authorized("0xunl")          # still authorized


def test_update_and_remove(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    s.add("0xA", budget_tokens=100)
    s.record_spend("0xA", 100)
    assert not s.is_authorized("0xA")
    s.update_budget("0xA", 500)              # raise the cap; spend unchanged
    assert s.is_authorized("0xA")
    assert s.remaining("0xA") == 400
    assert s.remove("0xA") is True
    assert s.get("0xA") is None
    assert s.remove("0xA") is False


def test_persistence(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    s.add("0xA", budget_tokens=1000, label="alice")
    s.record_spend("0xA", 250)
    s2 = SponsorBindingStore(tmp_path)       # reload from disk
    b = s2.get("0xA")
    assert b is not None
    assert b.spent_tokens == 250
    assert b.budget_tokens == 1000
    assert b.label == "alice"
    assert s2.remaining("0xA") == 750


def test_unbound_is_not_authorized(tmp_path: Path):
    s = SponsorBindingStore(tmp_path)
    assert not s.is_authorized("0xNEVER_BOUND")
    assert s.remaining("0xNEVER_BOUND") == 0


# ---------------------------------------------------------------------------
# Handler: authorize + meter
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Returns a fixed usage so we can assert metering."""
    def __init__(self, in_tok=10, out_tok=5):
        self._in, self._out = in_tok, out_tok

    async def send(self, **kwargs):
        return ProviderResponse(
            text="ok",
            tool_calls=None,
            usage=Usage(input_tokens=self._in, output_tokens=self._out),
            model=kwargs.get("model", "test-model"),
            stop_reason="end_turn",
        )


class _Runtime:
    def __init__(self, bindings):
        self.sponsor_bindings = bindings


def _make_handler(tmp_path: Path, store: SponsorBindingStore, provider):
    """Build a sponsor handler with provider resolution stubbed."""
    from atn.autonet_service import AutonetBridge
    from atn.config import RPBConfig

    bridge = AutonetBridge(RPBConfig())
    bridge._runtime = _Runtime(store)
    bridge._resolve_sponsor_provider = lambda rpb_cfg, model: provider  # stub
    cfg = bridge.config
    cfg.sponsor_model = "test-model"
    return bridge._create_sponsor_handler(cfg)


def test_handler_rejects_unbound(tmp_path: Path):
    store = SponsorBindingStore(tmp_path)
    handler = _make_handler(tmp_path, store, _FakeProvider())
    resp = asyncio.run(handler({"agent_address": "0xUNBOUND", "messages": []}))
    assert "error" in resp
    assert "not an authorized dependent" in resp["error"]


def test_handler_rejects_missing_address(tmp_path: Path):
    store = SponsorBindingStore(tmp_path)
    handler = _make_handler(tmp_path, store, _FakeProvider())
    resp = asyncio.run(handler({"messages": []}))
    assert "error" in resp
    assert "agent_address" in resp["error"]


def test_handler_serves_and_meters_bound(tmp_path: Path):
    store = SponsorBindingStore(tmp_path)
    store.add("0xBOUND", budget_tokens=100)
    handler = _make_handler(tmp_path, store, _FakeProvider(in_tok=10, out_tok=5))
    resp = asyncio.run(handler({"agent_address": "0xBOUND", "messages": []}))
    assert "error" not in resp, resp
    assert resp["text"] == "ok"
    # 10 + 5 spent → 85 remaining, surfaced in response
    assert store.remaining("0xBOUND") == 85
    assert resp["remaining_budget_tokens"] == 85


def test_handler_rejects_exhausted(tmp_path: Path):
    store = SponsorBindingStore(tmp_path)
    store.add("0xBOUND", budget_tokens=10)
    store.record_spend("0xBOUND", 10)        # already exhausted
    handler = _make_handler(tmp_path, store, _FakeProvider())
    resp = asyncio.run(handler({"agent_address": "0xBOUND", "messages": []}))
    assert "error" in resp
    assert "budget exhausted" in resp["error"]
