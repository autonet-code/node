"""Consumer side of the inference-as-a-service rail (docs/services_market.md,
Decision 2026-07-26 item 2).

Covers the ``service`` provider type end to end with a mocked transport and a
mocked chain client:

  - happy path: read ask (once) -> payForService -> service_request -> the
    buffered completion mapped into ProviderResponse;
  - payment failure with chain configured ABORTS the call (no request sent);
  - no chain config: payment SKIPPED with a loud warning, request still sent
    (the provider-side gate degrades open in the same condition, which is what
    makes a local two-daemon demo work);
  - the on-chain ask is cached across completions (one registry read);
  - provider_manager resolution + config surface.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from atn.providers.base import ProviderError
from atn.providers.service import ServiceProvider

PROVIDER_ADDR = "0x1111111111111111111111111111111111111111"
DIGEST = "ab" * 32
OWNER_KEY = "0x" + "11" * 32
CLIENT_ADDR = "0x2222222222222222222222222222222222222222"


class _Recorder:
    """Stand-in for atn.service_client: records calls, returns scripted values."""

    def __init__(self, *, ask=None, pay=None, reply=None):
        self.ask_result = ask if ask is not None else {
            "service_id": 7, "provider": PROVIDER_ADDR,
            "ask_amount": 1000, "active": True,
        }
        self.pay_result = pay if pay is not None else {
            "tx_hash": "0xdeadbeef", "request_id": "0x" + "aa" * 32,
        }
        # Provider-side success envelope: {request_id, content, model, usage,
        # stop_reason, max_tokens}.
        self.reply_result = reply if reply is not None else {
            "ok": True,
            "endpoint": "wss://provider.example/ws",
            "result": {
                "request_id": "0x" + "aa" * 32,
                "content": "hello from the seller",
                "model": "seller-model-1",
                "usage": {"input_tokens": 11, "output_tokens": 5},
                "stop_reason": "end_turn",
                "max_tokens": 256,
            },
        }
        self.ask_calls: list[tuple] = []
        self.pay_calls: list[dict] = []
        self.request_calls: list[dict] = []
        self._rid = 0

    def new_request_id(self):
        self._rid += 1
        return "0x" + f"{self._rid:064x}"

    async def lookup_service_ask(self, config, provider_address, spec_digest):
        self.ask_calls.append((provider_address, spec_digest))
        return dict(self.ask_result)

    async def pay_for_service(self, config, key, recipient, amount,
                              request_id=""):
        self.pay_calls.append({
            "key": key, "recipient": recipient, "amount": amount,
            "request_id": request_id,
        })
        return dict(self.pay_result)

    async def request_service(self, config, provider_address, spec_digest,
                              args, *, tx_hash="", request_id="",
                              client_address="", endpoint="",
                              timeout=None):  # noqa: ARG002
        self.request_calls.append({
            "provider_address": provider_address,
            "spec_digest": spec_digest,
            "args": args,
            "tx_hash": tx_hash,
            "request_id": request_id,
            "client_address": client_address,
            "endpoint": endpoint,
        })
        return dict(self.reply_result)


def _make_provider(rec: _Recorder, monkeypatch, *, chain: bool = True,
                   owner_key: str = OWNER_KEY) -> ServiceProvider:
    import atn.service_client as real_client

    for name in ("new_request_id", "lookup_service_ask", "pay_for_service",
                 "request_service"):
        monkeypatch.setattr(real_client, name, getattr(rec, name))

    prov = ServiceProvider(
        config=MagicMock(),
        provider_address=PROVIDER_ADDR,
        spec_digest=DIGEST,
        model="bought-model",
        owner_private_key=owner_key,
        client_address=CLIENT_ADDR,
    )
    monkeypatch.setattr(prov, "_chain_configured", lambda: chain)
    return prov


MESSAGES = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_pay_then_request_then_completion(self, monkeypatch):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)

        resp = await prov.send(messages=MESSAGES, system="be brief",
                               max_tokens=256)

        # Ask read, then paid at that ask, to the provider address, with the
        # OWNER key (spend is an owner-level act).
        assert rec.ask_calls == [(PROVIDER_ADDR, DIGEST)]
        assert len(rec.pay_calls) == 1
        assert rec.pay_calls[0]["amount"] == 1000
        assert rec.pay_calls[0]["recipient"] == PROVIDER_ADDR
        assert rec.pay_calls[0]["key"] == OWNER_KEY
        assert rec.pay_calls[0]["request_id"]  # fresh id supplied

        # Request carries the payment proof and the wire args.
        assert len(rec.request_calls) == 1
        call = rec.request_calls[0]
        assert call["tx_hash"] == "0xdeadbeef"
        assert call["request_id"] == "0x" + "aa" * 32
        assert call["client_address"] == CLIENT_ADDR
        assert call["spec_digest"] == DIGEST
        assert call["args"]["max_tokens"] == 256
        # Wire contract: {messages, max_tokens?, system?, temperature?}.
        assert call["args"]["system"] == "be brief"
        assert call["args"]["messages"] == [{"role": "user", "content": "hi"}]
        assert "temperature" not in call["args"]   # 0.0 default omitted

        # Completion mapped into the provider interface.
        assert resp.text == "hello from the seller"
        assert resp.model == "seller-model-1"
        assert resp.usage.input_tokens == 11
        assert resp.usage.output_tokens == 5
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_fresh_request_id_per_call(self, monkeypatch):
        """Replay protection is provider-side on request_id — never reuse."""
        rec = _Recorder()
        # Echo back whatever request_id the provider generated.
        async def _pay(config, key, recipient, amount, request_id=""):
            rec.pay_calls.append({"key": key, "recipient": recipient,
                                  "amount": amount, "request_id": request_id})
            return {"tx_hash": "0xfeed", "request_id": request_id}
        rec.pay_for_service = _pay
        prov = _make_provider(rec, monkeypatch)

        await prov.send(messages=MESSAGES)
        await prov.send(messages=MESSAGES)

        ids = [c["request_id"] for c in rec.pay_calls]
        assert len(ids) == 2
        assert ids[0] and ids[1] and ids[0] != ids[1]
        assert [c["request_id"] for c in rec.request_calls] == ids

    @pytest.mark.asyncio
    async def test_stream_yields_whole_completion_once(self, monkeypatch):
        """v1 is deliberately non-streaming: one chunk, the whole thing."""
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        chunks: list[str] = []

        async def on_chunk(text):
            chunks.append(text)

        resp = await prov.send_stream(messages=MESSAGES, on_chunk=on_chunk)
        assert chunks == ["hello from the seller"]
        assert resp.text == "hello from the seller"

    @pytest.mark.asyncio
    async def test_endpoint_cached_after_first_call(self, monkeypatch):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        await prov.send(messages=MESSAGES)
        await prov.send(messages=MESSAGES)
        assert rec.request_calls[0]["endpoint"] == ""
        assert rec.request_calls[1]["endpoint"] == "wss://provider.example/ws"


# ---------------------------------------------------------------------------
# Payment failure aborts
# ---------------------------------------------------------------------------

class TestPaymentFailureAborts:
    @pytest.mark.asyncio
    async def test_failed_payment_aborts_before_request(self, monkeypatch):
        rec = _Recorder(pay={"error": "Payment failed: insufficient balance"})
        prov = _make_provider(rec, monkeypatch)

        with pytest.raises(ProviderError, match="payment failed"):
            await prov.send(messages=MESSAGES)
        # Nothing was sent to the counterparty.
        assert rec.request_calls == []

    @pytest.mark.asyncio
    async def test_missing_owner_key_with_chain_aborts(self, monkeypatch):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch, owner_key="")
        with pytest.raises(ProviderError, match="owner signing key"):
            await prov.send(messages=MESSAGES)
        assert rec.pay_calls == []
        assert rec.request_calls == []

    @pytest.mark.asyncio
    async def test_unreadable_ask_aborts(self, monkeypatch):
        rec = _Recorder(ask={"error": "ServiceRegistry not configured"})
        prov = _make_provider(rec, monkeypatch)
        with pytest.raises(ProviderError, match="cannot read the on-chain ask"):
            await prov.send(messages=MESSAGES)
        assert rec.pay_calls == []
        assert rec.request_calls == []

    @pytest.mark.asyncio
    async def test_retired_service_aborts(self, monkeypatch):
        rec = _Recorder(ask={"service_id": 7, "provider": PROVIDER_ADDR,
                             "ask_amount": 1000, "active": False})
        prov = _make_provider(rec, monkeypatch)
        with pytest.raises(ProviderError, match="retired"):
            await prov.send(messages=MESSAGES)
        assert rec.pay_calls == []

    @pytest.mark.asyncio
    async def test_provider_side_error_surfaces(self, monkeypatch):
        rec = _Recorder(reply={"ok": False, "endpoint": "wss://x/ws",
                               "error": "Payment validation failed: replay"})
        prov = _make_provider(rec, monkeypatch)
        with pytest.raises(ProviderError, match="Payment validation failed"):
            await prov.send(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_error_inside_result_envelope_surfaces(self, monkeypatch):
        """Provider-side dispatch failures come back as {error} in result."""
        rec = _Recorder(reply={
            "ok": False, "endpoint": "wss://x/ws",
            "result": {"request_id": "0x1",
                       "error": "inference failed: upstream 429"},
        })
        prov = _make_provider(rec, monkeypatch)
        with pytest.raises(ProviderError, match="upstream 429"):
            await prov.send(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_unconfigured_provider_raises(self):
        prov = ServiceProvider(config=MagicMock(), provider_address="",
                               spec_digest="")
        assert prov.is_available is False
        with pytest.raises(ProviderError, match="not configured"):
            await prov.send(messages=MESSAGES)


# ---------------------------------------------------------------------------
# No-chain degrade (loud, open) — enables the local two-daemon demo
# ---------------------------------------------------------------------------

class TestNoChainDegrade:
    @pytest.mark.asyncio
    async def test_skips_payment_and_still_sends(self, monkeypatch, caplog):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch, chain=False, owner_key="")

        with caplog.at_level(logging.WARNING, logger="atn.providers.service"):
            resp = await prov.send(messages=MESSAGES)

        assert rec.pay_calls == []
        assert rec.ask_calls == []          # no ask read either — nothing to pay
        assert len(rec.request_calls) == 1
        # Request goes out WITHOUT payment proof.
        assert rec.request_calls[0]["tx_hash"] == ""
        assert rec.request_calls[0]["request_id"] == ""
        assert resp.text == "hello from the seller"
        # And it is LOUD.
        assert any("PAYMENT SKIPPED" in r.message or "PAYMENT SKIPPED" in r.getMessage()
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_zero_ask_needs_no_payment(self, monkeypatch):
        """A giveaway service (ask 0) is chain-configured but unpaid —
        payForService rejects a zero amount, so there is nothing to sign."""
        rec = _Recorder(ask={"service_id": 7, "provider": PROVIDER_ADDR,
                             "ask_amount": 0, "active": True})
        prov = _make_provider(rec, monkeypatch)
        resp = await prov.send(messages=MESSAGES)
        assert rec.pay_calls == []
        assert len(rec.request_calls) == 1
        assert resp.text == "hello from the seller"


# ---------------------------------------------------------------------------
# Ask caching
# ---------------------------------------------------------------------------

class TestAskCaching:
    @pytest.mark.asyncio
    async def test_ask_read_once_across_completions(self, monkeypatch):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)

        for _ in range(3):
            await prov.send(messages=MESSAGES)

        assert len(rec.ask_calls) == 1          # one registry scan
        assert len(rec.pay_calls) == 3          # but a payment per call
        assert prov.cached_ask == 1000
        assert all(c["amount"] == 1000 for c in rec.pay_calls)


# ---------------------------------------------------------------------------
# Wire shaping
# ---------------------------------------------------------------------------

class TestWireShaping:
    def test_block_content_flattened_to_text(self):
        wire = ServiceProvider._wire_messages([
            {"role": "assistant", "content": [
                {"type": "text", "text": "part one"},
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "{\"ok\": true}"},
            ]},
        ])
        assert wire[0] == {"role": "assistant", "content": "part one"}
        assert wire[1] == {"role": "user", "content": '{"ok": true}'}

    def test_plain_messages_pass_through(self):
        assert ServiceProvider._wire_messages(MESSAGES) == [
            {"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_temperature_passed_through_when_set(self, monkeypatch):
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        await prov.send(messages=MESSAGES, temperature=0.7)
        assert rec.request_calls[0]["args"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_empty_messages_rejected_before_paying(self, monkeypatch):
        """The provider side requires a non-empty messages list — don't pay
        for a request that is guaranteed to be refused."""
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        with pytest.raises(ProviderError, match="non-empty"):
            await prov.send(messages=[])
        assert rec.pay_calls == []
        assert rec.request_calls == []

    @pytest.mark.asyncio
    async def test_empty_served_model_is_not_an_error(self, monkeypatch):
        """An empty spec model means the seller resolves its own default."""
        rec = _Recorder(reply={
            "ok": True, "endpoint": "wss://x/ws",
            "result": {"request_id": "0x1", "content": "ok", "model": "",
                       "usage": {}, "stop_reason": "end_turn"},
        })
        prov = _make_provider(rec, monkeypatch)
        resp = await prov.send(messages=MESSAGES)
        assert resp.text == "ok"
        assert resp.model == "bought-model"   # falls back to our config label

    @pytest.mark.asyncio
    async def test_tools_are_dropped_loudly(self, monkeypatch, caplog):
        from atn.providers.base import ToolDefinition
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        with caplog.at_level(logging.WARNING, logger="atn.providers.service"):
            await prov.send(
                messages=MESSAGES,
                tools=[ToolDefinition(name="t", description="d",
                                      input_schema={})],
            )
        assert "tools" not in rec.request_calls[0]["args"]
        assert any("tool definition" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# provider_manager resolution / config surface
# ---------------------------------------------------------------------------

class TestProviderManagerWiring:
    def _mgr(self, extra: dict | None):
        from atn.config import ATNConfig, ProviderConfig
        from atn.runtime.provider_manager import ProviderManager

        config = ATNConfig()
        config.autonet.private_key = OWNER_KEY
        config.autonet.owner_wallet = CLIENT_ADDR
        if extra is not None:
            config.providers["service"] = ProviderConfig(
                name="service", default_model="bought-model", extra=extra)
        mgr = ProviderManager(config=config, credential_store=MagicMock(),
                              executors={}, events=MagicMock())
        # Normally attached by the runtime; provider_list aggregates over it.
        mgr._execution_log = MagicMock(_logs={})
        return mgr

    def test_resolves_from_config(self):
        mgr = self._mgr({"provider_address": PROVIDER_ADDR,
                         "spec_digest": DIGEST})
        prov = mgr._resolve_provider_by_name("service", "bought-model", "a1")
        assert isinstance(prov, ServiceProvider)
        assert prov.name == "service"
        assert prov.provider_address == PROVIDER_ADDR
        assert prov.spec_digest == DIGEST
        assert prov.is_available
        # Owner key + owner wallet threaded through (owner-level spend).
        assert prov._owner_key == OWNER_KEY
        assert prov._client_address == CLIENT_ADDR

    def test_missing_config_raises(self):
        mgr = self._mgr(None)
        with pytest.raises(ProviderError, match="provider_address"):
            mgr._resolve_provider_by_name("service", "m", "a1")

    def test_partial_config_raises(self):
        mgr = self._mgr({"provider_address": PROVIDER_ADDR})
        with pytest.raises(ProviderError, match="spec_digest"):
            mgr._resolve_provider_by_name("service", "m", "a1")

    @pytest.mark.asyncio
    async def test_provider_list_reports_configured(self):
        mgr = self._mgr({"provider_address": PROVIDER_ADDR,
                         "spec_digest": DIGEST})
        entries = {e["id"]: e for e in await mgr.provider_list()}
        assert entries["service"]["auth_type"] == "service"
        assert entries["service"]["configured"] is True

    @pytest.mark.asyncio
    async def test_provider_list_reports_unconfigured(self):
        mgr = self._mgr(None)
        entries = {e["id"]: e for e in await mgr.provider_list()}
        assert entries["service"]["configured"] is False


# ---------------------------------------------------------------------------
# Per-agent binding (docs/services_market.md, ratified 2026-07-26:
# employer-chooses-the-tool). A PARENT may buy a marketplace inference service
# and provision a CHILD bound to it; the child pays each call from its OWN
# wallet, and can never set or change the binding itself.
# ---------------------------------------------------------------------------

CHILD_KEY = "0x" + "33" * 32
CHILD_ADDR = "0x3333333333333333333333333333333333333333"
BINDING = {"provider_address": PROVIDER_ADDR, "spec_digest": DIGEST}


class TestBindingNormalization:
    def test_canonicalizes_address_and_digest(self):
        from atn.models import normalize_service_binding
        out = normalize_service_binding(
            {"provider_address": PROVIDER_ADDR, "spec_digest": ("AB" * 32)})
        assert out == {"provider_address": PROVIDER_ADDR,
                       "spec_digest": "ab" * 32}

    def test_strips_0x_from_digest(self):
        from atn.models import normalize_service_binding
        out = normalize_service_binding(
            {"provider_address": PROVIDER_ADDR, "spec_digest": "0x" + DIGEST})
        assert out["spec_digest"] == DIGEST

    @pytest.mark.parametrize("bad", [
        None,
        "not-an-object",
        {},
        {"provider_address": PROVIDER_ADDR},                    # no digest
        {"spec_digest": DIGEST},                                # no provider
        {"provider_address": "0xshort", "spec_digest": DIGEST},
        {"provider_address": PROVIDER_ADDR, "spec_digest": "zz" * 32},
        {"provider_address": PROVIDER_ADDR, "spec_digest": "ab" * 10},
    ])
    def test_partial_or_malformed_is_refused(self, bad):
        """A half-binding is worse than none: it would fall back to the
        daemon-level purchase and spend the OWNER's wallet."""
        from atn.models import normalize_service_binding
        with pytest.raises(ValueError):
            normalize_service_binding(bad)


class TestAgentKeySigning:
    """The bound agent's OWN key signs — spend authority is token custody."""

    def _bound_provider(self, rec, monkeypatch, *, key=CHILD_KEY):
        import atn.service_client as real_client
        for name in ("new_request_id", "lookup_service_ask", "pay_for_service",
                     "request_service"):
            monkeypatch.setattr(real_client, name, getattr(rec, name))
        prov = ServiceProvider(
            config=MagicMock(),
            provider_address=PROVIDER_ADDR,
            spec_digest=DIGEST,
            signing_key=key,
            client_address=CHILD_ADDR,
            payer_kind="agent",
            payer_id="parent-1.child-1",
        )
        monkeypatch.setattr(prov, "_chain_configured", lambda: True)
        return prov

    @pytest.mark.asyncio
    async def test_pays_with_the_child_key_not_the_owner_key(self, monkeypatch):
        rec = _Recorder()
        prov = self._bound_provider(rec, monkeypatch)
        resp = await prov.send(messages=MESSAGES)
        assert rec.pay_calls[0]["key"] == CHILD_KEY
        assert rec.pay_calls[0]["key"] != OWNER_KEY
        # The child is the payer on the wire, not the daemon owner.
        assert rec.request_calls[0]["client_address"] == CHILD_ADDR
        assert resp.text == "hello from the seller"

    def test_payer_provenance_is_reported(self, monkeypatch):
        rec = _Recorder()
        prov = self._bound_provider(rec, monkeypatch)
        assert prov.payer_kind == "agent"
        assert prov.payer_id == "parent-1.child-1"

    @pytest.mark.asyncio
    async def test_missing_agent_key_aborts_naming_the_agent(self, monkeypatch):
        """A bound child with no stored wallet key must fail loud — falling
        back to the owner key would spend the wrong wallet."""
        rec = _Recorder()
        prov = self._bound_provider(rec, monkeypatch, key="")
        with pytest.raises(ProviderError, match="child-1"):
            await prov.send(messages=MESSAGES)
        assert rec.pay_calls == []
        assert rec.request_calls == []

    @pytest.mark.asyncio
    async def test_owner_path_still_signs_with_the_owner_key(self, monkeypatch):
        """The daemon-level purchase is unchanged by the binding work."""
        rec = _Recorder()
        prov = _make_provider(rec, monkeypatch)
        await prov.send(messages=MESSAGES)
        assert rec.pay_calls[0]["key"] == OWNER_KEY
        assert prov.payer_kind == "owner"


class TestBindingResolution:
    """provider_manager: a bound agent gets a ServiceProvider from ITS binding;
    an unbound agent is entirely unaffected."""

    def _mgr(self, *, service_extra=None, keys=None, addresses=None):
        from atn.config import ATNConfig, ProviderConfig
        from atn.runtime.provider_manager import ProviderManager

        config = ATNConfig()
        config.autonet.private_key = OWNER_KEY
        config.autonet.owner_wallet = CLIENT_ADDR
        if service_extra is not None:
            config.providers["service"] = ProviderConfig(
                name="service", default_model="bought-model",
                extra=service_extra)
        mgr = ProviderManager(config=config, credential_store=MagicMock(),
                              executors={}, events=MagicMock())
        mgr._execution_log = MagicMock(_logs={})
        keys = keys if keys is not None else {"child-1": CHILD_KEY}
        addresses = (addresses if addresses is not None
                     else {"child-1": CHILD_ADDR})
        mgr._agent_key_resolver = lambda aid: keys.get(aid, "")
        mgr._agent_address_resolver = lambda aid: addresses.get(aid, "")
        return mgr

    def _defn(self, agent_id, *, binding=None, provider="ollama",
              model="qwen3:4b"):
        from atn.models import AgentDefinition, AgentMode
        return AgentDefinition(
            id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
            provider=provider, cognitive_model=model,
            service_provider=binding,
        )

    def test_bound_agent_resolves_to_service_provider(self):
        mgr = self._mgr()
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding=dict(BINDING)))
        assert isinstance(prov, ServiceProvider)
        assert prov.provider_address == PROVIDER_ADDR
        assert prov.spec_digest == DIGEST
        assert prov.is_available

    def test_binding_signs_with_the_agents_own_key(self):
        mgr = self._mgr()
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding=dict(BINDING)))
        assert prov._owner_key == CHILD_KEY
        assert prov._client_address == CHILD_ADDR
        assert prov.payer_kind == "agent"
        assert prov.payer_id == "child-1"

    def test_binding_overrides_provider_and_model(self):
        """A binding is the parent's choice of substrate — not one candidate in
        a fallback chain. It must win over an explicit provider."""
        mgr = self._mgr()
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding=dict(BINDING),
                       provider="anthropic", model="claude-sonnet-5"))
        assert isinstance(prov, ServiceProvider)
        assert prov.name == "service"

    def test_binding_does_not_read_the_daemon_purchase(self):
        """A bound child must never inherit the OWNER's purchase or key."""
        other = "0x9999999999999999999999999999999999999999"
        mgr = self._mgr(service_extra={"provider_address": other,
                                       "spec_digest": "cd" * 32})
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding=dict(BINDING)))
        # The binding's purchase, not config's.
        assert prov.provider_address == PROVIDER_ADDR
        assert prov.spec_digest == DIGEST
        assert prov._owner_key == CHILD_KEY             # never OWNER_KEY

    def test_unbound_agent_is_unaffected(self):
        from atn.providers.ollama import OllamaProvider
        mgr = self._mgr()
        prov = mgr.resolve_provider_with_fallback(self._defn("plain-1"))
        assert isinstance(prov, OllamaProvider)
        assert not isinstance(prov, ServiceProvider)

    def test_unbound_agent_still_gets_the_daemon_purchase(self):
        """providers.service (owner use) keeps working via provider='service'."""
        mgr = self._mgr(service_extra={"provider_address": PROVIDER_ADDR,
                                       "spec_digest": DIGEST})
        prov = mgr.resolve_provider_with_fallback(
            self._defn("plain-1", provider="service", model=""))
        assert isinstance(prov, ServiceProvider)
        assert prov._owner_key == OWNER_KEY
        assert prov.payer_kind == "owner"

    def test_malformed_binding_raises(self):
        mgr = self._mgr()
        with pytest.raises(ProviderError, match="malformed"):
            mgr.resolve_provider_with_fallback(
                self._defn("child-1",
                           binding={"provider_address": PROVIDER_ADDR}))

    def test_non_dict_binding_is_ignored(self):
        """Only a real dict is a binding. A stray truthy value (a test double,
        a half-migrated definition) must NOT reroute an agent's cognition —
        this is how the rpb routing tests broke when the check was truthiness."""
        from unittest.mock import MagicMock as MM

        from atn.providers.ollama import OllamaProvider
        mgr = self._mgr()
        defn = self._defn("child-1")
        defn.service_provider = MM()          # truthy, but not a binding
        prov = mgr.resolve_provider_with_fallback(defn)
        assert isinstance(prov, OllamaProvider)
        assert not isinstance(prov, ServiceProvider)

    def test_empty_dict_binding_is_ignored(self):
        from atn.providers.ollama import OllamaProvider
        mgr = self._mgr()
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding={}))
        assert isinstance(prov, OllamaProvider)

    def test_missing_agent_key_still_builds(self):
        """Resolution must not explode for an agent with no key yet — the
        payment step is where that fails, loudly and per-call."""
        mgr = self._mgr(keys={})
        prov = mgr.resolve_provider_with_fallback(
            self._defn("child-1", binding=dict(BINDING)))
        assert isinstance(prov, ServiceProvider)
        assert prov._owner_key == ""


class TestBindingPersistence:
    """The binding must survive a save/load round trip: dropping it on save
    would silently re-route the spend to the daemon owner after a restart."""

    def test_yaml_round_trip(self, tmp_path):
        from atn.loader import load_agent_file, save_agent
        from atn.models import AgentDefinition, AgentMode

        defn = AgentDefinition(
            id="child-1", name="child-1", mode=AgentMode.COGNITIVE,
            cognitive_model="bought", parent_id="parent-1",
            service_provider=dict(BINDING),
        )
        save_agent(defn, tmp_path)
        loaded, errors = load_agent_file(tmp_path / "child-1")
        assert errors == []
        assert loaded.service_provider == BINDING

    def test_pipeline_agent_binding_also_persists(self, tmp_path):
        """save_agent gates cognitive fields behind is_cognitive; the binding
        is deliberately outside that guard."""
        from atn.loader import load_agent_file, save_agent
        from atn.models import (AgentDefinition, AgentMode, StepDefinition,
                               StepType)

        defn = AgentDefinition(
            id="pipe-1", name="pipe-1", mode=AgentMode.PIPELINE,
            steps=[StepDefinition(type=StepType.SCRIPT,
                                  config={"command": "echo hi"})],
            service_provider=dict(BINDING),
        )
        save_agent(defn, tmp_path)
        loaded, errors = load_agent_file(tmp_path / "pipe-1")
        assert errors == []
        assert loaded.service_provider == BINDING

    def test_unbound_agent_writes_no_key(self, tmp_path):
        import yaml
        from atn.loader import save_agent
        from atn.models import AgentDefinition, AgentMode

        defn = AgentDefinition(id="plain-1", name="plain-1",
                               mode=AgentMode.COGNITIVE,
                               cognitive_model="sonnet")
        path = save_agent(defn, tmp_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "service_provider" not in raw

    def test_malformed_yaml_binding_is_a_load_error(self, tmp_path):
        """A dropped binding would fall back to the owner's wallet, so a bad
        one must fail the load rather than degrade silently."""
        from atn.loader import load_agent_file

        d = tmp_path / "bad-1"
        d.mkdir()
        (d / "agent.yaml").write_text(
            "id: bad-1\nname: bad-1\nmode: cognitive\n"
            "service_provider:\n  provider_address: '0xtooshort'\n"
            "  spec_digest: '" + DIGEST + "'\n",
            encoding="utf-8")
        loaded, errors = load_agent_file(d)
        assert loaded is None
        assert any("service_provider" in e.message for e in errors)
