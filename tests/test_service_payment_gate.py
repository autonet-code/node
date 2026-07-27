"""Stage B — real service-payment gate (ws_server._validate_service_payment).

Unit coverage with MOCKED on_chain responses (no live chain): the gate's
direct-payForService path and voucher path, their accept/reject branches,
replay protection, per-channel cumulative tracking, and the no-chain-config
degrade escape hatch. Also smoke-covers the ServiceStore payment-state
sidecars (seen-request-id set + channel voucher ledger) the gate persists
through.

Follows the tests/atn/test_service_store.py runtime harness (real Runtime,
autonet disabled) and monkeypatches the on_chain client methods the gate
calls, mirroring tests/test_service_market_client.py's mocking style.

Run: pytest tests/test_service_payment_gate.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode

import atn.on_chain as oc
from atn.ws_server import WebSocketBridge


PROVIDER_ADDR = "0x1111111111111111111111111111111111111111"
CLIENT_ADDR = "0x2222222222222222222222222222222222222222"
SUBSTRATE_ADDR = "0x4C4dAEd19B98ddaEc6cC421b6a781Bd3fBB7af25"
CHANNEL_ADDR = "0x97956A322dC5585157c91840d39292c396032508"
ASK_AMOUNT = 1000

# ATN-denominated by construction (the vestigial `ask.token` was dropped
# 2026-07-26); SUBSTRATE_ADDR is still the payment target the gate verifies.
ASK = {"amount": str(ASK_AMOUNT), "unit": "per_item"}
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def _make_runtime(tmp_path: Path, *, chain: bool):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    if chain:
        # Make OnChainService.available true (substrate_address + rpc_url).
        config.autonet.substrate_address = SUBSTRATE_ADDR
        config.autonet.payment_channel_address = CHANNEL_ADDR
        config.autonet.service_registry_address = SUBSTRATE_ADDR
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _register_service(rt, *, backing_tool="deadbeef", provider=PROVIDER_ADDR):
    """Register a service whose spec carries a provider address (author_pubkey)."""
    res = rt.service_store.register(
        name="echo_svc", description="Echo service.", input_schema=SCHEMA,
        author="user", ask=ASK, backing_tool=backing_tool)
    record = rt.service_store.get(res["digest"])
    # The provider address is stamped as author_pubkey at registration; the
    # owner-authored path leaves it empty, so set it explicitly for the gate.
    record.spec["author_pubkey"] = provider
    return record


def _patch_chain(monkeypatch, *, verify=None, voucher=None):
    """Patch the two chain-verify methods the gate calls so no real web3 is
    touched. Real ``available`` / ``channel_available`` resolve from config
    (chain=True sets substrate_address + payment_channel_address + rpc_url), so
    only the verify methods need mocking. `verify` -> verify_service_payment
    result; `voucher` -> verify_voucher result."""
    async def _verify(self, *, request_id, recipient, min_amount, tx_hash=None,
                      payer=None):
        return verify

    async def _verify_voucher(self, channel_id, cumulative_amount, signature):
        return voucher

    monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)
    monkeypatch.setattr(oc.ServiceMarketClient, "verify_voucher",
                        _verify_voucher)


# ---------------------------------------------------------------------------
# Direct payForService path
# ---------------------------------------------------------------------------

class TestDirectPath:
    @pytest.mark.asyncio
    async def test_accept(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={
            "verified": True, "reason": "ok",
            "amount": ASK_AMOUNT, "recipient": PROVIDER_ADDR})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"tx_hash": "0x" + "ab" * 32, "request_id": "0x" + "01" * 32},
            record)
        assert out["ok"] is True, out
        # Replay guard consumed the request_id.
        assert rt.service_store.has_served_request("0x" + "01" * 32)

    @pytest.mark.asyncio
    async def test_reject_wrong_amount(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={
            "verified": False, "reason": "amount 100 < min_amount 1000",
            "amount": 100})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": "0x" + "02" * 32}, record)
        assert out["ok"] is False
        assert "min_amount" in out["reason"]
        # A rejected request_id is NOT consumed (payer can retry with a fix).
        assert not rt.service_store.has_served_request("0x" + "02" * 32)

    @pytest.mark.asyncio
    async def test_reject_wrong_recipient(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={
            "verified": False,
            "reason": "no ServicePayment event matched requestId+recipient"})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": "0x" + "03" * 32}, record)
        assert out["ok"] is False
        assert "matched" in out["reason"] or "recipient" in out["reason"]

    @pytest.mark.asyncio
    async def test_reject_replayed_request_id(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={
            "verified": True, "reason": "ok", "amount": ASK_AMOUNT})
        server = WebSocketBridge(rt)

        rid = "0x" + "04" * 32
        first = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": rid}, record)
        assert first["ok"] is True
        # Second time with the SAME request_id -> replay rejected, even though
        # the (patched) chain would still verify it.
        second = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": rid}, record)
        assert second["ok"] is False
        assert "replay" in second["reason"]

    @pytest.mark.asyncio
    async def test_reject_no_provider_address(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt, provider="")  # no author_pubkey
        _patch_chain(monkeypatch, verify={"verified": True})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": "0x" + "05" * 32}, record)
        assert out["ok"] is False
        assert "provider address" in out["reason"]


# ---------------------------------------------------------------------------
# Voucher (PaymentChannel) path
# ---------------------------------------------------------------------------

class TestVoucherPath:
    @pytest.mark.asyncio
    async def test_accept(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "reason": "ok",
            "provider": PROVIDER_ADDR, "client": CLIENT_ADDR, "status": 1})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"channel_id": 7, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record)
        assert out["ok"] is True, out
        # Latest cumulative recorded + voucher stored for later settlement.
        assert rt.service_store.last_channel_cumulative(7) == ASK_AMOUNT
        stored = rt.service_store.get_channel_voucher(7)
        assert stored and stored["signature"] == "0xsig"

    @pytest.mark.asyncio
    async def test_accept_second_increment(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "provider": PROVIDER_ADDR, "status": 1})
        server = WebSocketBridge(rt)

        first = await server._validate_service_payment(
            {"channel_id": 9, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig1"}, record)
        assert first["ok"] is True
        # Next voucher must add at least another ask on top of the last.
        second = await server._validate_service_payment(
            {"channel_id": 9, "cumulative_amount": ASK_AMOUNT * 2,
             "signature": "0xsig2"}, record)
        assert second["ok"] is True
        assert rt.service_store.last_channel_cumulative(9) == ASK_AMOUNT * 2

    @pytest.mark.asyncio
    async def test_reject_bad_signature(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": False, "reason": "signer is not the channel client"})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"channel_id": 1, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xbad"}, record)
        assert out["ok"] is False
        assert "signer" in out["reason"] or "client" in out["reason"]

    @pytest.mark.asyncio
    async def test_reject_increment_below_ask(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "provider": PROVIDER_ADDR, "status": 1})
        server = WebSocketBridge(rt)

        # First voucher: fine (increment == ask).
        first = await server._validate_service_payment(
            {"channel_id": 3, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xs1"}, record)
        assert first["ok"] is True
        # Second voucher: increment only 1, below the ask -> rejected, and the
        # stored cumulative does NOT regress or advance.
        second = await server._validate_service_payment(
            {"channel_id": 3, "cumulative_amount": ASK_AMOUNT + 1,
             "signature": "0xs2"}, record)
        assert second["ok"] is False
        assert "increment" in second["reason"]
        assert rt.service_store.last_channel_cumulative(3) == ASK_AMOUNT

    @pytest.mark.asyncio
    async def test_reject_channel_closed(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": False, "reason": "channel not Open (status=3)"})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"channel_id": 2, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record)
        assert out["ok"] is False
        assert "not Open" in out["reason"]

    @pytest.mark.asyncio
    async def test_reject_wrong_channel_provider(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)  # provider = PROVIDER_ADDR
        _patch_chain(monkeypatch, voucher={
            "valid": True, "status": 1,
            "provider": "0x9999999999999999999999999999999999999999"})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"channel_id": 4, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record)
        assert out["ok"] is False
        assert "provider" in out["reason"]


# ---------------------------------------------------------------------------
# No proof / no-chain-config degrade
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_chain_config_degrades_open(self, tmp_path, monkeypatch):
        # chain=False -> OnChainService.available is False -> loud allow.
        rt = _make_runtime(tmp_path, chain=False)
        record = _register_service(rt)
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": "0xdead"}, record)
        assert out["ok"] is True
        assert "not configured" in out["reason"] or "dev" in out["reason"]

    @pytest.mark.asyncio
    async def test_missing_proof_rejected(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={"verified": True})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment({}, record)
        assert out["ok"] is False
        assert "no payment proof" in out["reason"]


# ---------------------------------------------------------------------------
# ServiceStore payment-state sidecars (persistence)
# ---------------------------------------------------------------------------

class TestPaymentStatePersistence:
    @pytest.mark.asyncio
    async def test_seen_requests_survive_reload(self, tmp_path):
        from atn.service_store import ServiceStore

        rt = _make_runtime(tmp_path, chain=True)
        rt.service_store.mark_request_served("0xreq1")
        rt.service_store.mark_request_served("0xreq2")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.has_served_request("0xreq1")
        assert reloaded.has_served_request("0xreq2")
        assert not reloaded.has_served_request("0xreq3")

    @pytest.mark.asyncio
    async def test_channel_vouchers_survive_reload(self, tmp_path):
        from atn.service_store import ServiceStore

        rt = _make_runtime(tmp_path, chain=True)
        rt.service_store.record_channel_voucher(5, 500, "0xsigA")
        rt.service_store.record_channel_voucher(5, 900, "0xsigB")
        # Monotone: a lower cumulative is ignored.
        rt.service_store.record_channel_voucher(5, 100, "0xstale")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.last_channel_cumulative(5) == 900
        assert reloaded.get_channel_voucher(5)["signature"] == "0xsigB"
