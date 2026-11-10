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

import asyncio

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
        # No jurisdiction is chosen in the test home (#50), so there is no
        # packaged RPC default to fall back on.
        config.autonet.rpc_url = "http://127.0.0.1:8545"
        config.autonet.payment_channel_address = CHANNEL_ADDR
        config.autonet.service_registry_address = SUBSTRATE_ADDR
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _register_service(rt, *, backing_tool="deadbeef", provider=PROVIDER_ADDR,
                      ask_amount=ASK_AMOUNT, active=True, name="echo_svc"):
    """Register a service and give it the ServiceRegistry entry the gate reads
    (stubbed by ``_patch_listing``): the only reference for its price, its
    active flag and its payee (Decision 2026-11-10)."""
    res = rt.service_store.register(
        name=name, description="Echo service.", input_schema=SCHEMA,
        author="user", backing_tool=backing_tool)
    record = rt.service_store.get(res["digest"])
    record.service_id = 7
    record.test_listing = {"service_id": 7, "provider": provider,
                           "ask_amount": ask_amount, "active": active}
    return record


def _patch_listing(monkeypatch):
    """Serve each record's ``test_listing`` as its on-chain entry."""
    async def _chain(self, record):
        return dict(getattr(record, "test_listing", None)
                    or {"error": "no on-chain listing"})
    monkeypatch.setattr(WebSocketBridge, "_chain_service", _chain)


@pytest.fixture(autouse=True)
def _listing(monkeypatch):
    _patch_listing(monkeypatch)


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
        # The channel's client is the test payer unless a test says otherwise.
        return {"client": _payer(), **voucher} if isinstance(voucher, dict) else voucher

    monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)
    monkeypatch.setattr(oc.ServiceMarketClient, "verify_voucher",
                        _verify_voucher)


PAYER_KEY = "0x" + "11" * 32


def _payer() -> str:
    from eth_account import Account
    return Account.from_key(PAYER_KEY).address


def _signed(req: dict, record, key: str = PAYER_KEY) -> dict:
    """A direct-path request signed by the paying key, as a buyer sends it."""
    from atn.service_client import sign_request
    provider = record.test_listing["provider"] or PROVIDER_ADDR
    try:
        sig = sign_request(key, req["request_id"], record.digest, {}, provider)
    except ValueError:
        return dict(req)   # a malformed id the gate refuses before any signature
    return {**req, "payer_signature": sig}

_RIDS = iter(range(1, 10 ** 6))


def _vsigned(req: dict, record, key: str = PAYER_KEY, rid: str = "") -> dict:
    """A voucher request signed by the channel's client, as a buyer sends it
    (a fresh request id each time, as the buyer makes one per request)."""
    rid = rid or "0x" + f"{next(_RIDS):064x}"
    return _signed({**req, "request_id": rid}, record, key)



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
            _signed({"tx_hash": "0x" + "ab" * 32, "request_id": "0x" + "01" * 32}, record),
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
            _signed({"tx_hash": "0xabc", "request_id": "0x" + "02" * 32}, record), record)
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
            _signed({"tx_hash": "0xabc", "request_id": "0x" + "03" * 32}, record), record)
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
            _signed({"tx_hash": "0xabc", "request_id": rid}, record), record)
        assert first["ok"] is True
        # Second time with the SAME request_id -> replay rejected, even though
        # the (patched) chain would still verify it.
        second = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": rid}, record), record)
        assert second["ok"] is False
        assert "replay" in second["reason"]

    @pytest.mark.asyncio
    async def test_reject_no_provider_address(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt, provider="")  # no on-chain provider
        _patch_chain(monkeypatch, verify={"verified": True})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": "0x" + "05" * 32}, record), record)
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
            "provider": PROVIDER_ADDR, "client": _payer(), "status": 1})
        server = WebSocketBridge(rt)

        out = await server._validate_service_payment(
            _vsigned({"channel_id": 7, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record), record)
        assert out["ok"] is True, out
        # Reserved: it counts at once, but is kept to settle with only once
        # the request is served.
        assert rt.service_store.last_channel_cumulative(7) == ASK_AMOUNT
        assert rt.service_store.get_channel_voucher(7) is None
        assert out["proof"] == "voucher" and out["reservation"]
        rt.service_store.keep_channel_voucher(out["reservation"])
        stored = rt.service_store.get_channel_voucher(7)
        assert stored and stored["signature"] == "0xsig"
        assert stored["service_id"] == 7 and stored["ask"] == ASK_AMOUNT

    @pytest.mark.asyncio
    async def test_voucher_request_must_be_signed_by_the_channel_client(
            self, tmp_path, monkeypatch):
        # A relay that saw a voucher cannot spend it on its own request.
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "reason": "ok",
            "provider": PROVIDER_ADDR, "client": _payer(), "status": 1})
        server = WebSocketBridge(rt)
        bare = {"channel_id": 7, "cumulative_amount": ASK_AMOUNT, "signature": "0xsig"}
        out = await server._validate_service_payment(bare, record)
        assert out["ok"] is False and "signed by the channel" in out["reason"]
        other = _vsigned(bare, record, key="0x" + "22" * 32)
        out = await server._validate_service_payment(other, record)
        assert out["ok"] is False
        assert rt.service_store.last_channel_cumulative(7) == 0

    @pytest.mark.asyncio
    async def test_accept_second_increment(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "provider": PROVIDER_ADDR, "status": 1})
        server = WebSocketBridge(rt)

        first = await server._validate_service_payment(
            _vsigned({"channel_id": 9, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig1"}, record), record)
        assert first["ok"] is True
        # Next voucher must add at least another ask on top of the last.
        second = await server._validate_service_payment(
            _vsigned({"channel_id": 9, "cumulative_amount": ASK_AMOUNT * 2,
             "signature": "0xsig2"}, record), record)
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
            _vsigned({"channel_id": 1, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xbad"}, record), record)
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
            _vsigned({"channel_id": 3, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xs1"}, record), record)
        assert first["ok"] is True
        # Second voucher: increment only 1, below the ask -> rejected, and the
        # stored cumulative does NOT regress or advance.
        second = await server._validate_service_payment(
            _vsigned({"channel_id": 3, "cumulative_amount": ASK_AMOUNT + 1,
             "signature": "0xs2"}, record), record)
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
            _vsigned({"channel_id": 2, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record), record)
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
            _vsigned({"channel_id": 4, "cumulative_amount": ASK_AMOUNT,
             "signature": "0xsig"}, record), record)
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
            {"tx_hash": "0xabc", "request_id": "0xdead"}, record, local=True)
        assert out["ok"] is True
        assert "not configured" in out["reason"] or "dev" in out["reason"]
        # The remote listener fails closed instead (payment is the credential).
        remote = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": "0xdead"}, record), record)
        assert remote["ok"] is False and "cannot verify" in remote["reason"]

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


# ---------------------------------------------------------------------------
# Holes found by the 2026-10-26 survey (economy report, findings 1, 3, 4, 5)
# ---------------------------------------------------------------------------

class TestSurveyHoles:
    @pytest.mark.asyncio
    async def test_one_payment_cannot_be_respelled_into_many_requests(self, tmp_path,
                                                                     monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, verify={"verified": True, "reason": "ok"})
        server = WebSocketBridge(rt)
        rid = "0x" + "07" * 32
        first = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": rid}, record), record)
        assert first["ok"] is True
        # The chain pads or cuts to 32 bytes, so these all name one payment.
        for spelling in (rid + "0", rid + "00", rid.upper().replace("0X", "0x"),
                         rid.removeprefix("0x"), "0xab"):
            out = await server._validate_service_payment(
                _signed({"tx_hash": "0xabc", "request_id": spelling}, record), record)
            assert out["ok"] is False, spelling

    @pytest.mark.asyncio
    async def test_two_copies_sent_at_once_are_served_once(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        import threading
        gate = threading.Event()   # the check runs off-loop, in a worker thread

        async def slow_verify(self, *, request_id, recipient, min_amount, tx_hash=None,
                              payer=None):
            while not gate.is_set():
                await asyncio.sleep(0.01)
            return {"verified": True, "reason": "ok"}
        monkeypatch.setattr(oc.OnChainService, "verify_service_payment", slow_verify)
        server = WebSocketBridge(rt)
        req = _signed({"tx_hash": "0xabc", "request_id": "0x" + "08" * 32}, record)
        a = asyncio.create_task(server._validate_service_payment(dict(req), record))
        b = asyncio.create_task(server._validate_service_payment(dict(req), record))
        await asyncio.sleep(0.05)
        gate.set()
        results = await asyncio.gather(a, b)
        assert sorted(r["ok"] for r in results) == [False, True]

    @pytest.mark.asyncio
    async def test_only_the_payer_redeems_its_payment(self, tmp_path, monkeypatch):
        """A request id read off the chain is worthless without the payer's
        key: unsigned requests are refused, and the gate checks the payment
        against whoever signed (the chain says who paid)."""
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        seen_payers: list = []

        async def _verify(self, *, request_id, recipient, min_amount,
                          tx_hash=None, payer=None):
            seen_payers.append(payer)
            if payer != _payer():
                return {"verified": False, "reason": "payer mismatch"}
            return {"verified": True, "reason": "ok"}

        monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)
        server = WebSocketBridge(rt)
        rid = "0x" + "0a" * 32
        unsigned = await server._validate_service_payment(
            {"tx_hash": "0xabc", "request_id": rid}, record)
        assert unsigned["ok"] is False and "signature" in unsigned["reason"]
        thief = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": rid}, record,
                    key="0x" + "22" * 32), record)
        assert thief["ok"] is False
        # The thief's attempt did not burn the payer's request id.
        assert not rt.service_store.has_served_request(rid)
        owner = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": rid}, record), record)
        assert owner["ok"] is True
        assert seen_payers[-1] == _payer()

    @pytest.mark.asyncio
    async def test_voucher_beyond_the_deposit_is_refused(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "status": 1, "provider": PROVIDER_ADDR,
            "deposit": "1", "exceeds_deposit": True})
        server = WebSocketBridge(rt)
        out = await server._validate_service_payment(
            _vsigned({"channel_id": 3, "cumulative_amount": ASK_AMOUNT * 50,
             "signature": "0xsig"}, record), record)
        assert out["ok"] is False and "deposit" in out["reason"]
        assert rt.service_store.last_channel_cumulative(3) == 0

    @pytest.mark.asyncio
    async def test_free_service_needs_no_proof(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt, ask_amount=0, name="free_svc")
        server = WebSocketBridge(rt)
        out = await server._validate_service_payment({}, record)
        assert out["ok"] is True and "free" in out["reason"]


# ---------------------------------------------------------------------------
# The chain is the only reference (Decision 2026-11-10)
# ---------------------------------------------------------------------------

class TestChainIsTheReference:
    @pytest.mark.asyncio
    async def test_price_is_read_from_the_chain(self, tmp_path, monkeypatch):
        """A price changed by updateServiceAsk applies at once: the gate
        charges the on-chain ask, for both proofs."""
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt, ask_amount=ASK_AMOUNT * 3)
        minimums: list = []

        async def _verify(self, *, request_id, recipient, min_amount,
                          tx_hash=None, payer=None):
            minimums.append(min_amount)
            return {"verified": True, "reason": "ok"}
        monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "status": 1, "provider": PROVIDER_ADDR})
        monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)
        server = WebSocketBridge(rt)  # verify re-patched after _patch_chain

        out = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": "0x" + "0e" * 32},
                    record), record)
        assert out["ok"] is True and minimums == [ASK_AMOUNT * 3]
        assert out["ask_amount"] == ASK_AMOUNT * 3

        low = await server._validate_service_payment(
            _vsigned({"channel_id": 4, "cumulative_amount": ASK_AMOUNT,
                      "signature": "0xsig"}, record), record)
        assert low["ok"] is False
        assert f"< ask {ASK_AMOUNT * 3}" in low["reason"]

    @pytest.mark.asyncio
    async def test_retired_on_chain_is_refused(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt, active=False)
        _patch_chain(monkeypatch, verify={"verified": True, "reason": "ok"})
        server = WebSocketBridge(rt)
        out = await server._validate_service_payment(
            _signed({"tx_hash": "0xabc", "request_id": "0x" + "0f" * 32},
                    record), record)
        assert out["ok"] is False and "retired on chain" in out["reason"]

    @pytest.mark.asyncio
    async def test_unreadable_listing_is_refused(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        req = _signed({"tx_hash": "0xabc", "request_id": "0x" + "10" * 32},
                      record)
        record.test_listing = None
        _patch_chain(monkeypatch, verify={"verified": True, "reason": "ok"})
        server = WebSocketBridge(rt)
        out = await server._validate_service_payment(req, record)
        assert out["ok"] is False and "on-chain listing" in out["reason"]


class TestChainService:
    """WebSocketBridge._chain_service itself: which entry is a service's."""

    def _server(self, rt, monkeypatch, *, entries, signer=PROVIDER_ADDR):
        monkeypatch.setattr(oc.ServiceMarketClient, "registry_available", True,
                            raising=False)

        async def _get(self, service_id):
            return entries.get(int(service_id))

        async def _list(self, start=1, limit=None):
            return [dict(v) for v in entries.values()]
        monkeypatch.setattr(oc.ServiceMarketClient, "get_service", _get)
        monkeypatch.setattr(oc.ServiceMarketClient, "list_services", _list)
        server = WebSocketBridge(rt)
        monkeypatch.setattr(server, "_service_provider_address",
                            lambda author: signer)
        return server

    @staticmethod
    def _record(rt):
        res = rt.service_store.register(
            name="svc", description="A service.", input_schema=SCHEMA,
            author="user", backing_tool="deadbeef")
        return rt.service_store.get(res["digest"])

    @pytest.mark.asyncio
    async def test_reads_the_registered_number(self, tmp_path, monkeypatch):
        monkeypatch.undo()
        rt = _make_runtime(tmp_path, chain=True)
        record = self._record(rt)
        rt.service_store.set_service_id(record.digest, 5)
        server = self._server(rt, monkeypatch, entries={5: {
            "service_id": 5, "provider": PROVIDER_ADDR,
            "spec_digest": record.digest, "ask_amount": "42", "active": True}})
        out = await server._chain_service(record)
        assert out == {"service_id": 5, "provider": PROVIDER_ADDR,
                       "ask_amount": 42, "active": True}

    @pytest.mark.asyncio
    async def test_refuses_an_entry_for_another_spec(self, tmp_path, monkeypatch):
        monkeypatch.undo()
        rt = _make_runtime(tmp_path, chain=True)
        record = self._record(rt)
        rt.service_store.set_service_id(record.digest, 5)
        server = self._server(rt, monkeypatch, entries={5: {
            "service_id": 5, "provider": PROVIDER_ADDR,
            "spec_digest": "ab" * 32, "ask_amount": "42", "active": True}})
        out = await server._chain_service(record)
        assert "another spec" in out["error"]

    @pytest.mark.asyncio
    async def test_ignores_a_copy_registered_by_another_address(
            self, tmp_path, monkeypatch):
        """Anyone can register a public spec's digest at price 0: without a
        number of its own, the daemon matches only its signer's entries."""
        monkeypatch.undo()
        rt = _make_runtime(tmp_path, chain=True)
        record = self._record(rt)
        server = self._server(rt, monkeypatch, entries={
            1: {"service_id": 1, "provider": CLIENT_ADDR,
                "spec_digest": record.digest, "ask_amount": "0", "active": True},
            2: {"service_id": 2, "provider": PROVIDER_ADDR,
                "spec_digest": record.digest, "ask_amount": "42",
                "active": True}})
        out = await server._chain_service(record)
        assert out["service_id"] == 2 and out["ask_amount"] == 42
        # The number is kept, so the next read goes straight to it.
        assert rt.service_store.get(record.digest).service_id == 2

    @pytest.mark.asyncio
    async def test_refuses_a_numbered_entry_of_another_address(
            self, tmp_path, monkeypatch):
        monkeypatch.undo()
        rt = _make_runtime(tmp_path, chain=True)
        record = self._record(rt)
        rt.service_store.set_service_id(record.digest, 1)
        server = self._server(rt, monkeypatch, entries={1: {
            "service_id": 1, "provider": CLIENT_ADDR,
            "spec_digest": record.digest, "ask_amount": "0", "active": True}})
        out = await server._chain_service(record)
        assert "another address" in out["error"]


# ---------------------------------------------------------------------------
# Not served = not paid: a voucher is kept only for a request that was served
# ---------------------------------------------------------------------------

class TestNotServedNotPaid:
    """_handle_service_request keeps a voucher the gate reserved only when
    the dispatch succeeds, and every reply says whether it charged, which
    is what the buyer's book follows."""

    @staticmethod
    def _setup(tmp_path, monkeypatch, outcomes):
        rt = _make_runtime(tmp_path, chain=True)
        record = _register_service(rt)
        _patch_chain(monkeypatch, voucher={
            "valid": True, "status": 1, "provider": PROVIDER_ADDR})
        server = WebSocketBridge(rt)
        calls = []
        # The backing is checked before the payment gate: make it resolve
        # (the dispatch itself is stubbed below).
        monkeypatch.setattr(rt.tool_store, "get", lambda digest: object())

        async def dispatch(self, record, args, msg_id, request_id, **_backing):
            calls.append(request_id)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                outcome = await outcome()
            return {"msg_id": msg_id, "ok": outcome,
                    "result": {"request_id": request_id,
                               **({} if outcome else {"error": "backing failed"})}}
        monkeypatch.setattr(WebSocketBridge, "_dispatch_service", dispatch)
        return rt, record, server, calls

    @staticmethod
    def _frame(record, cid, cum, rid=""):
        return {"spec_digest": record.digest, "args": {},
                **_vsigned({"channel_id": cid, "cumulative_amount": cum,
                            "signature": f"0xsig{cum}"}, record, rid=rid)}

    @pytest.mark.asyncio
    async def test_failed_dispatch_is_not_charged(self, tmp_path, monkeypatch):
        rt, record, server, calls = self._setup(tmp_path, monkeypatch, [False, True])
        store = rt.service_store
        frame = self._frame(record, 5, ASK_AMOUNT)
        out = await server._handle_service_request(frame, "m1")
        assert out["ok"] is False and out["charged"] is False
        assert store.last_channel_cumulative(5) == 0
        assert store.get_channel_voucher(5) is None
        # A copy of the failed request is refused, and never dispatched.
        again = await server._handle_service_request(dict(frame), "m2")
        assert again["ok"] is False and "replay" in again["error"]
        assert len(calls) == 1
        # The buyer did not count it, so its next voucher is the same total:
        # accepted, served and kept.
        nxt = await server._handle_service_request(
            self._frame(record, 5, ASK_AMOUNT), "m3")
        assert nxt["ok"] is True and nxt["charged"] is True
        assert store.get_channel_voucher(5)["cumulative"] == ASK_AMOUNT

    @pytest.mark.asyncio
    async def test_served_request_keeps_its_voucher(self, tmp_path, monkeypatch):
        rt, record, server, _ = self._setup(tmp_path, monkeypatch, [True, False])
        store = rt.service_store
        out = await server._handle_service_request(
            self._frame(record, 6, ASK_AMOUNT), "m1")
        assert out["ok"] is True and out["charged"] is True
        assert store.get_channel_voucher(6)["cumulative"] == ASK_AMOUNT
        # A failed request after it leaves the kept voucher where it was.
        out = await server._handle_service_request(
            self._frame(record, 6, ASK_AMOUNT * 2), "m2")
        assert out["charged"] is False
        assert store.last_channel_cumulative(6) == ASK_AMOUNT
        assert store.get_channel_voucher(6)["cumulative"] == ASK_AMOUNT

    @pytest.mark.asyncio
    async def test_a_copy_sent_while_serving_is_refused(self, tmp_path, monkeypatch):
        release = asyncio.Event()

        async def slow():
            await release.wait()
            return True
        rt, record, server, calls = self._setup(tmp_path, monkeypatch, [slow])
        frame = self._frame(record, 7, ASK_AMOUNT)
        first = asyncio.create_task(server._handle_service_request(frame, "m1"))
        while not calls:
            await asyncio.sleep(0.01)
        # Same total on another request id: it adds nothing to the reserved one.
        copy = await server._handle_service_request(
            self._frame(record, 7, ASK_AMOUNT), "m2")
        assert copy["ok"] is False and copy["charged"] is False
        assert "increment" in copy["error"]
        release.set()
        assert (await first)["charged"] is True
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_crash_in_the_backing_releases_the_voucher(self, tmp_path,
                                                              monkeypatch):
        rt, record, server, _ = self._setup(
            tmp_path, monkeypatch, [RuntimeError("backing crashed")])
        with pytest.raises(RuntimeError):
            await server._handle_service_request(
                self._frame(record, 8, ASK_AMOUNT), "m1")
        assert rt.service_store.last_channel_cumulative(8) == 0
        assert not rt.service_store.has_pending_voucher(8)

    @pytest.mark.asyncio
    async def test_refusals_say_nothing_was_charged(self, tmp_path, monkeypatch):
        rt, record, server, calls = self._setup(tmp_path, monkeypatch, [])
        low = await server._handle_service_request(
            self._frame(record, 9, ASK_AMOUNT - 1), "m1")
        assert low["ok"] is False and low["charged"] is False
        unknown = await server._handle_service_request(
            {"spec_digest": "ab" * 32, "request_id": "0x" + "01" * 32}, "m2")
        assert unknown["charged"] is False and calls == []

    @pytest.mark.asyncio
    async def test_a_direct_payment_is_charged_whatever_the_dispatch(
            self, tmp_path, monkeypatch):
        rt, record, server, _ = self._setup(tmp_path, monkeypatch, [False])
        _patch_chain(monkeypatch, verify={"verified": True, "reason": "ok"})
        frame = {"spec_digest": record.digest, "args": {},
                 **_signed({"tx_hash": "0xabc", "request_id": "0x" + "1a" * 32},
                           record)}
        out = await server._handle_service_request(frame, "m1")
        assert out["ok"] is False and out["charged"] is True
