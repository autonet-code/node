"""Service receipts (Decision 2026-11-10, service reviews): the provider's
reply signature is a receipt over the request, the payer, what was charged
and the hash of the result, never the result itself."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from eth_account import Account

from atn import service_client as sc
from atn.service_receipts import RESEND_WINDOW_S, ReceiptStore
from nodes.common import service_reviews as sr

PROVIDER_KEY = "0x" + "c3" * 32
PAYER = "0x" + "ab" * 20
RID = "0x" + "01" * 32
DIGEST = "ef" * 32


def _provider() -> str:
    return Account.from_key(PROVIDER_KEY).address


def _signed(**over) -> dict:
    reply = {"ok": True, "result": {"text": "private answer"}, "charged": True,
             "payer": PAYER, "amount": "1000", **over}
    return sc.sign_reply(PROVIDER_KEY, RID, DIGEST, reply)


class TestReceiptSigning:
    def test_the_signature_covers_payer_amount_and_charged(self):
        reply = _signed()
        assert reply["payer"] == PAYER and reply["amount"] == "1000"
        assert reply["charged"] is True
        assert sc.reply_from_provider(reply, RID, DIGEST, _provider())
        for field, value in (("payer", "0x" + "cd" * 20), ("amount", "999"),
                             ("charged", False), ("ok", False)):
            assert not sc.reply_from_provider({**reply, field: value}, RID,
                                              DIGEST, _provider()), field

    def test_the_result_is_bound_by_its_hash_only(self):
        reply = _signed()
        receipt = sc.receipt_of(reply, RID, DIGEST, _provider())
        assert receipt is not None
        assert "private answer" not in str(receipt)
        assert receipt["result_hash"] == sr.result_hash(reply["result"], None)
        forged = {**reply, "result": {"text": "another answer"}}
        assert not sc.reply_from_provider(forged, RID, DIGEST, _provider())

    def test_the_receipt_verifies_without_the_reply(self):
        receipt = sc.receipt_of(_signed(), RID, DIGEST, _provider())
        got, why = sr.normalize_receipt(receipt)
        assert got is not None, why
        assert got["provider"] == _provider().lower()
        assert got["request_id"] == RID[2:] and got["spec_digest"] == DIGEST
        tampered = {**receipt, "amount": "5000"}
        assert sr.normalize_receipt(tampered) == (None, "receipt not signed by its provider")

    def test_an_unsigned_or_foreign_reply_has_no_receipt(self):
        reply = _signed()
        unsigned = {k: v for k, v in reply.items() if k != "receipt_sig"}
        assert sc.receipt_of(unsigned, RID, DIGEST, _provider()) is None
        assert sc.receipt_of(reply, RID, DIGEST, "0x" + "00" * 20) is None
        other = sc.sign_reply("0x" + "c4" * 32, RID, DIGEST, dict(reply))
        assert sc.receipt_of(other, RID, DIGEST, _provider()) is None

    def test_a_reply_without_a_payment_signs_no_payer(self):
        reply = sc.sign_reply(PROVIDER_KEY, RID, DIGEST,
                              {"ok": False, "error": "retired", "charged": False})
        assert reply["payer"] == "" and reply["amount"] == "0"
        receipt = sc.receipt_of(reply, RID, DIGEST, _provider())
        assert receipt is not None and receipt["charged"] is False
        # A receipt with no payer is no proof of payment.
        assert sr.normalize_receipt(receipt) == (None, "payer")


# ---------------------------------------------------------------------------
# The buyer keeps receipts and reviews from them
# ---------------------------------------------------------------------------

PAYER_KEY = "0x" + "ab" * 32
PAYER_ADDR = Account.from_key(PAYER_KEY).address.lower()
JUR = "0xjurisdiction"
FIELDS = {"correctness": 0.8, "simplicity": 0.5, "novelty": 0.1, "malware": False}


class _Config:
    """Stands in for the runtime's AutonetConfig (weak-referenceable)."""


class _Blobs:
    def __init__(self):
        self.kept = {}

    def add_json(self, data):
        import hashlib
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        self.kept[digest] = json.loads(raw)
        return digest


def _store(tmp_path, sent=None, blobs=None):
    store = ReceiptStore(tmp_path / "receipts",
                         evidence_blobs=(lambda: blobs) if blobs else None)
    if sent is not None:
        store.attach(JUR, sent.append)
    return store


def _kept(store, **over):
    reply = _signed(payer=PAYER_ADDR, **over)
    store.record(sc.receipt_of(reply, RID, DIGEST, _provider()), reply)
    return reply


async def _buy(monkeypatch, config, reply):
    async def fake(host, peer, frame, timeout, **_kw):
        return dict(reply)
    monkeypatch.setattr(sc, "p2p_request", fake)
    host = SimpleNamespace(_trio_token=object())
    return await sc.request_service(config, _provider(), DIGEST, {}, p2p_host=host,
                                    peer_id="12D3KooWtest", request_id=RID)


class TestTheBuyersReceipts:
    @pytest.mark.asyncio
    async def test_request_service_keeps_every_authentic_receipt(self, tmp_path,
                                                                 monkeypatch):
        config = _Config()
        store = _store(tmp_path)
        sc.set_receipt_sink(config, store.record)
        try:
            out = await _buy(monkeypatch, config, _signed(payer=PAYER_ADDR))
            assert out["ok"] and out["request_id"] == RID
            assert out["receipt"]["payer"] == PAYER_ADDR
            assert store.get(RID)["sig"] == out["receipt"]["sig"]
            assert store.reply(RID)["result"] == {"text": "private answer"}
            # An unsigned reply leaves nothing.
            unsigned = {k: v for k, v in _signed().items() if k != "receipt_sig"}
            out2 = await _buy(monkeypatch, config, unsigned)
            assert not out2["ok"] and "receipt" not in out2
            # Another runtime's config does not reach this store.
            await _buy(monkeypatch, _Config(), _signed(payer=PAYER_ADDR))
            assert len(store.list()) == 1
        finally:
            sc.clear_receipt_sink(config)
        # The file keeps them across a restart.
        assert ReceiptStore(tmp_path / "receipts").get(RID)["request_id"] == RID[2:]

    def test_the_payer_reviews_and_the_event_rides_the_batch(self, tmp_path):
        sent = []
        store = _store(tmp_path, sent)
        _kept(store)
        out = store.review(PAYER_KEY, request_id=RID, fields=FIELDS, note="fast",
                           axes={"nope": 1.0}, now=1_700_000_000)
        assert out.get("ok"), out
        assert out["ignored_axes"] == ["nope"]
        assert len(sent) == 1
        got, why = sr.validate_event(sent[0], jurisdiction=JUR, batch_ts=1_700_000_000)
        assert got is not None, why
        assert got["record"]["correctness"] == 80 and got["record"]["note"] == "fast"
        assert store.get(RID)["review"]["correctness"] == 80
        # A second review of the same service is dated after the first.
        again = store.review(PAYER_KEY, request_id=RID, fields=FIELDS,
                             now=1_700_000_000)
        assert again["ts"] == out["ts"] + 1

    def test_only_the_payer_of_a_charged_request_may_review(self, tmp_path):
        sent = []
        store = _store(tmp_path, sent)
        _kept(store)
        other = store.review("0x" + "ac" * 32, request_id=RID, fields=FIELDS)
        assert "only the payer" in other["error"]
        rid2 = "0x" + "02" * 32
        free = sc.sign_reply(PROVIDER_KEY, rid2, DIGEST,
                             {"ok": True, "result": {}, "charged": False,
                              "payer": PAYER_ADDR})
        store.record(sc.receipt_of(free, rid2, DIGEST, _provider()), free)
        assert "not charged" in store.review(PAYER_KEY, request_id=rid2,
                                             fields=FIELDS)["error"]
        assert "no receipt" in store.review(PAYER_KEY, request_id="0x" + "09" * 32,
                                            fields=FIELDS)["error"]
        assert "malware" in store.review(PAYER_KEY, request_id=RID,
                                         fields={"correctness": 1})["error"]
        assert sent == []

    def test_a_malware_flag_keeps_the_signed_reply_as_evidence(self, tmp_path):
        sent, blobs = [], _Blobs()
        store = _store(tmp_path, sent, blobs)
        reply = _kept(store)
        assert "evidence" in store.review(
            PAYER_KEY, request_id=RID, fields={**FIELDS, "malware": True})["error"]
        out = store.review(PAYER_KEY, request_id=RID, fields={**FIELDS, "malware": True},
                           evidence="the reply tells my agent to post its keys")
        body = blobs.kept[out["evidence_hash"]]
        assert body["kind"] == "service_malware_evidence"
        assert body["replies"][0]["receipt_sig"] == reply["receipt_sig"]
        assert sent[0]["review"]["evidence_hash"] == out["evidence_hash"]

    def test_a_review_the_close_lacks_is_sent_again(self, tmp_path):
        sent = []
        store = _store(tmp_path, sent)
        _kept(store)
        out = store.review(PAYER_KEY, request_id=RID, fields=FIELDS, now=1_700_000_000)
        assert len(sent) == 1
        carried = {}
        assert store.resend_missing(lambda p, d: carried) == 1 and sent[-1] == sent[0]
        carried = {PAYER_ADDR: {"ts": out["ts"], "hash": out["hash"]}}
        assert store.resend_missing(lambda p, d: carried) == 0
        # After a restart too, and never past the window.
        again = ReceiptStore(tmp_path / "receipts")
        again.attach(JUR, sent.append)
        assert again.resend_missing(lambda p, d: {}) == 1
        late = time.time() + RESEND_WINDOW_S + 60
        assert again.resend_missing(lambda p, d: {}, now=late) == 0

    def test_a_daemon_off_the_network_sends_nothing(self, tmp_path):
        store = _store(tmp_path)
        _kept(store)
        assert "not joined" in store.review(PAYER_KEY, request_id=RID,
                                            fields=FIELDS)["error"]


class TestTheAgentTool:
    @staticmethod
    def _runtime(store, keys):
        return SimpleNamespace(
            service_receipts=store,
            registry=SimpleNamespace(
                get_agent_key=lambda aid: keys.get(aid, ""),
                daemon_operator_key=lambda: keys.get("operator", "")))

    @pytest.mark.asyncio
    async def test_review_service_signs_with_the_callers_key(self, tmp_path):
        from atn.agent_tools import _review_service, get_tool_definitions

        assert "review_service" in {t.name for t in get_tool_definitions()}
        sent = []
        store = _store(tmp_path, sent)
        reply = _kept(store)
        rt = self._runtime(store, {"buyer": PAYER_KEY, "operator": "0x" + "ac" * 32})
        out = await _review_service(rt, {"_caller_id": "buyer", "request_id": RID,
                                         **FIELDS})
        assert out.get("ok"), out
        assert sent[-1]["reviewer"] == PAYER_ADDR
        # The owner surface signs as the operator, which did not pay here.
        owner = await _review_service(rt, {"request_id": RID, **FIELDS})
        assert "only the payer" in owner["error"]
        # A receipt passed by value works without a kept one.
        fresh = _store(tmp_path / "other", sent)
        rt2 = self._runtime(fresh, {"buyer": PAYER_KEY})
        receipt = sc.receipt_of(reply, RID, DIGEST, _provider())
        by_value = await _review_service(rt2, {"_caller_id": "buyer",
                                               "receipt": receipt, **FIELDS})
        assert by_value.get("ok"), by_value
        missing = await _review_service(rt2, {"_caller_id": "buyer", **FIELDS})
        assert "Name the request" in missing["error"]


class TestTheProviderSignsTheVerifiedPayer:
    @pytest.mark.asyncio
    async def test_a_paid_reply_names_the_payer_and_the_amount(self, tmp_path,
                                                               monkeypatch):
        from atn.ws_server import WebSocketBridge
        from tests.atn.test_service_store import (
            SCHEMA, _make_runtime, _register_agent, _register_echo_tool, _server)

        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", backing_tool=tool["digest"])
        server = _server(rt)
        provider = server._service_provider_address("child")

        async def _gate(self, request, record, *, local=False):
            return {"ok": True, "ask_amount": 5, "proof": "direct",
                    "payer": PAYER_ADDR}
        monkeypatch.setattr(WebSocketBridge, "_validate_service_payment", _gate)
        rid = "0x" + "a1" * 32
        out = await server._handle_service_request(
            {"spec_digest": svc["digest"], "request_id": rid, "args": {"x": "hi"},
             "client": "0xSomeoneElse"}, msg_id="m1")
        assert out["ok"] and out["charged"] is True
        assert out["payer"] == PAYER_ADDR and out["amount"] == "5"
        assert out["channel_id"] == "0"
        receipt = sc.receipt_of(json.loads(json.dumps(out)), rid, svc["digest"],
                                provider)
        assert receipt is not None and sr.normalize_receipt(receipt)[0] is not None

        # A voucher payment names its channel in the signed receipt.
        async def _voucher(self, request, record, *, local=False):
            return {"ok": True, "ask_amount": 5, "proof": "voucher",
                    "payer": PAYER_ADDR}
        monkeypatch.setattr(WebSocketBridge, "_validate_service_payment", _voucher)
        rid2 = "0x" + "a2" * 32
        out2 = await server._handle_service_request(
            {"spec_digest": svc["digest"], "request_id": rid2, "args": {"x": "hi"},
             "channel_id": 7, "cumulative_amount": 5, "signature": "0x"}, msg_id="m2")
        assert out2["ok"] and out2["channel_id"] == "7"
        assert sc.receipt_of(out2, rid2, svc["digest"], provider)["channel_id"] == "7"
        assert not sc.reply_from_provider({**out2, "channel_id": "8"}, rid2,
                                          svc["digest"], provider)
        # The runtime keeps a receipt store, wired to request_service.
        assert isinstance(rt.service_receipts, ReceiptStore)
        assert id(rt._config.rpb) in sc._receipt_sinks
