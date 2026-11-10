"""Service receipts (Decision 2026-11-10, service reviews): the provider's
reply signature is a receipt over the request, the payer, what was charged
and the hash of the result, never the result itself."""

from __future__ import annotations

from eth_account import Account

from atn import service_client as sc
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
