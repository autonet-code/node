"""The provider signs every reply (``reply_sig``, atn/service_client.py) so
a relay or a wss impostor cannot make up an answer; the signature covers
the request id, the spec, the provider, the outcome, the result's hash and
the credit counts it reports. It authenticates the reply; it is not a
receipt."""

from __future__ import annotations

from eth_account import Account

from atn import service_client as sc

PROVIDER_KEY = "0x" + "c3" * 32
RID = "0x" + "01" * 32
DIGEST = "ef" * 32


def _provider() -> str:
    return Account.from_key(PROVIDER_KEY).address


def _signed(**over) -> dict:
    reply = {"msg_id": "m", "ok": True, "result": {"request_id": RID, "result": "hi"},
             "credit": {"channel_id": 5, "consumed": 1000, "cumulative": 5000}, **over}
    return sc.sign_reply(PROVIDER_KEY, RID, DIGEST, reply)


def test_a_signed_reply_is_the_providers():
    reply = _signed()
    assert sc.reply_signer(reply, RID, DIGEST, _provider()) == _provider()
    assert sc.reply_from_provider(reply, RID, DIGEST, _provider())
    # Read back after the wire (JSON) and with the digest spelled 0x-prefixed.
    import json
    wire = json.loads(json.dumps(reply))
    assert sc.reply_from_provider(wire, RID, "0x" + DIGEST.upper(), _provider())


def test_the_signature_covers_the_credit_counts_and_the_answer():
    for field, value in (("credit", {"channel_id": 5, "consumed": 0, "cumulative": 5000}),
                         ("credit", {"channel_id": 5, "consumed": 1000, "cumulative": 9000}),
                         ("credit", {"channel_id": 6, "consumed": 1000, "cumulative": 5000}),
                         ("result", {"request_id": RID, "result": "made up"}),
                         ("ok", False),
                         ("error", "made up")):
        tampered = {**_signed(), field: value}
        assert not sc.reply_from_provider(tampered, RID, DIGEST, _provider()), field
    # Another request, another spec, another provider: not this reply.
    reply = _signed()
    assert not sc.reply_from_provider(reply, "0x" + "02" * 32, DIGEST, _provider())
    assert not sc.reply_from_provider(reply, RID, "aa" * 32, _provider())
    other = Account.from_key("0x" + "c4" * 32).address
    assert not sc.reply_from_provider(reply, RID, DIGEST, other)


def test_an_unsigned_or_malformed_reply_is_nobodys():
    assert sc.reply_signer({"ok": True}, RID, DIGEST, _provider()) == ""
    assert sc.reply_signer({"ok": True, "reply_sig": "0xnope"}, RID, DIGEST,
                           _provider()) == ""
    assert sc.reply_signer("not a reply", RID, DIGEST, _provider()) == ""
    # A refusal without credits is signed too, and stays the provider's.
    refusal = sc.sign_reply(PROVIDER_KEY, RID, DIGEST,
                            {"ok": False, "error": "Payment refused: no payment"})
    assert sc.reply_from_provider(refusal, RID, DIGEST, _provider())


def test_a_request_is_signed_by_its_payer():
    payer_key = "0x" + "11" * 32
    args = {"x": 1}
    sig = sc.sign_request(payer_key, RID, DIGEST, args, _provider())
    payer = Account.from_key(payer_key).address
    assert sc.recover_request_signer(sig, RID, DIGEST, args, _provider()) == payer
    # Other args, or another provider: someone else signed it.
    assert sc.recover_request_signer(sig, RID, DIGEST, {"x": 2}, _provider()) != payer
    assert sc.recover_request_signer(sig, RID, DIGEST, args, "0x" + "33" * 20) != payer
