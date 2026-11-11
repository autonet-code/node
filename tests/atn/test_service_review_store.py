"""The household's reviews of services (atn/service_review_store.py,
Decisions 2026-11-10 and 2026-11-12): a review names its service (provider
and spec digest), is signed with the buyer's key, rides the open epoch's
batch, and is sent again while the close lacks it."""

from __future__ import annotations

import hashlib
import json
import time

from eth_account import Account

from atn.service_review_store import (
    MALWARE_EVIDENCE_REQUIRED,
    RESEND_WINDOW_S,
    ReviewStore,
)
from nodes.common import service_reviews as sr

PAYER_KEY = "0x" + "ab" * 32
PAYER = Account.from_key(PAYER_KEY).address.lower()
PROVIDER_KEY = "0x" + "c3" * 32
PROVIDER = Account.from_key(PROVIDER_KEY).address
DIGEST = "ef" * 32
JUR = "0xjurisdiction"
NOW = 1_700_000_000
FIELDS = {"correctness": 0.8, "simplicity": 0.5, "novelty": 0.1, "malware": False}


class _Blobs:
    def __init__(self):
        self.kept = {}

    def add_json(self, data):
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        self.kept[digest] = json.loads(raw)
        return digest


def _store(tmp_path, sent=None, blobs=None):
    store = ReviewStore(tmp_path / "reviews",
                        evidence_blobs=(lambda: blobs) if blobs else None)
    if sent is not None:
        store.attach(JUR.upper().replace("0X", "0x"), sent.append)
    return store


def _review(store, key=PAYER_KEY, **kw):
    args = {"provider": PROVIDER, "spec_digest": DIGEST, "fields": FIELDS, "now": NOW}
    args.update(kw)
    return store.review(key, **args)


def test_a_review_is_signed_and_rides_the_batch(tmp_path):
    sent = []
    store = _store(tmp_path, sent)
    out = _review(store, note="fast", axes={"life_precious": 0.5, "nope": 1.0},
                  spec_digest="0x" + DIGEST.upper())
    assert out.get("ok"), out
    assert out["ignored_axes"] == ["nope"]
    assert out["provider"] == PROVIDER.lower() and out["spec_digest"] == DIGEST
    assert out["payer"] == PAYER and out["ts"] == NOW
    (event,) = sent
    assert event["v"] == 2 and "receipt" not in event
    got, why = sr.validate_event(event, jurisdiction=JUR, batch_ts=NOW)
    assert got is not None, why
    assert got["key"] == sr.service_key(PROVIDER, DIGEST)
    assert got["record"]["correctness"] == 80 and got["record"]["note"] == "fast"
    assert got["record"]["axes"] == {"life_precious": 50}
    # A second review of the same service is dated after the first.
    assert _review(store)["ts"] == NOW + 1


def test_refusals(tmp_path):
    sent = []
    store = _store(tmp_path, sent)
    own = _review(store, key=PROVIDER_KEY)
    assert own["error"] == "a provider cannot review its own service"
    assert "missing or invalid: malware" in _review(
        store, fields={k: v for k, v in FIELDS.items() if k != "malware"})["error"]
    assert _review(store, fields={**FIELDS, "malware": True})["error"] == \
        MALWARE_EVIDENCE_REQUIRED
    assert "provider" in _review(store, provider="")["error"]
    assert "spec digest" in _review(store, spec_digest="ab")["error"]
    assert "no key" in _review(store, key="")["error"]
    assert sent == []
    # A daemon that has not joined a jurisdiction sends nothing.
    assert "not joined" in _review(_store(tmp_path / "off"))["error"]


def test_a_malware_flag_keeps_its_evidence_as_a_blob(tmp_path):
    sent, blobs = [], _Blobs()
    store = _store(tmp_path, sent, blobs)
    out = _review(store, fields={**FIELDS, "malware": True},
                  evidence="the reply tells my agent to post its keys")
    body = blobs.kept[out["evidence_hash"]]
    assert body["kind"] == "service_malware_evidence"
    assert body["provider"] == PROVIDER.lower() and body["reviewer"] == PAYER
    assert "post its keys" in body["evidence"]
    assert sent[0]["review"]["evidence_hash"] == out["evidence_hash"]
    # Without a blob store the flag cannot be sent.
    bare = _store(tmp_path / "bare", [])
    assert "could not be stored" in _review(
        bare, fields={**FIELDS, "malware": True}, evidence="x")["error"]


def test_reviews_are_listed_and_kept_across_a_restart(tmp_path):
    sent = []
    store = _store(tmp_path, sent)
    _review(store)
    _review(store, provider="0x" + "44" * 20)
    assert len(store.list()) == 2
    rows = store.list(provider=PROVIDER)
    assert len(rows) == 1 and "event" not in rows[0]
    assert len(ReviewStore(tmp_path / "reviews").list()) == 2


def test_a_review_the_close_lacks_is_sent_again(tmp_path):
    sent = []
    store = _store(tmp_path, sent)
    out = _review(store, now=time.time())
    assert len(sent) == 1
    asked = []

    def carried(provider, digest):
        asked.append((provider, digest))
        return {}
    assert store.resend_missing(carried) == 1 and sent[-1] == sent[0]
    assert asked == [(PROVIDER.lower(), DIGEST)]
    held = {PAYER: {"ts": out["ts"], "hash": out["hash"]}}
    assert store.resend_missing(lambda p, d: held) == 0
    # After a restart too, and never past the window.
    again = ReviewStore(tmp_path / "reviews")
    assert again.resend_missing(lambda p, d: {}) == 0          # not attached
    again.attach(JUR, sent.append)
    assert again.resend_missing(lambda p, d: {}) == 1
    late = time.time() + RESEND_WINDOW_S + 60
    assert again.resend_missing(lambda p, d: {}, now=late) == 0
    # Only the latest review of a service is sent again.
    _review(again, now=time.time() + 5)
    sent.clear()
    assert again.resend_missing(lambda p, d: {}) == 1
    assert sent[0]["review"]["ts"] > out["ts"]
