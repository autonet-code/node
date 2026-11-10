"""Service reviews in the epoch-close (Decision 2026-11-10): payer-only,
receipt-proven, one review per (provider, spec, payer), the latest wins,
ratings keyed by (provider, spec), carried in the anchored bundle."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest
from eth_account import Account

from nodes.common import service_reviews as sr
from nodes.common.authoritative_encoding import decode_carry_bundle, encode_carry_bundle
from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver
from nodes.common.federated_reconcile import federated_epoch_close

JUR = "0xjurisdiction"
TS = 1_700_000_000
PROVIDER_KEY = "0x" + "11" * 32
COPIER_KEY = "0x" + "12" * 32
PAYER_KEYS = ["0x" + f"{0x21 + i:02x}" * 32 for i in range(4)]
DIGEST = "ab" * 32


def addr(key: str) -> str:
    return Account.from_key(key).address.lower()


def receipt(provider_key=PROVIDER_KEY, payer_key=PAYER_KEYS[0], rid="01" * 32,
            digest=DIGEST, charged=True, amount=1000, ok=True):
    fields = {"request_id": rid, "spec_digest": digest,
              "provider": addr(provider_key),
              "payer": addr(payer_key) if payer_key else "",
              "ok": ok, "charged": charged, "amount": str(amount),
              "result_hash": sr.result_hash({"text": "answer"}, None)}
    fields["sig"] = sr.sign_hash(provider_key, sr.receipt_hash(fields))
    return fields


def event(payer_key=PAYER_KEYS[0], *, rcpt=None, correctness=80, simplicity=60,
          novelty=20, malware=False, evidence="", ts=TS, note="", axes=None,
          jurisdiction=JUR, signer=None, **receipt_kw):
    rcpt = rcpt or receipt(payer_key=payer_key, **receipt_kw)
    return sr.build_event(signer or payer_key, rcpt, jurisdiction=jurisdiction,
                          correctness=correctness, simplicity=simplicity,
                          novelty=novelty, malware=malware, axes=axes,
                          evidence_hash=evidence, note=note, ts=ts)


def check(ev, batch_ts=TS):
    return sr.validate_event(ev, jurisdiction=JUR, batch_ts=batch_ts)


def close(events, carried=None, **kw):
    return sr.close_service_reviews([(JUR, TS, e) for e in events],
                                    carried=carried, **kw)


KEY = sr.service_key(addr(PROVIDER_KEY), DIGEST)


class TestValidation:
    def test_a_paid_review_is_valid(self):
        got, why = check(event())
        assert got is not None, why
        assert got["key"] == KEY and got["payer"] == addr(PAYER_KEYS[0])
        assert got["record"]["amount"] == 1000 and got["record"]["correctness"] == 80

    def test_a_forged_receipt_is_refused(self):
        forged = receipt(amount=1000)
        forged["amount"] = "9999"
        assert check(event(rcpt=forged)) == (None, "receipt not signed by its provider")
        # Signed by someone else than the provider it names.
        other = receipt()
        other["sig"] = sr.sign_hash(COPIER_KEY, sr.receipt_hash(other))
        assert check(event(rcpt=other))[1] == "receipt not signed by its provider"

    def test_only_the_payer_may_review(self):
        rcpt = receipt(payer_key=PAYER_KEYS[0])
        ev = event(rcpt=rcpt, signer=PAYER_KEYS[1])
        assert check(ev) == (None, "the reviewer is not the payer")
        # Naming the payer as reviewer without its key fails the signature.
        ev["reviewer"] = addr(PAYER_KEYS[0])
        assert check(ev) == (None, "review not signed by the payer")

    def test_a_receipt_that_charged_nothing_is_no_proof(self):
        assert check(event(charged=False)) == (None, "the receipt says nothing was charged")

    def test_unsigned_reviews_and_receipts_are_refused(self):
        ev = event()
        ev["sig"] = ""
        assert check(ev) == (None, "review signature")
        ev = event()
        ev["receipt"]["sig"] = ""
        assert check(ev) == (None, "receipt signature")
        ev = event()
        ev["review"]["correctness"] = 100       # changed after signing
        assert check(ev) == (None, "review not signed by the payer")

    def test_other_refusals(self):
        assert check(event(jurisdiction="0xother")) == (None, "another jurisdiction")
        assert check(event(malware=True)) == (None, "a malware flag needs evidence")
        assert check(event(malware=True, evidence="cd" * 32))[0] is not None
        assert check(event(correctness=101)) == (None, "correctness")
        assert check(event(ts=TS + sr.MAX_FUTURE_S + 1)) == (None, "dated after its batch")
        assert check(event(note="x" * (sr.MAX_NOTE_CHARS + 1))) == (None, "note")
        self_review = event(payer_key=PROVIDER_KEY, provider_key=PROVIDER_KEY)
        assert check(self_review) == (None, "the provider reviewing itself")
        extra = event()
        extra["review"]["bonus"] = 1
        assert check(extra) == (None, "review fields")


class TestCounting:
    def test_a_later_review_replaces_the_earlier(self):
        first = close([event(correctness=-50, ts=TS)])
        assert first["ratings"][KEY]["correctness"] == -0.5
        second = close([event(correctness=90, ts=TS + 10, rid="02" * 32)],
                       carried=first["reviews"])
        r = second["ratings"][KEY]
        assert r["payers"] == 1 and r["correctness"] == 0.9
        assert second["reviews"][KEY][addr(PAYER_KEYS[0])]["request_id"] == "02" * 32
        # A replayed older review changes nothing.
        third = close([event(correctness=-50, ts=TS)], carried=second["reviews"])
        assert third["ratings"][KEY]["correctness"] == 0.9 and third["accepted"] == 0

    def test_the_rating_is_the_mean_over_payers(self):
        out = close([event(PAYER_KEYS[0], correctness=100, simplicity=40, novelty=0,
                           amount=1000),
                     event(PAYER_KEYS[1], correctness=0, simplicity=80, novelty=50,
                           malware=True, evidence="cd" * 32, amount=3000),
                     event(PAYER_KEYS[2], correctness=-40, simplicity=60, novelty=10,
                           amount=500)])
        r = out["ratings"][KEY]
        assert r["payers"] == 3
        assert r["correctness"] == pytest.approx(0.2)
        assert r["simplicity"] == pytest.approx(0.6)
        assert r["novelty"] == pytest.approx(0.2)
        assert r["malware"] == pytest.approx(1 / 3, abs=1e-9)
        assert r["charged"] == 4500
        assert r["provider"] == addr(PROVIDER_KEY) and r["spec_digest"] == DIGEST

    def test_a_copy_under_another_address_is_rated_apart(self):
        out = close([event(PAYER_KEYS[0], correctness=100),
                     event(PAYER_KEYS[1], provider_key=COPIER_KEY, correctness=-100)])
        copy_key = sr.service_key(addr(COPIER_KEY), DIGEST)
        assert out["ratings"][KEY]["correctness"] == 1.0
        assert out["ratings"][copy_key]["correctness"] == -1.0
        assert out["ratings"][KEY]["payers"] == out["ratings"][copy_key]["payers"] == 1

    def test_the_providers_household_and_sanctioned_payers_do_not_count(self):
        evs = [event(PAYER_KEYS[0], correctness=100), event(PAYER_KEYS[1], correctness=-100),
               event(PAYER_KEYS[2], correctness=50)]
        owners = {addr(PROVIDER_KEY): "0xowner", addr(PAYER_KEYS[0]): "0xowner",
                  addr(PAYER_KEYS[1]): "0xbanned"}
        out = close(evs, agent_owner_map=owners, sanctioned_households=["0xbanned"])
        assert out["ratings"][KEY]["payers"] == 1
        assert out["ratings"][KEY]["correctness"] == 0.5
        assert len(out["reviews"][KEY]) == 3      # kept, only not counted

    def test_charter_axes_count_unscored_at_half(self):
        out = close([event(PAYER_KEYS[0], axes={"care": 100}),
                     event(PAYER_KEYS[1])], axis_ids=["care", "truth"])
        r = out["ratings"][KEY]
        assert r["axes"] == {"care": 0.75, "truth": 0.5}
        assert r["value"] == 0.625

    def test_the_payers_kept_per_service_are_bounded(self, monkeypatch):
        monkeypatch.setattr(sr, "MAX_PAYERS_PER_SERVICE", 2)
        out = close([event(k, ts=TS + i) for i, k in enumerate(PAYER_KEYS[:3])])
        assert sorted(out["reviews"][KEY]) == sorted(addr(k) for k in PAYER_KEYS[1:3])

    def test_the_result_does_not_depend_on_the_order(self):
        evs = [event(k, correctness=c, ts=TS + t, rid=f"{t + 1:02x}" * 32)
               for k in PAYER_KEYS for c, t in ((10, 1), (-20, 2), (70, 3))]
        evs += [event(PAYER_KEYS[0], provider_key=COPIER_KEY, correctness=5)]
        base = close(evs[:5])
        want = json.dumps(close(evs[5:], carried=base["reviews"]), sort_keys=True)
        rng = random.Random(7)
        for _ in range(5):
            shuffled = list(evs)
            rng.shuffle(shuffled)
            cut = rng.randint(0, len(shuffled))
            a = close(shuffled[:cut])
            carried = json.loads(json.dumps(a["reviews"]))
            b = close(shuffled[cut:], carried=dict(reversed(list(carried.items()))))
            got = json.dumps(b, sort_keys=True)
            assert json.loads(got)["ratings"] == json.loads(want)["ratings"]
            assert json.loads(got)["reviews"] == json.loads(want)["reviews"]


def _batches(events, *, kp=None, ts=TS):
    kp = kp or Keypair.generate()
    out, prev = [], b""
    for i, ev in enumerate(events, start=1):
        b = EventBatch(rpb_address=JUR, sender_pubkey=kp.public_key, batch_seq=i,
                       events=[ev], prev_batch_hash=prev, timestamp=float(ts + i))
        out.append(b)
        prev = b.content_hash()
    return kp, out


class TestTheClose:
    def test_federated_close_rates_services_from_canonical_events(self):
        _, batches = _batches([event(PAYER_KEYS[0], correctness=60),
                               event(PAYER_KEYS[1], correctness=20)])
        out = federated_epoch_close(canonical_order(batches))
        assert out["service_ratings"][KEY]["payers"] == 2
        assert out["service_ratings"][KEY]["correctness"] == 0.4
        assert out["agent_mint"] == {} and out["total_mint"] == 0
        again = federated_epoch_close(canonical_order(list(reversed(batches))))
        assert json.dumps(again["service_reviews"], sort_keys=True) == \
            json.dumps(out["service_reviews"], sort_keys=True)

    def test_the_anchored_bundle_carries_the_rating_to_a_joiner(self, tmp_path):
        kp, batches = _batches([event(PAYER_KEYS[0], correctness=60)])

        def driver(name, batches):
            gossip = MagicMock(spec=EventGossip)
            gossip.known_senders.return_value = [kp.public_key]
            gossip.sender_pubkey = kp.public_key
            gossip.drain_epoch_batches.return_value = list(batches)
            d = tmp_path / name
            return FederatedCloseDriver(
                gossip=gossip, tool_registrations_path=d / "regs.json",
                tool_rounds_path=d / "rounds.json", tool_ratings_path=d / "ratings.json",
                service_reviews_path=d / "svc_reviews.json",
                service_ratings_path=d / "svc_ratings.json")

        always_on = driver("a", batches)
        fed = always_on.run({"epoch_id": "e1"})
        assert fed is not None
        carry = decode_carry_bundle(fed.carry_bundle_blob)["carry"]
        assert carry["service_ratings"][KEY]["correctness"] == 0.6
        assert fed.close_result["authoritative_payload"]["carry_cid"] == fed.carry_cid
        assert encode_carry_bundle(carry, epoch_id="e1") == fed.carry_bundle_blob

        joiner = driver("j", [])
        joiner.install_carry_over(carry)
        assert joiner.service_rating(addr(PROVIDER_KEY), "0x" + DIGEST.upper()) == \
            always_on.service_rating(addr(PROVIDER_KEY), DIGEST)
        assert joiner.service_reviews_of(addr(PROVIDER_KEY), DIGEST)

        # The next close builds on the carried reviews, on both daemons.
        _, later = _batches([event(PAYER_KEYS[1], correctness=0, ts=TS + 50)],
                            ts=TS + 50)
        for d in (always_on, joiner):
            d.gossip.drain_epoch_batches.return_value = list(later)
        a2, j2 = always_on.run({"epoch_id": "e2"}), joiner.run({"epoch_id": "e2"})
        assert a2.carry_bundle_blob == j2.carry_bundle_blob
        assert always_on.service_rating(addr(PROVIDER_KEY), DIGEST)["payers"] == 2
        # A restart reads the same state back.
        again = driver("a", [])
        assert again.carry_over_maps() == always_on.carry_over_maps()
