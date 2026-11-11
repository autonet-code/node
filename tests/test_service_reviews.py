"""Service reviews in the epoch-close (Decisions 2026-11-10 and 2026-11-12):
a review names its service (provider + spec digest) and carries no
receipt; it counts once its reviewer has paid that provider on chain, for
any service of that provider; one review per (provider, spec, payer), the
latest wins; ratings keyed by (provider, spec), carried in the anchored
bundle."""

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
OTHER_DIGEST = "cd" * 32


def addr(key: str) -> str:
    return Account.from_key(key).address.lower()


PROVIDER = addr(PROVIDER_KEY)
KEY = sr.service_key(PROVIDER, DIGEST)
OTHER_KEY = sr.service_key(PROVIDER, OTHER_DIGEST)


def event(payer_key=PAYER_KEYS[0], *, provider_key=PROVIDER_KEY, digest=DIGEST,
          correctness=80, simplicity=60, novelty=20, malware=False, evidence="",
          ts=TS, note="", axes=None, jurisdiction=JUR):
    return sr.build_event(payer_key, provider=addr(provider_key), spec_digest=digest,
                          jurisdiction=jurisdiction, correctness=correctness,
                          simplicity=simplicity, novelty=novelty, malware=malware,
                          axes=axes, evidence_hash=evidence, note=note, ts=ts)


def check(ev, batch_ts=TS):
    return sr.validate_event(ev, jurisdiction=JUR, batch_ts=batch_ts)


def close(events, carried=None, **kw):
    return sr.close_service_reviews([(JUR, TS, e) for e in events],
                                    carried=carried, **kw)


def snap(*, registered=(KEY,), paid=None, unscored_bps=None):
    return {"registered": list(registered), "paid": dict(paid or {}),
            "unscored_bps": unscored_bps}


def paid_by(payer_key, amount=1000, provider=PROVIDER):
    return {f"{addr(payer_key)}:{provider}": amount}


class TestValidation:
    def test_a_signed_review_is_valid(self):
        got, why = check(event(axes={"care": -30}, note="fast"))
        assert got is not None, why
        assert got["key"] == KEY and got["payer"] == addr(PAYER_KEYS[0])
        rec = got["record"]
        assert (rec["correctness"], rec["simplicity"], rec["novelty"]) == (80, 60, 20)
        assert rec["axes"] == {"care": -30} and rec["note"] == "fast"
        assert rec["paid"] is False and len(rec["hash"]) == 64
        ev = event()
        assert ev["v"] == 2 and set(ev) == {"kind", "v", "service", "review",
                                            "reviewer", "sig"}
        assert ev["service"] == {"provider": PROVIDER, "spec_digest": DIGEST}

    def test_only_the_reviewer_signs_its_review(self):
        ev = event()
        ev["sig"] = ""
        assert check(ev) == (None, "review signature")
        ev = event()
        ev["review"]["correctness"] = 100          # changed after signing
        assert check(ev) == (None, "review not signed by the reviewer")
        ev = event()
        ev["reviewer"] = addr(PAYER_KEYS[1])         # someone else's name
        assert check(ev) == (None, "review not signed by the reviewer")
        ev = event()
        ev["service"]["spec_digest"] = OTHER_DIGEST  # moved to another service
        assert check(ev) == (None, "review not signed by the reviewer")

    def test_the_provider_cannot_review_itself(self):
        assert check(event(PROVIDER_KEY)) == (None, "the provider reviewing itself")

    def test_the_service_must_be_well_formed(self):
        ev = event()
        ev["service"]["note"] = "x"
        assert check(ev) == (None, "service fields")
        ev = event()
        ev["service"]["provider"] = sr.ZERO_ADDRESS
        assert check(ev) == (None, "provider")
        ev = event()
        ev["service"]["spec_digest"] = "ab"
        assert check(ev) == (None, "spec_digest")
        old = event()
        old["v"] = 1
        old["receipt"] = {}
        assert check(old) == (None, "version")

    def test_other_refusals(self):
        assert check(event(jurisdiction="0xother")) == (None, "another jurisdiction")
        assert check(event(malware=True)) == (None, "a malware flag needs evidence")
        assert check(event(malware=True, evidence="cd" * 32))[0] is not None
        assert check(event(correctness=101)) == (None, "correctness")
        assert check(event(simplicity=-1)) == (None, "simplicity")
        assert check(event(axes={"care": 101})) == (None, "axis score")
        assert check(event(ts=TS + sr.MAX_FUTURE_S + 1)) == (None, "dated after its batch")
        assert check(event(note="x" * (sr.MAX_NOTE_CHARS + 1))) == (None, "note")
        bad_evidence = event()
        bad_evidence["review"]["evidence_hash"] = "zz"
        assert check(bad_evidence) == (None, "evidence_hash")
        extra = event()
        extra["review"]["bonus"] = 1
        assert check(extra) == (None, "review fields")


class TestCounting:
    """Without a snapshot (no chain) every valid review counts."""

    def test_a_later_review_replaces_the_earlier(self):
        first = close([event(correctness=-50, ts=TS)])
        assert first["ratings"][KEY]["correctness"] == -0.5
        second = close([event(correctness=90, ts=TS + 10)], carried=first["reviews"])
        r = second["ratings"][KEY]
        assert r["payers"] == 1 and r["correctness"] == 0.9 and r["latest_ts"] == TS + 10
        # A replayed older review changes nothing.
        third = close([event(correctness=-50, ts=TS)], carried=second["reviews"])
        assert third["ratings"][KEY]["correctness"] == 0.9 and third["accepted"] == 0

    def test_the_rating_is_the_mean_over_payers(self):
        out = close([event(PAYER_KEYS[0], correctness=100, simplicity=40, novelty=0),
                     event(PAYER_KEYS[1], correctness=0, simplicity=80, novelty=50,
                           malware=True, evidence="cd" * 32),
                     event(PAYER_KEYS[2], correctness=-40, simplicity=60, novelty=10)])
        r = out["ratings"][KEY]
        assert r["payers"] == 3 and r["pending"] == 0
        assert r["correctness"] == pytest.approx(0.2)
        assert r["simplicity"] == pytest.approx(0.6)
        assert r["novelty"] == pytest.approx(0.2)
        assert r["malware"] == pytest.approx(1 / 3, abs=1e-9)
        assert r["provider"] == PROVIDER and r["spec_digest"] == DIGEST
        assert r["paid"] == 0                       # no chain, nothing to sum

    def test_a_copy_under_another_address_is_rated_apart(self):
        out = close([event(PAYER_KEYS[0], correctness=100),
                     event(PAYER_KEYS[1], provider_key=COPIER_KEY, correctness=-100)])
        copy_key = sr.service_key(addr(COPIER_KEY), DIGEST)
        assert out["ratings"][KEY]["correctness"] == 1.0
        assert out["ratings"][copy_key]["correctness"] == -1.0

    def test_the_providers_household_and_sanctioned_payers_do_not_count(self):
        evs = [event(PAYER_KEYS[0], correctness=100), event(PAYER_KEYS[1], correctness=-100),
               event(PAYER_KEYS[2], correctness=50)]
        owners = {PROVIDER: "0xowner", addr(PAYER_KEYS[0]): "0xowner",
                  addr(PAYER_KEYS[1]): "0xbanned"}
        out = close(evs, agent_owner_map=owners, sanctioned_households=["0xbanned"])
        assert out["ratings"][KEY]["payers"] == 1
        assert out["ratings"][KEY]["correctness"] == 0.5
        assert len(out["reviews"][KEY]) == 3      # kept, only not counted

    def test_charter_axes_count_unscored_at_the_default(self):
        out = close([event(PAYER_KEYS[0], axes={"care": 100}),
                     event(PAYER_KEYS[1])], axis_ids=["care", "truth"])
        r = out["ratings"][KEY]
        assert r["axes"] == {"care": 0.75, "truth": 0.5}
        assert r["value"] == 0.625
        assert close([event()])["ratings"][KEY]["value"] is None   # no charter

    def test_invalid_events_are_counted_by_reason(self):
        out = close([event(correctness=101), event(correctness=101), event()])
        assert out["accepted"] == 1 and out["rejected"] == {"correctness": 2}

    def test_the_result_does_not_depend_on_the_order(self):
        evs = [event(k, correctness=c, ts=TS + t, digest=d)
               for k in PAYER_KEYS for c, t in ((10, 1), (-20, 2), (70, 3))
               for d in (DIGEST, OTHER_DIGEST)]
        evs += [event(PAYER_KEYS[0], provider_key=COPIER_KEY, correctness=5)]
        paid = snap(registered=(KEY, OTHER_KEY, sr.service_key(addr(COPIER_KEY), DIGEST)),
                    paid={**paid_by(PAYER_KEYS[0]), **paid_by(PAYER_KEYS[2], 7)})
        want = json.dumps(close(evs, snapshot=paid), sort_keys=True)
        rng = random.Random(7)
        for _ in range(5):
            shuffled = list(evs)
            rng.shuffle(shuffled)
            cut = rng.randint(0, len(shuffled))
            a = close(shuffled[:cut], snapshot=paid)
            carried = json.loads(json.dumps(a["reviews"]))
            b = close(shuffled[cut:], carried=dict(reversed(list(carried.items()))),
                      snapshot=paid)
            got = json.loads(json.dumps(b, sort_keys=True))
            assert got["ratings"] == json.loads(want)["ratings"]
            assert got["reviews"] == json.loads(want)["reviews"]


class TestBounds:
    def test_the_payers_kept_per_service_are_the_newest(self, monkeypatch):
        monkeypatch.setattr(sr, "MAX_PAYERS_PER_SERVICE", 2)
        out = close([event(k, ts=TS + i) for i, k in enumerate(PAYER_KEYS[:3])])
        assert sorted(out["reviews"][KEY]) == sorted(addr(k) for k in PAYER_KEYS[1:3])

    def test_waiting_reviews_cannot_push_out_paid_ones(self, monkeypatch):
        monkeypatch.setattr(sr, "MAX_PAYERS_PER_SERVICE", 3)
        monkeypatch.setattr(sr, "MAX_PENDING_PER_SERVICE", 1)
        evs = [event(PAYER_KEYS[0], ts=TS)] + [event(k, ts=TS + 10 + i)
                                               for i, k in enumerate(PAYER_KEYS[1:])]
        out = close(evs, snapshot=snap(paid=paid_by(PAYER_KEYS[0])))
        kept = out["reviews"][KEY]
        assert kept[addr(PAYER_KEYS[0])]["paid"] is True
        assert len(kept) == 2 and addr(PAYER_KEYS[3]) in kept     # the newest waiting one
        assert out["ratings"][KEY]["payers"] == 1 and out["ratings"][KEY]["pending"] == 1

    def test_the_services_kept_per_provider_are_bounded(self, monkeypatch):
        monkeypatch.setattr(sr, "MAX_SERVICES_PER_PROVIDER", 1)
        out = close([event(PAYER_KEYS[0], ts=TS), event(PAYER_KEYS[1], digest=OTHER_DIGEST,
                                                         ts=TS + 5)])
        assert list(out["reviews"]) == [OTHER_KEY]              # the latest
        # A service with paid reviews outranks a newer one without.
        out = close([event(PAYER_KEYS[0], ts=TS), event(PAYER_KEYS[1], digest=OTHER_DIGEST,
                                                         ts=TS + 5)],
                    snapshot=snap(registered=(KEY, OTHER_KEY), paid=paid_by(PAYER_KEYS[0])))
        assert list(out["reviews"]) == [KEY]


class TestPaymentOnChain:
    """With the chain snapshot, authority comes from payment."""

    def test_a_review_of_an_unregistered_service_is_dropped(self):
        out = close([event()], snapshot=snap(registered=()))
        assert out["reviews"] == {} and out["ratings"] == {}
        assert out["rejected"] == {"unregistered service": 1}
        # Carried reviews of a pair the chain never registered go too.
        lenient = close([event()])
        assert close([], carried=lenient["reviews"],
                     snapshot=snap(registered=()))["reviews"] == {}

    def test_a_review_waits_until_its_reviewer_has_paid_the_provider(self):
        payer = PAYER_KEYS[0]
        free = close([event(payer, correctness=100)], snapshot=snap())
        assert free["ratings"] == {}
        assert free["reviews"][KEY][addr(payer)]["paid"] is False
        # Another payer's payment, or one to another provider, is not its own.
        other = close([], carried=free["reviews"], snapshot=snap(paid={
            **paid_by(PAYER_KEYS[1]), **paid_by(payer, provider=addr(COPIER_KEY))}))
        assert other["ratings"] == {}
        # Any amount to this provider counts it, at the first close that has it.
        paid = close([], carried=free["reviews"], snapshot=snap(paid=paid_by(payer, 1)))
        r = paid["ratings"][KEY]
        assert r["payers"] == 1 and r["correctness"] == 1.0 and r["paid"] == 1
        assert paid["reviews"][KEY][addr(payer)]["paid"] is True

    def test_paying_a_provider_counts_for_any_of_its_services(self):
        """Credits are per provider: one payment lets the payer review every
        service of that provider."""
        two = (KEY, OTHER_KEY)
        out = close([event(PAYER_KEYS[0], digest=DIGEST),
                     event(PAYER_KEYS[0], digest=OTHER_DIGEST, correctness=-10)],
                    snapshot=snap(registered=two, paid=paid_by(PAYER_KEYS[0], 2500)))
        assert set(out["ratings"]) == set(two)
        assert out["ratings"][OTHER_KEY]["correctness"] == -0.1
        # `paid` is what the counted payers paid the provider in all.
        assert out["ratings"][KEY]["paid"] == out["ratings"][OTHER_KEY]["paid"] == 2500

    def test_the_rating_sums_what_its_payers_paid(self):
        out = close([event(PAYER_KEYS[0]), event(PAYER_KEYS[1]), event(PAYER_KEYS[2])],
                    snapshot=snap(paid={**paid_by(PAYER_KEYS[0], 300),
                                        **paid_by(PAYER_KEYS[1], 700)}))
        r = out["ratings"][KEY]
        assert (r["payers"], r["pending"], r["paid"]) == (2, 1, 1000)

    def test_unscored_axes_count_at_the_pinned_registry_value(self):
        out = close([event(axes={"care": 100})], axis_ids=["care", "truth"],
                    snapshot=snap(paid=paid_by(PAYER_KEYS[0]), unscored_bps=2000))
        assert out["ratings"][KEY]["axes"] == {"care": 1.0, "truth": 0.2}

    def test_the_snapshot_from_logs(self):
        channel = "0x" + "99" * 20
        stranger = "0x" + "77" * 20
        payer, other = addr(PAYER_KEYS[0]), addr(PAYER_KEYS[1])
        got = sr.snapshot_from_logs(
            registered=[{"provider": Account.from_key(PROVIDER_KEY).address,
                         "specDigest": bytes.fromhex(DIGEST)},
                        {"provider": PROVIDER, "specDigest": "0x" + OTHER_DIGEST}],
            payments=[{"payer": payer, "recipient": PROVIDER, "amount": 600},
                      {"payer": payer, "recipient": PROVIDER, "amount": 400},
                      # The channel's own payout through Substrate.
                      {"payer": channel, "recipient": PROVIDER, "amount": 50},
                      # To an address that registered no service.
                      {"payer": payer, "recipient": stranger, "amount": 9}],
            payouts=[{"client": other, "provider": PROVIDER, "amount": 50},
                     {"client": other, "provider": PROVIDER, "amount": 25},
                     {"client": payer, "provider": stranger, "amount": 5}],
            unscored_bps="3000", channel_address=channel)
        assert got == {"registered": sorted([KEY, OTHER_KEY]),
                       "paid": {f"{payer}:{PROVIDER}": 1000, f"{other}:{PROVIDER}": 75},
                       "unscored_bps": 3000}

    def test_a_snapshot_is_normalized(self):
        assert sr.normalize_snapshot(None) is None
        empty = sr.normalize_snapshot({})
        assert empty == {"registered": [], "paid": {},
                         "unscored_bps": sr.UNSCORED_BPS_DEFAULT}
        assert sr.normalize_snapshot({"unscored_bps": 99_999})["unscored_bps"] == sr.BPS
        assert sr.normalize_snapshot({"paid": {"a:b": "12", "c:d": "x"}})["paid"] == \
            {"a:b": 12}


class TestTheChainRead:
    """``voice_state.read_service_snapshot``: the logs it reads and how."""

    def test_the_event_abi_matches_the_contracts(self):
        import re
        from pathlib import Path

        from nodes.common.voice_state import _SERVICE_EVENTS_ABI

        root = Path(__file__).resolve().parent.parent / "contracts" / "core"
        source = ((root / "ServiceMarket.sol").read_text(encoding="utf-8")
                  + (root / "Substrate.sol").read_text(encoding="utf-8"))
        for group in _SERVICE_EVENTS_ABI.values():
            for ev in group:
                m = re.search(r"event\s+" + ev["name"] + r"\s*\(([^)]*)\)", source)
                assert m, ev["name"]
                declared = [p.split() for p in m.group(1).split(",")]
                assert [(p[0], "indexed" in p, p[-1]) for p in declared] == \
                    [(i["type"], i["indexed"], i["name"]) for i in ev["inputs"]], ev["name"]

    def test_it_reads_the_registry_the_payments_and_the_channel(self, monkeypatch):
        from web3 import Web3

        from nodes.common import voice_state as vs

        sub, reg, channel = ("0x" + c * 20 for c in ("51", "52", "53"))
        payer, other = addr(PAYER_KEYS[0]), addr(PAYER_KEYS[1])
        logs = {
            "ServiceRegistered": [{"serviceId": 1, "provider": PROVIDER,
                                   "specDigest": bytes.fromhex(DIGEST), "askAmount": 0}],
            "ServicePayment": [{"payer": payer, "recipient": PROVIDER, "amount": 5},
                               {"payer": channel, "recipient": PROVIDER, "amount": 9}],
            "ChannelPaid": [{"channelId": 1, "client": other, "provider": PROVIDER,
                             "amount": 9, "paidTotal": 9}],
        }
        read = []

        def get_logs(event, w3, address, block, start):
            read.append((event.event_name, address.lower(), block, start))
            return [{"args": a} for a in logs[event.event_name]]
        monkeypatch.setattr(vs, "_get_logs", get_logs)
        got = vs.read_service_snapshot(
            Web3(), block=10, substrate_address=sub, service_registry_address=reg,
            payment_channel_address=channel, jurisdiction_id="j1", start_block=3,
            registry_writes={"jurisdiction.j1.unscored_axis_bps": "2500"})
        assert got == {"registered": [KEY], "unscored_bps": 2500,
                       "paid": {f"{payer}:{PROVIDER}": 5, f"{other}:{PROVIDER}": 9}}
        assert sorted(read) == [("ChannelPaid", channel, 10, 3),
                                ("ServicePayment", sub, 10, 3),
                                ("ServiceRegistered", reg, 10, 3)]
        # Without a ServiceRegistry nothing is registered and nothing is read
        # from the Substrate: no review counts.
        read.clear()
        bare = vs.read_service_snapshot(Web3(), block=10, substrate_address=sub)
        assert bare == sr.normalize_snapshot({}) and read == []


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
        assert joiner.service_rating(PROVIDER, "0x" + DIGEST.upper()) == \
            always_on.service_rating(PROVIDER, DIGEST)
        assert joiner.service_reviews_of(PROVIDER, DIGEST)

        # The next close builds on the carried reviews, on both daemons.
        _, later = _batches([event(PAYER_KEYS[1], correctness=0, ts=TS + 50)],
                            ts=TS + 50)
        for d in (always_on, joiner):
            d.gossip.drain_epoch_batches.return_value = list(later)
        a2, j2 = always_on.run({"epoch_id": "e2"}), joiner.run({"epoch_id": "e2"})
        assert a2.carry_bundle_blob == j2.carry_bundle_blob
        assert always_on.service_rating(PROVIDER, DIGEST)["payers"] == 2
        # A restart reads the same state back.
        again = driver("a", [])
        assert again.carry_over_maps() == always_on.carry_over_maps()

    def test_the_driver_closes_on_the_pinned_snapshot(self):
        kp, batches = _batches([event(PAYER_KEYS[0]), event(PAYER_KEYS[1])])
        gossip = MagicMock(spec=EventGossip)
        gossip.known_senders.return_value = [kp.public_key]
        gossip.sender_pubkey = kp.public_key
        gossip.drain_epoch_batches.return_value = list(batches)
        d = FederatedCloseDriver(gossip=gossip)
        d.voice_source = lambda: {"snapshot_block": 10, "owner_map": {},
                                  "service_snapshot": snap(paid=paid_by(PAYER_KEYS[1]))}
        d._refresh_voice()
        fed = d._close(list(batches), "e1", {})
        r = fed.close_result["service_ratings"][KEY]
        assert r["payers"] == 1 and r["pending"] == 1 and r["paid"] == 1000

    def test_missing_service_maps_start_empty(self, tmp_path):
        names = ("tool_registrations", "tool_rounds", "tool_ratings")
        for n in names:
            (tmp_path / f"{n}.json").write_text("{}", encoding="utf-8")
        d = FederatedCloseDriver(
            gossip=MagicMock(spec=EventGossip),
            **{f"{n}_path": tmp_path / f"{n}.json" for n in names},
            service_reviews_path=tmp_path / "service_reviews.json",
            service_ratings_path=tmp_path / "service_ratings.json")
        d.voice_source = lambda: {"snapshot_block": 10, "owner_map": {}}
        d._refresh_voice()
        assert d._chain_at_genesis is False
        assert d.participation_blockers() == []
        assert d.carry_over_maps()["service_reviews"] == {}
        # No state dir for them at all is fine too.
        bare = FederatedCloseDriver(
            gossip=MagicMock(spec=EventGossip),
            **{f"{n}_path": tmp_path / f"{n}.json" for n in names})
        bare.voice_source = d.voice_source
        bare._refresh_voice()
        assert bare.participation_blockers() == []
