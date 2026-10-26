"""Decision 2026-10-26, items 1-3 and 7: the epoch the network agrees on.

  - the epoch id comes from the anchor count, the previous anchor's root
    and the cutoff, so every daemon names an epoch the same;
  - one epoch in flight: no close until the anchor before the next lands;
  - membership by signed timestamp against the agreed cutoff, independent
    of when each daemon's timer fires; an empty window slides by grace;
  - batches signed further than the skew from their arrival are refused;
  - the rules are governed Registry keys pinned to the anchor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nodes.common.event_gossip import (
    EventBatch,
    EventGossip,
    InMemoryHub,
    InMemoryTransport,
    Keypair,
    SignedBatch,
)
from nodes.common.federated_close_driver import FederatedCloseDriver
from nodes.common.voice_state import (
    EPOCH_RULE_DEFAULTS,
    epoch_id_cutoff,
    epoch_rule_params,
    epoch_window,
    governor_lifecycle,
    make_epoch_id,
    retarget_length,
)

BASE = 1_000_000
ROOT = "ab" * 32


def _event(seq: int, agent: str):
    return {
        "kind": "observation_added",
        "seq": seq,
        "author_agent": agent,
        "obs_id": f"obs_{agent}_{seq}",
        "coords": [0.5, 0.1 * seq, 0.0, 0.0, 0.0, 0.0],
        "label": f"{agent}_{seq}",
    }


_LAST: dict = {}


def _batch(kp: Keypair, seq: int, ts: float) -> EventBatch:
    """Batch ``seq`` of ``kp``'s hash chain (batches are built in order)."""
    prev = b"" if seq == 1 else _LAST[kp.public_key]
    b = EventBatch(
        rpb_address="rpb_test", sender_pubkey=kp.public_key, batch_seq=seq,
        events=[_event(seq, f"a{seq}")], prev_batch_hash=prev, timestamp=ts,
    )
    _LAST[kp.public_key] = b.content_hash()
    return b


class _Buffer:
    """The slice of EventGossip the window path uses."""

    def __init__(self, kp, batches):
        self.sender_pubkey = kp.public_key
        self._b = list(batches)
        self.max_skew_seconds = None

    def pending_epoch_batches(self):
        return list(self._b)

    def take_epoch_batches(self, selected, *, drop_at_or_before=None):
        ids = {id(b) for b in selected}
        self._b = [b for b in self._b if id(b) not in ids and not (
            drop_at_or_before is not None
            and b.timestamp <= drop_at_or_before)]

    def known_senders(self):
        return [self.sender_pubkey]


def _window(number=2, start=BASE - 500, length=0, grace=120, skew=60):
    return {"number": number, "prev_root": ROOT, "base": BASE,
            "start": start, "length": length, "grace": grace, "skew": skew}


def _driver(buffer, window, now, tmp_path=None):
    d = FederatedCloseDriver(
        gossip=buffer, embedding_dim=8,
        tool_registrations_path=(tmp_path / "tool_registrations.json"
                                 if tmp_path else None))
    d.voice_source = lambda: {"epoch_window": dict(window),
                              "snapshot_block": 5,
                              "snapshot_anchor_count": window["number"] - 1}
    d.clock = lambda: now
    return d


class TestEpochId:
    def test_round_trip_and_agreement(self):
        eid = make_epoch_id(7, "0x" + ROOT, BASE + 30)
        assert eid == f"epoch-7-{ROOT[:16]}-{BASE + 30}"
        assert epoch_id_cutoff(eid) == BASE + 30
        assert epoch_id_cutoff(f"epoch_{123456789}") is None

    def test_two_daemons_with_different_timers_agree(self, tmp_path):
        kp = Keypair.generate()
        batches = [_batch(kp, 1, BASE - 100), _batch(kp, 2, BASE - 10),
                   _batch(kp, 3, BASE + 50)]
        # One daemon's timer fires 2 minutes past the cutoff, the other's
        # 40 minutes past: same members, same id, same payload.
        a = _driver(_Buffer(kp, batches), _window(), BASE + 125)
        b = _driver(_Buffer(kp, batches), _window(), BASE + 2400)
        fa, fb = a.run({"epoch_id": "local_a"}), b.run({"epoch_id": "local_b"})
        assert fa.epoch_id == fb.epoch_id == make_epoch_id(2, ROOT, BASE)
        assert fa.n_batches == fb.n_batches == 2
        assert (fa.close_result["authoritative_payload"]
                == fb.close_result["authoritative_payload"])
        # The batch after the cutoff stays buffered for the next epoch.
        assert [x.batch_seq for x in a.gossip.pending_epoch_batches()] == [3]


class TestTiming:
    def test_waits_for_grace(self):
        kp = Keypair.generate()
        buf = _Buffer(kp, [_batch(kp, 1, BASE - 10)])
        d = _driver(buf, _window(), BASE + 119)
        assert d.run({}) is None
        assert len(buf.pending_epoch_batches()) == 1
        d.clock = lambda: BASE + 120
        assert d.run({}) is not None

    def test_one_epoch_in_flight(self, tmp_path):
        kp = Keypair.generate()
        buf = _Buffer(kp, [_batch(kp, 1, BASE - 10), _batch(kp, 2, BASE + 5)])
        d = _driver(buf, _window(), BASE + 1000, tmp_path)
        assert d.run({}) is not None
        # The anchor has not landed: same window, no second close.
        assert d.run({}) is None
        assert len(buf.pending_epoch_batches()) == 1
        # A restart mid-flight remembers it.
        d2 = _driver(buf, _window(), BASE + 1000, tmp_path)
        assert d2.run({}) is None
        # The anchor lands: the next window opens after this cutoff.
        nxt = _window(number=3, start=BASE)
        nxt["base"] = BASE + 300
        d2.voice_source = lambda: {"epoch_window": nxt, "snapshot_block": 6,
                                   "snapshot_anchor_count": 2}
        out = d2.run({})
        assert out is not None and out.n_batches == 1

    def test_empty_window_slides_by_grace(self):
        kp = Keypair.generate()
        buf = _Buffer(kp, [_batch(kp, 1, BASE + 250)])
        d = _driver(buf, _window(), BASE + 300)
        assert d.run({}) is None          # slot BASE+360 not over yet
        d.clock = lambda: BASE + 480
        out = d.run({})
        assert out.epoch_id == make_epoch_id(2, ROOT, BASE + 360)

    def test_late_batch_before_start_is_dropped(self):
        kp = Keypair.generate()
        stale = _batch(kp, 1, BASE - 600)       # before start = BASE - 500
        buf = _Buffer(kp, [stale, _batch(kp, 2, BASE - 10)])
        out = _driver(buf, _window(), BASE + 200).run({})
        assert out.n_batches == 1
        assert buf.pending_epoch_batches() == []

    def test_failed_refresh_takes_nothing(self):
        kp = Keypair.generate()
        buf = _Buffer(kp, [_batch(kp, 1, BASE - 10)])
        d = _driver(buf, _window(), BASE + 500)
        d._refresh_voice()
        d.voice_source = MagicMock(side_effect=ConnectionError("down"))
        assert d.run({}) is None
        assert len(buf.pending_epoch_batches()) == 1

    def test_refresh_only_on_new_anchor(self):
        kp = Keypair.generate()
        d = _driver(_Buffer(kp, []), _window(), BASE)
        calls = []
        src = d.voice_source
        d.voice_source = lambda: calls.append(1) or src()
        d.anchor_count_source = lambda: 1
        d.run({})
        d.run({})
        assert len(calls) == 1
        d.anchor_count_source = lambda: 2
        d.run({})
        assert len(calls) == 2


class TestSkew:
    def test_far_off_batches_refused(self):
        transport = InMemoryTransport(InMemoryHub())
        ws = MagicMock()
        kp_a, kp_b = Keypair.generate(), Keypair.generate()
        g = EventGossip(rpb_address="r", keypair=kp_a, transport=transport,
                        world_service=ws)
        g.max_skew_seconds = 60
        g.clock = lambda: 5000.0
        from nodes.common.event_gossip import _sign

        def deliver(ts, seq):
            b = EventBatch(rpb_address="r", sender_pubkey=kp_b.public_key,
                           batch_seq=seq, events=[_event(seq, "b")],
                           prev_batch_hash=b"", timestamp=ts)
            sig = _sign(kp_b.private_key, b.canonical_payload())
            g._on_message(g.topic, SignedBatch(batch=b, signature=sig).to_wire())

        deliver(5000.0 - 61, 1)     # backdated
        deliver(5000.0 + 61, 2)     # post-dated
        deliver(5000.0 - 30, 3)
        assert [b.batch_seq for b in g.pending_epoch_batches()] == [3]
        assert g.stats.batches_skewed == 2


class TestRules:
    def test_governed_keys_pinned_to_block(self):
        recs = [
            {"block": 10, "index": 0, "key": "jurisdiction.autonet.epoch_grace",
             "value": "300"},
            {"block": 20, "index": 0, "key": "jurisdiction.autonet.epoch_skew",
             "value": "90"},
            {"block": 20, "index": 1, "key": "jurisdiction.autonet.epoch_length",
             "value": "-5"},
        ]
        assert epoch_rule_params(recs, 15) == dict(
            EPOCH_RULE_DEFAULTS, epoch_grace=300)
        assert epoch_rule_params(recs, 20) == dict(
            EPOCH_RULE_DEFAULTS, epoch_grace=300, epoch_skew=90)
        assert epoch_rule_params(recs, None) == EPOCH_RULE_DEFAULTS

    def test_window_from_chain(self):
        contract = MagicMock()
        contract.functions.deployedAt.return_value.call.return_value = 777
        gen = epoch_window(contract, number=1, prev_root="00" * 32,
                           start=None, params_block=None, records=[])
        assert gen["number"] == 1 and gen["base"] == 777
        assert gen["start"] is None and gen["prev_root"] == "00" * 32
        lc = {"voting_delay": 10, "voting_period": 100, "min_delay": 20}
        win = epoch_window(contract, number=4, prev_root=ROOT, start=4242,
                           params_block=50, records=[], lifecycle=lc)
        # Cutoffs run on their own schedule: the next one counts from the
        # previous cutoff, not from when its anchor happened to land.
        assert win["number"] == 4 and win["base"] == 4242
        assert win["start"] == 4242 and win["prev_root"] == ROOT
        # No target share set: the length is the floor, one lifecycle.
        assert win["floor"] == 130 and win["length"] == 130
        assert win["ceiling"] == 1000


class TestRetarget:
    def test_bounds(self):
        kw = dict(floor=100, ceiling=1000, span_seconds=3600, supply=10**9)
        assert retarget_length(target_share_ppm=0, burned=5, **kw) == 100
        assert retarget_length(target_share_ppm=10, burned=0, **kw) == 1000
        # 10 ppm of 1e9 = 10_000 per epoch; 36_000 burned per hour means
        # 10_000 burns in 1000 s.
        assert retarget_length(target_share_ppm=10, burned=36_000, **kw) == 1000
        assert retarget_length(target_share_ppm=10, burned=72_000, **kw) == 500
        assert retarget_length(target_share_ppm=10, burned=10**9, **kw) == 100
        # A ceiling below the floor never undercuts the floor.
        assert retarget_length(floor=100, ceiling=50, span_seconds=3600,
                               supply=10**9, target_share_ppm=10,
                               burned=0) == 100

    def test_reads_only_the_retarget_window(self):
        contract = MagicMock()
        contract.functions.deployedAt.return_value.call.return_value = 0
        recs = [{"block": 1, "index": 0,
                 "key": "jurisdiction.autonet.epoch_target_share",
                 "value": "10"},
                {"block": 1, "index": 1,
                 "key": "jurisdiction.autonet.epoch_retarget_window",
                 "value": "2"}]
        lc = {"voting_delay": 0, "voting_period": 1_000, "min_delay": 0}
        # Epochs cut at t=1000 (huge burn, outside the window), 2000, 3000.
        history = [(1000, 10**12), (2000, 9_000), (3000, 9_000)]
        win = epoch_window(contract, number=4, prev_root=ROOT, start=3000,
                           params_block=1, records=recs, lifecycle=lc,
                           history=history, supply=10**9)
        # 18_000 burned over 2000 s; 10_000 per epoch takes 1111 s.
        assert win["length"] == 1111
        assert win["base"] == 3000

    def test_lifecycle_from_settings_events(self):
        from nodes.common import voice_state

        def log(block, idx, **args):
            return {"blockNumber": block, "logIndex": idx, "args": args}

        logs = {
            "VotingDelaySet": [log(1, 0, newVotingDelay=60)],
            "VotingPeriodSet": [log(1, 1, newVotingPeriod=600),
                                log(9, 0, newVotingPeriod=900)],
            "TimelockChange": [log(1, 2, newTimelock="0x" + "11" * 20)],
            "MinDelayChange": [log(1, 0, newDuration=120)],
        }

        class Ev:
            def __init__(self, name):
                self.name = name

        class Events:
            def __getattr__(self, name):
                return Ev(name)

        w3 = MagicMock()
        w3.eth.contract.return_value.events = Events()
        w3.eth.contract.return_value.address = "0x" + "22" * 20

        def fake_get_logs(event, _w3, _addr, to_block, _start=None):
            return [x for x in logs[event.name] if x["blockNumber"] <= to_block]

        orig = voice_state._get_logs
        voice_state._get_logs = fake_get_logs
        try:
            assert governor_lifecycle(w3, "0x" + "33" * 20, 5) == {
                "voting_delay": 60, "voting_period": 600, "min_delay": 120}
            assert governor_lifecycle(w3, "0x" + "33" * 20, 9)[
                "voting_period"] == 900
            assert governor_lifecycle(w3, "", 9) == {
                "voting_delay": 0, "voting_period": 0, "min_delay": 0}
        finally:
            voice_state._get_logs = orig
