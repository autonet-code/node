"""Batch catch-up (issue #41): a daemon that missed an epoch's batches
(joined late, slept, lost its peers) asks peers for them before closing,
so it closes the same set as everyone else."""

from __future__ import annotations

from pathlib import Path

from nodes.common.event_gossip import (
    BatchLog,
    EventGossip,
    InMemoryHub,
    InMemoryTransport,
    Keypair,
)
from nodes.common.library_service import LibraryService
from tests.test_epoch_window import BASE, ROOT, _batch, _Buffer, _driver, _window
from nodes.common.voice_state import make_epoch_id

RPB = "0x" + "ab" * 20


def _peer(hub, tmp_path: Path, name: str, now: float) -> EventGossip:
    svc = LibraryService(rpb_address=RPB, data_root=tmp_path / name)
    g = EventGossip(rpb_address=RPB, keypair=Keypair.generate(),
                    transport=InMemoryTransport(hub, node_id=name), library=svc)
    g.clock = lambda: now
    g.max_skew_seconds = 60
    return g


def _events(label):
    return [{"type": "noop", "label": label, "id": f"ev_{label}"}]


class TestGossipCatchup:
    def test_late_joiner_gets_the_missed_batches(self, tmp_path):
        hub, now = InMemoryHub(), 10_000.0
        a = _peer(hub, tmp_path, "a", now)
        b_hub = InMemoryHub()   # b is not on a's hub yet: it misses both
        a._sign_and_publish(_events("one"))
        a._sign_and_publish(_events("two"))
        late = _peer(b_hub, tmp_path, "late", now + 300)   # 5 min later, past the skew
        reply = a.batch_log.serve(late.catchup_request(now - 600))
        assert len(reply["batches"]) == 2
        assert late.ingest_catchup(reply["batches"], floor=now - 600) == 2
        assert {b.batch_seq for b in late.pending_epoch_batches()} == {1, 2}
        # Asking again sends only what it lacks; re-ingesting adds nothing.
        assert a.batch_log.serve(late.catchup_request(now - 600))["batches"] == []
        assert late.ingest_catchup(reply["batches"], floor=now - 600) == 0

    def test_catchup_window(self, tmp_path):
        hub, now = InMemoryHub(), 10_000.0
        a = _peer(hub, tmp_path, "a", now)
        a._sign_and_publish(_events("old"))
        b = _peer(InMemoryHub(), tmp_path, "b", now + 30)
        wires = a.batch_log.serve({"since": 0})["batches"]
        # Signed at or before the previous cutoff: belongs to a closed epoch.
        assert b.ingest_catchup(wires, floor=now) == 0
        # Post-dated past the skew: refused like gossip.
        b.clock = lambda: now - 120
        assert b.ingest_catchup(wires, floor=now - 600) == 0
        b.clock = lambda: now + 30
        assert b.ingest_catchup(wires, floor=now - 600) == 1

    def test_tampered_batch_refused(self, tmp_path):
        hub, now = InMemoryHub(), 10_000.0
        a = _peer(hub, tmp_path, "a", now)
        a._sign_and_publish(_events("x"))
        wire = a.batch_log.serve({"since": 0})["batches"][0]
        bad = wire.replace('"x"', '"y"')
        b = _peer(InMemoryHub(), tmp_path, "b", now)
        assert b.ingest_catchup([bad], floor=0) == 0
        assert b.stats.batches_invalid_signature == 1

    def test_log_retention(self):
        t = [1000.0]
        log = BatchLog(retention=100, clock=lambda: t[0])
        log.add(b"\x01" * 32, 950.0, b"w1")
        log.add(b"\x02" * 32, 990.0, b"w2")
        t[0] = 1080.0
        assert log.since(0, set()) == [b"w2"]


class TestCloseCatchup:
    def test_close_includes_caught_up_batch_and_agrees(self, tmp_path):
        kp = Keypair.generate()
        b1, b2 = _batch(kp, 1, BASE - 100), _batch(kp, 2, BASE - 10)
        full = _driver(_Buffer(kp, [b1, b2]), _window(), BASE + 125)
        short_buf = _Buffer(kp, [b1])
        short = _driver(short_buf, _window(), BASE + 125)
        calls = []

        def catchup(floor):
            calls.append(floor)
            if b2 in short_buf._b:
                return 0
            short_buf._b.append(b2)
            return 1
        short.catchup = catchup
        fa, fb = full.run({"epoch_id": "a"}), short.run({"epoch_id": "b"})
        # Asked again after a reply with something new, until nothing is.
        assert calls == [float(_window()["start"])] * 2
        assert fa.epoch_id == fb.epoch_id == make_epoch_id(2, ROOT, BASE)
        assert fa.n_batches == fb.n_batches == 2
        assert (fa.close_result["authoritative_payload"]
                == fb.close_result["authoritative_payload"])

    def test_catchup_runs_once_per_window_and_failure_is_harmless(self):
        kp = Keypair.generate()
        buf = _Buffer(kp, [_batch(kp, 1, BASE - 10)])
        d = _driver(buf, _window(), BASE + 119)
        calls = []

        def boom(floor):
            calls.append(floor)
            raise ConnectionError("no peers")
        d.catchup = boom
        assert d.run({}) is None and calls == []   # not time yet: no request
        d.clock = lambda: BASE + 120
        assert d.run({}) is not None and len(calls) == 1
