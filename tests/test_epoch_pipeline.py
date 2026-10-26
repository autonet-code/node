"""Pipelined elections in the close driver (Decision 2026-10-26, items 5-6
design): epochs close on their own schedule, each on this daemon's own view
of the unanchored ones before it; an epoch anchored as closed here is
pruned; one anchored differently rewinds onto the anchored state and its
batches close again."""

from __future__ import annotations

from types import SimpleNamespace

from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver

BASE = 1_000_000
L = 300          # epoch length
G = 120          # grace


def _event(seq: int, agent: str):
    return {"kind": "observation_added", "seq": seq, "author_agent": agent,
            "obs_id": f"obs_{agent}_{seq}",
            "coords": [0.5, 0.01 * seq, 0.0, 0.0, 0.0, 0.0],
            "label": f"{agent}_{seq}"}


class _Chain:
    """One sender's hash-chained batches."""

    def __init__(self):
        self.kp = Keypair.generate()
        self.last = b""
        self.seq = 0

    def batch(self, ts: float) -> EventBatch:
        self.seq += 1
        b = EventBatch(rpb_address="rpb", sender_pubkey=self.kp.public_key,
                       batch_seq=self.seq, events=[_event(self.seq, "a")],
                       prev_batch_hash=self.last, timestamp=ts)
        self.last = b.content_hash()
        return b


class _Buffer:
    def __init__(self, kp, batches=()):
        self.sender_pubkey = kp.public_key
        self.b = list(batches)
        self.max_skew_seconds = None

    def pending_epoch_batches(self):
        return list(self.b)

    def take_epoch_batches(self, selected, *, drop_at_or_before=None):
        ids = {x.content_hash() for x in selected}
        self.b = [x for x in self.b if x.content_hash() not in ids and not (
            drop_at_or_before is not None and x.timestamp <= drop_at_or_before)]

    def restore_epoch_batches(self, batches):
        have = {x.content_hash() for x in self.b}
        self.b += [x for x in batches if x.content_hash() not in have]

    def known_senders(self):
        return [self.sender_pubkey]


def _driver(buf, chain_state, now):
    d = FederatedCloseDriver(gossip=buf, embedding_dim=8)

    def voice(pin=None):
        start = pin["cutoffs"][-1] if pin["cutoffs"] else None
        return {"epoch_window": {
            "number": pin["number"], "prev_root": pin["prev_root"],
            "start": start, "base": BASE if start is None else start,
            "length": L, "grace": G, "skew": 60},
            "snapshot_block": 1}

    d.voice_source = voice
    d.chain_source = lambda: dict(chain_state)
    d.clock = lambda: now[0]
    return d


def test_closes_ahead_of_the_anchor_and_prunes_on_agreement():
    c = _Chain()
    buf = _Buffer(c.kp, [c.batch(BASE + 10), c.batch(BASE + 310),
                         c.batch(BASE + 620)])
    chain = {"anchor_count": 0, "anchors": []}
    now = [BASE + L + G]
    d = _driver(buf, chain, now)

    e1 = d.run({})
    assert e1.epoch_number == 1 and e1.n_batches == 1
    assert e1.epoch_id.endswith(f"-{BASE + L}")
    # Epoch 2 closes before epoch 1 is anchored, on epoch 1 as closed here.
    now[0] = BASE + 2 * L + G
    e2 = d.run({})
    assert e2.epoch_number == 2 and e2.n_batches == 1
    assert e2.parent_root == e1.close_result["epoch_root"]
    assert e2.epoch_id.endswith(f"-{BASE + 2 * L}")
    assert [e["number"] for e in d.local_epochs] == [1, 2]

    # Epoch 1 anchored as closed here: pruned, the chain continues.
    chain["anchor_count"] = 1
    chain["anchors"] = [{"number": 1, "epoch_id": e1.epoch_id,
                         "root": e1.close_result["epoch_root"],
                         "cutoff": BASE + L}]
    now[0] = BASE + 3 * L + G
    e3 = d.run({})
    assert e3.epoch_number == 3
    assert [e["number"] for e in d.local_epochs] == [2, 3]


def test_a_different_winner_rewinds_and_recloses():
    c = _Chain()
    b1, b2 = c.batch(BASE + 10), c.batch(BASE + 310)
    buf = _Buffer(c.kp, [b1, b2])
    chain = {"anchor_count": 0, "anchors": []}
    now = [BASE + L + G]
    d = _driver(buf, chain, now)
    e1 = d.run({})
    now[0] = BASE + 2 * L + G
    e2 = d.run({})
    assert e2 is not None and buf.pending_epoch_batches() == []

    # The network anchored a different epoch 1 (another option won).
    other_root = "ee" * 32
    chain["anchor_count"] = 1
    chain["anchors"] = [{"number": 1, "epoch_id": f"epoch-1-{'0' * 16}-{BASE + L}",
                         "root": other_root, "cutoff": BASE + L}]
    # The anchored state is not retrievable yet: wait, batches back in buffer.
    d.carry_source = lambda: SimpleNamespace(status="unavailable")
    assert d.run({}) is None
    assert d.local_epochs == []
    assert {x.batch_seq for x in buf.pending_epoch_batches()} == {1, 2}

    # Once it is, install it and close epoch 2 again on the anchored root.
    d.carry_source = lambda: SimpleNamespace(
        status="ok", epoch_id=chain["anchors"][0]["epoch_id"],
        carry={"tool_registrations": {}, "tool_vetting": {},
               "tool_positions": {}})
    re2 = d.run({})
    assert re2 is not None and re2.epoch_number == 2
    assert re2.parent_root == other_root
    assert re2.epoch_id.split("-")[2] == other_root[:16]
    # Batch 1 belonged to the anchored epoch 1's window: not re-closed.
    assert re2.n_batches == 1
