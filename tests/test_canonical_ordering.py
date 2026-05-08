"""Phase 5.3: canonical batch ordering at network epoch close.

Determinism contract: given the same set of batches, every daemon's
canonical_order(...) call produces the same ordered_batches, the
same per_sender_root map, and the same epoch_root().

What's tested:

  1. Two daemons that received the same batches in **different
     orders** compute the same canonical sequence.
  2. Replaying the canonical sequence on a fresh charter from each
     daemon produces a bit-identical world (node count, root score
     ranking).
  3. A sender with a broken chain (gap or hash mismatch) is dropped
     entirely; honest senders' positions are unaffected.
  4. Empty input produces an empty sequence with a stable epoch_root.
  5. Sorting by Merkle root genuinely changes when sender content
     changes — confirms the order is content-determined, not arbitrary.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import (
    EventBatch,
    EventGossip,
    InMemoryHub,
    InMemoryTransport,
    Keypair,
    SignedBatch,
)
from nodes.common.world_service import WorldService
from nodes.common.world_persistence import (
    PersistenceConfig,
    WorldPersistence,
)
from world_model.generalized import Observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _make_batch_chain(
    rpb: str,
    keypair: Keypair,
    n: int,
    label_prefix: str,
) -> List[EventBatch]:
    """Generate a chain of n linked batches with synthetic events."""
    chain: List[EventBatch] = []
    prev_hash = b""
    for i in range(1, n + 1):
        coord = _coord((0.0, 0.0, 0.5, 0.0), embed_idx=i, mag=0.5)
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": label_prefix,
            "obs_id": f"obs_{label_prefix}_{i}",
            "coords": list(coord),
            "label": f"{label_prefix}_{i}",
        }
        b = EventBatch(
            rpb_address=rpb,
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_arrival_orders_produce_same_canonical_sequence():
    """Same set of batches, different arrival orders, same output."""
    rpb = "rpb_canon_a"
    kp_a = Keypair.generate()
    kp_b = Keypair.generate()
    chain_a = _make_batch_chain(rpb, kp_a, 3, "alice")
    chain_b = _make_batch_chain(rpb, kp_b, 2, "bob")

    arrival_1 = chain_a + chain_b              # alice's chain, then bob's
    arrival_2 = list(reversed(chain_b + chain_a))  # interleaved differently
    arrival_3 = chain_a[1::] + [chain_b[1]] + [chain_a[0], chain_b[0]]
    random.shuffle(arrival_3)

    res1 = canonical_order(arrival_1)
    res2 = canonical_order(arrival_2)
    res3 = canonical_order(arrival_3)

    # Same ordered_batches by content hash sequence.
    seq_1 = [b.content_hash() for b in res1.ordered_batches]
    seq_2 = [b.content_hash() for b in res2.ordered_batches]
    seq_3 = [b.content_hash() for b in res3.ordered_batches]
    assert seq_1 == seq_2 == seq_3

    # Same per-sender roots and epoch root.
    assert res1.per_sender_root == res2.per_sender_root == res3.per_sender_root
    assert res1.epoch_root() == res2.epoch_root() == res3.epoch_root()


def test_canonical_replay_produces_bit_identical_worlds(tmp_path: Path):
    """Two daemons replay the canonical sequence on a fresh charter
    and end up with the same node count and same charter ranking."""
    rpb = "rpb_canon_b"
    kp_x = Keypair.generate()
    kp_y = Keypair.generate()
    chain_x = _make_batch_chain(rpb, kp_x, 2, "xena")
    chain_y = _make_batch_chain(rpb, kp_y, 3, "yuli")

    canonical = canonical_order(chain_x + chain_y)
    event_dicts = canonical.event_dicts()

    def _replay(label: str) -> WorldService:
        svc = WorldService(rpb_address=f"{rpb}_{label}", data_root=tmp_path)
        # Apply the canonical event sequence directly through
        # submit_events. Each batch's events + the trailing
        # __equilibrate__ marker make submit_events run with
        # equilibrate_after, so the per-batch equilibrate timing
        # reproduces.
        for b in canonical.ordered_batches:
            svc.submit_events(b.events, equilibrate_after=True)
        return svc

    a = _replay("a")
    b = _replay("b")
    try:
        assert a.stats()["n_nodes"] == b.stats()["n_nodes"], (
            f"a={a.stats()['n_nodes']}, b={b.stats()['n_nodes']}"
        )
        a_rank = sorted(a.read_root_scores(), key=a.read_root_scores().__getitem__)
        b_rank = sorted(b.read_root_scores(), key=b.read_root_scores().__getitem__)
        assert a_rank == b_rank
    finally:
        a.shutdown()
        b.shutdown()


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------


def test_broken_sender_chain_dropped_others_unaffected():
    """A sender whose chain has a gap is dropped; honest senders'
    positions in the canonical order are unchanged."""
    rpb = "rpb_canon_c"
    kp_good = Keypair.generate()
    kp_bad = Keypair.generate()
    chain_good = _make_batch_chain(rpb, kp_good, 2, "good")
    chain_bad = _make_batch_chain(rpb, kp_bad, 3, "bad")
    # Drop the middle batch from bad's chain — creates a gap.
    chain_bad_broken = [chain_bad[0], chain_bad[2]]

    full = chain_good + chain_bad_broken
    res = canonical_order(full)

    # Bad sender dropped.
    assert kp_bad.public_key in res.dropped_senders
    assert "sequence gap" in res.dropped_senders[kp_bad.public_key]
    # Good sender accepted.
    assert kp_good.public_key in res.per_sender_root
    # Output sequence contains only good's batches.
    assert all(b.sender_pubkey == kp_good.public_key for b in res.ordered_batches)
    assert len(res.ordered_batches) == len(chain_good)


def test_broken_hash_link_dropped():
    """Tamper with a batch's prev_batch_hash so the chain doesn't link."""
    rpb = "rpb_canon_d"
    kp = Keypair.generate()
    chain = _make_batch_chain(rpb, kp, 3, "xena")
    # Replace chain[1] with a corrupted prev_batch_hash.
    bad = chain[1]
    chain[1] = EventBatch(
        rpb_address=bad.rpb_address,
        sender_pubkey=bad.sender_pubkey,
        batch_seq=bad.batch_seq,
        events=bad.events,
        prev_batch_hash=b"\x99" * 32,  # garbage
        timestamp=bad.timestamp,
    )

    res = canonical_order(chain)
    assert kp.public_key in res.dropped_senders
    assert "prev_batch_hash mismatch" in res.dropped_senders[kp.public_key]
    assert res.ordered_batches == []


def test_first_batch_must_have_empty_prev_hash():
    """If batch_seq=1 carries a non-empty prev_batch_hash, the chain
    is rejected (no genesis predecessor)."""
    rpb = "rpb_canon_e"
    kp = Keypair.generate()
    bad_first = EventBatch(
        rpb_address=rpb,
        sender_pubkey=kp.public_key,
        batch_seq=1,
        events=[{"kind": "observation_added", "seq": 1,
                 "author_agent": "x", "obs_id": "o", "coords": [0]*1028,
                 "label": "x"}],
        prev_batch_hash=b"\xaa" * 32,
        timestamp=0.0,
    )
    res = canonical_order([bad_first])
    assert kp.public_key in res.dropped_senders


# ---------------------------------------------------------------------------
# Edge cases / determinism extras
# ---------------------------------------------------------------------------


def test_empty_input_yields_empty_canonical_order():
    from nodes.common.canonical_ordering import EMPTY_EPOCH_ROOT
    res = canonical_order([])
    assert res.ordered_batches == []
    assert res.per_sender_root == {}
    assert res.dropped_senders == {}
    # Stable, domain-separated empty epoch root for chain anchoring.
    # Distinguishable from sha256(b"") so a chain reader can tell
    # "epoch was canonically empty" from "no anchor submitted yet".
    assert res.epoch_root() == EMPTY_EPOCH_ROOT
    assert res.epoch_root() != hashlib.sha256(b"").digest()


def test_single_sender_passthrough():
    """Single sender's canonical = their batch_seq order."""
    rpb = "rpb_canon_f"
    kp = Keypair.generate()
    chain = _make_batch_chain(rpb, kp, 4, "solo")
    res = canonical_order(chain)
    seqs = [b.batch_seq for b in res.ordered_batches]
    assert seqs == [1, 2, 3, 4]


def test_sender_position_is_content_determined():
    """Two senders whose ONLY difference is the events they emit get
    different positions in the canonical order. Confirms ordering is
    truly content-derived, not arbitrary."""
    rpb = "rpb_canon_g"
    kp_x = Keypair.generate()
    kp_y = Keypair.generate()

    # Both chains length 2, same RPB, same timestamps — only event
    # contents differ.
    chain_x = _make_batch_chain(rpb, kp_x, 2, "xena")
    chain_y = _make_batch_chain(rpb, kp_y, 2, "yuli")
    res = canonical_order(chain_x + chain_y)

    # The two senders have distinct Merkle roots (different events
    # via author_agent label).
    root_x = res.per_sender_root[kp_x.public_key]
    root_y = res.per_sender_root[kp_y.public_key]
    assert root_x != root_y

    # The canonical sequence groups them: all of one sender, then all
    # of the other.
    senders_in_order = [b.sender_pubkey for b in res.ordered_batches]
    # Either xxx-then-yyy or yyy-then-xxx (depending on root sort);
    # senders are NOT interleaved.
    first = senders_in_order[0]
    block = []
    for pk in senders_in_order:
        if pk == first:
            block.append(pk)
        else:
            break
    assert len(block) == 2  # all of first sender's batches contiguous
    rest = senders_in_order[2:]
    assert all(pk != first for pk in rest)


def test_three_daemons_compute_same_canonical_order():
    """Three daemons receive the same batches in different orders;
    all three compute the same canonical sequence and epoch root."""
    rpb = "rpb_canon_h"
    senders = [Keypair.generate() for _ in range(4)]
    chains = [
        _make_batch_chain(rpb, kp, n, f"s{i}")
        for i, (kp, n) in enumerate(zip(senders, [1, 3, 2, 4]))
    ]
    all_batches = [b for c in chains for b in c]

    rng = random.Random(42)
    order_1 = list(all_batches); rng.shuffle(order_1)
    order_2 = list(all_batches); rng.shuffle(order_2)
    order_3 = list(all_batches); rng.shuffle(order_3)

    res_1 = canonical_order(order_1)
    res_2 = canonical_order(order_2)
    res_3 = canonical_order(order_3)

    seq_1 = [b.content_hash() for b in res_1.ordered_batches]
    seq_2 = [b.content_hash() for b in res_2.ordered_batches]
    seq_3 = [b.content_hash() for b in res_3.ordered_batches]
    assert seq_1 == seq_2 == seq_3
    assert res_1.epoch_root() == res_2.epoch_root() == res_3.epoch_root()
    # All four senders accepted.
    assert len(res_1.per_sender_root) == 4


# ---------------------------------------------------------------------------
# Phase 10.7: join-late tolerance
# ---------------------------------------------------------------------------


def test_chain_starting_above_one_is_accepted():
    """A daemon that joins after the network has been running sees a
    sender's batches starting mid-stream (e.g. seq=5, 6, 7). The
    chain is internally hash-linked but doesn't go back to seq=1.
    Phase 10.7 accepts these."""
    rpb = "rpb_canon_late_a"
    kp = Keypair.generate()

    # Build a full 1..7 chain, then drop the first 4 (simulating
    # what a late-joining daemon actually sees).
    full = _make_batch_chain(rpb, kp, 7, "alice")
    suffix = full[4:]  # seq 5, 6, 7
    assert suffix[0].batch_seq == 5
    assert suffix[0].prev_batch_hash == full[3].content_hash()

    res = canonical_order(suffix)
    assert kp.public_key not in res.dropped_senders, res.dropped_senders
    assert kp.public_key in res.per_sender_root
    assert len(res.ordered_batches) == 3
    assert [b.batch_seq for b in res.ordered_batches] == [5, 6, 7]


def test_internal_gap_still_rejected():
    """Sequence gaps WITHIN what we have are still rejected — the
    sender's chain is broken in our window, not just truncated."""
    rpb = "rpb_canon_late_b"
    kp = Keypair.generate()
    full = _make_batch_chain(rpb, kp, 5, "bob")
    # Take seq 2, 3, 5 — gap at 4
    gappy = [full[1], full[2], full[4]]

    res = canonical_order(gappy)
    assert kp.public_key in res.dropped_senders
    assert "sequence gap" in res.dropped_senders[kp.public_key]


def test_internal_hash_break_still_rejected():
    """Even if the chain starts above 1, internal hash links are
    still verified — a tampered prev_batch_hash within our window
    drops the sender."""
    rpb = "rpb_canon_late_c"
    kp = Keypair.generate()
    full = _make_batch_chain(rpb, kp, 5, "carol")
    # Take suffix seq 3, 4, 5 but tamper batch 4's prev_batch_hash.
    suffix = list(full[2:5])
    suffix[1] = EventBatch(
        rpb_address=suffix[1].rpb_address,
        sender_pubkey=suffix[1].sender_pubkey,
        batch_seq=suffix[1].batch_seq,
        events=suffix[1].events,
        prev_batch_hash=b"\xff" * 32,  # bogus
        timestamp=suffix[1].timestamp,
    )
    res = canonical_order(suffix)
    assert kp.public_key in res.dropped_senders
    assert "prev_batch_hash mismatch" in res.dropped_senders[kp.public_key]
