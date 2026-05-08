"""Phase 10.3a: federated-close driver tests.

The driver wires together gossip → canonical_order →
federated_epoch_close → submitter selection. The pieces it depends
on are already covered by their own tests; here we check only the
glue:

  - `pick_submitter` is deterministic across daemons given the same
    inputs (epoch_id, canonical_root, sender set).
  - `FederatedCloseDriver.run()` returns None when the gossip buffer
    is empty.
  - `FederatedCloseDriver.run()` produces a result whose
    `is_winner` flag is consistent with the deterministic-random
    selection rule (i.e. one of the daemons in a peer set is the
    winner; the others see is_winner=False).
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import (
    FederatedCloseDriver,
    pick_submitter,
)


def _stub_event(seq: int = 1, agent: str = "a", kind: str = "observation_added"):
    """Minimal valid event dict that survives canonical_order +
    apply_events + federated_reconcile_epoch."""
    return {
        "kind": kind,
        "seq": seq,
        "author_agent": agent,
        "obs_id": f"obs_{agent}_{seq}",
        "coords": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "label": f"{agent}_{seq}",
    }


def test_pick_submitter_deterministic():
    senders = [b"a" * 32, b"b" * 32, b"c" * 32]
    w1 = pick_submitter("epoch_42", senders, b"root_xyz")
    w2 = pick_submitter("epoch_42", list(reversed(senders)), b"root_xyz")
    assert w1 == w2
    assert w1 in senders


def test_pick_submitter_changes_with_epoch():
    senders = [b"a" * 32, b"b" * 32, b"c" * 32]
    seen = {pick_submitter(f"epoch_{i}", senders, b"") for i in range(20)}
    # With 20 epochs and 3 senders, we expect the deterministic
    # mapping to land on at least 2 distinct senders. (Not all 3
    # guaranteed in 20 trials, but at least 2 is.)
    assert len(seen) >= 2


def test_pick_submitter_empty_returns_none():
    assert pick_submitter("epoch_x", [], b"") is None


def test_driver_skips_when_no_batches():
    """No buffered batches → driver returns None, doesn't try to
    replay an empty canonical sequence."""
    gossip = MagicMock(spec=EventGossip)
    gossip.drain_epoch_batches.return_value = []
    gossip.known_senders.return_value = []
    gossip.sender_pubkey = b"x" * 32

    driver = FederatedCloseDriver(gossip=gossip, embedding_dim=8)
    out = driver.run(local_close_result={"epoch_id": "e1"})
    assert out is None


def test_driver_winner_is_consistent_across_daemons(tmp_path):
    """Two daemons see the same batches → both compute the same
    canonical order → both run pick_submitter with the same inputs
    → exactly one of them sees is_winner=True (or both see False if
    the chosen winner is some third party they both know about)."""
    # Build two keypairs (two daemons) and a shared batch set.
    kp_a = Keypair.generate()
    kp_b = Keypair.generate()

    # Each daemon publishes one batch with one event.
    rpb = "rpb_test"
    b_a = EventBatch(
        rpb_address=rpb,
        sender_pubkey=kp_a.public_key,
        batch_seq=1,
        events=[_stub_event(seq=1, agent="alice")],
        prev_batch_hash=b"",
        timestamp=1.0,
    )
    b_b = EventBatch(
        rpb_address=rpb,
        sender_pubkey=kp_b.public_key,
        batch_seq=1,
        events=[_stub_event(seq=1, agent="bob")],
        prev_batch_hash=b"",
        timestamp=1.0,
    )
    shared_batches = [b_a, b_b]
    senders = sorted([kp_a.public_key, kp_b.public_key])

    # Stub gossip A: pretends those are its buffered batches.
    gossip_a = MagicMock(spec=EventGossip)
    gossip_a.drain_epoch_batches.return_value = list(shared_batches)
    gossip_a.known_senders.return_value = senders
    gossip_a.sender_pubkey = kp_a.public_key

    # Same for gossip B.
    gossip_b = MagicMock(spec=EventGossip)
    gossip_b.drain_epoch_batches.return_value = list(shared_batches)
    gossip_b.known_senders.return_value = senders
    gossip_b.sender_pubkey = kp_b.public_key

    driver_a = FederatedCloseDriver(gossip=gossip_a, embedding_dim=8)
    driver_b = FederatedCloseDriver(gossip=gossip_b, embedding_dim=8)

    fed_a = driver_a.run(local_close_result={"epoch_id": "e1"})
    fed_b = driver_b.run(local_close_result={"epoch_id": "e1"})

    assert fed_a is not None and fed_b is not None
    # Both daemons saw the same canonical order -> same winner.
    assert fed_a.winner == fed_b.winner
    # Exactly one of them sees themselves as the winner.
    assert (fed_a.is_winner != fed_b.is_winner), (
        f"both daemons claim winner={fed_a.is_winner}/{fed_b.is_winner}"
    )
    # Both produced a non-empty close_result with a payload.
    assert "authoritative_payload" in fed_a.close_result
    assert (
        fed_a.close_result["authoritative_payload"]
        == fed_b.close_result["authoritative_payload"]
    ), "authoritative_payload must be bit-identical across daemons"
