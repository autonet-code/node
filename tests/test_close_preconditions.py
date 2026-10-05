"""Federated-close participation preconditions.

A daemon must not take part in a federated close it cannot compute
identically to its peers. With ``enforce_preconditions`` on (as
AutonetService sets it), ``FederatedCloseDriver.run()`` logs and skips
(returns None, no crash, batches still drained) unless:

  - chain read access is wired (``voice_source``: substrate + rpc), and
  - the five carry-over files exist, OR the chain has no anchors yet
    (genesis, where an empty prior is the correct prior).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver

CARRY = [
    "tool_registrations",
    "tool_vetting",
    "tool_positions",
    "tool_credibility",
    "tool_review_book",
]


def _gossip():
    kp = Keypair.generate()
    batch = EventBatch(
        rpb_address="rpb_test",
        sender_pubkey=kp.public_key,
        batch_seq=1,
        events=[{
            "kind": "observation_added",
            "seq": 1,
            "author_agent": "alice",
            "obs_id": "obs_alice_1",
            "coords": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            "label": "alice_1",
        }],
        prev_batch_hash=b"",
        timestamp=1.0,
    )
    g = MagicMock(spec=EventGossip)
    g.drain_epoch_batches.return_value = [batch]
    g.known_senders.return_value = [kp.public_key]
    g.sender_pubkey = kp.public_key
    return g


def _driver(tmp_path, *, voice_state=None, enforce=True, paths=True):
    kwargs = {}
    if paths:
        kwargs = {f"{k}_path": tmp_path / f"{k}.json" for k in CARRY}
    d = FederatedCloseDriver(gossip=_gossip(), embedding_dim=8, **kwargs)
    d.enforce_preconditions = enforce
    if voice_state is not None:
        d.voice_source = lambda: dict(voice_state)
    return d


# read_voice_state shapes: snapshot_block None <=> anchorCount == 0.
GENESIS = {"owner_map": {}, "voice_weights": {}, "snapshot_block": None,
           "emission_pool": 0.0}
ANCHORED = {"owner_map": {}, "voice_weights": {}, "snapshot_block": 123,
            "emission_pool": 0.0}


def _write_all(tmp_path):
    for k in CARRY:
        (tmp_path / f"{k}.json").write_text("{}", encoding="utf-8")


def test_refuses_without_chain_rpc(tmp_path, caplog):
    _write_all(tmp_path)
    d = _driver(tmp_path)  # no voice_source
    with caplog.at_level(logging.WARNING):
        out = d.run({"epoch_id": "e1"})
    assert out is None
    assert "chain RPC not configured" in caplog.text
    # Batches drained anyway, so they can't leak into the next epoch.
    d.gossip.drain_epoch_batches.assert_called_once()


def test_refuses_missing_carry_over_after_genesis(tmp_path, caplog):
    d = _driver(tmp_path, voice_state=ANCHORED)
    with caplog.at_level(logging.WARNING):
        out = d.run({"epoch_id": "e2"})
    assert out is None
    assert "missing carry-over files (chain has prior anchors)" in caplog.text
    for k in CARRY:
        assert k in caplog.text


def test_refuses_partial_carry_over(tmp_path):
    _write_all(tmp_path)
    (tmp_path / "tool_review_book.json").unlink()
    d = _driver(tmp_path, voice_state=ANCHORED)
    assert d.run({"epoch_id": "e2"}) is None
    d._refresh_voice()
    assert d.participation_blockers() == [
        "missing carry-over files (chain has prior anchors): tool_review_book"
    ]


def test_refuses_when_genesis_status_unknown(tmp_path):
    def boom():
        raise RuntimeError("rpc down")
    d = _driver(tmp_path)
    d.voice_source = boom
    assert d.run({"epoch_id": "e1"}) is None
    assert any("genesis status unknown" in b
               for b in d.participation_blockers())


def test_refuses_without_state_dir(tmp_path):
    d = _driver(tmp_path, voice_state=GENESIS, paths=False)
    assert d.run({"epoch_id": "e1"}) is None
    d._refresh_voice()
    assert any(b.startswith("no state dir for carry-over")
               for b in d.participation_blockers())


def test_genesis_closes_without_carry_over_and_writes_it(tmp_path):
    d = _driver(tmp_path, voice_state=GENESIS)
    out = d.run({"epoch_id": "e1"})
    assert out is not None
    # The close persisted every carry-over file, so the next close
    # (post-genesis) passes the file check.
    for k in CARRY:
        assert (tmp_path / f"{k}.json").exists()
    d._chain_at_genesis = False
    assert d.participation_blockers() == []


def test_closes_with_chain_and_carry_over(tmp_path):
    _write_all(tmp_path)
    d = _driver(tmp_path, voice_state=ANCHORED)
    assert d.run({"epoch_id": "e3"}) is not None


def test_default_off_keeps_library_behavior(tmp_path):
    d = _driver(tmp_path, enforce=False, paths=False)
    assert d.run({"epoch_id": "e1"}) is not None


def test_joiner_after_genesis_recovers_via_seed(tmp_path):
    # A fresh daemon on a chain with anchors: refused until seeded, then
    # closes and keeps closing (the refusal is not permanent).
    peer_dir = tmp_path / "peer"
    peer_dir.mkdir()
    peer = _driver(peer_dir, voice_state=GENESIS)
    assert peer.run({"epoch_id": "e1"}) is not None

    me_dir = tmp_path / "me"
    me_dir.mkdir()
    me = _driver(me_dir, voice_state=ANCHORED)
    assert me.run({"epoch_id": "e2"}) is None

    assert sorted(me.import_carry_over(peer_dir)) == sorted(CARRY)
    me._refresh_voice()
    assert me.participation_blockers() == []
    assert me._tool_positions == peer._tool_positions
    assert me._tool_review_book == peer._tool_review_book
    me.gossip = _gossip()
    assert me.run({"epoch_id": "e3"}) is not None


def test_seed_import_never_overwrites_local_state(tmp_path):
    peer_dir = tmp_path / "peer"
    peer_dir.mkdir()
    (peer_dir / "tool_credibility.json").write_text('{"h": 0.5}', encoding="utf-8")
    (peer_dir / "tool_vetting.json").write_text("[1]", encoding="utf-8")
    me_dir = tmp_path / "me"
    me_dir.mkdir()
    (me_dir / "tool_credibility.json").write_text('{"h": 0.9}', encoding="utf-8")
    me = _driver(me_dir, voice_state=ANCHORED)
    # Local credibility kept; non-object vetting seed rejected.
    assert me.import_carry_over(peer_dir) == []
    assert me._tool_credibility == {"h": 0.9}
    assert not (me_dir / "tool_vetting.json").exists()
