"""Federated-close participation preconditions.

A daemon must not take part in a federated close it cannot compute
identically to its peers. With ``enforce_preconditions`` on (as
AutonetService sets it), ``FederatedCloseDriver.run()`` logs and skips
(returns None, no crash, batches still drained) unless:

  - chain read access is wired (``voice_source``: substrate + rpc), and
  - the five carry-over files exist, OR the chain has no anchors yet
    (genesis, where an empty prior is the correct prior).

A joiner (files missing, chain anchored) fetches the verified carry-over
through ``carry_source`` and observes until it succeeds.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from nodes.common.authoritative_encoding import decode_carry_bundle
from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver
from nodes.common.state_sync import CarryOverFetch

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


def _peer_bundle(tmp_path):
    peer_dir = tmp_path / "peer"
    peer_dir.mkdir()
    peer = _driver(peer_dir, voice_state=GENESIS)
    first = peer.run({"epoch_id": "e1"})
    assert first is not None and first.carry_cid
    return peer, first


def _observing_joiner(tmp_path):
    """A fresh daemon on an anchored chain that has observed one epoch."""
    me_dir = tmp_path / "me"
    me_dir.mkdir()
    me = _driver(me_dir, voice_state=ANCHORED)
    me.carry_source = lambda: CarryOverFetch(status="unavailable")
    assert me.run({"epoch_id": "local_1"}) is None
    assert me._last_observed_root is not None
    return me, me_dir


def test_joiner_installs_verified_carry_and_closes(tmp_path):
    # A fresh daemon on a chain with anchors: the carry_source hook (in
    # production: fetch_carry_over_from_chain, hash-verified against the
    # anchor) supplies the bundle; the driver installs it and closes,
    # because the anchor is the epoch it observed just before.
    peer, first = _peer_bundle(tmp_path)
    me, me_dir = _observing_joiner(tmp_path)
    observed = me._last_observed_root
    me.carry_source = lambda: CarryOverFetch(
        status="ok", epoch_id="e1", epoch_root_hex=observed,
        carry_cid=first.carry_cid,
        carry=decode_carry_bundle(first.carry_bundle_blob)["carry"])
    assert me.run({"epoch_id": "local_2"}) is not None
    for k in CARRY:
        assert (me_dir / f"{k}.json").exists()
    assert me.participation_blockers() == []
    assert me._tool_positions == peer._tool_positions
    assert me._tool_review_book == peer._tool_review_book


def test_joiner_first_epoch_only_observes(tmp_path, caplog):
    # No prior epoch observed: never fetch, even if a bundle exists.
    calls = []
    me_dir = tmp_path / "me"
    me_dir.mkdir()
    me = _driver(me_dir, voice_state=ANCHORED)
    me.carry_source = lambda: calls.append(1)
    with caplog.at_level(logging.INFO):
        assert me.run({"epoch_id": "local_1"}) is None
    assert calls == []
    assert "has not observed a prior epoch yet" in caplog.text


def test_joiner_refuses_lagging_anchor(tmp_path, caplog):
    # The latest anchor is NOT the epoch observed just before this close
    # (the last winner failed to anchor): installing would be stale.
    _, first = _peer_bundle(tmp_path)
    me, me_dir = _observing_joiner(tmp_path)
    me.carry_source = lambda: CarryOverFetch(
        status="ok", epoch_id="e_old", epoch_root_hex="ff" * 32,
        carry_cid=first.carry_cid,
        carry=decode_carry_bundle(first.carry_bundle_blob)["carry"])
    with caplog.at_level(logging.WARNING):
        assert me.run({"epoch_id": "local_2"}) is None
    assert "is not the epoch observed immediately before this close" in caplog.text
    assert "refusing to install a stale carry-over" in caplog.text
    assert not any((me_dir / f"{k}.json").exists() for k in CARRY)


def test_joiner_keeps_refusing_on_pre_schema_anchor(tmp_path, caplog):
    me, me_dir = _observing_joiner(tmp_path)
    me.carry_source = lambda: CarryOverFetch(
        status="pre_schema", epoch_id="old", payload_schema=3)
    with caplog.at_level(logging.INFO):
        assert me.run({"epoch_id": "local_2"}) is None
    assert "predates carry-over commitments" in caplog.text
    assert "missing carry-over files (chain has prior anchors)" in caplog.text
    assert not any((me_dir / f"{k}.json").exists() for k in CARRY)


def test_joiner_rejects_failed_verification(tmp_path, caplog):
    def tampered():
        raise ValueError("carry bundle hash mismatch")
    me, me_dir = _observing_joiner(tmp_path)
    me.carry_source = tampered
    with caplog.at_level(logging.WARNING):
        assert me.run({"epoch_id": "local_2"}) is None
    assert "REJECTED anchored carry-over bundle" in caplog.text
    assert not any((me_dir / f"{k}.json").exists() for k in CARRY)


def test_carry_source_not_consulted_when_files_present(tmp_path):
    _write_all(tmp_path)
    calls = []
    me = _driver(tmp_path, voice_state=ANCHORED)
    me.carry_source = lambda: calls.append(1)
    assert me.run({"epoch_id": "e3"}) is not None
    assert calls == []
