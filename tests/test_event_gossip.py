"""Event-batch gossip between daemons.

Validates:
  1. Two daemons, same jurisdiction: one publishes a tool, the other's
     library service receives its tool_registered event via gossip (epoch
     buffer and discovery queue).
  2. Signature verification: a tampered batch is dropped on receive.
  3. Duplicate suppression: re-broadcasting the same batch ingests once.
  4. Origin tagging: a remote batch ingested locally does NOT trigger the
     local-events fan-out (no echo loops).
  5. Different jurisdictions are isolated by topic, even on a shared hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from nodes.common.event_gossip import (
    EventBatch,
    EventGossip,
    InMemoryHub,
    InMemoryTransport,
    Keypair,
    SignedBatch,
    topic_for_rpb,
)
from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library_service import LibraryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _manifest(name: str, author: str = "0x" + "a1" * 20):
    return build_tool_manifest(
        name=name,
        description=f"{name}: does a thing for the gossip test.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        author=author,
        trust_class="pinned",
        code_digest="c" * 64,
        created_ts=1780531200,
    )


def _peer(rpb: str, hub: InMemoryHub, tmp_path: Path, name: str) -> Tuple[LibraryService, EventGossip]:
    svc = LibraryService(rpb_address=rpb, data_root=tmp_path / name)
    transport = InMemoryTransport(hub, node_id=name)
    gossip = EventGossip(
        rpb_address=rpb,
        keypair=Keypair.generate(),
        transport=transport,
        library=svc,
    )
    return svc, gossip


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_daemons_converge_after_gossip(tmp_path: Path):
    """A publishes a tool; B receives the registration through gossip:
    the same event in both epoch buffers, and the digest queued on B for
    discovery."""
    hub = InMemoryHub()
    rpb = "rpb_conv"
    svc_a, gos_a = _peer(rpb, hub, tmp_path, "A")
    svc_b, gos_b = _peer(rpb, hub, tmp_path, "B")
    svc_a.open_epoch("e1")
    svc_b.open_epoch("e1")
    try:
        digest = svc_a.submit_tool_manifest(_manifest("intel_1"), agent_id="alice")["manifest_digest"]
        assert svc_a.epoch_status()["buffered_events"] == 1
        assert svc_b.epoch_status()["buffered_events"] == 1
        assert svc_a._epoch_events == svc_b._epoch_events
        assert svc_b._epoch_events[0]["kind"] == "tool_registered"
        assert svc_b.take_pending_remote_manifests() == [digest]
        assert gos_a.stats.batches_published == 1
        assert gos_b.stats.batches_received == 1
        assert gos_b.stats.events_ingested == 1
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_tampered_signature_is_dropped(tmp_path: Path):
    hub = InMemoryHub()
    rpb = "rpb_sig"
    svc_a, gos_a = _peer(rpb, hub, tmp_path, "A")
    svc_b, gos_b = _peer(rpb, hub, tmp_path, "B")

    # Bypass the legitimate publish path: hand-craft a batch claiming
    # to come from A but signed with bogus bytes.
    fake_batch = EventBatch(
        rpb_address=rpb,
        sender_pubkey=gos_a.sender_pubkey,
        batch_seq=999,
        events=[{"kind": "tool_registered", "artifact_digest": "ab" * 32,
                 "manifest_meta": {"author": "x"}}],
    )
    bogus = SignedBatch(batch=fake_batch, signature=b"\x00" * 64)

    try:
        attacker = InMemoryTransport(hub, node_id="attacker")
        attacker.publish(topic_for_rpb(rpb), bogus.to_wire())

        # The no-crypto fallback accepts any signature, so the drop is
        # only checked when cryptography is installed.
        from nodes.common.event_gossip import _HAVE_ED25519
        if _HAVE_ED25519:
            assert gos_b.stats.batches_invalid_signature == 1, gos_b.stats
            assert gos_b.stats.events_ingested == 0
        else:
            assert gos_b.stats.batches_received >= 1
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_duplicate_batch_ingested_once(tmp_path: Path):
    """Re-publishing the same signed batch must not ingest its events
    twice. The dedupe cache catches the second arrival."""
    hub = InMemoryHub()
    rpb = "rpb_dup"
    svc_a, gos_a = _peer(rpb, hub, tmp_path, "A")
    svc_b, gos_b = _peer(rpb, hub, tmp_path, "B")
    svc_b.open_epoch("e1")
    try:
        signed = gos_a.publish_events([
            {"kind": "tool_registered", "seq": 1, "author_agent": "alice",
             "artifact_digest": "ab" * 32,
             "manifest_meta": {"trust_class": "pinned", "author": "alice"}},
        ])
        first_count = svc_b.epoch_status()["buffered_events"]
        gos_a.transport.publish(gos_a.topic, signed.to_wire())
        assert svc_b.epoch_status()["buffered_events"] == first_count == 1
        assert gos_b.stats.batches_duplicate >= 1
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_remote_ingest_does_not_re_publish(tmp_path: Path):
    """When B ingests A's batch, B must NOT republish to the hub
    (otherwise an N^2 storm). Verified by counting publishes."""
    hub = InMemoryHub()
    rpb = "rpb_loop"
    svc_a, gos_a = _peer(rpb, hub, tmp_path, "A")
    svc_b, gos_b = _peer(rpb, hub, tmp_path, "B")
    try:
        svc_a.submit_tool_manifest(_manifest("x"), agent_id="alice")
        assert gos_a.stats.batches_published == 1
        assert gos_b.stats.batches_published == 0
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_different_rpbs_isolated(tmp_path: Path):
    """Two daemons on different jurisdictions sharing the same hub must
    not bleed events between them."""
    hub = InMemoryHub()
    svc_a, gos_a = _peer("rpb_X", hub, tmp_path, "A")
    svc_b, gos_b = _peer("rpb_Y", hub, tmp_path, "B")
    try:
        svc_a.submit_tool_manifest(_manifest("x"), agent_id="alice")
        assert gos_b.stats.batches_received == 0
        assert gos_b.stats.events_ingested == 0
        assert svc_b.take_pending_remote_manifests() == []
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_three_daemons_all_converge(tmp_path: Path):
    """Three daemons, one tool each from different agents: all three
    buffer the same three registrations."""
    hub = InMemoryHub()
    rpb = "rpb_3"
    peers = [_peer(rpb, hub, tmp_path, n)[0] for n in ("A", "B", "C")]
    for svc in peers:
        svc.open_epoch("e1")
    try:
        for svc, (name, agent) in zip(peers, [("a", "alice"), ("b", "bob"), ("c", "carol")]):
            svc.submit_tool_manifest(_manifest(f"tool_{name}"), agent_id=agent)
        digests = [sorted(e["artifact_digest"] for e in svc._epoch_events) for svc in peers]
        assert digests[0] == digests[1] == digests[2]
        assert len(digests[0]) == 3
    finally:
        for svc in peers:
            svc.shutdown()


class _GatedTransport(InMemoryTransport):
    """An InMemoryTransport that can report having no peers."""
    online = False

    def has_peers(self, topic):
        return self.online


def test_events_without_peers_wait_and_are_signed_when_sent(tmp_path: Path):
    """Events published before any peer is subscribed (a booting daemon)
    are held, not lost: signed only when a peer can hear them, and only
    then part of the sender's own epoch set, so the sender's close and its
    peers' closes count the same batches."""
    hub = InMemoryHub()
    rpb = "rpb_outbox"
    svc_a = LibraryService(rpb_address=rpb, data_root=tmp_path / "A")
    gos_a = EventGossip(rpb_address=rpb, keypair=Keypair.generate(),
                        transport=_GatedTransport(hub, node_id="A"), library=svc_a)
    svc_b, gos_b = _peer(rpb, hub, tmp_path, "B")
    now = [1000.0]
    gos_a.clock = lambda: now[0]
    ev = {"kind": "tool_registered", "seq": 1, "author_agent": "alice",
          "artifact_digest": "cd" * 32,
          "manifest_meta": {"trust_class": "pinned", "author": "alice"}}
    try:
        assert gos_a.publish_events([ev]) is None
        assert gos_a.pending_epoch_batches() == []
        assert gos_b.stats.batches_received == 0

        now[0] = 1090.0
        assert gos_a.flush_outbox() is None          # still nobody there
        gos_a.transport.online = True
        signed = gos_a.flush_outbox()
        assert signed is not None and signed.batch.timestamp == 1090.0
        assert [b.content_hash() for b in gos_a.pending_epoch_batches()] == \
            [b.content_hash() for b in gos_b.pending_epoch_batches()]
        assert gos_b.stats.batches_received == 1
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_sender_chain_survives_a_restart(tmp_path):
    """A daemon restart inside an epoch keeps its batch_seq and hash chain,
    so peers keep the whole of its epoch."""
    from nodes.common.event_gossip import EventGossip, InMemoryHub, InMemoryTransport, Keypair

    class Library:
        rpb_address = "0x" + "11" * 20

    path = tmp_path / "sender_seq.json"
    kp = Keypair.generate()

    def daemon():
        return EventGossip(rpb_address=Library.rpb_address, keypair=kp,
                           transport=InMemoryTransport(InMemoryHub()),
                           library=Library(), seq_path=path)

    first = daemon()
    a = first._sign_and_publish([{"n": 1}])
    b = first._sign_and_publish([{"n": 2}])
    second = daemon()
    c = second._sign_and_publish([{"n": 3}])
    assert (a.batch.batch_seq, b.batch.batch_seq, c.batch.batch_seq) == (1, 2, 3)
    assert c.batch.prev_batch_hash == b.batch.content_hash()


def test_open_epoch_batches_survive_a_restart(tmp_path):
    """A week's buffer is not lost to a reboot: batches come back from the
    log for the close and for peers catching up, and a cut last line is fine."""
    from nodes.common.event_gossip import EventGossip, InMemoryHub, InMemoryTransport, Keypair

    class Library:
        rpb_address = "0x" + "22" * 20

    path = tmp_path / "batches.jsonl"
    kp = Keypair.generate()

    def daemon():
        return EventGossip(rpb_address=Library.rpb_address, keypair=kp,
                           transport=InMemoryTransport(InMemoryHub()),
                           library=Library(), batch_log_path=path)

    first = daemon()
    a = first._sign_and_publish([{"n": 1}])
    b = first._sign_and_publish([{"n": 2}])
    with path.open("a", encoding="utf-8") as f:
        f.write('{"h": "ab", "ts"')
    second = daemon()
    held = {x.content_hash() for x in second.pending_epoch_batches()}
    assert held == {a.batch.content_hash(), b.batch.content_hash()}
    assert len(second.batch_log.since(0.0, set())) == 2
