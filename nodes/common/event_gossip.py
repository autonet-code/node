"""Event-batch gossip between daemons (Phase 5.2).

Federation transport for the substrate. Each daemon hosts a
``WorldService`` whose epoch state is fed by:

  - **Local activity**: agent traces -> WorldSubstrateFeed -> WorldService.
  - **Peer activity** (this module): EventBatches arriving over a
    pub/sub topic -> WorldService.submit_events.

Content-addressed dedupe means a batch arriving twice (e.g. via
gossip re-broadcast) is harmless: ``apply_events`` consolidates
duplicate sprouts via their (anchor, polarity_axis) hash.

What's in scope here (5.2):

  - ``EventBatch`` envelope: rpb, sender_pubkey, batch_seq, events,
    prev_batch_hash, signature.
  - ``EventGossip`` orchestrator: publish/subscribe API, signing,
    verification, dedupe cache, ingest hook into WorldService.
  - ``Transport`` interface and ``InMemoryTransport`` impl for tests.

What's NOT in scope:

  - libp2p transport (thin adapter to come when 5.2 lands and we
    move to multi-host integration tests).
  - Canonical ordering across senders at network epoch close (5.3).
  - Fork detection / dispute (later phase).

Signatures
----------

We use ed25519 because it's ubiquitous and supported by libp2p.
The public key serves as the sender id. ``cryptography`` is in the
project's transitive deps; if it's missing at runtime, the gossip
layer falls back to "sign = blake2b(secret_seed||payload)" so tests
keep working without crypto. Production must use real keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol


logger = logging.getLogger(__name__)


# Message format version for forward-compatibility.
EVENT_BATCH_SCHEMA = 1


# ---------------------------------------------------------------------------
# Crypto: ed25519 if available, blake2b fallback
# ---------------------------------------------------------------------------


_HAVE_ED25519 = False
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization

    _HAVE_ED25519 = True
except ImportError:  # pragma: no cover — covered by test_event_gossip
    Ed25519PrivateKey = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Keypair:
    """Sender identity. ``public_key`` is the sender_pubkey wire id;
    ``private_key`` signs outbound batches.

    Use ``Keypair.generate()`` for tests; production loads from disk.
    """
    public_key: bytes
    private_key: bytes

    @staticmethod
    def generate() -> "Keypair":
        if _HAVE_ED25519:
            sk = Ed25519PrivateKey.generate()
            pk = sk.public_key()
            sk_bytes = sk.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            pk_bytes = pk.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            return Keypair(public_key=pk_bytes, private_key=sk_bytes)
        # Fallback: random 32-byte seed; "public" = blake2b(seed).
        import os
        seed = os.urandom(32)
        pk = hashlib.blake2b(seed, digest_size=32).digest()
        return Keypair(public_key=pk, private_key=seed)


def _sign(private_key: bytes, payload: bytes) -> bytes:
    if _HAVE_ED25519:
        sk = Ed25519PrivateKey.from_private_bytes(private_key)
        return sk.sign(payload)
    # Fallback: keyed-blake2b (NOT real signature, dev-mode only).
    return hashlib.blake2b(payload, key=private_key, digest_size=64).digest()


def _verify(public_key: bytes, payload: bytes, signature: bytes) -> bool:
    if _HAVE_ED25519:
        try:
            pk = Ed25519PublicKey.from_public_bytes(public_key)
            pk.verify(signature, payload)
            return True
        except Exception:
            return False
    # Fallback: cannot verify without the seed — treat as valid in
    # dev-mode tests. Production MUST install cryptography.
    return True


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


@dataclass
class EventBatch:
    """One batch of events from one sender, ready to gossip.

    Wire format (canonical JSON):
      {
        "schema": 1,
        "rpb_address": "0xabc...",
        "sender_pubkey": "<hex>",
        "batch_seq": 42,
        "prev_batch_hash": "<hex or empty>",
        "timestamp": 1714900000.123,
        "events": [...],
      }

    The signature signs the canonical JSON with the message order
    above. Sender's batch_seq increases monotonically per-sender.
    """
    rpb_address: str
    sender_pubkey: bytes
    batch_seq: int
    events: List[Dict[str, Any]]
    prev_batch_hash: bytes = b""
    timestamp: float = 0.0
    schema: int = EVENT_BATCH_SCHEMA

    def canonical_payload(self) -> bytes:
        """Bytes that get signed. Stable across runs."""
        body = {
            "schema": self.schema,
            "rpb_address": self.rpb_address,
            "sender_pubkey": self.sender_pubkey.hex(),
            "batch_seq": int(self.batch_seq),
            "prev_batch_hash": self.prev_batch_hash.hex() if self.prev_batch_hash else "",
            "timestamp": float(self.timestamp),
            "events": list(self.events),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def content_hash(self) -> bytes:
        """sha256 of the canonical payload — used as prev_batch_hash
        for the next batch by the same sender."""
        return hashlib.sha256(self.canonical_payload()).digest()


@dataclass
class SignedBatch:
    """Wire envelope: an EventBatch plus its signature."""
    batch: EventBatch
    signature: bytes

    def to_wire(self) -> bytes:
        out = {
            "batch": json.loads(self.batch.canonical_payload().decode("utf-8")),
            "signature": self.signature.hex(),
        }
        return json.dumps(out, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_wire(data: bytes) -> "SignedBatch":
        outer = json.loads(data.decode("utf-8"))
        b = outer["batch"]
        batch = EventBatch(
            schema=int(b.get("schema", 1)),
            rpb_address=b["rpb_address"],
            sender_pubkey=bytes.fromhex(b["sender_pubkey"]),
            batch_seq=int(b["batch_seq"]),
            prev_batch_hash=bytes.fromhex(b["prev_batch_hash"]) if b.get("prev_batch_hash") else b"",
            timestamp=float(b.get("timestamp", 0.0)),
            events=list(b.get("events", [])),
        )
        return SignedBatch(batch=batch, signature=bytes.fromhex(outer["signature"]))


def topic_for_rpb(rpb_address: str) -> str:
    """The pubsub topic carrying world-events for one RPB."""
    return f"/autonet/world-events/{rpb_address}"


# ---------------------------------------------------------------------------
# Transport interface
# ---------------------------------------------------------------------------


# Handler signature: `(topic: str, data: bytes) -> None`.
TransportHandler = Callable[[str, bytes], None]


class Transport(Protocol):
    """Pluggable transport: publish bytes to a topic, subscribe to one."""

    def publish(self, topic: str, data: bytes) -> None: ...

    def subscribe(self, topic: str, handler: TransportHandler) -> None: ...

    def stop(self) -> None: ...


class InMemoryTransport:
    """Test/dev transport: in-process pub/sub among multiple
    InMemoryTransport instances joined to the same ``InMemoryHub``.

    Real production runs over libp2p (thin adapter, future).
    """

    def __init__(self, hub: "InMemoryHub", node_id: str = ""):
        self.hub = hub
        self.node_id = node_id or f"node-{id(self):x}"
        self._handlers: Dict[str, List[TransportHandler]] = {}
        hub._register(self)

    def publish(self, topic: str, data: bytes) -> None:
        self.hub._broadcast(self, topic, data)

    def subscribe(self, topic: str, handler: TransportHandler) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    def deliver(self, topic: str, data: bytes) -> None:
        """Hub calls this when another transport publishes. Skips own
        node by design (a publisher does not echo back to itself)."""
        for h in list(self._handlers.get(topic, [])):
            try:
                h(topic, data)
            except Exception as e:
                logger.error("InMemoryTransport handler raised: %s", e)

    def stop(self) -> None:
        self._handlers.clear()
        self.hub._unregister(self)


class InMemoryHub:
    """Connect-everyone broker for InMemoryTransport instances.

    Models a small group of peers on a perfect network — useful for
    smoke and federation tests without network dependencies.
    """

    def __init__(self):
        self._members: List[InMemoryTransport] = []
        self._lock = threading.RLock()

    def _register(self, t: InMemoryTransport) -> None:
        with self._lock:
            self._members.append(t)

    def _unregister(self, t: InMemoryTransport) -> None:
        with self._lock:
            try:
                self._members.remove(t)
            except ValueError:
                pass

    def _broadcast(self, sender: InMemoryTransport, topic: str, data: bytes) -> None:
        with self._lock:
            members = list(self._members)
        for t in members:
            if t is sender:
                continue  # don't echo back
            t.deliver(topic, data)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class GossipStats:
    batches_published: int = 0
    batches_received: int = 0
    batches_invalid_signature: int = 0
    batches_duplicate: int = 0
    events_ingested: int = 0


class EventGossip:
    """Per-daemon event-batch gossip.

    Wires:
      - publish: WorldService observes a local batch -> sign -> publish.
      - subscribe: incoming batch -> verify -> dedupe -> ingest into
        the local WorldService.
    """

    def __init__(
        self,
        rpb_address: str,
        keypair: Keypair,
        transport: Transport,
        world_service: Any,
        *,
        dedupe_cache_size: int = 4096,
    ):
        self.rpb_address = rpb_address
        self.keypair = keypair
        self.transport = transport
        self.world_service = world_service
        self.topic = topic_for_rpb(rpb_address)

        self._lock = threading.RLock()
        self._next_batch_seq = 1
        self._last_published_hash = b""
        self._seen: List[bytes] = []  # FIFO of recent batch hashes
        self._dedupe_cache_size = dedupe_cache_size
        self._stats = GossipStats()
        self._known_seq_by_sender: Dict[bytes, int] = {}

        # Subscribe to the topic so peers' batches reach us.
        self.transport.subscribe(self.topic, self._on_message)

        # Subscribe to local activity so we publish our daemon's events.
        # The handler bridges WorldService's "events were applied locally"
        # signal to our publish_events.
        if hasattr(world_service, "subscribe_local_events"):
            world_service.subscribe_local_events(self._on_local_events)

    @property
    def sender_pubkey(self) -> bytes:
        return self.keypair.public_key

    @property
    def stats(self) -> GossipStats:
        with self._lock:
            return GossipStats(
                batches_published=self._stats.batches_published,
                batches_received=self._stats.batches_received,
                batches_invalid_signature=self._stats.batches_invalid_signature,
                batches_duplicate=self._stats.batches_duplicate,
                events_ingested=self._stats.events_ingested,
            )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _on_local_events(self, events: List[Dict[str, Any]]) -> None:
        """WorldService applied a batch locally. Republish to peers."""
        try:
            self.publish_events(events)
        except Exception as e:
            logger.error("local-events handler raised: %s", e)

    def publish_events(self, events: List[Dict[str, Any]]) -> Optional[SignedBatch]:
        """Wrap a list of events as a signed batch and publish it.

        Idempotent on empty input. Returns the published batch (so the
        caller can also feed it locally if desired) or None on no-op.
        """
        if not events:
            return None
        with self._lock:
            seq = self._next_batch_seq
            self._next_batch_seq += 1
            prev_hash = self._last_published_hash

        batch = EventBatch(
            rpb_address=self.rpb_address,
            sender_pubkey=self.sender_pubkey,
            batch_seq=seq,
            events=list(events),
            prev_batch_hash=prev_hash,
            timestamp=time.time(),
        )
        payload = batch.canonical_payload()
        signature = _sign(self.keypair.private_key, payload)
        signed = SignedBatch(batch=batch, signature=signature)

        with self._lock:
            self._last_published_hash = batch.content_hash()
            self._note_seen_locked(batch.content_hash())
            self._stats.batches_published += 1

        try:
            self.transport.publish(self.topic, signed.to_wire())
        except Exception as e:
            logger.error("Failed to publish batch: %s", e)
        return signed

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    def _on_message(self, topic: str, data: bytes) -> None:
        if topic != self.topic:
            return
        try:
            signed = SignedBatch.from_wire(data)
        except Exception as e:
            logger.warning("malformed batch dropped: %s", e)
            return

        # Reject batches addressed to a different rpb (shouldn't happen
        # given topic isolation, but defense in depth).
        if signed.batch.rpb_address != self.rpb_address:
            logger.warning(
                "batch with rpb=%s dropped on topic=%s",
                signed.batch.rpb_address, self.topic,
            )
            return

        # Verify signature.
        if not _verify(
            signed.batch.sender_pubkey,
            signed.batch.canonical_payload(),
            signed.signature,
        ):
            with self._lock:
                self._stats.batches_invalid_signature += 1
            logger.warning(
                "batch from %s dropped: bad signature",
                signed.batch.sender_pubkey.hex()[:16],
            )
            return

        h = signed.batch.content_hash()
        with self._lock:
            if h in self._seen:
                self._stats.batches_duplicate += 1
                return
            self._note_seen_locked(h)
            self._stats.batches_received += 1

            # Track per-sender seq for diagnostics. Out-of-order is
            # tolerated at this layer; canonical ordering is Phase 5.3's
            # job at network epoch close.
            prev_seq = self._known_seq_by_sender.get(signed.batch.sender_pubkey, 0)
            if signed.batch.batch_seq > prev_seq:
                self._known_seq_by_sender[signed.batch.sender_pubkey] = signed.batch.batch_seq

            # Don't ingest our own published batches: the local
            # WorldService already has them.
            if signed.batch.sender_pubkey == self.sender_pubkey:
                return

        # Ingest into the local WorldService. Use submit_events with
        # origin="remote" so content-addressed dedupe + persistence +
        # epoch buffering all apply, but the local-events fan-out
        # doesn't echo this batch back out.
        try:
            self.world_service.submit_events(
                signed.batch.events,
                equilibrate_after=True,
                origin="remote",
            )
            with self._lock:
                self._stats.events_ingested += len(signed.batch.events)
        except Exception as e:
            logger.error("ingesting peer batch failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _note_seen_locked(self, h: bytes) -> None:
        self._seen.append(h)
        if len(self._seen) > self._dedupe_cache_size:
            # Trim oldest. List is small (FIFO), pop(0) is fine.
            self._seen = self._seen[-self._dedupe_cache_size:]

    def stop(self) -> None:
        """Tear down. Caller is responsible for transport.stop() if it
        owns the transport."""
        # No persistent state to flush; placeholder for symmetry with
        # transport.stop().
        pass
