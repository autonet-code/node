"""Event-batch gossip between daemons (Phase 5.2).

Federation transport for the tool library. Each daemon hosts a
``LibraryService`` whose epoch buffer is fed by:

  - **Local activity**: published tools -> LibraryService.
  - **Peer activity** (this module): EventBatches arriving over a
    pub/sub topic -> LibraryService.submit_events.

A batch arriving twice (e.g. via gossip re-broadcast) is dropped by the
content-hash dedupe cache; the close keeps the first registration of a
digest.

What's in scope here (5.2):

  - ``EventBatch`` envelope: rpb, sender_pubkey, batch_seq, events,
    prev_batch_hash, signature.
  - ``EventGossip`` coordinator: publish/subscribe API, signing,
    verification, dedupe cache, ingest hook into LibraryService.
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
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
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


class LibP2PTransport:
    """Production transport: gossipsub via ``AutonetHost``.

    Bridges the sync ``Transport`` Protocol to the host's async
    publish/subscribe surface. Cross-thread safe — the substrate feed
    runs in the daemon's blocking worker thread, the host runs in its
    own trio thread.
    """

    def __init__(self, host: Any):
        if host is None:
            raise ValueError("LibP2PTransport requires an AutonetHost")
        self._host = host
        self._handlers: Dict[str, List[TransportHandler]] = {}
        self._subscribed: set[str] = set()

    def publish(self, topic: str, data: bytes) -> None:
        self._host.publish_topic_threadsafe(topic, data)

    def has_peers(self, topic: str) -> bool:
        """Whether a publish on ``topic`` would reach anyone."""
        return self._host.topic_peer_count(topic) > 0

    def subscribe(self, topic: str, handler: TransportHandler) -> None:
        self._handlers.setdefault(topic, []).append(handler)
        if topic in self._subscribed:
            return
        self._subscribed.add(topic)

        # The host invokes _dispatch on its trio thread; we forward to
        # all registered handlers synchronously. Handlers are expected
        # to be cheap (they enqueue or write to a thread-safe state).
        def _dispatch(t: str, d: bytes) -> None:
            for h in list(self._handlers.get(t, [])):
                try:
                    h(t, d)
                except Exception as e:
                    logger.error(
                        "LibP2PTransport handler raised on %s: %s", t, e,
                    )

        self._host.subscribe_topic_threadsafe(topic, _dispatch)

    def stop(self) -> None:
        # Host shutdown cancels its nursery, which kills topic readers;
        # nothing to do here.
        self._handlers.clear()
        self._subscribed.clear()


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
# Coordinator
# ---------------------------------------------------------------------------


class BatchLog:
    """Recently seen signed batches, kept as wire bytes so a peer that
    missed them can ask (issue #41: GossipSub keeps no history, so a daemon
    that joins or loses its peers mid-epoch never sees the batches sent
    before). Entries older than ``retention`` seconds are dropped: two
    weeks, so a whole epoch stays available however late a peer asks. Daemons
    and relays both keep one; requesters verify every batch themselves. With
    a ``path`` the log survives a restart."""

    def __init__(self, retention: float = 14 * 86400.0, max_entries: int = 100000,
                 clock: Callable[[], float] = time.time,
                 path: Optional[Any] = None):
        self.retention = float(retention)
        self.max_entries = int(max_entries)
        self.clock = clock
        self._lock = threading.Lock()
        self._entries: Dict[bytes, tuple] = {}  # content hash -> (timestamp, wire)
        self._path = Path(path) if path else None
        self._load()

    def _load(self) -> None:
        """Read the file back, then rewrite it without what has expired."""
        if self._path is None or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    self._entries[bytes.fromhex(r["h"])] = (
                        float(r["ts"]), r["w"].encode("utf-8"))
                except Exception:   # a line cut short by a crash
                    continue
        self._prune_locked(force=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for h, (ts, w) in self._entries.items():
                f.write(self._line(h, ts, w))
        os.replace(tmp, self._path)

    @staticmethod
    def _line(h: bytes, ts: float, wire: bytes) -> str:
        return json.dumps({"h": h.hex(), "ts": ts, "w": wire.decode("utf-8")}) + chr(10)

    def wires(self) -> List[bytes]:
        """Every kept batch, oldest first."""
        with self._lock:
            return [w for _, w in sorted(self._entries.values(), key=lambda e: e[0])]

    def add(self, content_hash: bytes, timestamp: float, wire: bytes) -> None:
        with self._lock:
            self._entries[content_hash] = (float(timestamp), wire)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(self._line(content_hash, float(timestamp), wire))
            if len(self._entries) > self.max_entries:
                self._prune_locked(force=True)

    def _prune_locked(self, force: bool = False) -> None:
        floor = float(self.clock()) - self.retention
        self._entries = {h: e for h, e in self._entries.items() if e[0] >= floor}
        if force and len(self._entries) > self.max_entries:
            keep = sorted(self._entries.items(), key=lambda kv: kv[1][0])
            self._entries = dict(keep[-self.max_entries:])

    def since(self, start: float, have: set, limit: int = 500) -> List[bytes]:
        """Wire bytes of the batches signed after ``start`` whose hash is
        not in ``have``, oldest first, at most ``limit``."""
        with self._lock:
            self._prune_locked()
            picked = sorted(((ts, h, w) for h, (ts, w) in self._entries.items()
                             if ts > start and h.hex() not in have),
                            key=lambda t: (t[0], t[1]))
        return [w for _, _, w in picked[:limit]]

    def serve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a catch-up request ``{since, have, limit}``."""
        try:
            start = float(request.get("since") or 0.0)
            have = {str(h) for h in (request.get("have") or [])}
            limit = max(1, min(int(request.get("limit") or 500), 500))
        except (TypeError, ValueError):
            return {"error": "bad catch-up request"}
        return {"batches": [w.decode("utf-8") for w in self.since(start, have, limit)]}


@dataclass
class GossipStats:
    batches_published: int = 0
    batches_received: int = 0
    batches_invalid_signature: int = 0
    batches_duplicate: int = 0
    events_ingested: int = 0
    batches_skewed: int = 0


class EventGossip:
    """Per-daemon event-batch gossip.

    Wires:
      - publish: library service observes a local batch -> sign -> publish.
      - subscribe: incoming batch -> verify -> dedupe -> ingest into
        the local library service.
    """

    def __init__(
        self,
        rpb_address: str,
        keypair: Keypair,
        transport: Transport,
        library: Any,
        *,
        dedupe_cache_size: int = 4096,
        seq_path: Optional[Any] = None,
        batch_log_path: Optional[Any] = None,
    ):
        self.rpb_address = rpb_address
        self.keypair = keypair
        self.transport = transport
        self.library = library
        self.topic = topic_for_rpb(rpb_address)

        self._lock = threading.RLock()
        self._next_batch_seq = 1
        self._last_published_hash = b""
        # Where the sender chain survives a restart: peers reject a sender
        # whose batch_seq jumps back, which loses its whole epoch.
        self._seq_path = Path(seq_path) if seq_path else None
        self._load_seq()
        self._seen: List[bytes] = []  # FIFO of recent batch hashes
        self._dedupe_cache_size = dedupe_cache_size
        self._stats = GossipStats()
        self._known_seq_by_sender: Dict[bytes, int] = {}
        # Per-epoch buffer of accepted batches (own + peer). Drained
        # by the federated-close driver at epoch boundaries to build
        # the canonical order. Held under self._lock.
        self._epoch_batches: List[EventBatch] = []
        # Local events held while no peer is subscribed (publish_events).
        self._outbox: List[List[Dict[str, Any]]] = []
        # Largest allowed gap (seconds) between a batch's signed timestamp
        # and its arrival (Decision 2026-10-26, the governed epoch_skew,
        # set by the close driver from the pinned rules). None = no check.
        self.max_skew_seconds: Optional[float] = None
        self.clock = time.time
        # Batches we accepted, served to peers catching up (issue #41).
        self.batch_log = BatchLog(clock=lambda: self.clock(), path=batch_log_path)
        # The open epoch's batches come back after a restart; the close
        # keeps only those signed after its window's start.
        for wire in self.batch_log.wires():
            try:
                batch = SignedBatch.from_wire(wire).batch
            except Exception:
                continue
            self._note_seen_locked(batch.content_hash())
            self._epoch_batches.append(batch)

        # Subscribe to the topic so peers' batches reach us.
        self.transport.subscribe(self.topic, self._on_message)

        # Subscribe to local activity so we publish our daemon's events.
        # The handler bridges the library service's "events were applied locally"
        # signal to our publish_events.
        if hasattr(library, "subscribe_local_events"):
            library.subscribe_local_events(self._on_local_events)

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
                batches_skewed=self._stats.batches_skewed,
            )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _on_local_events(self, events: List[Dict[str, Any]]) -> None:
        """The library service took a local batch. Republish to peers."""
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
        # Nobody to hear it (booting, or cut off): hold the events and sign
        # them when a peer is back. A batch is signed only when it is sent,
        # so its timestamp is within every receiver's skew, and it joins our
        # own epoch set only when peers can have it too; otherwise our own
        # close counts batches nobody else ever saw.
        has_peers = getattr(self.transport, "has_peers", None)
        if has_peers is not None:
            with self._lock:
                if self._outbox or not has_peers(self.topic):
                    self._outbox.append(list(events))
                    held = True
                else:
                    held = False
            if held:
                return self.flush_outbox()
        return self._sign_and_publish(events)

    def flush_outbox(self) -> Optional[SignedBatch]:
        """Publish the held events, oldest first, once a peer can hear
        them. Called on every publish and by the service loop. Returns the
        last batch published, or None."""
        has_peers = getattr(self.transport, "has_peers", None)
        last = None
        while True:
            with self._lock:
                if not self._outbox or (has_peers is not None
                                        and not has_peers(self.topic)):
                    return last
                events = self._outbox.pop(0)
            last = self._sign_and_publish(events)

    def _load_seq(self) -> None:
        if self._seq_path is None or not self._seq_path.exists():
            return
        try:
            data = json.loads(self._seq_path.read_text(encoding="utf-8"))
            self._next_batch_seq = int(data["next_batch_seq"])
            self._last_published_hash = bytes.fromhex(data["last_hash"])
        except Exception as e:
            logger.warning("gossip sequence state unreadable (%s): "
                           "restarting the sender chain", e)

    def _save_seq(self) -> None:
        """Written before the batch goes out: a crash then costs one gap,
        never two different batches under one seq."""
        if self._seq_path is None:
            return
        tmp = self._seq_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "next_batch_seq": self._next_batch_seq,
            "last_hash": self._last_published_hash.hex(),
        }), encoding="utf-8")
        os.replace(tmp, self._seq_path)

    def _sign_and_publish(self, events: List[Dict[str, Any]]) -> Optional[SignedBatch]:
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
            timestamp=float(self.clock()),
        )
        payload = batch.canonical_payload()
        signature = _sign(self.keypair.private_key, payload)
        signed = SignedBatch(batch=batch, signature=signature)
        wire = signed.to_wire()

        with self._lock:
            self._last_published_hash = batch.content_hash()
            self._note_seen_locked(batch.content_hash())
            self._epoch_batches.append(batch)
            self._stats.batches_published += 1
            self._save_seq()
        self.batch_log.add(batch.content_hash(), batch.timestamp, wire)

        try:
            self.transport.publish(self.topic, wire)
        except Exception as e:
            logger.error("Failed to publish batch: %s", e)
        return signed

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    def _on_message(self, topic: str, data: bytes) -> None:
        if topic != self.topic:
            return
        self._accept(data)

    def ingest_catchup(self, wires: List[bytes], floor: float) -> int:
        """Accept batches a peer served for catch-up (issue #41). They were
        signed in the past, so the skew check is replaced by a window: after
        ``floor`` (the previous epoch's cutoff) and no later than now plus the
        skew. Signatures, rpb and dedupe are checked as for gossip. Returns
        how many were new."""
        n = 0
        for w in wires:
            if isinstance(w, str):
                w = w.encode("utf-8")
            if self._accept(w, catchup_floor=float(floor)):
                n += 1
        return n

    def catchup_request(self, start: float) -> Dict[str, Any]:
        """The request a catch-up sends: batches signed after ``start`` that
        this daemon does not already hold."""
        with self._lock:
            have = sorted({b.content_hash().hex() for b in self._epoch_batches})
        return {"topic": self.topic, "since": float(start), "have": have}

    def _accept(self, data: bytes, catchup_floor: Optional[float] = None) -> bool:
        """Verify one wire batch and buffer it for the close; True when it
        was new. ``catchup_floor`` marks a batch served for catch-up."""
        try:
            signed = SignedBatch.from_wire(data)
        except Exception as e:
            logger.warning("malformed batch dropped: %s", e)
            return False

        # Reject batches addressed to a different rpb (shouldn't happen
        # given topic isolation, but defense in depth).
        if signed.batch.rpb_address != self.rpb_address:
            logger.warning(
                "batch with rpb=%s dropped on topic=%s",
                signed.batch.rpb_address, self.topic,
            )
            return False

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
            return False

        # A batch signed further from now than the skew could be backdated
        # into an epoch that is closing (or post-dated into a later one).
        # A catch-up batch is old by nature: it must belong to the open
        # epoch (signed after the previous cutoff) and not be post-dated.
        skew = self.max_skew_seconds
        ts, now = float(signed.batch.timestamp), float(self.clock())
        if catchup_floor is not None:
            late = ts <= catchup_floor or (skew is not None and ts - now > skew)
        else:
            late = skew is not None and abs(ts - now) > skew
        if late:
            with self._lock:
                self._stats.batches_skewed += 1
            logger.warning(
                "batch from %s dropped: signed %.0fs from now (skew limit %ss%s)",
                signed.batch.sender_pubkey.hex()[:16], ts - now, skew,
                ", catch-up" if catchup_floor is not None else "",
            )
            return False

        h = signed.batch.content_hash()
        with self._lock:
            if h in self._seen:
                self._stats.batches_duplicate += 1
                return False
            self._note_seen_locked(h)
            self._stats.batches_received += 1

            # Track per-sender seq for diagnostics. Out-of-order is
            # tolerated at this layer; canonical ordering is Phase 5.3's
            # job at network epoch close.
            prev_seq = self._known_seq_by_sender.get(signed.batch.sender_pubkey, 0)
            if signed.batch.batch_seq > prev_seq:
                self._known_seq_by_sender[signed.batch.sender_pubkey] = signed.batch.batch_seq

            # Buffer for the next epoch's canonical-order computation.
            # Both peer batches and (in publish_events) own batches
            # land here so the federated close sees the full set.
            self._epoch_batches.append(signed.batch)
        self.batch_log.add(h, ts, data)

        # Don't ingest our own published batches: the local
        # library service already has them.
        if signed.batch.sender_pubkey == self.sender_pubkey:
            return True

        # Ingest into the local library service with origin="remote": the
        # epoch buffering and discovery queue apply, but the local-events
        # fan-out doesn't echo this batch back out.
        try:
            self.library.submit_events(signed.batch.events, origin="remote")
            with self._lock:
                self._stats.events_ingested += len(signed.batch.events)
        except Exception as e:
            logger.error("ingesting peer batch failed: %s", e, exc_info=True)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _note_seen_locked(self, h: bytes) -> None:
        self._seen.append(h)
        if len(self._seen) > self._dedupe_cache_size:
            # Trim oldest. List is small (FIFO), pop(0) is fine.
            self._seen = self._seen[-self._dedupe_cache_size:]

    def drain_epoch_batches(self) -> List[EventBatch]:
        """Return + clear the per-epoch batch buffer.

        Called by the federated-close driver at epoch close to build
        the canonical order. Atomic under self._lock.
        """
        with self._lock:
            out = list(self._epoch_batches)
            self._epoch_batches.clear()
        return out

    def pending_epoch_batches(self) -> List[EventBatch]:
        """A copy of the per-epoch batch buffer, left in place."""
        with self._lock:
            return list(self._epoch_batches)

    def take_epoch_batches(
        self, selected: List[EventBatch], *,
        drop_at_or_before: Optional[float] = None,
    ) -> None:
        """Remove ``selected`` (an epoch's members) from the buffer, plus
        any batch signed at or before ``drop_at_or_before`` (the epoch's
        cutoff: such a batch can no longer join any epoch). The rest stay
        buffered for the next epoch."""
        chosen = {b.content_hash() for b in selected}
        with self._lock:
            self._epoch_batches = [
                b for b in self._epoch_batches
                if b.content_hash() not in chosen
                and not (drop_at_or_before is not None
                         and float(b.timestamp) <= drop_at_or_before)
            ]

    def restore_epoch_batches(self, batches: List[EventBatch]) -> None:
        """Put batches back in the buffer (an epoch that was not anchored
        as this daemon closed it, or a close that did not complete), once
        each."""
        with self._lock:
            have = {b.content_hash() for b in self._epoch_batches}
            for b in batches:
                if b.content_hash() not in have:
                    self._epoch_batches.append(b)
                    have.add(b.content_hash())

    def known_senders(self) -> List[bytes]:
        """All distinct sender pubkeys we've seen in this lifetime.

        Includes our own pubkey. Used for deterministic-random
        submitter selection.
        """
        with self._lock:
            senders = set(self._known_seq_by_sender.keys())
            senders.add(self.sender_pubkey)
        return sorted(senders)

    def stop(self) -> None:
        """Tear down. Caller is responsible for transport.stop() if it
        owns the transport."""
        # No persistent state to flush; placeholder for symmetry with
        # transport.stop().
        pass
