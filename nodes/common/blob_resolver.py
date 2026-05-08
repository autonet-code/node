"""Pluggable resolver: fetch off-chain blobs by their CID.

Phase 5.6 of native world-model integration. Abstracts "where does
the blob actually live" so the chain-submission path doesn't have to
care.

Test impl: ``InMemoryBlobResolver`` — backed by a dict keyed on CID.
Production impl: a thin wrapper over ``nodes.common.blob_store``
serving via the existing libp2p ``/autonet/blob/1.0.0`` protocol.
That production wrapper isn't built here because Phase 5.6 stays on
hardhat + in-process; we'll add it when we wire to a multi-process
testbed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Protocol

from .authoritative_encoding import cid_for_blob


logger = logging.getLogger(__name__)


class BlobResolver(Protocol):
    """Fetch a blob by its CID. Returns ``None`` on miss."""

    def get(self, cid: str) -> Optional[bytes]: ...

    def put(self, blob: bytes) -> str:
        """Store a blob and return its CID. Optional — only the
        in-memory test resolver needs to be writable; production
        retrieval is read-only from the agent's perspective."""
        ...


class InMemoryBlobResolver:
    """Test/dev resolver: an in-process keyed map.

    Used by tests where the daemon and the agent live in the same
    process (or where we want to skip network setup). Thread-safe via
    a single lock; usage is low-volume in tests.
    """

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def get(self, cid: str) -> Optional[bytes]:
        with self._lock:
            return self._store.get(cid)

    def put(self, blob: bytes) -> str:
        cid = cid_for_blob(blob)
        with self._lock:
            self._store[cid] = blob
        return cid

    def has(self, cid: str) -> bool:
        with self._lock:
            return cid in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class BlobIntegrityError(RuntimeError):
    """Raised when a fetched blob's content hash doesn't match the
    expected CID — the resolver returned bytes, but they're not the
    bytes we asked for. Indicates tampering or a confused resolver."""


class LibP2PBlobResolver:
    """Cross-daemon blob resolver backed by an ``AutonetHost``.

    Writes go to a local in-memory map AND register a blob handler on
    the host so peers can fetch via ``/autonet/blob/1.0.0``.

    Reads: local map first; on miss, walk every known peer and try
    ``host.fetch_blob`` over libp2p. The host already verifies
    content-hash integrity inside ``fetch_blob``; this resolver
    additionally confirms the returned hex matches the requested
    CID so a buggy peer can't poison our cache.

    Phase 10.3d: minimal implementation. No DHT/announce — relies on
    peers being connected (via mDNS / bootstrap) and us iterating
    them. Sufficient for the two-daemon smoke test; production may
    want explicit content advertisement.
    """

    def __init__(self, host: Any, *, fetch_timeout: float = 10.0):
        if host is None:
            raise ValueError("LibP2PBlobResolver requires an AutonetHost")
        self._host = host
        self._local: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self._fetch_timeout = fetch_timeout

        # Wire ourselves as the blob handler so peers fetching from
        # this daemon hit our local store. The host invokes the
        # handler async (trio); we serve from the in-memory map.
        async def _serve(content_hash: str) -> Optional[bytes]:
            with self._lock:
                return self._local.get(content_hash)

        try:
            host.set_blob_handler(_serve)
        except Exception as e:
            logger.debug("LibP2PBlobResolver: set_blob_handler failed: %s", e)

    def put(self, blob: bytes) -> str:
        cid = cid_for_blob(blob)
        with self._lock:
            self._local[cid] = blob
        return cid

    def has(self, cid: str) -> bool:
        with self._lock:
            return cid in self._local

    def __len__(self) -> int:
        with self._lock:
            return len(self._local)

    def get(self, cid: str) -> Optional[bytes]:
        # Local first.
        with self._lock:
            blob = self._local.get(cid)
        if blob is not None:
            return blob

        # Walk known peers, try each.
        peer_ids = list(self._known_peer_ids())
        if not peer_ids:
            logger.debug("LibP2PBlobResolver: no known peers to query for %s", cid[:16])
            return None

        for pid in peer_ids:
            blob = self._fetch_from_peer(pid, cid)
            if blob is None:
                continue
            actual = cid_for_blob(blob)
            if actual != cid:
                logger.warning(
                    "LibP2PBlobResolver: peer %s returned wrong cid for %s",
                    str(pid)[:16], cid[:16],
                )
                continue
            # Cache so subsequent calls (including peers hitting us) hit local.
            with self._lock:
                self._local[cid] = blob
            return blob
        return None

    def _known_peer_ids(self) -> list:
        """Iterate distinct PeerIDs we currently know about."""
        try:
            caps = self._host.known_capabilities
        except Exception:
            return []
        out = []
        for cap in caps.values():
            pid_str = getattr(cap, "peer_id", "")
            if not pid_str:
                continue
            out.append(pid_str)
        return out

    def _fetch_from_peer(self, peer_id_str: str, cid: str) -> Optional[bytes]:
        """Cross-thread bridge: schedule host.fetch_blob via the host's
        captured trio token and block briefly waiting for the result."""
        token = getattr(self._host, "_trio_token", None)
        if token is None:
            return None
        try:
            import trio
            # Convert string peer_id to PeerID type lazily — the host
            # API accepts the PeerID instance.
            from libp2p.peer.id import ID as PeerID  # type: ignore
            try:
                pid = PeerID.from_base58(peer_id_str)
            except Exception:
                return None

            async def _do():
                return await self._host.fetch_blob(
                    pid, cid, timeout=self._fetch_timeout,
                )

            return trio.from_thread.run(_do, trio_token=token)
        except Exception as e:
            logger.debug(
                "LibP2PBlobResolver: fetch from %s failed: %s",
                peer_id_str[:16], e,
            )
            return None
