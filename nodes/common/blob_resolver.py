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
from typing import Optional, Protocol

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
