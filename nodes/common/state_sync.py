"""Network state sync: canonical world checkpoints + anchored catch-up.

The problem this solves: a rejoining (or brand-new) daemon should not
have to replay the network's entire event history to participate — it
should catch up from the latest synchronization point the way a chain
indexer does.

The synchronization point is the epoch anchor (Substrate.sol). Each
federated close already produces a bit-identical authoritative result
on every honest daemon. This module adds a bit-identical **cumulative
canonical world** alongside it:

    canon_world_0 = fresh charter
    canon_world_N = replay(canonical_events_N onto canon_world_{N-1})

Every daemon tracks it (``CanonicalWorldTracker``), snapshots it at
each close (exact snapshot via ``world_model.generalized.serialize``),
and embeds the checkpoint blob's cid into the authoritative payload as
``world_cid`` (encoding schema 2). Because the tracker replay is
deterministic, every daemon computes the SAME blob bytes → same cid →
payload stays consensus-identical.

Catch-up for a rejoiner is then pure chain + blob-store reads:

    latest anchor → payload bytes (cid == on-chain payloadHash)
                  → world_cid → checkpoint blob (sha256-verified)
                  → restore_world → live

Mint semantics are untouched: ``federated_epoch_close`` still scores
each epoch on a fresh charter world. The cumulative canonical world is
*additional* derived state for sync, retrieval, and cross-epoch
continuity — not a change to how rewards are computed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from world_model.generalized import (
    World,
    equilibrate,
    restore_world,
    snapshot_world,
)

from .authoritative_encoding import cid_for_blob
from .world_model_substrate.adapter import build_charter_world
from .world_model_substrate.aggregate import apply_events


logger = logging.getLogger(__name__)

WORLD_CHECKPOINT_SCHEMA = 1


# ---------------------------------------------------------------------------
# Checkpoint blob encoding
# ---------------------------------------------------------------------------


def encode_world_checkpoint(
    world: World,
    *,
    epoch_id: str,
    epoch_root_hex: str,
    prev_world_cid: str = "",
) -> bytes:
    """Canonical bytes for a world checkpoint blob.

    Compact JSON, NO key sorting: the snapshot's dict insertion order
    is semantic (tendency iteration order, claim-tree order, …) and is
    itself deterministic given a bit-identical world, so plain dumps
    is already canonical across daemons. Re-serializing with
    ``sort_keys=True`` would corrupt restore-order fidelity.
    """
    payload = {
        "schema": WORLD_CHECKPOINT_SCHEMA,
        "epoch_id": str(epoch_id),
        "epoch_root": str(epoch_root_hex),
        "prev_world_cid": str(prev_world_cid),
        "world": snapshot_world(world),
    }
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")


def decode_world_checkpoint(blob: bytes) -> Dict[str, Any]:
    payload = json.loads(blob.decode("utf-8"))
    schema = payload.get("schema")
    if schema != WORLD_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported world checkpoint schema: {schema!r}")
    return payload


# ---------------------------------------------------------------------------
# Publisher side: cumulative canonical world
# ---------------------------------------------------------------------------


@dataclass
class CanonicalCheckpoint:
    epoch_id: str
    cid: str
    blob: bytes
    prev_world_cid: str


class CanonicalWorldTracker:
    """Maintains the cumulative canonical world across epoch closes.

    Deterministic by construction: the world starts as a fresh charter
    (or a restored canonical checkpoint when rejoining) and only ever
    advances by replaying canonical-ordered batches with the same
    equilibration parameters ``federated_epoch_close`` uses. Two
    daemons that agree on the canonical sequences agree on every
    checkpoint blob byte.
    """

    def __init__(
        self,
        *,
        bandwidth: float = 1.5,
        embedding_dim: int = 1024,
        equilibrate_rounds: int = 8,
        equilibrate_tolerance: float = 1e-3,
        world: Optional[World] = None,
        prev_world_cid: str = "",
    ):
        self.bandwidth = bandwidth
        self.embedding_dim = embedding_dim
        self.equilibrate_rounds = equilibrate_rounds
        self.equilibrate_tolerance = equilibrate_tolerance
        self.world = world if world is not None else build_charter_world(
            bandwidth=bandwidth, embedding_dim=embedding_dim,
        )
        self.prev_world_cid = prev_world_cid

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        cid: str,
        **kwargs: Any,
    ) -> "CanonicalWorldTracker":
        """Resume the tracker from a decoded checkpoint (rejoin path)."""
        return cls(
            world=restore_world(checkpoint["world"]),
            prev_world_cid=cid,
            **kwargs,
        )

    def on_close(
        self,
        epoch_id: str,
        epoch_root_hex: str,
        batch_event_lists: List[List[Dict[str, Any]]],
    ) -> CanonicalCheckpoint:
        """Advance the canonical world by one closed epoch and encode
        its checkpoint. Mirrors federated_epoch_close's replay shape
        (per-batch apply + equilibrate) so the cumulative world evolves
        on the same kernel path.
        """
        for events in batch_event_lists:
            if events:
                apply_events(self.world, events)
            equilibrate(
                self.world,
                max_rounds=self.equilibrate_rounds,
                tolerance=self.equilibrate_tolerance,
            )
        blob = encode_world_checkpoint(
            self.world,
            epoch_id=epoch_id,
            epoch_root_hex=epoch_root_hex,
            prev_world_cid=self.prev_world_cid,
        )
        cid = cid_for_blob(blob)
        checkpoint = CanonicalCheckpoint(
            epoch_id=epoch_id,
            cid=cid,
            blob=blob,
            prev_world_cid=self.prev_world_cid,
        )
        self.prev_world_cid = cid
        logger.info(
            "canonical world checkpoint: epoch=%s cid=%s (%d bytes)",
            epoch_id, cid[:16], len(blob),
        )
        return checkpoint


# ---------------------------------------------------------------------------
# Catch-up side
# ---------------------------------------------------------------------------


@dataclass
class CatchUpResult:
    world: World
    epoch_id: str
    epoch_root_hex: str
    world_cid: str
    prev_world_cid: str
    checkpoint: Dict[str, Any] = field(repr=False, default_factory=dict)


def _blob_getter(source: Any) -> Callable[[str], Optional[bytes]]:
    """Normalize a blob source: BlobResolver (.get), BlobStore
    (.get_bytes), or a plain callable."""
    if callable(source):
        return source
    if hasattr(source, "get_bytes"):
        return source.get_bytes
    if hasattr(source, "get"):
        return source.get
    raise TypeError(f"unusable blob source: {type(source).__name__}")


def catch_up_from_chain(
    contract: Any,
    blob_source: Any,
) -> Optional[CatchUpResult]:
    """Restore the latest anchored canonical world from chain + blobs.

    ``contract`` is a web3 contract instance for Substrate.sol (or any
    object exposing the same ``functions.anchorCount/getAnchor`` call
    interface). ``blob_source`` is a BlobResolver, BlobStore, or
    ``cid -> bytes`` callable.

    Returns None when the chain has no anchors, or when the anchored
    payload predates world checkpoints (schema 1, no world_cid), or
    when required blobs aren't retrievable. Raises on integrity
    violations (a fetched blob contradicting the chain).
    """
    get_blob = _blob_getter(blob_source)

    count = int(contract.functions.anchorCount().call())
    if count == 0:
        logger.info("catch-up: chain has no anchors yet")
        return None

    anchor = contract.functions.getAnchor(count - 1).call()
    # Anchor struct order: (epochId, epochRoot, prevEpochRoot,
    # prevAnchorHash, agentMintCid, payloadHash, submitter,
    # blockNumber, timestamp)
    epoch_id = str(anchor[0])
    epoch_root_hex = bytes(anchor[1]).hex()
    payload_hash_hex = bytes(anchor[5]).hex()

    payload_bytes = get_blob(payload_hash_hex)
    if payload_bytes is None:
        logger.warning(
            "catch-up: authoritative payload blob %s not retrievable",
            payload_hash_hex[:16],
        )
        return None
    if cid_for_blob(payload_bytes) != payload_hash_hex:
        raise ValueError(
            "catch-up: payload blob hash mismatch against on-chain "
            f"payloadHash {payload_hash_hex[:16]}"
        )

    payload = json.loads(payload_bytes.decode("utf-8"))
    world_cid = str(payload.get("world_cid", ""))
    if not world_cid:
        logger.info(
            "catch-up: anchored payload for %s carries no world_cid "
            "(pre-state-sync epoch)", epoch_id,
        )
        return None

    blob = get_blob(world_cid)
    if blob is None:
        logger.warning(
            "catch-up: world checkpoint blob %s not retrievable",
            world_cid[:16],
        )
        return None
    if cid_for_blob(blob) != world_cid:
        raise ValueError(
            f"catch-up: world blob hash mismatch for cid {world_cid[:16]}"
        )

    checkpoint = decode_world_checkpoint(blob)
    if checkpoint.get("epoch_root") != epoch_root_hex:
        raise ValueError(
            "catch-up: world checkpoint epoch_root contradicts the "
            f"anchor for {epoch_id}"
        )

    world = restore_world(checkpoint["world"])
    logger.info(
        "catch-up: restored canonical world at epoch %s (cid=%s)",
        epoch_id, world_cid[:16],
    )
    return CatchUpResult(
        world=world,
        epoch_id=epoch_id,
        epoch_root_hex=epoch_root_hex,
        world_cid=world_cid,
        prev_world_cid=str(checkpoint.get("prev_world_cid", "")),
        checkpoint=checkpoint,
    )


def install_as_local_checkpoint(persistence: Any, result: CatchUpResult) -> None:
    """Seed a daemon's local world persistence from a catch-up result.

    Writes the canonical world as the local checkpoint at event offset
    0 so the normal boot path (``WorldPersistence.try_restore``) picks
    it up. Only valid on a daemon with no local event history — mixing
    a network world with an unrelated local log would replay foreign
    events onto it.
    """
    events_path = getattr(persistence, "events_path", None)
    if events_path is not None and events_path.exists():
        if events_path.stat().st_size > 0:
            raise RuntimeError(
                "refusing to install network checkpoint: local event log "
                "is non-empty (this daemon has its own history)"
            )
    persistence.write_checkpoint(result.world, events_applied=0)
