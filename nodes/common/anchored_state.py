"""The anchored carry-over: fetch and verify the state a close hands the next.

Each anchor's authoritative payload commits ``carry_cid``, the canonical
carry bundle (``authoritative_encoding.encode_carry_bundle``): the tool maps
the next close reads as inputs. A daemon that boots after anchors exist
fetches it the same way a chain indexer catches up:

    latest anchor -> payload bytes (cid == on-chain payloadHash)
                  -> carry_cid -> carry bundle blob (sha256-verified,
                    canonical form checked) -> install -> close
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .authoritative_encoding import cid_for_blob, decode_carry_bundle

logger = logging.getLogger(__name__)


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


@dataclass
class AnchoredPayload:
    """The latest anchor plus its hash-verified authoritative payload."""
    epoch_id: str
    epoch_root_hex: str
    payload_hash_hex: str
    payload: Dict[str, Any]
    # 0-based position of this anchor on chain (anchorCount - 1).
    anchor_index: int = 0


def read_latest_anchored_payload(
    contract: Any,
    get_blob: Callable[[str], Optional[bytes]],
) -> Tuple[str, Optional[AnchoredPayload]]:
    """Latest anchor → payload blob, verified against the on-chain
    ``payloadHash``.

    Returns ``(status, anchored)`` with status ``"genesis"`` (no anchors),
    ``"unavailable"`` (payload blob not retrievable) or ``"ok"``. Raises
    ``ValueError`` when the fetched payload contradicts the chain.
    """
    count = int(contract.functions.anchorCount().call())
    if count == 0:
        return "genesis", None

    anchor = contract.functions.getAnchor(count - 1).call()
    # Anchor struct order: (epochId, epochRoot, prevEpochRoot,
    # prevAnchorHash, agentMintCid, payloadHash, submitter,
    # blockNumber, timestamp, ...)
    epoch_id = str(anchor[0])
    epoch_root_hex = bytes(anchor[1]).hex()
    payload_hash_hex = bytes(anchor[5]).hex()

    payload_bytes = get_blob(payload_hash_hex)
    if payload_bytes is None:
        logger.warning(
            "authoritative payload blob %s (epoch %s) not retrievable",
            payload_hash_hex[:16], epoch_id,
        )
        return "unavailable", None
    if cid_for_blob(payload_bytes) != payload_hash_hex:
        raise ValueError(
            "payload blob hash mismatch against on-chain "
            f"payloadHash {payload_hash_hex[:16]}"
        )
    payload = json.loads(payload_bytes.decode("utf-8"))
    return "ok", AnchoredPayload(
        epoch_id=epoch_id,
        epoch_root_hex=epoch_root_hex,
        payload_hash_hex=payload_hash_hex,
        payload=payload,
        anchor_index=count - 1,
    )


@dataclass
class CarryOverFetch:
    """Outcome of ``fetch_carry_over_from_chain``.

    ``status``:
      - ``"ok"``: ``carry`` holds the five verified maps produced by the
        close of ``epoch_id`` (the inputs of the close after it).
      - ``"genesis"``: no anchors; an empty carry-over is correct.
      - ``"pre_schema"``: the latest anchor's payload has no
        ``carry_cid`` (payload schema <= 3); nothing to verify against.
      - ``"unavailable"``: a required blob is not retrievable (yet).

    ``seed`` is True when the anchor is a genesis seed (payload
    ``"seed": true`` at anchor index 0, docs/multi_jurisdiction.md
    decision 9): its bundle is an adopted snapshot, not the output of an
    observed epoch. A seed flag on any later anchor is ignored (logged).
    """
    status: str
    epoch_id: str = ""
    # The anchored epoch_root of ``epoch_id`` (canonical-batch merkle
    # root). The joiner installs only when this equals the root of the
    # epoch it observed immediately before the close it is about to run.
    epoch_root_hex: str = ""
    carry_cid: str = ""
    payload_schema: Any = None
    seed: bool = False
    anchor_index: int = -1
    carry: Dict[str, Any] = field(repr=False, default_factory=dict)


def fetch_carry_over_from_chain(
    contract: Any,
    blob_source: Any,
) -> CarryOverFetch:
    """Fetch + verify the latest anchored carry-over bundle.

    latest anchor → payload (cid == on-chain payloadHash) → carry_cid →
    bundle blob (sha256 == carry_cid, canonical form, epoch_id matches
    the anchor). Nothing here trusts the peer that served the bytes: the
    chain commits the payload hash, the payload commits the bundle cid.

    Raises ``ValueError`` on any integrity violation.
    """
    get_blob = _blob_getter(blob_source)
    status, anchored = read_latest_anchored_payload(contract, get_blob)
    if anchored is None:
        return CarryOverFetch(status=status)

    schema = anchored.payload.get("schema")
    idx = anchored.anchor_index
    seed = anchored.payload.get("seed") is True
    if seed and idx != 0:
        logger.warning(
            "anchor %d (epoch %s) carries a seed flag; only anchor 0 may "
            "seed a jurisdiction, treating it as an ordinary anchor",
            idx, anchored.epoch_id,
        )
        seed = False
    carry_cid = str(anchored.payload.get("carry_cid", "") or "")
    if not carry_cid:
        return CarryOverFetch(
            status="pre_schema", epoch_id=anchored.epoch_id,
            epoch_root_hex=anchored.epoch_root_hex, payload_schema=schema,
            seed=seed, anchor_index=idx,
        )

    blob = get_blob(carry_cid)
    if blob is None:
        logger.info(
            "carry-over bundle %s (epoch %s) not retrievable yet",
            carry_cid[:16], anchored.epoch_id,
        )
        return CarryOverFetch(
            status="unavailable", epoch_id=anchored.epoch_id,
            epoch_root_hex=anchored.epoch_root_hex,
            carry_cid=carry_cid, payload_schema=schema,
            seed=seed, anchor_index=idx,
        )
    if cid_for_blob(blob) != carry_cid:
        raise ValueError(
            f"carry bundle hash mismatch for cid {carry_cid[:16]} "
            f"(epoch {anchored.epoch_id})"
        )
    bundle = decode_carry_bundle(blob)
    if bundle["epoch_id"] != anchored.epoch_id:
        raise ValueError(
            f"carry bundle epoch {bundle['epoch_id']!r} contradicts the "
            f"anchor for {anchored.epoch_id!r}"
        )
    return CarryOverFetch(
        status="ok",
        epoch_id=anchored.epoch_id,
        epoch_root_hex=anchored.epoch_root_hex,
        carry_cid=carry_cid,
        payload_schema=schema,
        seed=seed,
        anchor_index=idx,
        carry=bundle["carry"],
    )
