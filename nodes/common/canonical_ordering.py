"""Canonical batch ordering for network epoch close (Phase 5.3).

When the network closes an epoch, every daemon needs to compute the
**same** global event sequence given the **same** set of batches —
even if those batches arrived via gossip in different orders, with
different latencies, on different hosts.

Algorithm
---------

1. Group batches by sender (sender_pubkey).
2. Per-sender, sort by batch_seq and validate the prev_batch_hash
   chain. A broken chain (missing seq, mismatched hash) drops the
   entire sender for this epoch — the strongest signal.
3. Compute each sender's epoch **Merkle root** over their ordered
   batches' content hashes. Standard binary Merkle: pairwise sha256,
   last leaf duplicated if odd, recurse to a single root.
4. Sort senders lexicographically by Merkle root.
5. Concatenate: each sender's full chain (in batch_seq order), in
   the sorted-by-root sender sequence.

The output is a deterministic global ordering of batches that all
honest daemons will reproduce given the same input set. The events
inside each batch retain their per-batch order. ``apply_events``
runs over the full concatenation, with ``__equilibrate__`` markers
inserted between batches so equilibration history reproduces.

Grind resistance
----------------

A malicious sender that wants to bias their position would need to
grind across **all** their batches simultaneously (because the root
covers all of them). That's combinatorially much harder than
grinding a single message hash. Honest senders who have nothing to
hide get a position determined entirely by the content of their
work — no per-sender cursor or scheduler needed.

What this module does NOT do
----------------------------

- Reject malformed batches (signature, schema). That's done at
  ingest by ``EventGossip``.
- Choose between competing chain forks for the same sender. A
  fork is a broken chain by definition; the sender is dropped.
  Phase 5+ may add explicit fork choice if needed.
- Replay or equilibrate. This module is pure ordering; the caller
  passes the sequence to ``apply_events`` etc.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .event_gossip import EventBatch


logger = logging.getLogger(__name__)


# Domain-separated empty-epoch root. Distinguishable from
# ``sha256(b"")`` (which is what an uninitialized chain slot or a
# default-zero hash hashes to) and from any real Merkle root over
# real content. Encodes both the protocol name and a version so
# future format changes don't collide with legacy anchors.
EMPTY_EPOCH_ROOT: bytes = hashlib.sha256(
    b"autonet:world-model:empty-epoch:v1"
).digest()


@dataclass
class CanonicalOrder:
    """Result of canonicalizing a set of batches for an epoch.

    ``ordered_batches`` is the global sequence to replay. Each entry
    is the original ``EventBatch`` (unchanged). ``per_sender_root``
    maps each accepted sender's pubkey to their epoch Merkle root.
    ``dropped_senders`` records senders rejected for chain
    integrity reasons, with a brief reason string.
    """
    ordered_batches: List[EventBatch] = field(default_factory=list)
    per_sender_root: Dict[bytes, bytes] = field(default_factory=dict)
    dropped_senders: Dict[bytes, str] = field(default_factory=dict)

    def epoch_root(self) -> bytes:
        """Single deterministic root over all included senders'
        epoch roots (sorted by sender root). Used as the input to the
        on-chain anchor in Phase 5.5.

        An epoch where no senders contributed (or all were dropped
        for chain-integrity reasons) returns ``EMPTY_EPOCH_ROOT``,
        which is distinguishable from ``sha256(b"")`` (the default
        zero hash) and from any real Merkle root over real content.
        That lets a chain reader distinguish "epoch closed empty"
        from "no anchor submitted yet".
        """
        if not self.per_sender_root:
            return EMPTY_EPOCH_ROOT
        # Sort by root bytes for determinism (same key as the
        # canonical ordering itself).
        roots = [self.per_sender_root[pk] for pk in
                 sorted(self.per_sender_root.keys(),
                        key=lambda p: self.per_sender_root[p])]
        return _merkle_root(roots)

    def event_dicts(self) -> List[Dict[str, Any]]:
        """Flatten the canonical sequence into the event dicts that
        ``apply_events`` consumes, with ``__equilibrate__`` markers
        between batches so per-batch equilibration history is
        reproducible during replay.
        """
        out: List[Dict[str, Any]] = []
        for b in self.ordered_batches:
            out.extend(b.events)
            out.append({"kind": "__equilibrate__", "max_rounds": 8, "tolerance": 1e-3})
        return out


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------


def _merkle_root(leaves: List[bytes]) -> bytes:
    """Binary Merkle root over a list of leaf hashes.

    - Empty list -> sha256(b"") (caller's choice; we never call it
      with empty in canonical_order, but the helper is defensive).
    - Single leaf -> that leaf as the root.
    - Odd number of nodes at any level -> last node duplicated.
    - Pair node -> sha256(left || right).
    """
    if not leaves:
        return hashlib.sha256(b"").digest()
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            h = hashlib.sha256(layer[i] + layer[i + 1]).digest()
            next_layer.append(h)
        layer = next_layer
    return layer[0]


# ---------------------------------------------------------------------------
# Per-sender chain validation
# ---------------------------------------------------------------------------


def _validate_sender_chain(
    batches: List[EventBatch],
) -> Tuple[Optional[List[EventBatch]], str]:
    """Sort by batch_seq and validate prev_batch_hash linkage.

    Rules:
      - Sequences must form a contiguous run starting at 1.
      - The first batch must have prev_batch_hash == b"" (empty).
      - Each subsequent batch must have prev_batch_hash equal to the
        previous batch's content_hash.

    Returns (chain, "") on success, (None, reason) on rejection.
    """
    if not batches:
        return ([], "")

    sorted_batches = sorted(batches, key=lambda b: b.batch_seq)

    # Sequence numbers must start at 1 and be contiguous.
    expected = 1
    for b in sorted_batches:
        if b.batch_seq != expected:
            return (
                None,
                f"sequence gap: expected batch_seq={expected}, got {b.batch_seq}",
            )
        expected += 1

    # Hash chain must link.
    prev_hash = b""
    for b in sorted_batches:
        if b.prev_batch_hash != prev_hash:
            return (
                None,
                f"prev_batch_hash mismatch at batch_seq={b.batch_seq}",
            )
        prev_hash = b.content_hash()

    return (sorted_batches, "")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def canonical_order(batches: Iterable[EventBatch]) -> CanonicalOrder:
    """Compute the canonical global ordering for an epoch's batches.

    Honest daemons running this on the same input set produce the
    same ``CanonicalOrder``: same ``ordered_batches``, same
    ``per_sender_root``, same ``epoch_root()``.
    """
    # Group by sender.
    by_sender: Dict[bytes, List[EventBatch]] = {}
    for b in batches:
        by_sender.setdefault(b.sender_pubkey, []).append(b)

    accepted: Dict[bytes, List[EventBatch]] = {}
    dropped: Dict[bytes, str] = {}

    for pk, sender_batches in by_sender.items():
        chain, reason = _validate_sender_chain(sender_batches)
        if chain is None:
            dropped[pk] = reason
            logger.warning(
                "canonical_order: dropping sender %s — %s",
                pk.hex()[:16], reason,
            )
            continue
        if not chain:
            # Empty chain — nothing to include but no error either.
            continue
        accepted[pk] = chain

    # Per-sender Merkle root.
    per_sender_root: Dict[bytes, bytes] = {}
    for pk, chain in accepted.items():
        leaves = [b.content_hash() for b in chain]
        per_sender_root[pk] = _merkle_root(leaves)

    # Sort senders by their root, concatenate chains.
    sorted_senders = sorted(
        accepted.keys(),
        key=lambda pk: per_sender_root[pk],
    )
    ordered: List[EventBatch] = []
    for pk in sorted_senders:
        ordered.extend(accepted[pk])

    return CanonicalOrder(
        ordered_batches=ordered,
        per_sender_root=per_sender_root,
        dropped_senders=dropped,
    )
