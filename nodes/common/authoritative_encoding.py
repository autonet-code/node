"""Canonical byte-level encoding for the authoritative epoch payload.

This is the **wire format** that gets hashed and committed to chain.
Once an anchor lands on chain, the encoding is permanent. Any change
must bump the schema version.

Spec
----

The authoritative payload is a dict with these fields, in this order:

  1. schema:           int (currently 2)
  2. epoch_id:         str
  3. epoch_root:       64-char lowercase hex string (sha256 digest)
  4. prev_epoch_root:  64-char lowercase hex string (sha256 digest)
  5. agent_mint:       dict[str, str] — agent_id -> fixed-format float string
  6. agent_novelty:    dict[str, str] — agent_id -> fixed-format float string
  7. total_mint:       fixed-format float string
  8. total_novelty:    fixed-format float string
  9. output_decimals:  int
  10. gate_applied:    bool
  11. n_batches:       int
  12. n_events:        int
  13. world_cid:       str — sha256 cid of the canonical-world checkpoint
                       blob for this epoch (state sync), or "" when no
                       canonical tracker is wired. Added in schema 2;
                       schema-1 anchors on chain predate it.

Float formatting
----------------

Floats are serialized as ``f"{x:.10f}"`` (10 decimal places, fixed
notation, no trailing/leading whitespace). Negative zero is normalized
to positive zero. This avoids ALL Python-version dependence in JSON
float formatting.

Dict ordering
-------------

agent_mint and agent_novelty have keys sorted ascending. The outer
dict has keys in the order listed above (NOT alphabetical) — this
is the order we serialize, and it matches the documented field
order in this spec.

JSON encoding
-------------

``json.dumps(payload, ensure_ascii=False, separators=(",", ":"))``
produces the canonical bytes. UTF-8 encoding.

agent_mint_cid
--------------

The CID for the off-chain agent_mint blob is just sha256 of the blob
bytes, hex-encoded. The blob is itself a canonical-encoded JSON of
the ``agent_mint`` dict alone (schema, agent_id->str, sorted keys),
NOT the full authoritative_payload.

The blob lets agents look up only what they need (their own mint
amount) without parsing the full payload, while the on-chain
``payload_hash`` covers everything for integrity.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Tuple


# Schema 3 (fees-only, Decision 2026-07-10): MONEY ONLY. The authoritative
# payload was always money-only; the off-chain mint blob drops its optional
# ``agent_rep`` key (REP is claimed DAO-side now, never anchored). Schema 2
# was the v4.1 era (3-field merkle leaf + agent_rep blob key); bumped so the
# blob shape change is explicit, not silently reusing a different-shape number.
SCHEMA_VERSION = 3
DEFAULT_DECIMALS = 10

# The fixed top-level field order. Encoding will lay these out in
# this sequence, NOT alphabetical. Anchored once → permanent.
_FIELD_ORDER = [
    "schema",
    "epoch_id",
    "epoch_root",
    "prev_epoch_root",
    "agent_mint",
    "agent_novelty",
    "total_mint",
    "total_novelty",
    "output_decimals",
    "gate_applied",
    "n_batches",
    "n_events",
    "world_cid",
]


def _format_float(x: float, decimals: int = DEFAULT_DECIMALS) -> str:
    """Fixed-decimal string form. Normalizes -0.0 -> 0.0."""
    v = float(x)
    if v == 0.0:
        v = 0.0  # collapse negative zero
    if not math.isfinite(v):
        # Reject non-finite values explicitly. Anchoring NaN/Inf
        # would be a footgun.
        raise ValueError(f"non-finite float in authoritative payload: {x!r}")
    return f"{v:.{decimals}f}"


def _format_agent_dict(d: Dict[str, float], decimals: int) -> Dict[str, str]:
    """Sort agent_id keys ascending, format values as fixed-decimal."""
    return {
        k: _format_float(d[k], decimals)
        for k in sorted(d.keys())
    }


def _validate_hex32(name: str, value: str) -> str:
    """Hex string must be 64 lowercase chars (sha256 digest as hex)."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hex string, got {type(value).__name__}")
    v = value.lower()
    if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise ValueError(f"{name} must be 64 lowercase hex chars, got {value!r}")
    return v


def encode_authoritative_payload(
    payload: Dict[str, Any],
    *,
    epoch_id: str,
    epoch_root_hex: str,
    prev_epoch_root_hex: str,
) -> bytes:
    """Encode an authoritative payload to canonical bytes.

    The input ``payload`` is the dict produced by
    ``federated_epoch_close``'s ``authoritative_payload`` field plus
    a few caller-supplied identifiers (``epoch_id``,
    ``prev_epoch_root``).
    """
    decimals = int(payload.get("output_decimals", DEFAULT_DECIMALS))

    canonical = {
        "schema": SCHEMA_VERSION,
        "epoch_id": str(epoch_id),
        "epoch_root": _validate_hex32("epoch_root", epoch_root_hex),
        "prev_epoch_root": _validate_hex32("prev_epoch_root", prev_epoch_root_hex),
        "agent_mint": _format_agent_dict(
            payload.get("agent_mint", {}), decimals,
        ),
        "agent_novelty": _format_agent_dict(
            payload.get("agent_novelty", {}), decimals,
        ),
        "total_mint": _format_float(payload.get("total_mint", 0.0), decimals),
        "total_novelty": _format_float(payload.get("total_novelty", 0.0), decimals),
        "output_decimals": decimals,
        "gate_applied": bool(payload.get("gate_applied", False)),
        "n_batches": int(payload.get("n_batches", 0)),
        "n_events": int(payload.get("n_events", 0)),
        "world_cid": str(payload.get("world_cid", "")),
    }

    # Canonical order: NOT alphabetical, but the explicit _FIELD_ORDER.
    # We rebuild a fresh dict in that order so dict-iteration order
    # matches the spec.
    ordered = {field: canonical[field] for field in _FIELD_ORDER}

    # Compact JSON, no whitespace, UTF-8.
    return json.dumps(
        ordered,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def encode_agent_mint_blob(
    agent_mint: Dict[str, float],
    *,
    decimals: int = DEFAULT_DECIMALS,
) -> bytes:
    """Encode the off-chain ``agent_mint`` blob (money only).

    Format:
      {
        "schema": 3,
        "agent_mint": {agent_id: float-as-str (sorted keys)}
      }

    Decision 2026-07-10 (money only): the v4.1 decoupled ``agent_rep``
    key is gone — REP is claimed DAO-side (RepToken) on ratified ATN
    earnings, never anchored on the close path. The submitter builds its
    merkle proof from the ``agent_mint`` map alone (2-field leaf).
    """
    blob = {
        "schema": SCHEMA_VERSION,
        "agent_mint": _format_agent_dict(agent_mint, decimals),
    }
    return json.dumps(
        blob,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def cid_for_blob(blob: bytes) -> str:
    """Content id for an off-chain blob: sha256 hex.

    This matches the ``*Cid`` field convention described in the
    project's CLAUDE.md — sha256 hex, NOT IPFS. Daemons serve blobs
    P2P over the existing libp2p blob protocol.
    """
    return hashlib.sha256(blob).hexdigest()


def payload_hash(payload_bytes: bytes) -> bytes:
    """sha256 digest (raw bytes, 32B) of the encoded authoritative
    payload. Goes on-chain as ``payload_hash``."""
    return hashlib.sha256(payload_bytes).digest()


def decode_agent_mint_blob(blob: bytes) -> Dict[str, float]:
    """Reverse of ``encode_agent_mint_blob`` (mint map only). Returns
    floats.

    Used by agents to read their mint amount after the chain anchor
    points them at a CID.
    """
    data = json.loads(blob.decode("utf-8"))
    out: Dict[str, float] = {}
    for k, v in data.get("agent_mint", {}).items():
        out[str(k)] = float(v)
    return out
