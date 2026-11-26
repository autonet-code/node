"""Canonical byte-level encoding for the authoritative epoch payload.

This is the **wire format** that gets hashed and committed to chain.
Once an anchor lands on chain, the encoding is permanent. Any change
must bump the schema version.

Spec
----

The authoritative payload is a dict with these fields, in this order:

  1. schema:           int (currently 10)
  2. epoch_id:         str
  3. epoch_root:       64-char lowercase hex string (sha256 digest)
  4. prev_epoch_root:  64-char lowercase hex string (sha256 digest)
  5. claims:           dict, earner -> {"tokens": {token: amount},
                       "value": value, "seed": seed}. The earner is the
                       household's lowercase owner wallet, each token a
                       lowercase 0x address, amounts in the token's base
                       units, the value in 6-decimal value units at the
                       pinned parity, the seed (value units) its part of
                       the REP floor. One leaf each under the anchor's
                       claimRoot, over the tokens, amounts and seed: the
                       value is informational (Substrate computes REP on
                       chain from the capped amounts at the parity it
                       stores at the anchor, and adds the seed).
  6. pools:            dict, token -> the epoch's pool of it (the window's
                       tool share plus the carry).
  7. parity:           dict, token -> the parity the close valued it at
                       (the tokens of ``pools``; 0 = out of the index).
  8. total_value:      the claims' values summed.
  9. rep_floor:        the REP floor the close read (Registry
                       ``jurisdiction.autonet.rep_floor`` / 1e12, value
                       units; 0 = none).
  10. total_seed:      the claims' seeds summed: the floor less
                       ``total_value`` when that is below it and a tool
                       concluded, less the split's dust; else 0.
  11. output_decimals: int, the precision weights are turned into
                       integers at before the split.
  12. n_batches:       int
  13. n_events:        int
  14. carry_cid:       str, sha256 cid of the carry-over bundle blob
                       (``encode_carry_bundle``): the tool maps this close
                       produced, i.e. the INPUTS of the next close. A
                       daemon that boots after anchors exist fetches and
                       verifies it instead of trusting a peer.
  15. charter_hash:    str, sha256 hex of the charter the close ran
                       against (a charter change is a fork boundary).
  16. sanctioned_households: list[str], sorted lowercase households under
                       a DAO sanction at the snapshot (their share is
                       withheld into the carry).
  17. concluded:       dict[str, int], each tool that concluded in this
                       close (digest hex) -> its malware share in bps; the
                       anchor's claimRoot holds one leaf per entry, from
                       which ToolBond opens the tool's verdict.
  18. seed:            true, ONLY on a genesis seed anchor (not the REP
                       floor's seeds, which live in ``claims``)
                       (docs/multi_jurisdiction.md decision 9): the
                       carry bundle is an adopted snapshot, not the output
                       of an observed epoch. Omitted entirely otherwise.

Integers
--------

Every amount, value and parity is written as its decimal string
(``str(int)``), so no JSON reader ever turns an 18-decimal amount into a
float. Maps have their keys sorted ascending; the outer dict has keys in
the order listed above (NOT alphabetical).

JSON encoding
-------------

``json.dumps(payload, ensure_ascii=False, separators=(",", ":"))``
produces the canonical bytes. UTF-8 encoding.

The claims blob
---------------

The blob the anchor's ``claimCid`` names (sha256 of its bytes, hex) holds
the claims and the concluded tools, everything a submitter needs to
rebuild the claimRoot tree and prove a leaf, while the on-chain
``payload_hash`` covers the whole payload.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple


# Schema 10 (the REP floor): each claim carries its ``seed``, in its leaf
# too, and the payload commits ``rep_floor`` and ``total_seed``.
# Schema 9 (the value index's tokens replace ATN): ``claims`` (per earner,
# an amount of each token's pool and the REP value), ``pools``, ``parity``
# and ``total_value`` replace agent_mint and total_mint; every money number
# is a decimal string.
# Schema 8 (Decision 2026-11-09): the payload commits ``concluded`` (each
# concluding tool's malware share in bps).
# Schema 7 (Decision 2026-11-08): the payload drops the world's fields.
# Schema 5 (Decision 2026-10-26): the payload commits ``charter_hash`` and
# ``sanctioned_households``.
# Schema 4: the payload gains ``carry_cid``.
SCHEMA_VERSION = 10
DEFAULT_DECIMALS = 10

# The fixed top-level field order. Encoding will lay these out in
# this sequence, NOT alphabetical. Anchored once, permanent.
_FIELD_ORDER = [
    "schema",
    "epoch_id",
    "epoch_root",
    "prev_epoch_root",
    "claims",
    "pools",
    "parity",
    "total_value",
    "rep_floor",
    "total_seed",
    "output_decimals",
    "n_batches",
    "n_events",
    "carry_cid",
    "charter_hash",
    "sanctioned_households",
    "concluded",
]


def _uint(value: Any) -> int:
    """A non-negative integer from an int or a decimal string; anything
    else (a float, a negative) is refused."""
    if isinstance(value, bool):
        raise TypeError(f"not an amount: {value!r}")
    if isinstance(value, int):
        v = value
    elif isinstance(value, str) and value.isdigit():
        v = int(value)
    else:
        raise TypeError(f"not an integer amount: {value!r}")
    if v < 0:
        raise ValueError(f"negative amount: {value!r}")
    return v


def _addr(value: Any) -> str:
    return str(value).lower()


def _format_amounts(d: Dict[str, Any]) -> Dict[str, str]:
    """{address: amount} with lowercase keys, sorted, decimal strings."""
    out = {_addr(k): str(_uint(v)) for k, v in (d or {}).items()}
    return {k: out[k] for k in sorted(out)}


def _format_claims(claims: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The claims in canonical form: earner (lowercase, sorted) ->
    {"tokens": {token: "amount"}, "value": "value", "seed": "seed"}."""
    out: Dict[str, Dict[str, Any]] = {}
    for earner in sorted((claims or {}), key=_addr):
        entry = claims[earner] or {}
        out[_addr(earner)] = {
            "tokens": _format_amounts(entry.get("tokens") or {}),
            "value": str(_uint(entry.get("value") or 0)),
            "seed": str(_uint(entry.get("seed") or 0)),
        }
    return {k: out[k] for k in sorted(out)}


def _format_concluded(d: Dict[str, Any]) -> Dict[str, int]:
    """Concluded tools: lowercase digest hex (no 0x) -> share bps, sorted."""
    out = {str(k).lower().removeprefix("0x"): int(v) for k, v in d.items()}
    return {k: out[k] for k in sorted(out)}


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
    canonical = {
        "schema": SCHEMA_VERSION,
        "epoch_id": str(epoch_id),
        "epoch_root": _validate_hex32("epoch_root", epoch_root_hex),
        "prev_epoch_root": _validate_hex32("prev_epoch_root", prev_epoch_root_hex),
        "claims": _format_claims(payload.get("claims") or {}),
        "pools": _format_amounts(payload.get("pools") or {}),
        "parity": _format_amounts(payload.get("parity") or {}),
        "total_value": str(_uint(payload.get("total_value") or 0)),
        "rep_floor": str(_uint(payload.get("rep_floor") or 0)),
        "total_seed": str(_uint(payload.get("total_seed") or 0)),
        "output_decimals": int(payload.get("output_decimals", DEFAULT_DECIMALS)),
        "n_batches": int(payload.get("n_batches", 0)),
        "n_events": int(payload.get("n_events", 0)),
        "carry_cid": str(payload.get("carry_cid", "")),
        "charter_hash": str(payload.get("charter_hash", "")).lower(),
        "sanctioned_households": sorted(
            {str(h).lower() for h in payload.get("sanctioned_households") or ()}),
        "concluded": _format_concluded(payload.get("concluded") or {}),
    }

    ordered = {field: canonical[field] for field in _FIELD_ORDER}
    # Genesis seed marker (decision 9): appended last and only when set.
    if payload.get("seed") is True:
        ordered["seed"] = True

    return json.dumps(
        ordered,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def encode_claims_blob(
    claims: Dict[str, Any],
    *,
    concluded: Dict[str, Any] | None = None,
) -> bytes:
    """Encode the off-chain claims blob the anchor's ``claimCid`` names.

    Format::

      {"schema": 10,
       "claims": {earner: {"tokens": {token: "amount"}, "value": "value",
                           "seed": "seed"}},
       "concluded": {digest: share bps}}   (only when any)

    ``concluded`` holds the tools whose leaves share the claimRoot tree
    with the claim leaves (Decision 2026-11-09), so a submitter can rebuild
    the tree and prove either kind.
    """
    blob: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "claims": _format_claims(claims),
    }
    if concluded:
        blob["concluded"] = _format_concluded(concluded)
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


def decode_claims_blob(blob: bytes) -> Dict[str, Dict[str, Any]]:
    """Reverse of ``encode_claims_blob`` (claims only): earner ->
    {"tokens": {token: int}, "value": int, "seed": int}. Raises ValueError
    on a blob of another schema."""
    data = json.loads(blob.decode("utf-8"))
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported claims blob schema: {data.get('schema')!r}")
    out: Dict[str, Dict[str, Any]] = {}
    for earner, entry in (data.get("claims") or {}).items():
        out[str(earner)] = {
            "tokens": {str(t): _uint(a) for t, a in (entry.get("tokens") or {}).items()},
            "value": _uint(entry.get("value") or 0),
            "seed": _uint(entry.get("seed") or 0),
        }
    return out


def decode_concluded(blob: bytes) -> Dict[str, int]:
    """The concluded tools a claims blob carries (digest -> share bps)."""
    data = json.loads(blob.decode("utf-8"))
    return _format_concluded(data.get("concluded") or {})


# ---------------------------------------------------------------------------
# Carry-over bundle (schema-4 payload ``carry_cid``)
# ---------------------------------------------------------------------------

CARRY_BUNDLE_SCHEMA = 1

# The maps a federated close produces and the next close consumes. Order is
# documentation only: the bundle serializes with sorted keys. Ratings are
# read from ToolBond; ``tool_ratings`` keeps each tool's plain means as of
# the close (Decision 2026-11-09), for followers and joiners to show.
# ``service_reviews`` keeps each payer's latest review of each service and
# ``service_ratings`` the per-service ratings (Decision 2026-11-10,
# nodes/common/service_reviews.py).
CARRY_KEYS: Tuple[str, ...] = (
    "tool_registrations",
    "tool_rounds",
    "tool_ratings",
    "service_reviews",
    "service_ratings",
)


def encode_carry_bundle(carry: Dict[str, Any], *, epoch_id: str) -> bytes:
    """Canonical bytes for the carry-over bundle a close produces.

    Format (compact JSON, UTF-8, ALL keys sorted at every level)::

      {"carry": {<the CARRY_KEYS maps>}, "epoch_id": str, "schema": 1}

    ``epoch_id`` is the epoch whose close PRODUCED these maps (they are
    the inputs of the following close).

    Why ``sort_keys`` is safe here: every consumer of these maps
    (``compute_tool_mint`` and its ``_normalize_*`` helpers) iterates
    them in sorted-key order, and the close already emits them
    key-sorted, so key order carries no meaning. Sorting makes the bytes
    a pure function of the values.

    Floats are emitted as Python's shortest round-trip ``repr`` (NOT the
    payload's fixed 10-decimal form): the joiner must load the EXACT
    values an always-on daemon holds, or its next close would drift. The
    close already rounds them (ratings to 9 places, round state to 12), so
    the repr is short and stable. NaN/Inf are rejected.
    """
    missing = [k for k in CARRY_KEYS if not isinstance(carry.get(k), dict)]
    if missing:
        raise ValueError(f"carry bundle missing maps: {', '.join(missing)}")
    blob = {
        "schema": CARRY_BUNDLE_SCHEMA,
        "epoch_id": str(epoch_id),
        "carry": {k: carry[k] for k in CARRY_KEYS},
    }
    return json.dumps(
        blob,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def decode_carry_bundle(blob: bytes) -> Dict[str, Any]:
    """Reverse of ``encode_carry_bundle``; strict.

    Returns ``{"epoch_id": str, "carry": {key: map}}``. Raises
    ``ValueError`` on a wrong schema, a missing/non-object map, or bytes
    that are not in canonical form (re-encoding must reproduce them
    exactly, so the cid commits to one unambiguous value).
    """
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"carry bundle is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("carry bundle is not a JSON object")
    if data.get("schema") != CARRY_BUNDLE_SCHEMA:
        raise ValueError(
            f"unsupported carry bundle schema: {data.get('schema')!r}")
    carry = data.get("carry")
    if not isinstance(carry, dict):
        raise ValueError("carry bundle has no carry object")
    missing = [k for k in CARRY_KEYS if not isinstance(carry.get(k), dict)]
    if missing:
        raise ValueError(f"carry bundle missing maps: {', '.join(missing)}")
    extra = sorted(set(carry) - set(CARRY_KEYS))
    if extra:
        raise ValueError(f"carry bundle has unknown maps: {', '.join(extra)}")
    epoch_id = str(data.get("epoch_id", ""))
    if encode_carry_bundle(carry, epoch_id=epoch_id) != blob:
        raise ValueError("carry bundle bytes are not canonical")
    return {"epoch_id": epoch_id, "carry": {k: carry[k] for k in CARRY_KEYS}}
