"""Service reviews in the epoch-close (Decision 2026-11-10,
docs/services_market.md).

A service review is a buyer's rating of a service it paid for. It has the
fields of a tool review (correctness, simplicity, novelty, malware, the
optional charter axes) and rides the epoch batches as one
``service_review`` event. Authority comes from payment, not REP: only the
address that paid for a request may review it, and every counted review
weighs the same. The close keeps each payer's latest review of each service
and commits the per-service ratings in the anchored carry bundle. Reviews
enforce nothing: no mint, no REP, no slashing.

**Receipt.** The provider signs every reply (``receipt_sig``) over
``receipt_hash``: keccak of the ABI encoding of (domain, request id, spec
digest, provider address, the payer the gate verified, ok, charged, the
amount charged, sha256 of the canonical ``{"error", "result"}`` JSON). The
result itself is not in it, so a review proves the receipt without
revealing what the provider returned.

**Event.**::

  {"kind": "service_review", "v": 1,
   "receipt": {"request_id", "spec_digest", "provider", "payer", "ok",
               "charged", "amount", "result_hash", "sig"},
   "review": {"jurisdiction", "correctness", "simplicity", "novelty",
              "malware", "axes", "evidence_hash", "note", "ts"},
   "reviewer": "0x...", "sig": "0x..."}

Scores are in hundredths, as on ToolBond's wire: correctness and charter
axes -100..100, simplicity and novelty 0..100, malware true or false. The
reviewer signs ``review_hash`` (keccak of the domain, the receipt hash and
the sha256 of the review's canonical JSON).

**Validity** (``validate_event``, a pure function, no chain reads): the
receipt is well formed, says ``charged: true`` and is signed by its
provider; the reviewer is the receipt's payer, is not the provider, and
signed the review; the review is for this jurisdiction, has all four
fields in range, evidence with a malware yes, and is not dated more than
``MAX_FUTURE_S`` after its batch.

**Counting** (``close_service_reviews``): one review per (provider, spec
digest, payer); a later review (larger signed ``ts``, then larger review
hash) replaces the earlier, so a replayed old review changes nothing. A
service's rating is the mean of each field over its payers' latest reviews,
the malware share as the share of payers flagging it, the number of
reviewing payers and the total the receipts behind them charged. Not
counted (kept, so a change of the pinned inputs counts them again): payers
of the provider's own household and sanctioned households. Ratings are
keyed by (provider address, spec digest), so a copy registered under
another address never shares a rating. Every sum is over integers and
every iteration over sorted keys, so every daemon computes the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

KIND = "service_review"
EVENT_VERSION = 1
RECEIPT_DOMAIN = "autonet:service_receipt:v1"
REVIEW_DOMAIN = "autonet:service_review:v1"
ZERO_ADDRESS = "0x" + "00" * 20

# A review may be dated at most this long after the batch carrying it.
MAX_FUTURE_S = 3600
MAX_NOTE_CHARS = 500
MAX_AXES = 32
MAX_AXIS_ID = 64
# Bounds on the carried state, enforced at every close (keep the most
# recent): the payers kept per service, the services kept per provider.
MAX_PAYERS_PER_SERVICE = 1000
MAX_SERVICES_PER_PROVIDER = 256
# An unscored charter axis counts as half the top score (the framework's
# default for tools, ``review_fields.UNSCORED_AXIS_DEFAULT``), in
# hundredths. The close reads no Registry value for it.
UNSCORED_AXIS_HUNDREDTHS = 50

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEXID = re.compile(r"^[0-9a-f]{1,64}$")
_ADDR = re.compile(r"^0x[0-9a-f]{40}$")
_SIG = re.compile(r"^0x[0-9a-f]{130}$")
_AMOUNT = re.compile(r"^(0|[1-9][0-9]{0,77})$")
_RECEIPT_KEYS = frozenset({"request_id", "spec_digest", "provider", "payer", "ok",
                           "charged", "amount", "result_hash", "sig"})
_REVIEW_KEYS = frozenset({"jurisdiction", "correctness", "simplicity", "novelty",
                          "malware", "axes", "evidence_hash", "note", "ts"})
_RANGES = {"correctness": (-100, 100), "simplicity": (0, 100), "novelty": (0, 100)}


# ---------------------------------------------------------------------------
# Canonical forms and hashes
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def norm_hex(value: Any) -> str:
    """Lowercase hex without ``0x``."""
    return str(value or "").strip().lower().removeprefix("0x")


def norm_address(value: Any) -> str:
    """Lowercase ``0x`` address, or "" when empty."""
    v = str(value or "").strip().lower()
    if not v:
        return ""
    return v if v.startswith("0x") else "0x" + v


def result_hash(result: Any, error: Any) -> str:
    """sha256 (hex) of the reply's result and error as canonical JSON: what a
    receipt commits to instead of the result itself."""
    return hashlib.sha256(canonical_json({"error": error, "result": result})).hexdigest()


def receipt_hash(receipt: Dict[str, Any]) -> bytes:
    """What the provider signs for one reply (see the module docstring)."""
    from eth_abi import encode
    from eth_utils import keccak, to_checksum_address

    payer = norm_address(receipt.get("payer")) or ZERO_ADDRESS
    return keccak(encode(
        ["string", "string", "string", "address", "address", "bool", "bool",
         "uint256", "bytes32"],
        [RECEIPT_DOMAIN, norm_hex(receipt.get("request_id")),
         norm_hex(receipt.get("spec_digest")),
         to_checksum_address(norm_address(receipt.get("provider")) or ZERO_ADDRESS),
         to_checksum_address(payer), bool(receipt.get("ok")),
         bool(receipt.get("charged")), int(receipt.get("amount") or 0),
         bytes.fromhex(norm_hex(receipt.get("result_hash")).rjust(64, "0"))]))


def review_hash(receipt_h: bytes, review: Dict[str, Any]) -> bytes:
    """What the reviewer signs: the review bound to its receipt."""
    from eth_abi import encode
    from eth_utils import keccak

    return keccak(encode(["string", "bytes32", "bytes32"],
                         [REVIEW_DOMAIN, bytes(receipt_h),
                          hashlib.sha256(canonical_json(review)).digest()]))


def sign_hash(private_key: str, digest: bytes) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    key = private_key if private_key.startswith("0x") else "0x" + private_key
    return "0x" + bytes(Account.sign_message(
        encode_defunct(primitive=bytes(digest)), key).signature).hex()


def recover(digest: bytes, signature: str) -> str:
    """The lowercase address that signed ``digest``, or "" when the
    signature is malformed."""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        return str(Account.recover_message(
            encode_defunct(primitive=bytes(digest)), signature=signature)).lower()
    except Exception:  # noqa: BLE001 (a bad signature is no signature)
        return ""


def service_key(provider: str, spec_digest: str) -> str:
    """The carried maps' key for one service: ``<provider>:<digest>``."""
    return f"{norm_address(provider)}:{norm_hex(spec_digest)}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _int_in(value: Any, lo: int, hi: int) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if lo <= value <= hi else None


def normalize_receipt(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """A receipt in canonical form, signature checked, or (None, why)."""
    if not isinstance(raw, dict) or set(raw) != _RECEIPT_KEYS:
        return None, "receipt fields"
    rid, digest = raw.get("request_id"), raw.get("spec_digest")
    provider, payer = raw.get("provider"), raw.get("payer")
    if not (isinstance(rid, str) and _HEXID.match(rid)):
        return None, "request_id"
    if not (isinstance(digest, str) and _HEX64.match(digest)):
        return None, "spec_digest"
    if not (isinstance(provider, str) and _ADDR.match(provider)):
        return None, "provider"
    if not (isinstance(payer, str) and _ADDR.match(payer)) or payer == ZERO_ADDRESS:
        return None, "payer"
    if not isinstance(raw.get("ok"), bool) or not isinstance(raw.get("charged"), bool):
        return None, "ok/charged"
    amount = raw.get("amount")
    if not (isinstance(amount, str) and _AMOUNT.match(amount)
            and int(amount) < 2 ** 256):
        return None, "amount"
    if not (isinstance(raw.get("result_hash"), str) and _HEX64.match(raw["result_hash"])):
        return None, "result_hash"
    sig = raw.get("sig")
    if not (isinstance(sig, str) and _SIG.match(sig)):
        return None, "receipt signature"
    receipt = dict(raw)
    if recover(receipt_hash(receipt), sig) != provider:
        return None, "receipt not signed by its provider"
    return receipt, ""


def validate_event(ev: Any, *, jurisdiction: str,
                   batch_ts: float) -> Tuple[Optional[Dict[str, Any]], str]:
    """``({"key", "payer", "record"}, "")`` for a valid ``service_review``
    event, or ``(None, why)``. Pure: no chain reads, no clock."""
    if not isinstance(ev, dict) or ev.get("kind") != KIND:
        return None, "not a service review"
    if ev.get("v") != EVENT_VERSION:
        return None, "version"
    receipt, why = normalize_receipt(ev.get("receipt"))
    if receipt is None:
        return None, why
    if not receipt["charged"]:
        return None, "the receipt says nothing was charged"
    reviewer = ev.get("reviewer")
    if reviewer != receipt["payer"]:
        return None, "the reviewer is not the payer"
    if reviewer == receipt["provider"]:
        return None, "the provider reviewing itself"
    review = ev.get("review")
    if not isinstance(review, dict) or set(review) != _REVIEW_KEYS:
        return None, "review fields"
    if str(review.get("jurisdiction")) != str(jurisdiction or "").lower():
        return None, "another jurisdiction"
    scores: Dict[str, int] = {}
    for name, (lo, hi) in _RANGES.items():
        v = _int_in(review.get(name), lo, hi)
        if v is None:
            return None, name
        scores[name] = v
    if not isinstance(review.get("malware"), bool):
        return None, "malware"
    axes = review.get("axes")
    if not isinstance(axes, dict) or len(axes) > MAX_AXES:
        return None, "axes"
    for axis_id, v in axes.items():
        if not isinstance(axis_id, str) or not 0 < len(axis_id) <= MAX_AXIS_ID:
            return None, "axis id"
        if _int_in(v, -100, 100) is None:
            return None, "axis score"
    evidence = review.get("evidence_hash")
    if not isinstance(evidence, str) or (evidence and not _HEX64.match(evidence)):
        return None, "evidence_hash"
    if review["malware"] and not evidence:
        return None, "a malware flag needs evidence"
    note = review.get("note")
    if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
        return None, "note"
    ts = review.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return None, "ts"
    if ts > float(batch_ts) + MAX_FUTURE_S:
        return None, "dated after its batch"
    sig = ev.get("sig")
    if not (isinstance(sig, str) and _SIG.match(sig)):
        return None, "review signature"
    h = review_hash(receipt_hash(receipt), review)
    if recover(h, sig) != reviewer:
        return None, "review not signed by the payer"
    record = {
        "request_id": receipt["request_id"],
        "amount": int(receipt["amount"]),
        **scores,
        "malware": bool(review["malware"]),
        "axes": {k: int(axes[k]) for k in sorted(axes)},
        "evidence_hash": evidence,
        "note": note,
        "ts": ts,
        "hash": h.hex(),
    }
    return {"key": service_key(receipt["provider"], receipt["spec_digest"]),
            "payer": reviewer, "record": record}, ""


# ---------------------------------------------------------------------------
# The close
# ---------------------------------------------------------------------------

def _order(record: Dict[str, Any]) -> Tuple[int, str]:
    return int(record.get("ts") or 0), str(record.get("hash") or "")


def _normalize_carried(carried: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """A fresh copy of the carried reviews (never the caller's dicts)."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in sorted(carried or {}):
        payers = (carried or {})[key]
        if not isinstance(payers, dict):
            continue
        kept = {str(p): json.loads(json.dumps(r)) for p, r in sorted(payers.items())
                if isinstance(r, dict)}
        if kept:
            out[str(key)] = kept
    return out


def _bound(reviews: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Apply the caps, keeping the most recent reviews and services."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_provider: Dict[str, List[Tuple[Tuple[int, str], str]]] = {}
    for key in sorted(reviews):
        payers = reviews[key]
        if len(payers) > MAX_PAYERS_PER_SERVICE:
            keep = sorted(payers, key=lambda p: (_order(payers[p]), p),
                          reverse=True)[:MAX_PAYERS_PER_SERVICE]
            payers = {p: payers[p] for p in sorted(keep)}
        out[key] = payers
        latest = max((_order(r) for r in payers.values()), default=(0, ""))
        by_provider.setdefault(key.split(":", 1)[0], []).append((latest, key))
    for provider in sorted(by_provider):
        services = by_provider[provider]
        if len(services) > MAX_SERVICES_PER_PROVIDER:
            for _, key in sorted(services, reverse=True)[MAX_SERVICES_PER_PROVIDER:]:
                out.pop(key, None)
    return {k: out[k] for k in sorted(out)}


def service_rating(key: str, payers: Dict[str, Dict[str, Any]], *,
                   axis_ids: List[str], household: Any,
                   sanctioned: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
    """One service's rating over its payers' latest reviews, or None when
    no review counts. ``household(address)`` maps an address to its
    household (the pinned owner map)."""
    provider, _, digest = key.partition(":")
    banned = set(sanctioned)
    own = household(provider)
    counted = [payers[p] for p in sorted(payers)
               if household(p) != own and household(p) not in banned]
    n = len(counted)
    if n == 0:
        return None
    sums = {name: sum(int(r[name]) for r in counted) for name in _RANGES}
    flagged = sum(1 for r in counted if r.get("malware"))
    axes: Dict[str, float] = {}
    total = 0
    for axis_id in sorted(axis_ids):
        s = sum(int((r.get("axes") or {}).get(axis_id, UNSCORED_AXIS_HUNDREDTHS))
                for r in counted)
        total += s
        axes[axis_id] = round(s / n / 100.0, 9)
    return {
        "provider": provider,
        "spec_digest": digest,
        "payers": n,
        "correctness": round(sums["correctness"] / n / 100.0, 9),
        "simplicity": round(sums["simplicity"] / n / 100.0, 9),
        "novelty": round(sums["novelty"] / n / 100.0, 9),
        "malware": round(flagged / n, 9),
        "axes": axes,
        "value": round(total / (n * len(axis_ids)) / 100.0, 9) if axis_ids else None,
        "charged": sum(int(r.get("amount") or 0) for r in counted),
        "latest_ts": max(int(r.get("ts") or 0) for r in counted),
    }


def close_service_reviews(
    events: Iterable[Tuple[str, float, Dict[str, Any]]],
    *,
    carried: Optional[Dict[str, Any]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    sanctioned_households: Iterable[str] = (),
    axis_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """The close's service reviews: ``events`` are (batch jurisdiction,
    batch timestamp, event) for every ``service_review`` event in the
    canonical batches. Returns ``{"reviews": {key: {payer: record}},
    "ratings": {key: rating}, "accepted": n, "rejected": {why: n}}``.
    The result does not depend on the order of ``events``."""
    reviews = _normalize_carried(carried)
    accepted = 0
    rejected: Dict[str, int] = {}
    for jurisdiction, batch_ts, ev in events:
        got, why = validate_event(ev, jurisdiction=jurisdiction, batch_ts=batch_ts)
        if got is None:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        payers = reviews.setdefault(got["key"], {})
        old = payers.get(got["payer"])
        if old is None or _order(got["record"]) > _order(old):
            payers[got["payer"]] = got["record"]
            accepted += 1
    reviews = _bound(reviews)

    owners = {str(k).lower(): str(v).lower() for k, v in sorted((agent_owner_map or {}).items())}

    def household(address: str) -> str:
        a = str(address).lower()
        return owners.get(a, a)

    banned = sorted({str(h).lower() for h in sanctioned_households})
    ids = sorted({str(a) for a in axis_ids})
    ratings: Dict[str, Dict[str, Any]] = {}
    for key in sorted(reviews):
        rating = service_rating(key, reviews[key], axis_ids=ids,
                                household=household, sanctioned=banned)
        if rating is not None:
            ratings[key] = rating
    return {"reviews": reviews, "ratings": ratings, "accepted": accepted,
            "rejected": dict(sorted(rejected.items()))}
