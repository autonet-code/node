"""Service reviews in the epoch-close (docs/services_market.md, Decisions
2026-11-10 and 2026-11-12).

A service review is a buyer's rating of a service. It has the fields of a
tool review (correctness, simplicity, novelty, malware, the optional
charter axes) and rides the epoch batches as one ``service_review`` event.
Authority comes from payment, not REP: a review counts only when its
reviewer has paid the service's provider on chain, and every counted review
weighs the same. A buyer's credits are per provider (one payment channel
for all its services), so paying a provider lets the buyer review any of
its services: the review names the service; nothing proves the reviewer
used that one rather than another of the same provider's. That still costs
a real payment to that provider. The close keeps each payer's latest
review of each service and commits the per-service ratings in the anchored
carry bundle. Reviews enforce nothing: no mint, no REP, no slashing.

**Event.**::

  {"kind": "service_review", "v": 2,
   "service": {"provider": "0x...", "spec_digest": "<64 hex>"},
   "review": {"jurisdiction", "correctness", "simplicity", "novelty",
              "malware", "axes", "evidence_hash", "note", "ts"},
   "reviewer": "0x...", "sig": "0x..."}

Scores are in hundredths, as on ToolBond's wire: correctness and charter
axes -100..100, simplicity and novelty 0..100, malware true or false. The
reviewer signs ``review_hash`` (keccak of the domain, the service key and
the sha256 of the review's canonical JSON).

**Validity** (``validate_event``, pure): the service is well formed; the
reviewer is not its provider and signed the review; the review is for this
jurisdiction, has all four fields in range, evidence with a malware yes,
and is not dated more than ``MAX_FUTURE_S`` after its batch.

**The chain, pinned to the snapshot block** (``snapshot_from_logs``, read
by ``voice_state.read_service_snapshot`` like the rest of the close's chain
inputs): the (provider, digest) pairs the ServiceRegistry registered, what
each payer has paid each provider (Substrate ``ServicePayment`` events
sent by the payer, and PaymentChannel ``ChannelPaid`` payouts from the
payer's channels), and the Registry's ``unscored_axis_bps``. With a
snapshot:

  - a review of a pair that is not registered is dropped, new or carried;
  - a review counts only once its reviewer has paid the provider (any
    amount above 0). Until then it waits (``paid: false``) and counts at
    the first close whose snapshot has the payment.

Without a snapshot (no chain: tests, local tools) every valid review counts.

**Counting** (``close_service_reviews``): one review per (provider, spec
digest, payer); a later review (larger signed ``ts``, then larger review
hash) replaces the earlier, so a replayed old review changes nothing. A
service's rating is over its payers' latest paid reviews: the mean of each
field, the malware share as the share of payers flagging it, each charter
axis's mean (an unscored axis at the pinned ``unscored_axis_bps``) and
their mean together, the number of payers, what those payers have paid the
provider in all (``paid``: for all its services, since credits are per
provider), and how many reviews wait for a payment. Not counted (kept, so a
change of the pinned inputs counts them again): payers of the provider's
own household and sanctioned households. Ratings are keyed by (provider
address, spec digest), so a copy registered under another address never
shares a rating. Every sum is over integers and every iteration over
sorted keys, so every daemon computes the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

KIND = "service_review"
EVENT_VERSION = 2
REVIEW_DOMAIN = "autonet:service_review:v2"
ZERO_ADDRESS = "0x" + "00" * 20

# A review may be dated at most this long after the batch carrying it.
MAX_FUTURE_S = 3600
MAX_NOTE_CHARS = 500
MAX_AXES = 32
MAX_AXIS_ID = 64
# Bounds on the carried state, enforced at every close: the reviews kept per
# service (paid ones first, then the most recent), of which at most
# MAX_PENDING_PER_SERVICE may wait for their payment, and the services kept
# per provider.
MAX_PAYERS_PER_SERVICE = 1000
MAX_PENDING_PER_SERVICE = 100
MAX_SERVICES_PER_PROVIDER = 256
# What an unscored charter axis counts as, in bps of the top score, when no
# Registry value is pinned (the tools' default, ToolBond DEFAULT_UNSCORED_BPS).
UNSCORED_BPS_DEFAULT = 5000
BPS = 10_000

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ADDR = re.compile(r"^0x[0-9a-f]{40}$")
_SIG = re.compile(r"^0x[0-9a-f]{130}$")
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


def review_hash(service_k: str, review: Dict[str, Any]) -> bytes:
    """What the reviewer signs: the review bound to the service it rates."""
    from eth_abi import encode
    from eth_utils import keccak

    return keccak(encode(["string", "string", "bytes32"],
                         [REVIEW_DOMAIN, str(service_k),
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


def normalize_service(raw: Any) -> Tuple[Optional[Dict[str, str]], str]:
    """A review's service in canonical form, or (None, why)."""
    if not isinstance(raw, dict) or set(raw) != {"provider", "spec_digest"}:
        return None, "service fields"
    provider, digest = norm_address(raw.get("provider")), norm_hex(raw.get("spec_digest"))
    if not _ADDR.match(provider) or provider == ZERO_ADDRESS:
        return None, "provider"
    if not _HEX64.match(digest):
        return None, "spec_digest"
    return {"provider": provider, "spec_digest": digest}, ""


def validate_event(ev: Any, *, jurisdiction: str,
                   batch_ts: float) -> Tuple[Optional[Dict[str, Any]], str]:
    """``({"key", "payer", "record"}, "")`` for a valid ``service_review``
    event, or ``(None, why)``. Pure: no chain reads, no clock."""
    if not isinstance(ev, dict) or ev.get("kind") != KIND:
        return None, "not a service review"
    if ev.get("v") != EVENT_VERSION:
        return None, "version"
    service, why = normalize_service(ev.get("service"))
    if service is None:
        return None, why
    reviewer = ev.get("reviewer")
    if not (isinstance(reviewer, str) and _ADDR.match(reviewer)):
        return None, "reviewer"
    if reviewer == service["provider"]:
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
    key = service_key(service["provider"], service["spec_digest"])
    h = review_hash(key, review)
    if recover(h, sig) != reviewer:
        return None, "review not signed by the reviewer"
    record = {
        **scores,
        "malware": bool(review["malware"]),
        "axes": {k: int(axes[k]) for k in sorted(axes)},
        "evidence_hash": evidence,
        "note": note,
        "ts": ts,
        "hash": h.hex(),
        "paid": False,
    }
    return {"key": key, "payer": reviewer, "record": record}, ""


def build_event(private_key: str, *, provider: str, spec_digest: str,
                jurisdiction: str, correctness: int, simplicity: int,
                novelty: int, malware: bool, axes: Optional[Dict[str, int]] = None,
                evidence_hash: str = "", note: str = "", ts: int) -> Dict[str, Any]:
    """A signed ``service_review`` event (scores in hundredths). The caller
    checks the inputs; ``validate_event`` is the judge."""
    from eth_account import Account

    key = private_key if private_key.startswith("0x") else "0x" + private_key
    review = {
        "jurisdiction": str(jurisdiction or "").lower(),
        "correctness": int(correctness),
        "simplicity": int(simplicity),
        "novelty": int(novelty),
        "malware": bool(malware),
        "axes": {str(k): int(v) for k, v in sorted((axes or {}).items())},
        "evidence_hash": norm_hex(evidence_hash),
        "note": str(note or ""),
        "ts": int(ts),
    }
    service = {"provider": norm_address(provider), "spec_digest": norm_hex(spec_digest)}
    return {
        "kind": KIND,
        "v": EVENT_VERSION,
        "service": service,
        "review": review,
        "reviewer": str(Account.from_key(key).address).lower(),
        "sig": sign_hash(key, review_hash(
            service_key(service["provider"], service["spec_digest"]), review)),
    }


# ---------------------------------------------------------------------------
# The chain snapshot
# ---------------------------------------------------------------------------

def snapshot_from_logs(
    *,
    registered: Iterable[Dict[str, Any]] = (),
    payments: Iterable[Dict[str, Any]] = (),
    payouts: Iterable[Dict[str, Any]] = (),
    unscored_bps: Any = None,
    channel_address: str = "",
) -> Dict[str, Any]:
    """The close's service snapshot from log args, all up to the snapshot
    block: ``ServiceRegistered`` (provider, specDigest), Substrate
    ``ServicePayment`` (payer, recipient, amount) and PaymentChannel
    ``ChannelPaid`` (client, provider, amount). Returns::

      {"registered": ["<provider>:<digest>", ...],
       "paid": {"<payer>:<provider>": amount},
       "unscored_bps": int}

    Payments count only to registered providers. A channel payout counts
    for the channel's client; the channel contract's own ``ServicePayment``
    (it pays providers through Substrate) is not counted twice."""
    pairs = sorted({service_key(a.get("provider"), _hex32(a.get("specDigest")))
                    for a in registered})
    providers = {p.split(":", 1)[0] for p in pairs}
    channel = norm_address(channel_address)
    paid: Dict[str, int] = {}

    def add(payer: str, provider: str, amount: Any) -> None:
        if provider not in providers or not payer:
            return
        who = f"{payer}:{provider}"
        paid[who] = paid.get(who, 0) + max(0, int(amount or 0))
    for a in payments:
        payer = norm_address(a.get("payer"))
        if channel and payer == channel:
            continue
        add(payer, norm_address(a.get("recipient")), a.get("amount"))
    for a in payouts:
        add(norm_address(a.get("client")), norm_address(a.get("provider")), a.get("amount"))
    return normalize_snapshot({"registered": pairs, "paid": paid,
                               "unscored_bps": unscored_bps})


def _hex32(value: Any) -> str:
    raw = value.hex() if isinstance(value, (bytes, bytearray)) else str(value or "")
    return norm_hex(raw).rjust(64, "0")


def normalize_snapshot(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Canonical snapshot (sorted, integers), or None for none."""
    if raw is None:
        return None
    try:
        bps = int(str(raw.get("unscored_bps")).strip())
    except (TypeError, ValueError):
        bps = UNSCORED_BPS_DEFAULT
    paid = {str(k): max(0, int(v)) for k, v in sorted((raw.get("paid") or {}).items())
            if isinstance(v, int) or str(v).isdigit()}
    return {"registered": sorted({str(p) for p in raw.get("registered") or ()}),
            "paid": paid, "unscored_bps": max(0, min(BPS, bps))}


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


def _mark_paid(reviews: Dict[str, Dict[str, Dict[str, Any]]],
               snapshot: Optional[Dict[str, Any]]) -> None:
    """Set each review's ``paid``: its reviewer has paid the provider as of
    the snapshot (every review without one)."""
    for key in sorted(reviews):
        provider = key.split(":", 1)[0]
        for payer in sorted(reviews[key]):
            r = reviews[key][payer]
            r["paid"] = (snapshot is None
                         or int(snapshot["paid"].get(f"{payer}:{provider}", 0)) > 0)


def _bound(reviews: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Apply the caps. Paid reviews are kept before waiting ones, so reviews
    that cost nothing cannot push out reviews that were paid for; then the
    most recent."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_provider: Dict[str, List[Tuple[int, Tuple[int, str], str]]] = {}
    for key in sorted(reviews):
        payers = reviews[key]
        ranked = sorted(payers, key=lambda p: (bool(payers[p].get("paid")),
                                               _order(payers[p]), p), reverse=True)
        kept: List[str] = []
        waiting = 0
        for p in ranked:
            if len(kept) >= MAX_PAYERS_PER_SERVICE:
                break
            if not payers[p].get("paid"):
                if waiting >= MAX_PENDING_PER_SERVICE:
                    continue
                waiting += 1
            kept.append(p)
        if not kept:
            continue
        out[key] = {p: payers[p] for p in sorted(kept)}
        n_paid = sum(1 for p in kept if payers[p].get("paid"))
        latest = max(_order(payers[p]) for p in kept)
        by_provider.setdefault(key.split(":", 1)[0], []).append((n_paid, latest, key))
    for provider in sorted(by_provider):
        services = by_provider[provider]
        if len(services) > MAX_SERVICES_PER_PROVIDER:
            for _, _, key in sorted(services, reverse=True)[MAX_SERVICES_PER_PROVIDER:]:
                out.pop(key, None)
    return {k: out[k] for k in sorted(out)}


def service_rating(key: str, payers: Dict[str, Dict[str, Any]], *,
                   axis_ids: List[str], household: Any,
                   sanctioned: Iterable[str] = (),
                   unscored_bps: int = UNSCORED_BPS_DEFAULT,
                   paid: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
    """One service's rating over its payers' latest paid reviews, or None
    when no review counts. ``household(address)`` maps an address to its
    household (the pinned owner map)."""
    provider, _, digest = key.partition(":")
    banned = set(sanctioned)
    own = household(provider)
    eligible_payers = [p for p in sorted(payers)
                       if household(p) != own and household(p) not in banned]
    eligible = [payers[p] for p in eligible_payers]
    counted = [r for r in eligible if r.get("paid")]
    counted_payers = [p for p in eligible_payers if payers[p].get("paid")]
    n = len(counted)
    if n == 0:
        return None
    sums = {name: sum(int(r[name]) for r in counted) for name in _RANGES}
    flagged = sum(1 for r in counted if r.get("malware"))
    axes: Dict[str, float] = {}
    total = 0
    for axis_id in sorted(axis_ids):
        # bps of the top score: a score in hundredths x 100, an unscored
        # axis the pinned default.
        s = 0
        for r in counted:
            v = (r.get("axes") or {}).get(axis_id)
            s += int(unscored_bps) if v is None else int(v) * 100
        total += s
        axes[axis_id] = round(s / n / BPS, 9)
    return {
        "provider": provider,
        "spec_digest": digest,
        "payers": n,
        "pending": len(eligible) - n,
        "correctness": round(sums["correctness"] / n / 100.0, 9),
        "simplicity": round(sums["simplicity"] / n / 100.0, 9),
        "novelty": round(sums["novelty"] / n / 100.0, 9),
        "malware": round(flagged / n, 9),
        "axes": axes,
        "value": round(total / (n * len(axis_ids)) / BPS, 9) if axis_ids else None,
        "paid": sum(int((paid or {}).get(f"{p}:{provider}", 0)) for p in counted_payers),
        "latest_ts": max(int(r.get("ts") or 0) for r in counted),
    }


def close_service_reviews(
    events: Iterable[Tuple[str, float, Dict[str, Any]]],
    *,
    carried: Optional[Dict[str, Any]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    sanctioned_households: Iterable[str] = (),
    axis_ids: Iterable[str] = (),
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The close's service reviews: ``events`` are (batch jurisdiction,
    batch timestamp, event) for every ``service_review`` event in the
    canonical batches; ``snapshot`` the chain as of the snapshot block
    (``snapshot_from_logs``), None without a chain. Returns ``{"reviews":
    {key: {payer: record}}, "ratings": {key: rating}, "accepted": n,
    "rejected": {why: n}}``. The result does not depend on the order of
    ``events``."""
    snap = normalize_snapshot(snapshot)
    registered = set(snap["registered"]) if snap is not None else None
    reviews = _normalize_carried(carried)
    if registered is not None:
        reviews = {k: v for k, v in reviews.items() if k in registered}
    accepted = 0
    rejected: Dict[str, int] = {}
    for jurisdiction, batch_ts, ev in events:
        got, why = validate_event(ev, jurisdiction=jurisdiction, batch_ts=batch_ts)
        if got is None:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        if registered is not None and got["key"] not in registered:
            rejected["unregistered service"] = rejected.get("unregistered service", 0) + 1
            continue
        payers = reviews.setdefault(got["key"], {})
        old = payers.get(got["payer"])
        if old is None or _order(got["record"]) > _order(old):
            payers[got["payer"]] = got["record"]
            accepted += 1
    _mark_paid(reviews, snap)
    reviews = _bound(reviews)

    owners = {str(k).lower(): str(v).lower() for k, v in sorted((agent_owner_map or {}).items())}

    def household(address: str) -> str:
        a = str(address).lower()
        return owners.get(a, a)

    banned = sorted({str(h).lower() for h in sanctioned_households})
    ids = sorted({str(a) for a in axis_ids})
    unscored = snap["unscored_bps"] if snap is not None else UNSCORED_BPS_DEFAULT
    ratings: Dict[str, Dict[str, Any]] = {}
    for key in sorted(reviews):
        rating = service_rating(key, reviews[key], axis_ids=ids, household=household,
                                sanctioned=banned, unscored_bps=unscored,
                                paid=snap["paid"] if snap is not None else None)
        if rating is not None:
            ratings[key] = rating
    return {"reviews": reviews, "ratings": ratings, "accepted": accepted,
            "rejected": dict(sorted(rejected.items()))}
