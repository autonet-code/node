"""The buyer's service receipts and reviews (Decision 2026-11-10, service
reviews in the epoch-close; docs/services_market.md).

Every authentic reply a buyer reads (``service_client.request_service``)
leaves a receipt here: the fields the provider signed (request id, spec
digest, provider, payer, ok, charged, amount, the result's hash) and the
signature, plus the whole signed reply as a blob in a store of this
daemon's own that no peer is served from (results can be private).

A review (``review``) is the payer's rating of one paid request: the four
fixed fields, the optional charter axes, a note, and for a malware flag the
evidence, a blob holding the reviewer's text and the provider's signed
reply, kept in the tool store's blob store so peers can read it with
``fetch_evidence`` like a tool flag's evidence. The review is signed with
the payer's key and handed to the library service (``attach``), so it
rides the open epoch's batch. The close counts it once its payment is on
chain as of the close's snapshot; one the carried state lacks (its service
was registered after the snapshot, or its batch was lost) is sent again
after each close for ``RESEND_WINDOW_S`` (``resend_missing``).

Daemon-local state: ``<data_dir>/services/receipts/receipts.jsonl``,
``reviews.jsonl`` and ``blobs/``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Receipts kept in memory (the newest); the file keeps every one.
MEMORY_CAP = 20_000
EVIDENCE_CHARS = 20_000
# A review the carried state lacks is sent again after each close for this
# long (its service registered too late for the snapshot, or its batch lost).
RESEND_WINDOW_S = 14 * 86400

MALWARE_EVIDENCE_REQUIRED = (
    "a malware flag needs evidence: say what the service did and how it "
    "shows in its reply (the provider's signed reply is attached to the "
    "evidence)")


def _rid(request_id: Any) -> str:
    return str(request_id or "").strip().lower().removeprefix("0x")


class ReceiptStore:
    """Receipts of the replies this daemon's buyers read, and their reviews."""

    def __init__(self, directory: Path, *,
                 evidence_blobs: Optional[Callable[[], Any]] = None) -> None:
        self._dir = Path(directory)
        self._receipts_path = self._dir / "receipts.jsonl"
        self._reviews_path = self._dir / "reviews.jsonl"
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, Any]] = {}
        self._reviews: dict[str, dict[str, Any]] = {}
        # (service key, payer) -> the latest review sent, with its event,
        # for ``resend_missing``.
        self._latest: dict[tuple[str, str], dict[str, Any]] = {}
        self._blobs: Any = None
        self._evidence_blobs = evidence_blobs
        # Set when the daemon joins a jurisdiction (``attach``): the
        # jurisdiction key reviews are signed for, and where they go.
        self.jurisdiction: str = ""
        self.review_sink: Optional[Callable[[dict[str, Any]], Any]] = None
        self._load()

    # ------------------------------------------------------------------
    # Receipts
    # ------------------------------------------------------------------

    def attach(self, jurisdiction: str, sink: Callable[[dict[str, Any]], Any]) -> None:
        """Send reviews to ``sink`` (the library service's submit_events),
        signed for ``jurisdiction`` (its rpb address)."""
        self.jurisdiction = str(jurisdiction or "").lower()
        self.review_sink = sink

    def _private_blobs(self) -> Any:
        if self._blobs is None:
            from nodes.common.blob_store import BlobStore
            self._blobs = BlobStore(data_dir=str(self._dir / "blobs"))
        return self._blobs

    def record(self, receipt: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
        """Keep one authentic reply's receipt and the reply itself."""
        rid = _rid(receipt.get("request_id"))
        if not rid:
            return {}
        try:
            reply_blob = self._private_blobs().add_json(
                json.loads(json.dumps(reply, default=str))) or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("could not keep the reply of %s: %s", rid[:16], exc)
            reply_blob = ""
        row = {**{k: receipt[k] for k in sorted(receipt)},
               "reply_blob": reply_blob, "kept_at": int(time.time())}
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._receipts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            self._keep_locked(row)
        return dict(row)

    def _keep_locked(self, row: dict[str, Any]) -> None:
        rid = _rid(row.get("request_id"))
        self._rows.pop(rid, None)
        self._rows[rid] = row
        while len(self._rows) > MEMORY_CAP:
            self._rows.pop(next(iter(self._rows)))

    def get(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._rows.get(_rid(request_id))
            if row is None:
                return None
            out = dict(row)
            review = self._reviews.get(_rid(request_id))
        if review is not None:
            out["review"] = dict(review)
        return out

    def list(self, *, payer: str = "", provider: str = "",
             limit: int = 50) -> list[dict[str, Any]]:
        """Receipts, newest first, optionally for one payer or provider."""
        p, v = str(payer or "").lower(), str(provider or "").lower()
        with self._lock:
            rows = [dict(r) for r in reversed(list(self._rows.values()))
                    if (not p or r.get("payer") == p)
                    and (not v or r.get("provider") == v)]
            reviews = dict(self._reviews)
        for r in rows[:limit]:
            review = reviews.get(_rid(r.get("request_id")))
            if review is not None:
                r["review"] = dict(review)
        return rows[:limit]

    def reply(self, request_id: str) -> Optional[dict[str, Any]]:
        """The whole signed reply a receipt was kept from."""
        row = self.get(request_id)
        digest = str((row or {}).get("reply_blob") or "")
        if not digest:
            return None
        raw = self._private_blobs().get_bytes_local(digest)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def review(self, private_key: str, *, request_id: str = "",
               receipt: Optional[dict[str, Any]] = None,
               fields: dict[str, Any], axes: Any = None, note: str = "",
               evidence: str = "", now: Optional[float] = None) -> dict[str, Any]:
        """Sign and send the payer's review of one paid request. ``fields``
        holds correctness (-1..1), simplicity (0..1, the buyer's
        convenience), novelty (0..1) and malware (bool); ``axes`` optional
        charter scores (-1..1) by axis id. Returns the review or
        ``{"error"}``."""
        from eth_account import Account

        from nodes.common import service_reviews as sr
        from nodes.common.library.charter import active_charter
        from nodes.common.library.review_fields import parse_fields

        if not private_key:
            return {"error": "no key is held for the payer, so the review "
                             "cannot be signed"}
        raw = receipt
        if raw is None:
            row = self.get(request_id) if request_id else None
            if row is None:
                return {"error": f"no receipt for request {request_id or '(none)'} "
                                 "on this daemon: review a request this "
                                 "daemon bought (request_id from "
                                 "request_service), or pass its receipt"}
            raw = row
        if not isinstance(raw, dict):
            return {"error": "'receipt' must be an object"}
        got, why = sr.normalize_receipt({k: raw.get(k) for k in sorted(sr.RECEIPT_KEYS)})
        if got is None:
            return {"error": f"the receipt is not valid ({why})"}
        if not got["charged"]:
            return {"error": "that request was not charged: only a paid request "
                             "can be reviewed"}
        key = private_key if private_key.startswith("0x") else "0x" + private_key
        reviewer = str(Account.from_key(key).address).lower()
        if reviewer != got["payer"]:
            return {"error": f"only the payer {got['payer']} may review this "
                             f"request (this key is {reviewer})"}
        if reviewer == got["provider"]:
            return {"error": "a provider cannot review its own service"}

        parsed, why = parse_fields(fields or {})
        if parsed is None:
            return {"error": why}
        if parsed["malware"] and not str(evidence or "").strip():
            return {"error": MALWARE_EVIDENCE_REQUIRED}
        known = set(active_charter().ids)
        scores: dict[str, int] = {}
        ignored: list[str] = []
        for axis_id, value in (axes.items() if isinstance(axes, dict) else ()):
            try:
                v = float(value) if not isinstance(value, bool) else None
            except (TypeError, ValueError):
                v = None
            if v is None or v != v or str(axis_id) not in known:
                ignored.append(str(axis_id))
                continue
            scores[str(axis_id)] = int(round(max(-1.0, min(1.0, v)) * 100))
        if not self.jurisdiction or self.review_sink is None:
            return {"error": "this daemon has not joined a jurisdiction's "
                             "network, so the review cannot be sent"}

        skey = sr.service_key(got["provider"], got["spec_digest"])
        with self._lock:
            last = max((int(r.get("ts") or 0) for r in self._reviews.values()
                        if r.get("service") == skey and r.get("payer") == reviewer),
                       default=0)
        ts = max(int(now if now is not None else time.time()), last + 1)

        evidence_hash = ""
        if parsed["malware"]:
            evidence_hash = self._store_evidence(got, reviewer, str(evidence), ts)
            if not evidence_hash:
                return {"error": "the evidence could not be stored"}
        event = sr.build_event(
            key, got, jurisdiction=self.jurisdiction,
            correctness=int(round(parsed["correctness"] * 100)),
            simplicity=int(round(parsed["simplicity"] * 100)),
            novelty=int(round(parsed["novelty"] * 100)),
            malware=parsed["malware"], axes=scores,
            evidence_hash=evidence_hash,
            note=str(note or "")[:sr.MAX_NOTE_CHARS], ts=ts)
        ok, why = sr.validate_event(event, jurisdiction=self.jurisdiction, batch_ts=ts)
        if ok is None:
            return {"error": f"the review is not valid ({why})"}
        try:
            self.review_sink(event)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"the review could not be sent: {exc}"}
        kept = {"request_id": got["request_id"], "service": skey,
                "provider": got["provider"], "spec_digest": got["spec_digest"],
                "payer": reviewer, **ok["record"], "sent_at": int(time.time())}
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._reviews_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**kept, "event": event}, sort_keys=True) + "\n")
            self._keep_review({**kept, "event": event})
        out = {"ok": True, "sent": True, **kept,
               "counted": "rides the open epoch's batch; it counts once its "
                          "payment is on chain as of a close's snapshot, and "
                          "replaces any earlier review of this service by "
                          "this payer"}
        out.pop("paid", None)
        if ignored:
            out["ignored_axes"] = sorted(ignored)
        return out

    def resend_missing(self, carried: Callable[[str, str], dict[str, Any]],
                       *, now: Optional[float] = None) -> int:
        """Send again each latest review, younger than ``RESEND_WINDOW_S``,
        that the carried state lacks: ``carried(provider, digest)`` is the
        close's payer -> review map for one service. A review the close
        dropped (its service was not registered as of the snapshot yet, or
        its batch never reached the closing daemons) gets in once it can.
        Returns how many were sent."""
        if self.review_sink is None:
            return 0
        t = float(now if now is not None else time.time())
        with self._lock:
            latest = dict(self._latest)
        sent = 0
        for (skey, payer), row in sorted(latest.items()):
            event = row.get("event")
            if not isinstance(event, dict) or t - float(row.get("sent_at") or 0) > RESEND_WINDOW_S:
                continue
            provider, _, digest = skey.partition(":")
            try:
                have = (carried(provider, digest) or {}).get(payer)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(have, dict) and (int(have.get("ts") or 0), str(have.get("hash") or "")) \
                    >= (int(row.get("ts") or 0), str(row.get("hash") or "")):
                continue
            try:
                self.review_sink(event)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("resending a service review failed: %s", exc)
        return sent

    def _store_evidence(self, receipt: dict[str, Any], reviewer: str,
                        text: str, ts: int) -> str:
        """The malware evidence as a blob peers can fetch (fetch_evidence):
        the reviewer's text, the receipt and the provider's signed reply."""
        if self._evidence_blobs is None:
            return ""
        try:
            blobs = self._evidence_blobs()
        except Exception:  # noqa: BLE001
            return ""
        body = {"kind": "service_malware_evidence",
                "provider": receipt["provider"],
                "spec_digest": receipt["spec_digest"],
                "request_id": receipt["request_id"],
                "reviewer": reviewer,
                "evidence": text[:EVIDENCE_CHARS],
                "receipt": receipt,
                "replies": [r for r in (self.reply(receipt["request_id"]),) if r],
                "ts": int(ts)}
        try:
            return blobs.add_json(body) or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("service malware evidence blob failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        for path, keep in ((self._receipts_path, self._keep_locked),
                           (self._reviews_path, self._keep_review)):
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.warning("service receipts unreadable (%s): %s", path.name, exc)
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("request_id"):
                    keep(row)

    def _keep_review(self, row: dict[str, Any]) -> None:
        public = {k: v for k, v in row.items() if k != "event"}
        self._reviews[_rid(row.get("request_id"))] = public
        key = (str(row.get("service") or ""), str(row.get("payer") or ""))
        old = self._latest.get(key)
        if old is None or (int(row.get("ts") or 0), str(row.get("hash") or "")) \
                > (int(old.get("ts") or 0), str(old.get("hash") or "")):
            self._latest[key] = row


def receipts_for(runtime: Any) -> Optional[ReceiptStore]:
    """The runtime's receipt store, or None (a runtime built without one)."""
    store = getattr(runtime, "service_receipts", None)
    return store if isinstance(store, ReceiptStore) else None
