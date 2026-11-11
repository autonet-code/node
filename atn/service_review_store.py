"""The household's reviews of services (docs/services_market.md, Decisions
2026-11-10 and 2026-11-12).

A review (``review``) is a buyer's rating of one service: the four fixed
fields, the optional charter axes, a note, and for a malware flag the
evidence (a blob in the tool store's blob store, so peers can read it with
``fetch_evidence`` like a tool flag's evidence). It is signed with the
buyer's key and handed to the library service (``attach``), so it rides
the open epoch's batch. The close counts it once the reviewer has paid the
service's provider on chain (``nodes/common/service_reviews.py``); credits
are per provider, so any payment to that provider counts. One the carried
state lacks (its service registered after the snapshot, or its batch lost)
is sent again after each close for ``RESEND_WINDOW_S``
(``resend_missing``).

Daemon-local state: ``<data_dir>/services/reviews/reviews.jsonl``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

EVIDENCE_CHARS = 20_000
RESEND_WINDOW_S = 14 * 86400

MALWARE_EVIDENCE_REQUIRED = (
    "a malware flag needs evidence: say what the service did and how it shows "
    "in what it returned")


class ReviewStore:
    """The reviews this household sent, and the sending."""

    def __init__(self, directory: Path, *,
                 evidence_blobs: Optional[Callable[[], Any]] = None) -> None:
        self._dir = Path(directory)
        self._path = self._dir / "reviews.jsonl"
        self._lock = threading.RLock()
        # (service key, payer) -> the latest review sent, with its event.
        self._latest: dict[tuple[str, str], dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        self._evidence_blobs = evidence_blobs
        self.jurisdiction: str = ""
        self.review_sink: Optional[Callable[[dict[str, Any]], Any]] = None
        self._load()

    def attach(self, jurisdiction: str, sink: Callable[[dict[str, Any]], Any]) -> None:
        """Send reviews to ``sink`` (the library service's submit_events),
        signed for ``jurisdiction`` (its rpb address)."""
        self.jurisdiction = str(jurisdiction or "").lower()
        self.review_sink = sink

    def list(self, *, provider: str = "", limit: int = 50) -> list[dict[str, Any]]:
        p = str(provider or "").lower()
        with self._lock:
            rows = [{k: v for k, v in r.items() if k != "event"}
                    for r in reversed(self._rows) if not p or r.get("provider") == p]
        return rows[:limit]

    def review(self, private_key: str, *, provider: str, spec_digest: str,
               fields: dict[str, Any], axes: Any = None, note: str = "",
               evidence: str = "", now: Optional[float] = None) -> dict[str, Any]:
        """Sign and send a review of the service (``provider``,
        ``spec_digest``). ``fields`` holds correctness (-1..1), simplicity
        (0..1, the buyer's convenience), novelty (0..1) and malware (bool);
        ``axes`` optional charter scores (-1..1) by axis id. Returns the
        review or ``{"error"}``."""
        from eth_account import Account

        from nodes.common import service_reviews as sr
        from nodes.common.library.charter import active_charter
        from nodes.common.library.review_fields import parse_fields

        if not private_key:
            return {"error": "no key is held for the buyer, so the review cannot be signed"}
        key = private_key if private_key.startswith("0x") else "0x" + private_key
        reviewer = str(Account.from_key(key).address).lower()
        service, why = sr.normalize_service({"provider": provider, "spec_digest": spec_digest})
        if service is None:
            return {"error": f"name the service by its provider and spec digest ({why})"}
        if reviewer == service["provider"]:
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
            return {"error": "this daemon has not joined a jurisdiction's network, so "
                             "the review cannot be sent"}
        skey = sr.service_key(service["provider"], service["spec_digest"])
        with self._lock:
            last = int((self._latest.get((skey, reviewer)) or {}).get("ts") or 0)
        ts = max(int(now if now is not None else time.time()), last + 1)
        evidence_hash = ""
        if parsed["malware"]:
            evidence_hash = self._store_evidence(service, reviewer, str(evidence), ts)
            if not evidence_hash:
                return {"error": "the evidence could not be stored"}
        event = sr.build_event(
            key, provider=service["provider"], spec_digest=service["spec_digest"],
            jurisdiction=self.jurisdiction,
            correctness=int(round(parsed["correctness"] * 100)),
            simplicity=int(round(parsed["simplicity"] * 100)),
            novelty=int(round(parsed["novelty"] * 100)),
            malware=parsed["malware"], axes=scores, evidence_hash=evidence_hash,
            note=str(note or "")[:sr.MAX_NOTE_CHARS], ts=ts)
        ok, why = sr.validate_event(event, jurisdiction=self.jurisdiction, batch_ts=ts)
        if ok is None:
            return {"error": f"the review is not valid ({why})"}
        try:
            self.review_sink(event)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"the review could not be sent: {exc}"}
        kept = {"service": skey, "provider": service["provider"],
                "spec_digest": service["spec_digest"], "payer": reviewer,
                **ok["record"], "sent_at": int(time.time())}
        kept.pop("paid", None)
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**kept, "event": event}, sort_keys=True) + "\n")
            self._keep({**kept, "event": event})
        out = {"ok": True, "sent": True, **kept,
               "counted": "rides the open epoch's batch; it counts once you have paid "
                          "this provider on chain (credits cashed in, or a direct "
                          "payment) as of a close's snapshot, and replaces any earlier "
                          "review of this service by you"}
        if ignored:
            out["ignored_axes"] = sorted(ignored)
        return out

    def resend_missing(self, carried: Callable[[str, str], dict[str, Any]],
                       *, now: Optional[float] = None) -> int:
        """Send again each latest review, younger than ``RESEND_WINDOW_S``,
        that the carried state lacks (``carried(provider, digest)`` is the
        close's payer -> review map for one service). Returns how many."""
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

    def _store_evidence(self, service: dict[str, Any], reviewer: str,
                        text: str, ts: int) -> str:
        if self._evidence_blobs is None:
            return ""
        try:
            blobs = self._evidence_blobs()
        except Exception:  # noqa: BLE001
            return ""
        body = {"kind": "service_malware_evidence", "provider": service["provider"],
                "spec_digest": service["spec_digest"], "reviewer": reviewer,
                "evidence": text[:EVIDENCE_CHARS], "ts": int(ts)}
        try:
            return blobs.add_json(body) or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("service malware evidence blob failed: %s", exc)
            return ""

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("service reviews unreadable: %s", exc)
            return
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("service"):
                self._keep(row)

    def _keep(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        del self._rows[:-5000]
        key = (str(row.get("service") or ""), str(row.get("payer") or ""))
        old = self._latest.get(key)
        if old is None or (int(row.get("ts") or 0), str(row.get("hash") or "")) \
                > (int(old.get("ts") or 0), str(old.get("hash") or "")):
            self._latest[key] = row
