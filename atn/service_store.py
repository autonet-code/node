"""ServiceStore — daemon-side registry of published service specs.

Design: ``docs/services_market.md``. A Service is a remote API an agent
(ultimately its human benefactor) offers to the market: general-purpose,
priced per work item in ATN. The price, whether the service is active
and who is paid live on chain only (``ServiceRegistry``, Decision
2026-11-10); the spec carries none of them. It rides the same
content-addressed blob rail as tool manifests, but — unlike a tool — a
service gets NO substrate standing, NO verdict claims, NO mint. Trust is
behavioral: signed identity, atomic payment, receipt-gated reviews.

This is the provider-side store: specs are AUTHORED here, publish =
register-on-chain + blob push. This store owns spec build/sign/persist,
the on-chain service number each registration returned, and the local
request log that later reviews ride.

The backing implementation of a v1 service is just a local tool run on
the provider's daemon — "a Service is a tool the OWNER chose to sell"
(spec §3). ``ServiceRecord.backing_tool`` names that tool by digest;
dispatch goes through ``runtime.tool_store``.

The second backing kind (decision 2026-07-26) is INFERENCE: the spec
carries ``{"inference": {"provider", "model", "max_tokens_cap",
"max_input_chars"}}`` and requests are served by exactly that upstream
provider of the provider daemon's OWN stack (their GPU, their key). The two
are mutually exclusive — a service has exactly one backing.

The substrate package (``nodes.*``) is imported lazily; on a standalone
atn install registration fails loudly instead of degrading silently —
mirroring ToolStore.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .service_spec import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_TOKENS_CAP,
    build_service_spec,
    canonical_service_bytes,
    validate_inference,
    validate_service_spec,
)

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)

# Author id recorded for owner-published services (WS surface, caller "").
OWNER_AUTHOR = "user"


@dataclass
class ServiceRecord:
    """One published service: spec digest + daemon-local provider state."""
    digest: str
    spec: dict[str, Any]
    backing_tool: str = ""      # digest of the local tool that fulfils requests
    retired: bool = False
    registered_ts: int = 0
    # ServiceRegistry's number for this spec, from the registration this
    # daemon sent (None until it lands). The price, the active flag and the
    # payee are read from that entry on chain, never from the spec.
    service_id: int | None = None

    @property
    def author(self) -> str:
        return str(self.spec.get("author") or "")

    @property
    def name(self) -> str:
        return str(self.spec.get("name") or "")

    @property
    def image_uri(self) -> str:
        """Display-plane banner image (advisory; may be absent)."""
        return str(self.spec.get("image_uri") or "")

    @property
    def inference(self) -> dict[str, Any]:
        """The declared inference backing (``{provider, model,
        max_tokens_cap, max_input_chars}``), or ``{}`` when this service is
        tool-backed. Lives in the signed spec, so it survives a store reload
        for free."""
        block = self.spec.get("inference")
        return block if isinstance(block, dict) else {}

    def _bound(self, key: str, default: int) -> int:
        block = self.inference
        if not block:
            return 0
        try:
            value = int(block.get(key) or 0)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @property
    def max_tokens_cap(self) -> int:
        """Output-token ceiling for an inference-backed service (0 when
        not inference-backed)."""
        return self._bound("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)

    @property
    def max_input_chars(self) -> int:
        """Input bound of an inference-backed service, in characters
        (``service_spec.inference_input_chars``); 0 when not
        inference-backed."""
        return self._bound("max_input_chars", DEFAULT_MAX_INPUT_CHARS)

    def to_row(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "backing_tool": self.backing_tool,
            "retired": self.retired,
            "registered_ts": self.registered_ts,
            "service_id": self.service_id,
        }


class ServiceStore:
    """Persistent registry of service specs (provider side)."""

    def __init__(self, runtime: Runtime, services_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(services_dir)
        self._registry_path = self._dir / "registry.jsonl"
        self._requests_path = self._dir / "requests.jsonl"
        self._summary_path = self._dir / "requests_summary.json"
        # Payment-gate sidecars (Stage B). The seen-request-id set is the
        # direct-payment replay guard; the voucher ledger tracks the latest
        # accepted cumulative per channel (so the increment must exceed the
        # ask) plus the latest voucher signature the provider settles with.
        self._seen_path = self._dir / "served_requests.jsonl"
        self._voucher_path = self._dir / "channel_vouchers.json"
        self._records: dict[str, ServiceRecord] = {}
        # Called with the record once the chain numbers a registration: the
        # daemon's service catalog announces the spec on gossip from here.
        self.on_registered: Any = None
        self._blobs: Any = None  # lazy; None until substrate package needed
        self._request_seq = 0
        # request id -> when it was used up. Kept for REPLAY_HORIZON_S: the
        # gate refuses a direct payment older than that, so an older id can
        # never be presented again and is forgotten.
        self._seen_requests: dict[str, float] = {}
        self._seen_lines = 0
        self._channel_vouchers: dict[str, dict[str, Any]] = {}
        # Vouchers the gate accepted for requests still being served, per
        # channel (memory only): kept when the dispatch succeeds, dropped
        # when it fails, so a buyer pays only for what was served.
        self._pending_vouchers: dict[str, dict[int, dict[str, Any]]] = {}
        self._pending_seq = 0
        # async digest -> bytes | None over the libp2p blob protocol, set by
        # the autonet wiring: specs of other daemons' services (#45).
        self.blob_fetcher: Any = None
        self._load()
        self._load_payment_state()
        self._load_summary()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        author: str,
        backing_tool: str = "",
        output_schema: dict[str, Any] | None = None,
        endpoint_hint: str = "",
        image_uri: str = "",
        inference: dict[str, Any] | None = None,
        version_of: str | None = None,
    ) -> dict[str, Any]:
        """Build, sign, blob-store, and persist a service spec. Returns
        ``{"digest", "spec"}`` or raises ValueError on bad input.

        ``inference`` declares an LLM backing instead of a backing tool
        (decision 2026-07-26): a paid request is served by this daemon's
        own provider stack. Exactly one backing kind per service — a
        spec naming both is a contradiction about what the buyer gets,
        so it's refused at registration rather than resolved at dispatch.
        """
        blobs = self._blob_store()
        author_pubkey = self._author_address(author)

        if inference is not None:
            if backing_tool:
                raise ValueError(
                    "invalid service spec: a service has ONE backing — "
                    "'inference' and 'backing_tool' are mutually exclusive"
                )
            problems = validate_inference(inference)
            if not problems:
                problem = self._unknown_upstream(inference)
                if problem:
                    problems = [problem]
            if problems:
                raise ValueError("invalid service spec: " + "; ".join(problems))

        spec = build_service_spec(
            name=name,
            description=description,
            input_schema=input_schema,
            author=author,
            output_schema=output_schema,
            author_pubkey=author_pubkey,
            endpoint_hint=endpoint_hint,
            image_uri=image_uri,
            inference=inference,
            version_of=version_of,
            created_ts=int(time.time()),
        )
        self._sign(author, spec)

        digest = blobs.add_json(spec)
        record = ServiceRecord(
            digest=digest,
            spec=spec,
            backing_tool=backing_tool,
            registered_ts=int(time.time()),
        )
        self._records[digest] = record
        self._persist()
        log.info("registered service %r by %s -> %s",
                 name, author, digest[:16])
        return {"digest": digest, "spec": spec}

    def set_service_id(self, digest: str, service_id: int) -> bool:
        """Record the ServiceRegistry number a registration returned."""
        record = self._records.get(digest)
        if record is None:
            return False
        record.service_id = int(service_id)
        self._persist()
        hook = self.on_registered
        if hook is not None:
            try:
                hook(record)
            except Exception:  # noqa: BLE001 - announcing never fails a registration
                log.debug("service registration hook failed", exc_info=True)
        return True

    def retire(self, digest: str) -> bool:
        """Soft-retire a service (no longer offered; record kept for the
        request-log history reviews ride on)."""
        record = self._records.get(digest)
        if record is None:
            return False
        record.retired = True
        self._persist()
        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, digest: str) -> ServiceRecord | None:
        return self._records.get(digest)

    def resolve(self, name_or_digest: str) -> ServiceRecord | None:
        """Resolve by full digest, ``svc_<digest-prefix>``, or spec name
        (None when the name is ambiguous across live records)."""
        if name_or_digest in self._records:
            return self._records[name_or_digest]
        if name_or_digest.startswith("svc_"):
            prefix = name_or_digest[len("svc_"):]
            matches = [r for d, r in self._records.items() if d.startswith(prefix)]
            return matches[0] if len(matches) == 1 else None
        named = [r for r in self._records.values()
                 if r.name == name_or_digest and not r.retired]
        return named[0] if len(named) == 1 else None

    def list(self, *, include_retired: bool = False) -> list[ServiceRecord]:
        return [r for r in self._records.values()
                if include_retired or not r.retired]

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Request log (provider-side history; reviews ride on it later)
    # ------------------------------------------------------------------
    #
    # Only requests that passed the payment gate get a row, and a row names
    # a client only as the payer the gate verified: the request's own
    # `client` field is whatever the sender wrote. Attempts the gate refused
    # (unpaid, a copied id, a bad voucher) are counted per service, never
    # attributed, and never written row by row, so a stranger cannot grow
    # the log or tank the success rate for free. The summary is kept as the
    # rows go in (`requests_summary.json`); the row file rotates at
    # REQUEST_LOG_MAX_BYTES, keeping one previous file.

    REQUEST_LOG_MAX_BYTES = 4 * 1024 * 1024
    # Refused attempts change only a counter: persisted at most this often.
    REFUSED_PERSIST_INTERVAL = 10.0

    def record_request(
        self,
        spec_digest: str,
        request_id: str,
        payer: str,
        ok: bool,
        amount: str = "",
        token: str = "ATN",
    ) -> None:
        """Log one request the payment gate let through. Local ledger
        first — never lost to a substrate outage.

        ``payer`` is the address the gate verified as having paid, or ""
        when nothing was paid (a free service, the no-chain dev degrade);
        never the client the request names. ``token`` is a fixed "ATN"
        label (settlement is ATN-only, ratified 2026-07-10)."""
        record = self._records.get(spec_digest)
        payer = str(payer or "").strip().lower()
        self._request_seq += 1
        row = {
            "seq": self._request_seq,
            "ts": int(time.time()),
            "spec_digest": spec_digest,
            "service_name": record.name if record else "",
            "service_author": record.author if record else "",
            "request_id": request_id,
            "payer": payer,
            "ok": bool(ok),
            "amount": amount,
            "token": token,
        }
        entry = self._summary_entry(spec_digest)
        entry["service_name"] = row["service_name"] or entry["service_name"]
        entry["service_author"] = (row["service_author"]
                                   or entry["service_author"])
        entry["count"] += 1
        if ok:
            entry["ok_count"] += 1
        if payer:
            entry["clients"].add(payer)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rotate_request_log()
        with self._requests_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self._persist_summary()

    def count_refused(self, spec_digest: str) -> None:
        """Count a request refused before it was served (payment not
        verified, or refused before the gate): no row, no client."""
        if spec_digest not in self._records:
            return
        self._summary_entry(spec_digest)["refused"] += 1
        if time.monotonic() - self._summary_saved >= self.REFUSED_PERSIST_INTERVAL:
            self._persist_summary()

    def summary(self) -> dict[str, Any]:
        """Per-service volume/success of requests that passed the payment
        gate, plus how many were refused, kept as requests are logged.

        This is the provider-side track record the spec's §4 reviews
        later ride on (success rate + volume, displayed not enforced)."""
        out: dict[str, Any] = {}
        for digest, entry in sorted(self._summary.items()):
            count = entry["count"]
            ok_count = entry["ok_count"]
            out[digest] = {
                "service_name": entry["service_name"],
                "service_author": entry["service_author"],
                "count": count,
                "ok_count": ok_count,
                "success_rate": round(ok_count / count, 6) if count else 0.0,
                "unique_clients": len(entry["clients"]),
                "refused": entry["refused"],
            }
        return out

    def _summary_entry(self, digest: str) -> dict[str, Any]:
        entry = self._summary.get(digest)
        if entry is None:
            record = self._records.get(digest)
            entry = self._summary[digest] = {
                "service_name": record.name if record else "",
                "service_author": record.author if record else "",
                "count": 0, "ok_count": 0, "refused": 0, "clients": set(),
            }
        return entry

    def _load_summary(self) -> None:
        self._summary = {}
        self._summary_saved = time.monotonic()
        if not self._summary_path.exists():
            return
        try:
            data = json.loads(self._summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ServiceStore could not read the request summary: %s",
                        exc)
            return
        if not isinstance(data, dict):
            return
        try:
            self._request_seq = int(data.get("seq") or 0)
        except (TypeError, ValueError):
            self._request_seq = 0
        for digest, raw in (data.get("services") or {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                self._summary[str(digest)] = {
                    "service_name": str(raw.get("service_name") or ""),
                    "service_author": str(raw.get("service_author") or ""),
                    "count": int(raw.get("count") or 0),
                    "ok_count": int(raw.get("ok_count") or 0),
                    "refused": int(raw.get("refused") or 0),
                    "clients": {str(c) for c in raw.get("clients") or []},
                }
            except (TypeError, ValueError):
                log.warning("ServiceStore skipping a corrupt summary entry")

    def _persist_summary(self) -> None:
        data = {
            "seq": self._request_seq,
            "services": {
                digest: {**{k: v for k, v in entry.items() if k != "clients"},
                         "clients": sorted(entry["clients"])}
                for digest, entry in self._summary.items()
            },
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._summary_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._summary_path)
        self._summary_saved = time.monotonic()

    def _rotate_request_log(self) -> None:
        """Move the row file aside once it reaches REQUEST_LOG_MAX_BYTES
        (one previous file is kept; the summary carries the totals)."""
        try:
            if self._requests_path.stat().st_size < self.REQUEST_LOG_MAX_BYTES:
                return
        except OSError:
            return
        try:
            os.replace(self._requests_path,
                       self._requests_path.with_name("requests.1.jsonl"))
        except OSError as exc:
            log.warning("ServiceStore could not rotate the request log: %s", exc)

    # ------------------------------------------------------------------
    # Payment gate state (Stage B) — replay guard + voucher ledger
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_request_id(request_id: Any) -> str:
        """Canonical form of a bytes32 request id: lowercase hex, no ``0x``.

        The replay guard keyed on the RAW string until 2026-07-26, which made it
        prefix- and case-sensitive — and the two sides of the wire genuinely
        spell the same id differently: ``service_client.new_request_id()`` emits
        ``0x<64 hex>``, while the on-chain ``ServicePayment.requestId`` decodes
        to bare hex (case varying by web3 version). A replayer only had to flip
        the ``0x`` or the case to have ONE payment serve a second work item; the
        cross-machine E2E caught exactly that (two served rows, one payment).
        Normalizing here closes it for every caller at once, including the
        persisted set (see ``_load_payment_state``).
        """
        rid = str(request_id or "").strip().lower()
        return rid[2:] if rid.startswith("0x") else rid

    # How long a used-up request id is remembered. The payment gate refuses a
    # direct payment older than this (``ServicePayment`` block time), so a
    # replay of an older one is refused there and its id need not be kept.
    REPLAY_HORIZON_S = 30 * 86400

    def has_served_request(self, request_id: str) -> bool:
        """True if this request_id was already used up (replay)."""
        return self.normalize_request_id(request_id) in self._seen_requests

    def mark_request_served(self, request_id: str) -> None:
        """Record a request_id as used up: one line appended to the log
        (not a rewrite of the whole set). Idempotent."""
        rid = self.normalize_request_id(request_id)
        if not rid or rid in self._seen_requests:
            return
        now = time.time()
        self._seen_requests[rid] = now
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._seen_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": rid, "ts": now}) + "\n")
        self._seen_lines += 1
        if self._seen_lines > 2 * max(len(self._seen_requests), 1024):
            self._compact_seen(now)

    def _compact_seen(self, now: float) -> None:
        """Forget ids past the replay horizon and rewrite the log with the
        rest (rare: when the file holds twice the live ids)."""
        cutoff = now - self.REPLAY_HORIZON_S
        self._seen_requests = {r: t for r, t in self._seen_requests.items()
                               if t >= cutoff}
        self._persist_seen()

    def kept_channel_cumulative(self, channel_id: str | int) -> int:
        """The cumulative of the best voucher kept for a channel (0 if
        none): what the provider can settle with."""
        entry = self._channel_vouchers.get(str(channel_id))
        if not entry:
            return 0
        try:
            return int(entry.get("cumulative", 0))
        except (TypeError, ValueError):
            return 0

    def last_channel_cumulative(self, channel_id: str | int) -> int:
        """The highest cumulative accepted for a channel, kept or reserved
        by a request still being served (0 if never seen). A new voucher
        must add at least one price on top of it."""
        pending = self._pending_vouchers.get(str(channel_id)) or {}
        return max([self.kept_channel_cumulative(channel_id)]
                   + [int(p["cumulative"]) for p in pending.values()])

    def record_channel_voucher(
        self,
        channel_id: str | int,
        cumulative: int,
        signature: str,
        provider: str = "",
        service_id: int | None = None,
        ask: int = 0,
    ) -> None:
        """Store the latest accepted voucher for a channel so the provider can
        settle later via close_channel. Monotone: a lower cumulative than the
        stored one is ignored (never regress the settlement position).
        ``service_id`` and ``ask`` name the service it paid for, so the
        settlement sweep can tell when the deposit no longer covers it."""
        cid = str(channel_id)
        prev = self.kept_channel_cumulative(cid)
        if int(cumulative) < prev:
            return
        self._channel_vouchers[cid] = {
            "cumulative": int(cumulative),
            "signature": str(signature),
            "provider": str(provider or ""),
            "service_id": None if service_id is None else int(service_id),
            "ask": int(ask or 0),
            "ts": int(time.time()),
        }
        self._persist_vouchers()

    def reserve_channel_voucher(
        self,
        channel_id: str | int,
        cumulative: int,
        signature: str,
        provider: str = "",
        service_id: int | None = None,
        ask: int = 0,
    ) -> dict[str, Any]:
        """Hold a voucher the gate accepted while its request is served.
        Counts toward ``last_channel_cumulative`` at once (a copy sent
        meanwhile is refused), but is kept only by
        ``keep_channel_voucher``; ``release_channel_voucher`` drops it and
        the channel is back at its previous cumulative. Returns the handle
        both take."""
        cid = str(channel_id)
        self._pending_seq += 1
        token = self._pending_seq
        self._pending_vouchers.setdefault(cid, {})[token] = {
            "cumulative": int(cumulative), "signature": str(signature),
            "provider": str(provider or ""),
            "service_id": None if service_id is None else int(service_id),
            "ask": int(ask or 0)}
        return {"channel_id": cid, "token": token}

    def _pop_pending(self, reservation: dict[str, Any]) -> dict[str, Any] | None:
        cid = str(reservation.get("channel_id"))
        pending = self._pending_vouchers.get(cid)
        if not pending:
            return None
        entry = pending.pop(reservation.get("token"), None)
        if not pending:
            self._pending_vouchers.pop(cid, None)
        return entry

    def keep_channel_voucher(self, reservation: dict[str, Any]) -> None:
        """The request was served: keep its voucher to settle with."""
        entry = self._pop_pending(reservation)
        if entry is not None:
            self.record_channel_voucher(
                reservation["channel_id"], entry["cumulative"],
                entry["signature"], provider=entry["provider"],
                service_id=entry["service_id"], ask=entry["ask"])

    def release_channel_voucher(self, reservation: dict[str, Any]) -> None:
        """The request was not served: drop its voucher (not paid)."""
        self._pop_pending(reservation)

    def has_pending_voucher(self, channel_id: str | int) -> bool:
        """A request on this channel is being served right now."""
        return bool(self._pending_vouchers.get(str(channel_id)))

    def channel_vouchers(self) -> dict[str, dict[str, Any]]:
        """Every channel this provider holds a voucher for (settlement)."""
        return {k: dict(v) for k, v in self._channel_vouchers.items()}

    def forget_channel_voucher(self, channel_id: str | int) -> None:
        """Drop a settled channel's voucher."""
        if self._channel_vouchers.pop(str(channel_id), None) is not None:
            self._persist_vouchers()

    def get_channel_voucher(self, channel_id: str | int) -> dict[str, Any] | None:
        """Return the latest stored voucher for a channel, or None."""
        return self._channel_vouchers.get(str(channel_id))

    def _load_payment_state(self) -> None:
        if self._seen_path.exists():
            cutoff = time.time() - self.REPLAY_HORIZON_S
            lines = 0
            try:
                for line in self._seen_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    lines += 1
                    try:
                        row = json.loads(line)
                        rid = self.normalize_request_id(row.get("id"))
                        ts = float(row.get("ts") or 0.0)
                    except (ValueError, TypeError, AttributeError):
                        continue
                    if rid and ts >= cutoff:
                        self._seen_requests[rid] = ts
            except OSError as exc:
                log.warning("ServiceStore could not read served requests: %s", exc)
            self._seen_lines = lines
            if lines > len(self._seen_requests):
                self._persist_seen()
        if self._voucher_path.exists():
            try:
                data = json.loads(self._voucher_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._channel_vouchers = {
                        str(k): v for k, v in data.items() if isinstance(v, dict)
                    }
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("ServiceStore could not read channel vouchers: %s", exc)

    def _persist_seen(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._seen_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rid, ts in sorted(self._seen_requests.items()):
                fh.write(json.dumps({"id": rid, "ts": ts}) + "\n")
        os.replace(tmp, self._seen_path)
        self._seen_lines = len(self._seen_requests)

    def _persist_vouchers(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._voucher_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._channel_vouchers, fh)
        os.replace(tmp, self._voucher_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unknown_upstream(self, inference: dict[str, Any]) -> str:
        """Why this daemon cannot serve ``inference`` (its provider id is
        not one the provider manager knows), or ""."""
        manager = getattr(self._runtime, "providers", None)
        check = getattr(manager, "routing_error", None)
        if check is None:
            return ""
        try:
            err = check(str(inference.get("provider") or "").strip(),
                        str(inference.get("model") or "").strip())
        except Exception:  # noqa: BLE001 - the serving path refuses instead
            return ""
        return f"inference: {err}" if err else ""

    def _signing_key(self, author: str) -> str:
        """The key that signs ``author``'s specs and registers them: the
        operator key for the owner surface, the agent's stored key for an
        agent (an external agent's is the operator's). Empty when none is
        held (a dev daemon without a vault)."""
        registry = getattr(self._runtime, "registry", None)
        if registry is None:
            return ""
        try:
            if not author or author == OWNER_AUTHOR:
                return str(registry.daemon_operator_key() or "").strip()
            return str(registry.get_agent_key(author) or "").strip()
        except Exception:  # noqa: BLE001 - no vault: unsigned
            return ""

    def _author_address(self, author: str) -> str:
        """The address the spec names as ``author_pubkey``: the signer of
        the spec and the address that registers it on chain, so the
        catalog takes a registry entry as genuine only when its provider
        is this address (anyone can register a copy of a public spec).
        Without a signing key: the agent's identity address, else the
        owner wallet."""
        key = self._signing_key(author)
        if key:
            try:
                from eth_account import Account
                return str(Account.from_key(key).address)
            except Exception:  # noqa: BLE001
                pass
        if author != OWNER_AUTHOR:
            defn = self._runtime.get_agent(author)
            identity = getattr(defn, "identity", None) if defn else None
            addr = str(getattr(identity, "address", "") or "")
            if addr:
                return addr
        return self._owner_wallet()

    def _owner_wallet(self) -> str:
        return str(getattr(
            getattr(getattr(self._runtime, "_config", None),
                    "autonet", None), "owner_wallet", "") or "")

    def _sign(self, author: str, spec: dict[str, Any]) -> None:
        """Sign the canonical spec bytes with the author agent's key.

        Best-effort: owner/vendor authors have no agent key, and the
        eth_account dependency is optional — an unsigned spec is valid,
        it just carries no cryptographic authorship proof. Mirrors
        ToolStore._sign."""
        key = self._signing_key(author)
        if not key:
            return
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            signed = Account.sign_message(
                encode_defunct(canonical_service_bytes(spec)),
                private_key=key,
            )
            spec["author_sig"] = signed.signature.hex()
        except Exception as exc:  # optional-dep path; never block registration
            log.debug("service spec signing skipped for %s: %s", author, exc)

    async def spec_for(self, digest: str, timeout: float = 10.0) -> dict[str, Any] | None:
        """A service's spec by digest: ours, one fetched before, or one
        fetched now from the network (digest-verified, then kept). None
        when nobody online holds it."""
        import asyncio
        import hashlib

        d = str(digest or "").lower().removeprefix("0x")
        rec = self.get(d)
        if rec is not None:
            return rec.spec
        try:
            blobs = self._blob_store()
        except RuntimeError:
            return None
        raw = blobs.get_bytes_local(d)
        if raw is None and self.blob_fetcher is not None:
            try:
                raw = await asyncio.wait_for(self.blob_fetcher(d), timeout)
            except Exception:  # noqa: BLE001 — unknown is a normal answer
                raw = None
            if raw is not None and hashlib.sha256(raw).hexdigest() == d:
                blobs.add_bytes(raw)
            else:
                raw = None
        try:
            spec = json.loads(raw) if raw else None
        except ValueError:
            return None
        return spec if isinstance(spec, dict) else None

    def _blob_store(self) -> Any:
        if self._blobs is None:
            try:
                from nodes.common.blob_store import BlobStore
            except ImportError as exc:
                raise RuntimeError(
                    "service registration requires the autonet substrate "
                    "package (nodes.*); it is not importable here"
                ) from exc
            self._blobs = BlobStore(data_dir=str(self._dir / "blobs"))
        return self._blobs

    # ---- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            lines = self._registry_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("ServiceStore could not read %s: %s",
                        self._registry_path, exc)
            return
        try:
            blobs = self._blob_store()
        except RuntimeError as exc:
            log.warning("ServiceStore: %d persisted services unusable: %s",
                        len(lines), exc)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                digest = row["digest"]
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("ServiceStore skipping corrupt registry row")
                continue
            spec = blobs.get_json(digest)
            if spec is None:
                log.warning("ServiceStore: spec blob %s... missing, dropping",
                            digest[:16])
                continue
            self._records[digest] = ServiceRecord(
                digest=digest,
                spec=spec,
                backing_tool=str(row.get("backing_tool") or ""),
                retired=bool(row.get("retired", False)),
                registered_ts=int(row.get("registered_ts") or 0),
                service_id=(int(row["service_id"])
                            if row.get("service_id") is not None else None),
            )

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in self._records.values():
                fh.write(json.dumps(record.to_row()) + "\n")
        os.replace(tmp, self._registry_path)
