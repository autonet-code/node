"""ServiceStore — daemon-side registry of published service specs.

Design: ``docs/services_market.md``. A Service is a remote API an agent
(ultimately its human benefactor) offers to the market: general-purpose,
priced per work item, payable in any ERC20. It rides the same
content-addressed blob rail as tool manifests, but — unlike a tool — a
service gets NO substrate standing, NO verdict claims, NO mint. Trust is
behavioral: signed identity, atomic payment, receipt-gated reviews.

This is the provider-side store: specs are AUTHORED here, ask managed
here, publish = register-on-chain + blob push (chain wiring lives in the
contracts workstream; this store owns spec build/sign/persist and the
local request log that later reviews ride).

The backing implementation of a v1 service is just a local tool run on
the provider's daemon — "a Service is a tool the OWNER chose to sell"
(spec §3). ``ServiceRecord.backing_tool`` names that tool by digest;
dispatch goes through ``runtime.tool_store``.

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
    build_service_spec,
    canonical_service_bytes,
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

    @property
    def author(self) -> str:
        return str(self.spec.get("author") or "")

    @property
    def name(self) -> str:
        return str(self.spec.get("name") or "")

    @property
    def ask(self) -> dict[str, Any]:
        ask = self.spec.get("ask")
        return ask if isinstance(ask, dict) else {}

    def to_row(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "backing_tool": self.backing_tool,
            "retired": self.retired,
            "registered_ts": self.registered_ts,
        }


class ServiceStore:
    """Persistent registry of service specs (provider side)."""

    def __init__(self, runtime: Runtime, services_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(services_dir)
        self._registry_path = self._dir / "registry.jsonl"
        self._requests_path = self._dir / "requests.jsonl"
        self._records: dict[str, ServiceRecord] = {}
        self._blobs: Any = None  # lazy; None until substrate package needed
        self._request_seq = 0
        self._load()
        self._request_seq = self._count_requests()

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
        ask: dict[str, Any],
        backing_tool: str = "",
        output_schema: dict[str, Any] | None = None,
        endpoint_hint: str = "",
        version_of: str | None = None,
    ) -> dict[str, Any]:
        """Build, sign, blob-store, and persist a service spec. Returns
        ``{"digest", "spec"}`` or raises ValueError on bad input."""
        blobs = self._blob_store()
        author_pubkey = self._author_address(author)

        spec = build_service_spec(
            name=name,
            description=description,
            input_schema=input_schema,
            author=author,
            ask=ask,
            output_schema=output_schema,
            author_pubkey=author_pubkey,
            endpoint_hint=endpoint_hint,
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
        log.info("registered service %r by %s (ask %s) -> %s",
                 name, author, ask.get("token", "?"), digest[:16])
        return {"digest": digest, "spec": spec}

    def update_ask(
        self,
        digest: str,
        ask: dict[str, Any],
        *,
        backing_tool: str | None = None,
    ) -> dict[str, Any]:
        """Publish a new spec version with a changed ask.

        Content-addressed identity means a changed ask IS a new digest —
        there is no in-place mutation. The new spec carries
        ``version_of = <old digest>`` (artifact lineage). The predecessor
        is soft-retired so the market resolves to the current price.
        Returns ``{"digest", "spec"}`` or raises ValueError."""
        record = self._records.get(digest)
        if record is None:
            raise ValueError(f"unknown service digest: {digest[:16]}")
        spec = record.spec
        new = self.register(
            name=str(spec.get("name") or ""),
            description=str(spec.get("description") or ""),
            input_schema=dict(spec.get("input_schema") or {}),
            author=record.author,
            ask=ask,
            backing_tool=(backing_tool if backing_tool is not None
                          else record.backing_tool),
            output_schema=(dict(spec["output_schema"])
                           if isinstance(spec.get("output_schema"), dict) else None),
            endpoint_hint=str(spec.get("endpoint_hint") or ""),
            version_of=digest,
        )
        record.retired = True
        self._persist()
        return new

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

    def record_request(
        self,
        spec_digest: str,
        request_id: str,
        client: str,
        ok: bool,
        amount: str = "",
        token: str = "",
    ) -> None:
        """Append a provider-side request-log row. Local ledger first —
        never lost to a substrate outage."""
        record = self._records.get(spec_digest)
        self._request_seq += 1
        row = {
            "seq": self._request_seq,
            "ts": int(time.time()),
            "spec_digest": spec_digest,
            "service_name": record.name if record else "",
            "service_author": record.author if record else "",
            "request_id": request_id,
            "client": client,
            "ok": ok,
            "amount": amount,
            "token": token,
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._requests_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def summary(self) -> dict[str, Any]:
        """Per-service volume/success aggregated from the request log.

        This is the provider-side track record the spec's §4 reviews
        later ride on (success rate + volume, displayed not enforced)."""
        services: dict[str, dict[str, Any]] = {}
        for row in self._iter_requests():
            digest = row.get("spec_digest", "")
            entry = services.setdefault(digest, {
                "service_name": row.get("service_name", ""),
                "service_author": row.get("service_author", ""),
                "count": 0, "ok_count": 0,
                "clients": set(),
            })
            entry["service_name"] = row.get("service_name") or entry["service_name"]
            entry["service_author"] = (row.get("service_author")
                                       or entry["service_author"])
            entry["count"] += 1
            if row.get("ok"):
                entry["ok_count"] += 1
            client = row.get("client")
            if client:
                entry["clients"].add(client)
        out: dict[str, Any] = {}
        for digest, entry in sorted(services.items()):
            count = entry["count"]
            ok_count = entry["ok_count"]
            out[digest] = {
                "service_name": entry["service_name"],
                "service_author": entry["service_author"],
                "count": count,
                "ok_count": ok_count,
                "success_rate": round(ok_count / count, 6) if count else 0.0,
                "unique_clients": len(entry["clients"]),
            }
        return out

    def _iter_requests(self):
        if not self._requests_path.exists():
            return
        try:
            lines = self._requests_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("ServiceStore could not read requests: %s", exc)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("ServiceStore skipping corrupt request row")

    def _count_requests(self) -> int:
        return sum(1 for _ in self._iter_requests())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _author_address(self, author: str) -> str:
        defn = self._runtime.get_agent(author)
        identity = getattr(defn, "identity", None) if defn else None
        return str(getattr(identity, "address", "") or "")

    def _sign(self, author: str, spec: dict[str, Any]) -> None:
        """Sign the canonical spec bytes with the author agent's key.

        Best-effort: owner/vendor authors have no agent key, and the
        eth_account dependency is optional — an unsigned spec is valid,
        it just carries no cryptographic authorship proof. Mirrors
        ToolStore._sign."""
        key = None
        registry = getattr(self._runtime, "registry", None)
        if registry is not None and hasattr(registry, "get_agent_key"):
            key = registry.get_agent_key(author)
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
            )

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in self._records.values():
                fh.write(json.dumps(record.to_row()) + "\n")
        os.replace(tmp, self._registry_path)
