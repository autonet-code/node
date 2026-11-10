"""The jurisdiction's service catalog: every registered service this daemon
knows, its spec, and search over them (Decision 2026-11-10, services are
published and found the way tools are).

Three sources feed it:

- **The chain** (``ServiceRegistry``): the only reference for each
  service's spec digest, price, active flag and payee (the provider that
  registered it). The catalog follows it: the first sync enumerates every
  entry, later syncs read the entries numbered past the last one known and
  the ``ServiceAskUpdated`` / ``ServiceRetired`` logs since the last block,
  in chunks of ``LOG_CHUNK`` blocks. One sync every ``SYNC_INTERVAL``.
- **Gossip** (``/autonet/services/<jurisdiction>``): a provider's daemon
  announces each of its specs whole (a spec has no code to fetch) when it
  registers one and again every ``ANNOUNCE_INTERVAL``. Anyone can gossip,
  so a spec is accepted only once the chain has an entry naming its
  digest: registering costs gas and needs a registered address. An
  announcement for an entry not yet synced waits (up to ``PENDING_CAP``).
  Accepting one costs no chain read.
- **Blob fetch**: an active entry whose spec no announcement brought is
  fetched by digest from whichever daemon holds it, retried with a
  doubling backoff (``RETRY_BASE`` up to ``RETRY_CAP``), like a tool
  manifest.

An entry counts only when its provider is the spec's signer (the spec's
``author_pubkey``, with a valid ``author_sig``): anyone can register a copy
of a public spec, at any price, and the catalog ignores it. An entry whose
spec is not here yet is not listed until the spec arrives.

Search is the tools' ranking (``atn/tool_search.py``): BM25 over the name,
description and input-schema property names, fused with semantic
similarity from the local search model; keyword only until the model is
loaded. Order is relevance only; price and provider come back as metadata.

All of it is daemon-local state (``<data_dir>/services/catalog.json`` and
``<data_dir>/search/service_vectors.jsonl``): never anchored, never read by
the epoch-close. Services earn no mint, so nothing here is consensus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from .service_spec import (
    is_service_spec,
    service_embedding_text,
    signed_by_author,
    validate_service_spec,
)

log = logging.getLogger(__name__)

SYNC_INTERVAL = 60.0
ANNOUNCE_INTERVAL = 1800.0
LOG_CHUNK = 1000
MAX_CHUNKS_PER_SYNC = 50
MAX_SPEC_BYTES = 64 * 1024
PENDING_CAP = 512
RETRY_BASE = 60.0
RETRY_CAP = 3600.0
ANNOUNCEMENT_VERSION = 1


def topic_for(jurisdiction_key: str) -> str:
    """The pubsub topic carrying one jurisdiction's service specs (relays
    subscribe to it too, so specs cross NAT'd daemons)."""
    from nodes.common.event_gossip import topic_for_services
    return topic_for_services(jurisdiction_key)


def spec_digest(spec: dict[str, Any]) -> str:
    """The digest the blob store gives a spec (``BlobStore.add_json``)."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


def _norm(digest: Any) -> str:
    return str(digest or "").lower().removeprefix("0x")


def announcement(spec: dict[str, Any], service_id: int) -> bytes:
    """The gossip message announcing one registered spec."""
    return json.dumps({"v": ANNOUNCEMENT_VERSION,
                       "service_id": int(service_id),
                       "spec": spec}, sort_keys=True).encode("utf-8")


class ServiceCatalog:
    """Every service in the jurisdiction this daemon knows, with search."""

    def __init__(self, runtime: Any, path: Optional[Path] = None, *,
                 index: Any = None) -> None:
        self._runtime = runtime
        self._path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._entries: dict[int, dict[str, Any]] = {}
        self._specs: dict[str, dict[str, Any]] = {}
        self._pending: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._last_block: Optional[int] = None
        self._fetch_lease: dict[str, tuple[int, float]] = {}
        self._index = index
        self._load()
        if self._index is None:
            from .tool_search import ToolSearchIndex
            vec_path = None
            if self._path is not None:
                vec_path = self._path.parent.parent / "search" / "service_vectors.jsonl"
            self._index = ToolSearchIndex(vec_path, text_of=service_embedding_text)
        for digest, spec in self._specs.items():
            self._index.note(digest, spec)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def synced(self) -> bool:
        return self._last_block is not None

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for _, e in sorted(self._entries.items())]

    def spec(self, digest: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._specs.get(_norm(digest))

    def _own_digests(self) -> set:
        store = getattr(self._runtime, "service_store", None)
        if store is None:
            return set()
        return {r.digest for r in store.list(include_retired=True)}

    def _genuine_locked(self, entry: dict[str, Any], own: set) -> bool:
        """An entry registered by its spec's signer (this daemon's own
        specs count when unsigned, on a daemon without keys)."""
        d = _norm(entry.get("spec_digest"))
        spec = self._specs.get(d)
        if spec is None:
            return False
        author = str(spec.get("author_pubkey") or "").lower()
        if not author or str(entry.get("provider") or "").lower() != author:
            return False
        return d in own or signed_by_author(spec)

    def entry_for(self, digest: str, provider: str = "") -> Optional[dict[str, Any]]:
        """The newest genuine entry naming ``digest`` (under ``provider``
        when given), active ones first. None while the spec is not here."""
        d = _norm(digest)
        p = str(provider or "").lower()
        own = self._own_digests()
        with self._lock:
            hits = [e for e in self._entries.values()
                    if _norm(e.get("spec_digest")) == d
                    and (not p or str(e.get("provider") or "").lower() == p)
                    and self._genuine_locked(e, own)]
        if not hits:
            return None
        hits.sort(key=lambda e: (bool(e.get("active")), int(e["service_id"])))
        return dict(hits[-1])

    # ------------------------------------------------------------------
    # The chain
    # ------------------------------------------------------------------

    def _config(self) -> Any:
        return self._runtime._config.rpb

    def sync_chain(self) -> dict[str, Any]:
        """Bring the entries up to the chain's latest block. Blocking (web3
        HTTP): run it off the event loop. Returns counts or ``{error}``."""
        from .on_chain import ServiceMarketClient
        smc = ServiceMarketClient(self._config())
        if not smc.registry_available:
            return {"error": "ServiceRegistry not configured"}
        try:
            w3 = smc._get_web3()
            reg = smc._registry_contract(w3)
            latest = int(w3.eth.block_number)
            count = int(reg.functions.serviceCount().call())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ServiceRegistry read failed: {exc}"}
        with self._lock:
            known = set(self._entries)
            last = self._last_block
        changed = {sid for sid in range(1, count + 1) if sid not in known}
        reached = latest
        if last is not None and last < latest and known:
            reached, logged = self._changed_since(w3, reg, last + 1, latest)
            if logged is None:
                # Logs unavailable on this RPC: read every known entry again.
                changed |= known
                reached = latest
            else:
                changed |= logged
        updated = 0
        for sid in sorted(changed):
            try:
                provider, digest, ask, active, reg_at, upd_at = \
                    reg.functions.getService(int(sid)).call()
            except Exception as exc:  # noqa: BLE001
                log.debug("getService(%s) failed: %s", sid, exc)
                return {"error": f"getService({sid}) failed: {exc}"}
            if int(reg_at) == 0:
                continue
            entry = {
                "service_id": int(sid),
                "provider": str(provider),
                "spec_digest": (digest.hex() if isinstance(digest, (bytes, bytearray))
                                else _norm(digest)).removeprefix("0x"),
                "ask_amount": int(ask),
                "active": bool(active),
                "registered_at": int(reg_at),
                "updated_at": int(upd_at),
            }
            with self._lock:
                self._entries[int(sid)] = entry
            updated += 1
        with self._lock:
            self._last_block = reached
            adopted = self._adopt_pending_locked()
            self._save_locked()
        return {"entries": len(self._entries), "updated": updated,
                "adopted_specs": adopted, "block": reached}

    @staticmethod
    def _topic(sig: str) -> str:
        from web3 import Web3
        return "0x" + Web3.keccak(text=sig).hex().removeprefix("0x")

    def _changed_since(self, w3: Any, reg: Any, start: int,
                       latest: int) -> tuple[int, Optional[set[int]]]:
        """Service numbers whose price or active flag changed in blocks
        ``start..`` (at most ``MAX_CHUNKS_PER_SYNC`` chunks per sync), and
        the last block read. ``None`` when the RPC refuses the logs."""
        topics = [[self._topic("ServiceAskUpdated(uint256,address,uint256)"),
                   self._topic("ServiceRetired(uint256,address)")]]
        out: set[int] = set()
        chunks = 0
        block = start
        while block <= latest and chunks < MAX_CHUNKS_PER_SYNC:
            end = min(latest, block + LOG_CHUNK - 1)
            try:
                logs = w3.eth.get_logs({"address": reg.address, "fromBlock": block,
                                        "toBlock": end, "topics": topics})
            except Exception as exc:  # noqa: BLE001
                log.debug("ServiceRegistry logs %d..%d refused: %s", block, end, exc)
                return latest, None
            for lg in logs:
                t = lg["topics"][1]
                raw = bytes(t) if not isinstance(t, str) else bytes.fromhex(
                    t.removeprefix("0x"))
                out.add(int.from_bytes(raw, "big"))
            block = end + 1
            chunks += 1
        return block - 1, out

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def accept_announcement(self, data: bytes) -> bool:
        """Take a gossiped spec. Kept when a synced entry names its digest,
        held (bounded) when no entry does yet. No chain read. Returns
        whether the spec was accepted now."""
        if not data or len(data) > MAX_SPEC_BYTES:
            return False
        try:
            msg = json.loads(data.decode("utf-8"))
            spec = msg.get("spec")
        except (ValueError, UnicodeDecodeError, AttributeError):
            return False
        if not isinstance(spec, dict) or not is_service_spec(spec):
            return False
        if validate_service_spec(spec):
            return False
        digest = spec_digest(spec)
        with self._lock:
            if digest in self._specs:
                return True
            if any(_norm(e.get("spec_digest")) == digest
                   for e in self._entries.values()):
                self._keep_locked(digest, spec)
                self._save_locked()
                return True
            self._pending[digest] = spec
            self._pending.move_to_end(digest)
            while len(self._pending) > PENDING_CAP:
                self._pending.popitem(last=False)
        return False

    def note_spec(self, digest: str, spec: Any) -> bool:
        """Keep a spec fetched by digest (verified here) when an entry
        names it."""
        d = _norm(digest)
        if not isinstance(spec, dict) or spec_digest(spec) != d:
            return False
        if not is_service_spec(spec) or validate_service_spec(spec):
            return False
        with self._lock:
            if not any(_norm(e.get("spec_digest")) == d
                       for e in self._entries.values()):
                return False
            self._keep_locked(d, spec)
            self._save_locked()
        return True

    def due_fetches(self, limit: int = 32) -> list[str]:
        """Digests of active entries without a spec, due for a fetch. Each
        returned digest is leased for a doubling backoff."""
        now = time.time()
        out: list[str] = []
        with self._lock:
            want = sorted({_norm(e.get("spec_digest")) for e in self._entries.values()
                           if e.get("active")} - set(self._specs))
            for d in want:
                if len(out) >= limit:
                    break
                attempts, due = self._fetch_lease.get(d, (0, 0.0))
                if due > now:
                    continue
                self._fetch_lease[d] = (attempts + 1, now + min(
                    RETRY_BASE * (2 ** attempts), RETRY_CAP))
                out.append(d)
        return out

    def _keep_locked(self, digest: str, spec: dict[str, Any]) -> None:
        self._specs[digest] = spec
        self._pending.pop(digest, None)
        self._fetch_lease.pop(digest, None)
        if self._index is not None:
            self._index.note(digest, spec)

    def _adopt_pending_locked(self) -> int:
        named = {_norm(e.get("spec_digest")) for e in self._entries.values()}
        adopted = 0
        for digest in [d for d in self._pending if d in named]:
            self._keep_locked(digest, self._pending[digest])
            adopted += 1
        return adopted

    def own_announcements(self) -> list[bytes]:
        """Announcements for this daemon's own registered, unretired
        services (``ServiceStore`` records with a service number)."""
        store = getattr(self._runtime, "service_store", None)
        if store is None:
            return []
        out = []
        for rec in store.list(include_retired=False):
            if rec.service_id is not None:
                out.append(announcement(rec.spec, rec.service_id))
        return out

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str = "", limit: int = 25, *,
               include_inactive: bool = False) -> list[dict[str, Any]]:
        """Services matching ``query`` in relevance order (newest first
        without a query): a row per genuine entry whose spec is here."""
        own = self._own_digests()
        with self._lock:
            entries = [dict(e) for e in self._entries.values()
                       if (include_inactive or e.get("active"))
                       and self._genuine_locked(e, own)]
            specs = dict(self._specs)
        q = str(query or "").strip()
        if not q:
            ordered = sorted(entries, key=lambda e: -int(e["service_id"]))
            return [self._row(e, specs, own) for e in ordered[:limit]]
        by_digest: dict[str, list[dict[str, Any]]] = {}
        for e in entries:
            by_digest.setdefault(_norm(e.get("spec_digest")), []).append(e)
        cands = [d for d in by_digest if d in specs]
        ranked = self._index.rank(q, cands) if cands else []
        rows: list[dict[str, Any]] = []
        for hit in ranked:
            for e in sorted(by_digest.get(hit["digest"], []),
                            key=lambda e: -int(e["service_id"])):
                row = self._row(e, specs, own)
                row["score"] = hit["score"]
                rows.append(row)
                if len(rows) >= limit:
                    return rows
        return rows

    @staticmethod
    def _row(entry: dict[str, Any], specs: dict[str, dict[str, Any]],
             own: set) -> dict[str, Any]:
        digest = _norm(entry.get("spec_digest"))
        spec = specs.get(digest) or {}
        return {
            "service_id": entry["service_id"],
            "provider": entry.get("provider", ""),
            "ask": {"amount": str(entry.get("ask_amount") or 0), "unit": "per_item"},
            "ask_amount": int(entry.get("ask_amount") or 0),
            "digest": digest,
            "active": bool(entry.get("active")),
            "name": str(spec.get("name") or "") or f"service {digest[:10]}",
            "description": str(spec.get("description") or ""),
            "input_schema": spec.get("input_schema"),
            "output_schema": spec.get("output_schema"),
            "image_uri": str(spec.get("image_uri") or ""),
            "inference": spec.get("inference") if isinstance(spec.get("inference"), dict) else {},
            "owned": digest in own,
            "known_spec": bool(spec),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("service catalog unreadable (%s); syncing afresh", exc)
            return
        for e in data.get("entries") or []:
            try:
                self._entries[int(e["service_id"])] = dict(e)
            except (KeyError, TypeError, ValueError):
                continue
        for d, spec in (data.get("specs") or {}).items():
            if isinstance(spec, dict) and spec_digest(spec) == _norm(d):
                self._specs[_norm(d)] = spec
        lb = data.get("last_block")
        self._last_block = int(lb) if isinstance(lb, int) else None

    def _save_locked(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "last_block": self._last_block,
            "entries": [self._entries[k] for k in sorted(self._entries)],
            "specs": self._specs,
        }), encoding="utf-8")
        os.replace(tmp, self._path)


def catalog_for(runtime: Any) -> ServiceCatalog:
    """The runtime's catalog (created on first use; tests may pre-set
    ``runtime.service_catalog``)."""
    cat = runtime.__dict__.get("service_catalog")
    if isinstance(cat, ServiceCatalog):
        return cat
    path = None
    try:
        path = Path(runtime._config.data_dir) / "services" / "catalog.json"
    except Exception:  # noqa: BLE001 (no data dir: in memory)
        path = None
    cat = ServiceCatalog(runtime, path)
    runtime.__dict__["service_catalog"] = cat
    return cat


async def ensure_synced(catalog: ServiceCatalog) -> dict[str, Any]:
    """Sync once if this catalog never has (a daemon whose background
    sync has not run yet, or that has no p2p). Off the event loop."""
    import asyncio
    if catalog.synced:
        return {}
    return await asyncio.to_thread(catalog.sync_chain)


async def fetch_missing(catalog: ServiceCatalog, fetch, *, limit: int = 32,
                        concurrency: int = 8) -> int:
    """Fetch up to ``limit`` due specs with ``fetch(digest) -> spec|None``;
    returns how many were kept."""
    import asyncio
    sem = asyncio.Semaphore(concurrency)

    async def _one(d: str) -> bool:
        async with sem:
            try:
                spec = await fetch(d)
            except Exception:  # noqa: BLE001
                return False
        return catalog.note_spec(d, spec)

    digests = catalog.due_fetches(limit)
    if not digests:
        return 0
    results = await asyncio.gather(*(_one(d) for d in digests))
    return sum(1 for r in results if r)
