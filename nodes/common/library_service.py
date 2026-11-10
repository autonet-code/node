"""The daemon's tool library service: the epoch tick, the event plumbing
between the daemon and the gossip, search over the library, and the
display copy of each federated close.

One per jurisdiction key (``rpb_address``). It holds:

  - the epoch tick the scheduler drives (``open_epoch`` / ``close_epoch``).
    A tick's record carries no mint: closing it triggers the federated
    close (``FederatedCloseDriver``), the only place mint is computed;
  - the local-event fan-out the gossip republishes (``subscribe_local_events``)
    and the ingest of peer events (``submit_events(origin="remote")``),
    buffered for the open epoch;
  - ``submit_tool_manifest``: a published tool goes into the artifact index
    and out as one ``tool_registered`` event;
  - the blob store and the daemon-local artifact index (search), and the
    queue of remote manifests whose blobs discovery still has to fetch;
  - the latest federated close's registrations, ratings and per-tool
    mint (``note_federated_close``), which the economy graph reads.

Nothing here is consensus state: the close's inputs live with the driver
(the anchored carry-over) and the gossip batch log.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .blob_store import BlobStore
from .library.artifact_index import ArtifactIndex
from .library.usefulness_coords import DEFAULT_DIM

logger = logging.getLogger(__name__)

# The directory name predates the library service; keeping it keeps every
# daemon's blobs and carry files where they are.
_DEFAULT_DATA_ROOT = Path.home() / ".autonet" / "world"

# Local epoch-tick records kept in memory and on disk.
EPOCH_HISTORY_KEEP = 500
# Federated close summaries kept for the economy graph's recent mint.
CLOSES_KEEP = 50


def _safe_name(s: str, default: str = "default") -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s or default)
    return s or default


def _write_json_atomic(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class LibraryService:
    """The tool library and the epoch tick of one jurisdiction."""

    def __init__(
        self,
        rpb_address: str,
        *,
        data_root: Optional[Path] = None,
        blob_store: Optional[BlobStore] = None,
        embedding_dim: int = DEFAULT_DIM,
    ):
        self.rpb_address = rpb_address
        self.embedding_dim = embedding_dim
        self.state_dir = Path(data_root or _DEFAULT_DATA_ROOT) / _safe_name(rpb_address)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_dir = self.state_dir / "epochs"
        self.epochs_dir.mkdir(exist_ok=True)
        self._economy_path = self.state_dir / "economy.json"
        self._pruned_path = self.state_dir / "pruned.json"
        # The close's pruned tools (manifest digests) and their blobs, each
        # mapped to its tool (Decision 2026-11-10), persisted across restarts.
        self._pruned_tools: set = set()
        self._blob_owners: Dict[str, str] = {}
        self._lock = threading.RLock()
        # Close records wait here for their subscribers, which run AFTER
        # the lock is released: a subscriber that waits on another thread
        # (the federated close fetching blobs over the trio host) while the
        # lock is held would deadlock with the gossip ingest (issue #40).
        # The second lock only keeps deliveries in close order.
        self._pending_close_records: "collections.deque" = collections.deque()
        self._close_delivery_lock = threading.Lock()
        self._created_at = time.time()
        self._events_seen = 0

        # The open epoch: its id, start, and the events that arrived in it
        # (with arrival times, for the candle cutoff). Events after a
        # cutoff roll into the next epoch.
        self._current_epoch_id: Optional[str] = None
        self._epoch_started_at: float = 0.0
        self._epoch_events: List[Dict[str, Any]] = []
        self._epoch_event_ts: List[float] = []
        self._carry_events: List[Dict[str, Any]] = []
        self._carry_event_ts: List[float] = []
        self._epoch_history: List[Dict[str, Any]] = []
        self._epochs_recorded = 0

        self._epoch_subscribers: List[Any] = []
        self._local_event_subscribers: List[Any] = []

        if blob_store is None:
            blob_store = BlobStore(data_dir=str(self.state_dir / "blobs"))
        self._blob_store = blob_store
        self._artifact_index = ArtifactIndex(
            self._blob_store,
            index_path=self.state_dir / "artifact_index.jsonl",
            dim=embedding_dim,
        )
        # Manifest digests seen on remote tool_registered events whose blob
        # this daemon does not hold yet. Ingest records the digest (cheap,
        # on the gossip hot path); a worker outside this class pulls the
        # blob digest-verified and indexes it.
        self._pending_remote_manifests: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
        self._pending_remote_manifests_cap = 2048

        # The latest federated close, for the economy graph.
        self._registrations: Dict[str, Dict[str, str]] = {}
        self._ratings: Dict[str, Dict[str, Any]] = {}
        self._closes: List[Dict[str, Any]] = []
        self._load_economy()
        self._load_epoch_history()
        self._load_pruned()

    # ------------------------------------------------------------------
    # Epoch tick
    # ------------------------------------------------------------------

    def open_epoch(self, epoch_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new epoch tick; events that arrive are buffered until
        ``close_epoch``."""
        try:
            with self._lock:
                if self._current_epoch_id is not None:
                    logger.warning(
                        "open_epoch called while %s is still open; "
                        "closing it first", self._current_epoch_id)
                    self._close_epoch_locked()
                eid = epoch_id or f"epoch_{time.time_ns()}"
                self._current_epoch_id = eid
                self._epoch_started_at = time.time()
                self._epoch_events = list(self._carry_events)
                self._epoch_event_ts = list(self._carry_event_ts)
                self._carry_events = []
                self._carry_event_ts = []
                return {"epoch_id": eid, "started_at": self._epoch_started_at}
        finally:
            self._deliver_epoch_closed()

    def close_epoch(self, *, cutoff_ts: Optional[float] = None) -> Dict[str, Any]:
        """Close the open epoch tick and hand its record to the subscribers
        (the federated close driver among them).

        ``cutoff_ts`` (candle close): only events that arrived at or before
        it belong to this tick; later ones roll into the next.
        """
        try:
            with self._lock:
                return self._close_epoch_locked(cutoff_ts=cutoff_ts)
        finally:
            self._deliver_epoch_closed()

    def _close_epoch_locked(self, *, cutoff_ts: Optional[float] = None) -> Dict[str, Any]:
        if self._current_epoch_id is None:
            logger.warning("close_epoch called with no open epoch; ignoring")
            return {"epoch_id": None, "n_events": 0}
        events_rolled = 0
        events = list(self._epoch_events)
        if cutoff_ts is not None and self._epoch_events:
            kept = []
            for ev, ts in zip(self._epoch_events, self._epoch_event_ts):
                if ts <= cutoff_ts:
                    kept.append(ev)
                else:
                    self._carry_events.append(ev)
                    self._carry_event_ts.append(ts)
            events_rolled = len(self._epoch_events) - len(kept)
            events = kept
        record = {
            "epoch_id": self._current_epoch_id,
            "rpb_address": self.rpb_address,
            "started_at": self._epoch_started_at,
            "closed_at": time.time(),
            "n_events": len(events),
            "cutoff_ts": cutoff_ts,
            "events_rolled_forward": events_rolled,
            "authoritative": False,
            "scope": "local",
        }
        self._epoch_history.append(record)
        self._epochs_recorded += 1
        self._write_epoch_record(record)
        while len(self._epoch_history) > EPOCH_HISTORY_KEEP:
            old = self._epoch_history.pop(0)
            self._delete_epoch_record(str(old.get("epoch_id", "")))
        self._pending_close_records.append(record)
        self._current_epoch_id = None
        self._epoch_started_at = 0.0
        self._epoch_events = []
        self._epoch_event_ts = []
        (logger.info if events else logger.debug)(
            "epoch tick closed: rpb=%s, epoch=%s, %d events%s",
            self.rpb_address, record["epoch_id"], len(events),
            f" ({events_rolled} rolled forward)" if events_rolled else "")
        return dict(record)

    @property
    def current_epoch_id(self) -> Optional[str]:
        return self._current_epoch_id

    def epoch_status(self) -> Dict[str, Any]:
        """This service's half of the epoch_status surface (the scheduler
        owns the timing). Lock-free: a status poll never queues behind a
        close or an ingest (issue #40)."""
        return {
            "epoch_id": self._current_epoch_id,
            "opened_at": self._epoch_started_at or None,
            "buffered_events": len(self._epoch_events),
            "rpb_address": self.rpb_address,
        }

    @property
    def epochs_recorded(self) -> int:
        return self._epochs_recorded

    @property
    def epoch_history(self) -> List[Dict[str, Any]]:
        return list(self._epoch_history)

    def read_recent_epochs(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in reversed(self._epoch_history[-n:])]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def submit_events(self, events: List[Dict[str, Any]], *,
                      origin: str = "local") -> Dict[str, Any]:
        """Take events into the open epoch. ``origin="local"`` events go
        out to the local-event subscribers (the gossip); ``"remote"`` ones
        (peer batches) do not, and their tool registrations are queued for
        discovery."""
        if not events:
            return {"events_applied": 0}
        with self._lock:
            self._buffer_locked(events)
            if origin == "remote":
                self._note_remote_manifests_locked(events)
            else:
                self._broadcast_local_events_locked(events)
            self._events_seen += len(events)
            return {"events_applied": len(events)}

    def submit_tool_manifest(self, manifest: Dict[str, Any], *,
                             agent_id: str = "default") -> Dict[str, Any]:
        """Register a published tool manifest: it goes into the artifact
        index (search) and one ``tool_registered`` event goes to the epoch
        buffer and to the gossip; the close reads its digest and metadata."""
        from .library.tool_manifest import registration_event, validate_manifest

        errors = validate_manifest(manifest)
        if errors:
            return {"error": "invalid tool manifest: " + "; ".join(errors)}
        with self._lock:
            digest = self._artifact_index.add_artifact(manifest)
            event = registration_event(manifest, digest, agent_id=agent_id)
            self._buffer_locked([event])
            self._broadcast_local_events_locked([event])
            self._events_seen += 1
            return {"events_applied": 1, "manifest_digest": digest}

    def subscribe_local_events(self, handler: Any) -> None:
        """``handler(events)`` for every locally originated batch (the
        gossip republishes them). Called under the lock."""
        with self._lock:
            self._local_event_subscribers.append(handler)

    def unsubscribe_local_events(self, handler: Any) -> None:
        with self._lock:
            try:
                self._local_event_subscribers.remove(handler)
            except ValueError:
                pass

    def subscribe_epoch_closed(self, handler: Any) -> None:
        """``handler(record)`` after each epoch tick closes, run with the
        lock released (it may block on blob fetches and chain reads)."""
        with self._lock:
            self._epoch_subscribers.append(handler)

    def unsubscribe_epoch_closed(self, handler: Any) -> None:
        with self._lock:
            try:
                self._epoch_subscribers.remove(handler)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Search and discovery
    # ------------------------------------------------------------------

    def artifact_digests(self) -> List[str]:
        """Every manifest digest in the artifact index."""
        with self._lock:
            return list(self._artifact_index._vectors.keys())

    def blob_json(self, digest: str) -> Any:
        return self._blob_store.get_json(digest)

    def take_pending_remote_manifests(self, limit: int = 32) -> List[str]:
        """Pop up to ``limit`` queued remote manifest digests."""
        out: List[str] = []
        with self._lock:
            while self._pending_remote_manifests and len(out) < limit:
                digest, _meta = self._pending_remote_manifests.popitem(last=False)
                if self._artifact_index.has(digest) or digest in self._pruned_tools:
                    continue
                out.append(digest)
        return out

    def index_remote_manifest(self, manifest: Dict[str, Any]) -> Optional[str]:
        """Index a manifest fetched from a peer (the caller verified its
        digest). Touches only the daemon-local index."""
        from .library.tool_manifest import is_tool_manifest

        if not is_tool_manifest(manifest):
            return None
        with self._lock:
            # The digest add_json would give it (json.dumps, sorted keys).
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
            if digest in self._pruned_tools:
                return None
            try:
                return self._artifact_index.add_artifact(manifest)
            except Exception:
                logger.warning("indexing remote manifest failed", exc_info=True)
                return None

    # ------------------------------------------------------------------
    # The economy graph (display only)
    # ------------------------------------------------------------------

    def note_federated_close(self, close_result: Dict[str, Any], *,
                             epoch_id: str = "") -> None:
        """Keep the latest federated close's registrations, ratings and
        per-tool mint for the economy graph."""
        regs = close_result.get("tool_registrations")
        ratings = close_result.get("tool_ratings")
        if "pruned" in close_result:
            self.apply_pruned(close_result.get("pruned") or [])
        with self._lock:
            if isinstance(regs, dict):
                self._registrations = {str(k): dict(v) for k, v in regs.items()
                                       if isinstance(v, dict)}
            if isinstance(ratings, dict):
                self._ratings = {str(k): dict(v) for k, v in ratings.items()
                                 if isinstance(v, dict)}
            self._closes.append({
                "epoch_id": str(epoch_id or close_result.get("epoch_id") or ""),
                "closed_at": time.time(),
                "total_mint": float(close_result.get("total_mint") or 0.0),
                "tool_mint": dict(close_result.get("tool_mint") or {}),
            })
            self._closes = self._closes[-CLOSES_KEEP:]
            self._save_economy()

    def apply_pruned(self, digests: List[str]) -> Dict[str, int]:
        """Make the close's pruned list this daemon's (Decision 2026-11-10):
        drop newly pruned manifests from the index and delete their blobs
        (the manifest and the code it names), except blobs the household
        keeps for a tool it installed; stop serving and fetching them all. A
        tool no longer listed (the DAO cleared its verdict) is fetchable
        again. Returns counts for logging."""
        from . import pruned_tools

        tools = {str(d).lower().removeprefix("0x") for d in digests if d}
        with self._lock:
            new = sorted(tools - self._pruned_tools)
            # blob -> the pruned tool it belongs to; a cleared tool's go.
            owners = {b: o for b, o in self._blob_owners.items() if o in tools}
            for d in sorted(tools):
                owners[d] = d
                payload = self._local_json(d)
                code = str(payload.get("code_digest") or "") if isinstance(payload, dict) else ""
                if code:
                    owners[code] = d
            dropped = self._artifact_index.remove(new)
            deleted = 0
            for b in sorted(owners):
                if self._blob_store.has(b) and not pruned_tools.kept(b):
                    deleted += 1 if self._blob_store.delete(b) else 0
            self._pruned_tools = tools
            self._blob_owners = owners
            for d in new:
                self._pending_remote_manifests.pop(d, None)
            pruned_tools.replace(set(tools) | set(owners))
            self._save_pruned()
        if new or deleted:
            logger.info("pruned %d tool(s) (%d newly), %d index entries, %d blob(s) deleted",
                        len(tools), len(new), dropped, deleted)
        return {"tools": len(tools), "new": len(new), "indexed": dropped, "deleted": deleted}

    def pruned_tool_digests(self) -> List[str]:
        with self._lock:
            return sorted(self._pruned_tools)

    def _local_json(self, digest: str) -> Any:
        raw = getattr(self._blob_store, "get_bytes_local", None)
        data = raw(digest) if raw is not None else None
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _load_pruned(self) -> None:
        from . import pruned_tools

        try:
            data = json.loads(self._pruned_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._pruned_tools = {str(d) for d in data.get("tools") or []}
        self._blob_owners = {str(b): str(o) for b, o in (data.get("blobs") or {}).items()}
        pruned_tools.replace(self._pruned_tools | set(self._blob_owners))

    def _save_pruned(self) -> None:
        try:
            _write_json_atomic(self._pruned_path, {
                "tools": sorted(self._pruned_tools),
                "blobs": dict(sorted(self._blob_owners.items()))})
        except OSError:
            logger.warning("failed to persist the pruned tools", exc_info=True)

    def read_economy_graph(self, *, last_n_epochs: int = 10) -> Dict[str, Any]:
        """The tool-economy graph from the latest federated closes.

        Returns ``{registrations: {digest: {author, trust_class, deps}},
        recent_tool_mint: {digest: float}, ratings: {digest: rating},
        last_epoch: {epoch_id, closed_at, total_mint, tool_mint} | None,
        epochs_considered: int}``. A rating is the ToolBond means as of the
        latest close (``federated_reconcile.tool_rating``: correctness,
        simplicity, novelty, malware share, charter ``axes`` and their
        ``value``, ``mass``) plus ``score``, the factor on the tool's author
        side (V x C x S).
        """
        from .federated_reconcile import tool_score
        with self._lock:
            registrations = {
                digest: {
                    "author": str(meta.get("author") or ""),
                    "trust_class": str(meta.get("trust_class") or ""),
                    "deps": [d for d in str(meta.get("deps") or "").split(",") if d],
                }
                for digest, meta in self._registrations.items()
            }
            ratings = {digest: {**r, "score": tool_score(r)}
                       for digest, r in self._ratings.items()}
            recent = list(self._closes[-last_n_epochs:]) if last_n_epochs > 0 else []
        recent_mint: Dict[str, float] = {}
        for record in recent:
            for digest, entry in (record.get("tool_mint") or {}).items():
                if not isinstance(entry, dict):
                    continue
                try:
                    recent_mint[digest] = (recent_mint.get(digest, 0.0)
                                           + float(entry.get("mint") or 0.0))
                except (TypeError, ValueError):
                    continue
        last = recent[-1] if recent else None
        return {
            "registrations": registrations,
            "recent_tool_mint": recent_mint,
            "ratings": ratings,
            "last_epoch": None if last is None else {
                "epoch_id": last.get("epoch_id"),
                "closed_at": last.get("closed_at"),
                "total_mint": last.get("total_mint", 0.0),
                "tool_mint": last.get("tool_mint") or {},
            },
            "epochs_considered": len(recent),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rpb_address": self.rpb_address,
                "events_seen": self._events_seen,
                "artifacts": len(self._artifact_index._vectors),
                "uptime_seconds": time.time() - self._created_at,
            }

    def shutdown(self) -> None:
        with self._lock:
            try:
                self._save_economy()
            except Exception as e:  # noqa: BLE001
                logger.warning("economy save during shutdown failed: %s", e)
        logger.info("LibraryService shut down: rpb=%s", self.rpb_address)

    # ------------------------------------------------------------------
    # Internal (call under self._lock)
    # ------------------------------------------------------------------

    def _buffer_locked(self, events: List[Dict[str, Any]]) -> None:
        if self._current_epoch_id is not None and events:
            self._epoch_events.extend(events)
            now = time.time()
            self._epoch_event_ts.extend([now] * len(events))

    def _note_remote_manifests_locked(self, events: List[Dict[str, Any]]) -> None:
        for ev in events:
            try:
                if ev.get("kind") != "tool_registered":
                    continue
                digest = str(ev.get("artifact_digest") or "")
                meta = ev.get("manifest_meta")
                if not digest or not isinstance(meta, dict) or not meta:
                    continue
                if (self._artifact_index.has(digest) or digest in self._pending_remote_manifests
                        or digest in self._pruned_tools):
                    continue
                self._pending_remote_manifests[digest] = {
                    str(k): str(v) for k, v in meta.items()}
                while len(self._pending_remote_manifests) > self._pending_remote_manifests_cap:
                    self._pending_remote_manifests.popitem(last=False)
            except Exception:  # pragma: no cover - never break ingest
                logger.debug("remote manifest note failed", exc_info=True)

    def _broadcast_local_events_locked(self, events: List[Dict[str, Any]]) -> None:
        for handler in list(self._local_event_subscribers):
            try:
                handler(events)
            except Exception as e:
                logger.error("local event subscriber failed: %s", e, exc_info=True)

    def _deliver_epoch_closed(self) -> None:
        with self._close_delivery_lock:
            while True:
                try:
                    record = self._pending_close_records.popleft()
                except IndexError:
                    return
                for handler in list(self._epoch_subscribers):
                    try:
                        handler(record)
                    except Exception as e:
                        logger.error("epoch close subscriber failed: %s", e,
                                     exc_info=True)

    def _epoch_record_path(self, epoch_id: str) -> Path:
        return self.epochs_dir / f"{_safe_name(str(epoch_id), 'unknown')}.json"

    def _write_epoch_record(self, record: Dict[str, Any]) -> None:
        try:
            _write_json_atomic(self._epoch_record_path(str(record.get("epoch_id"))), record)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to persist epoch record: %s", e)

    def _delete_epoch_record(self, epoch_id: str) -> None:
        try:
            self._epoch_record_path(epoch_id).unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug("failed to delete epoch record: %s", e)

    def _load_epoch_history(self) -> None:
        loaded = []
        for path in sorted(self.epochs_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict):
                loaded.append(rec)
        loaded.sort(key=lambda r: r.get("closed_at", 0.0))
        for r in loaded[:-EPOCH_HISTORY_KEEP]:
            self._delete_epoch_record(str(r.get("epoch_id", "")))
        self._epoch_history = loaded[-EPOCH_HISTORY_KEEP:]
        self._epochs_recorded = len(self._epoch_history)

    def _load_economy(self) -> None:
        if not self._economy_path.exists():
            return
        try:
            data = json.loads(self._economy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("economy graph state unreadable: %s", e)
            return
        if isinstance(data, dict):
            self._registrations = dict(data.get("registrations") or {})
            self._ratings = dict(data.get("ratings") or {})
            self._closes = list(data.get("closes") or [])[-CLOSES_KEEP:]

    def _save_economy(self) -> None:
        try:
            _write_json_atomic(self._economy_path, {
                "registrations": self._registrations,
                "ratings": self._ratings,
                "closes": self._closes,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("economy graph state save failed: %s", e)
