"""On-disk persistence for the persistent World.

Layout (one tree per RPB address):

  <data_root>/<rpb>/events.jsonl        append-only event log (source of truth)
  <data_root>/<rpb>/scores.json         equilibrated score snapshot (cache only)
  <data_root>/<rpb>/checkpoint.json     full world state at an event offset

Recovery rules
--------------

The **event log is the source of truth**. Per the substrate's design
(see ``world_model_substrate/events.py``), score is a derived equilibrium
of the events that have landed; the world topology can be reconstructed
deterministically by replaying events on a fresh charter.

Full replay is O(events) and gets slow fast (the equilibrate cost per
event grows with world size — a ~2000-event log took >1h to boot), so
restore works like a chain indexer:

  1. If ``checkpoint.json`` exists and is consistent with the event
     log (same rpb, log holds at least the checkpointed event count),
     restore the world exactly from the checkpoint and replay only the
     tail events past its offset, honoring ``__equilibrate__`` markers.
  2. On ANY checkpoint problem (missing, corrupt, foreign, claims more
     events than the log has), fall back to full replay on a fresh
     charter — the pre-checkpoint behavior, always correct.
  3. The score cache (``scores.json``) remains a display/sanity cache.

The checkpoint fast path is exact, not approximate: node ids are fully
deterministic (content-addressed children, ``root_<tendency_id>``
roots, ``tree_<tendency_id>`` trees) and
``world_model.generalized.serialize`` round-trips every piece of
evolution-relevant state, including score caches (net_score reads in
cyclic co-parented graphs are evaluation-order dependent). The
``tests/test_world_snapshot_roundtrip.py`` harness pins the
checkpoint-equals-full-replay contract on the production rail.

Format choice
-------------

JSON-lines for events, JSON for the score cache. Both are debuggable
by eye. Move to binary once log size demands it; deferred until
measurements say so.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from world_model.generalized import (
    World,
    equilibrate,
    restore_world,
    snapshot_world,
)

from .world_model_substrate.adapter import build_charter_world
from .world_model_substrate.aggregate import apply_events
from .world_model_substrate.events import snapshot_node_scores


logger = logging.getLogger(__name__)


_DEFAULT_DATA_ROOT = Path.home() / ".autonet" / "world"


def _sanitize_rpb(rpb_address: str) -> str:
    """Make an RPB address safe to use as a directory name."""
    s = rpb_address or "default"
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    return s or "default"


@dataclass
class PersistenceConfig:
    rpb_address: str
    data_root: Optional[Path] = None
    snapshot_every_n_events: int = 100
    snapshot_every_seconds: float = 60.0
    embedding_dim: int = 0
    bandwidth: float = 1.5


@dataclass
class RestoredWorld:
    world: World
    events_replayed: int            # total events represented in the world
    from_checkpoint: bool = False
    tail_events: int = 0            # events replayed past the checkpoint


class WorldPersistence:
    """File-backed event log + score snapshot."""

    def __init__(self, config: PersistenceConfig):
        self.config = config
        root = config.data_root or _DEFAULT_DATA_ROOT
        self._dir = Path(root) / _sanitize_rpb(config.rpb_address)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self._dir / "events.jsonl"
        self.scores_path = self._dir / "scores.json"
        self.checkpoint_path = self._dir / "checkpoint.json"
        self.epochs_dir = self._dir / "epochs"
        self.epochs_dir.mkdir(exist_ok=True)
        self._events_handle = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Snapshot I/O
    # ------------------------------------------------------------------

    def write_snapshot(self, world: World, *, events_applied: int) -> Path:
        """Write the equilibrated score cache AND the full world
        checkpoint. The event log stays as-is — it's the source of
        truth, not invalidated by snapshots.
        """
        payload = {
            "schema": 2,
            "rpb_address": self.config.rpb_address,
            "events_applied": int(events_applied),
            "root_scores": dict(world.root_scores()),
            "node_scores": snapshot_node_scores(world),
        }
        with self._lock:
            self._write_json_atomic(payload, self.scores_path, prefix="scores.")
            try:
                self.write_checkpoint(world, events_applied=events_applied)
            except Exception as e:
                # A failed checkpoint must never take down the caller:
                # boot falls back to full replay without it.
                logger.error("checkpoint write failed: %s", e)

        logger.info(
            "score snapshot written: %s (events_applied=%d)",
            self.scores_path, events_applied,
        )
        return self.scores_path

    def write_checkpoint(self, world: World, *, events_applied: int) -> Path:
        """Write the full world state at an event offset (atomic).

        ``try_restore`` uses this as the indexer-style fast path:
        restore the checkpointed world exactly, then replay only the
        events past ``events_applied``.
        """
        payload = {
            "schema": 1,
            "rpb_address": self.config.rpb_address,
            "events_applied": int(events_applied),
            "world": snapshot_world(world),
        }
        with self._lock:
            self._write_json_atomic(
                payload, self.checkpoint_path, prefix="checkpoint.",
            )
        logger.info(
            "world checkpoint written: %s (events_applied=%d)",
            self.checkpoint_path, events_applied,
        )
        return self.checkpoint_path

    def append_events(
        self,
        events: List[Dict[str, Any]],
        *,
        equilibrate_after: bool = True,
        equilibrate_rounds: int = 8,
        equilibrate_tolerance: float = 1e-3,
    ) -> None:
        """Append a batch to the JSONL log. One JSON object per line.

        After the batch's events, write a synthetic ``__equilibrate__``
        marker carrying the parameters used at runtime. Replay reproduces
        the same equilibrate calls — making score history deterministic.
        """
        if not events and not equilibrate_after:
            return
        with self._lock:
            handle = self._open_events_handle()
            for ev in events:
                handle.write(json.dumps(ev) + "\n")
            if equilibrate_after:
                marker = {
                    "kind": "__equilibrate__",
                    "max_rounds": int(equilibrate_rounds),
                    "tolerance": float(equilibrate_tolerance),
                }
                handle.write(json.dumps(marker) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass

    def try_restore(self) -> Optional[RestoredWorld]:
        """Restore the world from disk.

        Fast path: load ``checkpoint.json`` and replay only the events
        past its offset. Falls back to full event-log replay onto a
        fresh charter on any checkpoint inconsistency. Both paths honor
        ``__equilibrate__`` markers between event batches so
        equilibration history (and therefore per-node score) reproduces
        bit-for-bit.

        Returns ``None`` if no event log exists.
        """
        with self._lock:
            log = self._read_events_log()

            checkpoint = self._read_checkpoint()
            if checkpoint is not None:
                try:
                    return self._restore_from_checkpoint(checkpoint, log)
                except Exception as e:
                    logger.warning(
                        "checkpoint restore failed (%s); falling back to "
                        "full event replay", e,
                    )

            if not log:
                return None

            world = build_charter_world(
                bandwidth=self.config.bandwidth,
                embedding_dim=self.config.embedding_dim,
            )

            # Group consecutive events between markers; ``apply_events``
            # runs equilibrate internally so the marker is purely a
            # batch boundary, not an equilibrate trigger.
            batch: List[Dict[str, Any]] = []
            event_count = 0

            def _flush() -> None:
                nonlocal batch
                if batch:
                    apply_events(world, batch)
                    batch = []

            for entry in log:
                if entry.get("kind") == "__equilibrate__":
                    _flush()
                else:
                    batch.append(entry)
                    event_count += 1

            # Trailing events past the last marker (legacy log, or
            # in-progress batch from a crash). Apply once so they land
            # in the world.
            if batch:
                _flush()

            logger.info(
                "restored world from %d events for rpb=%s (full replay)",
                event_count, self.config.rpb_address,
            )
            return RestoredWorld(world=world, events_replayed=event_count)

    def _restore_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        log: List[Dict[str, Any]],
    ) -> RestoredWorld:
        """Exact restore from a checkpoint + tail replay.

        Raises on any inconsistency; the caller falls back to full
        replay. The event log stays authoritative: a checkpoint that
        claims more events than the log holds is treated as foreign.
        """
        offset = int(checkpoint["events_applied"])
        total_events = sum(
            1 for e in log if e.get("kind") != "__equilibrate__"
        )
        if total_events < offset:
            raise ValueError(
                f"event log has {total_events} events but checkpoint "
                f"claims {offset}"
            )

        world = restore_world(checkpoint["world"])
        # Rehydrate the dynamic artifact_digest attribute from its
        # metadata mirror (viz kind tags + ratings-lift ranking would
        # otherwise degrade until the next close).
        from .world_model_substrate.aggregate import lift_artifact_digests
        lift_artifact_digests(world)

        # Tail replay past the checkpoint offset. Markers before the
        # boundary are no-ops (their batches are inside the checkpoint);
        # markers after it flush batches exactly like a full replay.
        batch: List[Dict[str, Any]] = []
        seen = 0
        for entry in log:
            if entry.get("kind") == "__equilibrate__":
                if batch:
                    apply_events(world, batch)
                    batch = []
                continue
            seen += 1
            if seen <= offset:
                continue
            batch.append(entry)
        if batch:
            apply_events(world, batch)

        tail = total_events - offset
        logger.info(
            "restored world from checkpoint at offset %d + %d tail "
            "events for rpb=%s", offset, tail, self.config.rpb_address,
        )
        return RestoredWorld(
            world=world,
            events_replayed=total_events,
            from_checkpoint=True,
            tail_events=tail,
        )

    def _read_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint.json if present, valid, and ours."""
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("unreadable checkpoint %s: %s",
                           self.checkpoint_path, e)
            return None
        if payload.get("schema") != 1:
            logger.warning("unsupported checkpoint schema: %r",
                           payload.get("schema"))
            return None
        if payload.get("rpb_address") != self.config.rpb_address:
            logger.warning(
                "checkpoint rpb mismatch: %r != %r",
                payload.get("rpb_address"), self.config.rpb_address,
            )
            return None
        return payload

    # ------------------------------------------------------------------
    # Epoch records (Phase 4: projection-level surface)
    # ------------------------------------------------------------------

    def write_epoch_record(self, record: Dict[str, Any]) -> Path:
        """Atomically write a closed-epoch record to disk.

        These are projection-level results (computed locally by this
        daemon, not yet federated-authoritative). Phase 5's network
        close will stamp ``authoritative: true`` on the canonical
        version after federation reconciles.
        """
        eid = str(record.get("epoch_id", "unknown"))
        # Sanitize the epoch id same way we sanitize rpb addresses.
        safe_eid = re.sub(r"[^A-Za-z0-9_.-]", "_", eid) or "unknown"
        path = self.epochs_dir / f"{safe_eid}.json"
        with self._lock:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"{safe_eid}.", suffix=".tmp", dir=str(self.epochs_dir),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(record, fh)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        logger.debug("epoch record written: %s", path)
        return path

    def read_epoch_record(self, epoch_id: str) -> Optional[Dict[str, Any]]:
        safe_eid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(epoch_id)) or "unknown"
        path = self.epochs_dir / f"{safe_eid}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("failed to read epoch record %s: %s", path, e)
            return None

    def list_epoch_records(self) -> List[Path]:
        """Sorted list of all on-disk epoch record paths."""
        if not self.epochs_dir.exists():
            return []
        return sorted(self.epochs_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._close_events_handle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_json_atomic(
        self, payload: Dict[str, Any], path: Path, *, prefix: str,
    ) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=prefix, suffix=".json.tmp", dir=str(self._dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _read_events_log(self) -> List[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(self.events_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "skipping malformed event line: %s", e,
                        )
        except OSError as e:
            logger.error("failed to read events log: %s", e)
        return out

    def _open_events_handle(self):
        if self._events_handle is None or self._events_handle.closed:
            self._events_handle = open(
                self.events_path, "a", encoding="utf-8",
            )
        return self._events_handle

    def _close_events_handle(self) -> None:
        if self._events_handle is not None and not self._events_handle.closed:
            try:
                self._events_handle.flush()
                self._events_handle.close()
            except OSError as e:
                logger.warning("error closing events handle: %s", e)
        self._events_handle = None
