"""secret_audit — the names-only secret-access audit trail.

The broker deliberately never logs accesses; the daemon is where "which agent
used which secret, when" is recorded. Three hook points feed this log (all in
``runtime/__init__.py``):

  * ``granted`` — a broker session was minted for a worker PID (Seam B,
    ``_on_pid_bound``): the agent CAN now request these services.
  * ``staged``  — the broker actually decrypted + staged one secret for the
    worker, observed via the owner-authenticated value-push
    (``_value_push_sink``). Push receipt == access: the broker fails staging
    closed if the push does not land, so every stage produces a row.
  * ``revoked`` — the worker was reaped and its session torn down
    (``_on_pid_revoked``).

Tool-bound sessions (docs/tool_secret_binding.md) record ``granted`` /
``revoked`` the same way, but carry a ``tool`` digest: the secret was bound to
a TOOL subprocess, not to the agent's own worker. That field is what makes the
log able to answer "what consumed this secret", which the PID-only rows cannot
— a worker that reads its staged file ten thousand times still produces exactly
one ``staged`` row.

Owner mutations from the WS surface (``secrets_put`` / ``secrets_delete``)
also record here (``added`` / ``rotated`` / ``deleted``) so the log is the
single timeline of everything that happened to the vault.

Data-flow discipline (same contract as security_monitor):
  * Rows carry secret NAMES only — NEVER a value, NEVER a staged path (the
    path is a live scan token; logging it would hand a reader the tripwire's
    needle).
  * The emitted ``SECRET_ACCESS`` event carries the agent as ``source`` and
    NO ``agent_id`` key in ``data`` (the subtree-scoping trap; see
    security_monitor.py).

Thread model: ``record()`` is called from the asyncio loop (WS handlers), the
``atn-vpush`` listener thread (staged), and whatever thread the supervisor
fires the bind/revoke hooks on. The JSONL append is guarded by a lock; the
event emit hops to the loop bound via ``bind_loop()`` (called in
``Runtime.start()``) with ``run_coroutine_threadsafe``. No loop bound yet =>
the row is still written, only the live event is skipped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..events import Event, EventBus, EventType

log = logging.getLogger(__name__)

_FILENAME = "secret_access.jsonl"

# Actions recorded by the runtime hooks / WS surface. Free-form by design —
# consumers filter on the string — but keep this list in sync with the docstring.
KNOWN_ACTIONS = ("granted", "staged", "revoked", "added", "rotated", "deleted")


class SecretAuditLog:
    """Append-only JSONL audit trail + live SECRET_ACCESS event emitter."""

    def __init__(self, events: EventBus, data_dir: Path) -> None:
        self._events = events
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._path: Optional[Path] = None
        try:
            sec_dir = Path(data_dir) / "security"
            sec_dir.mkdir(parents=True, exist_ok=True)
            self._path = sec_dir / _FILENAME
        except OSError:
            log.warning("secret_audit: could not create security dir under %s",
                        data_dir, exc_info=True)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the runtime's asyncio loop so off-loop record() calls can emit
        the live event. Called once from Runtime.start()."""
        self._loop = loop

    # -- write path -----------------------------------------------------------
    def record(self, action: str, *, agent_id: str,
               services: list[str], pid: Optional[int] = None,
               tool: Optional[str] = None) -> None:
        """Append one names-only row and emit a live SECRET_ACCESS event.

        ``tool`` is the manifest digest when the secret was bound to a TOOL
        subprocess rather than the agent's worker — the call-site dimension.
        Truncated to 16 hex chars: enough to identify the tool in the UI,
        short enough to keep rows small.

        Best-effort at every step: an audit failure must never break the
        grant/stage/revoke path it is observing.
        """
        services = [str(s) for s in (services or []) if s]
        row: dict[str, Any] = {
            "ts": time.time(),
            "action": str(action),
            "agent_id": str(agent_id or ""),
            "services": services,
        }
        if pid is not None:
            try:
                row["pid"] = int(pid)
            except (TypeError, ValueError):
                pass
        if tool:
            row["tool"] = str(tool)[:16]

        if self._path is not None:
            try:
                line = json.dumps(row, separators=(",", ":"))
                with self._lock:
                    with open(self._path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
            except OSError:
                log.debug("secret_audit: append failed", exc_info=True)

        self._emit(row)

    def _emit(self, row: dict[str, Any]) -> None:
        """Emit SECRET_ACCESS on the bound loop, from any thread.

        The agent travels as ``source``; data carries NO agent_id key.
        """
        event = Event(
            type=EventType.SECRET_ACCESS,
            source=row.get("agent_id") or "runtime",
            data={
                "action": row["action"],
                "services": row["services"],
                "pid": row.get("pid"),
                "tool": row.get("tool"),
                "ts": row["ts"],
            },
        )
        coro = self._events.emit(event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is not None:
                running.create_task(coro)
            elif self._loop is not None and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                coro.close()  # no loop yet (boot) — row is on disk, drop the event
        except Exception:  # noqa: BLE001 — audit must never break the hot path
            log.debug("secret_audit: event emit failed", exc_info=True)

    # -- read path ------------------------------------------------------------
    def tail(self, limit: int = 200, *, agent_id: Optional[str] = None,
             service: Optional[str] = None) -> list[dict[str, Any]]:
        """The most recent rows (newest last), optionally filtered."""
        if self._path is None or not self._path.exists():
            return []
        try:
            limit = max(1, min(int(limit), 2000))
        except (TypeError, ValueError):
            limit = 200
        rows: list[dict[str, Any]] = []
        try:
            with self._lock:
                lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if agent_id and row.get("agent_id") != agent_id:
                continue
            if service and service not in (row.get("services") or []):
                continue
            rows.append(row)
        return rows[-limit:]
