"""PIDs of processes the daemon spawned to run lower-trust code.

The local owner listener (:7700) pre-auths every connection as the OWNER, so
no process the daemon itself started to run a tool or an agent's command may
be served there. ws_auth.local_peer_denied refuses a client whose owning PID
is tracked here, or is a descendant of a tracked PID.

What gets tracked, each for the life of its run:

- guest tools (origin="integration", atn/guest_sandbox.py, same-uid path),
- authored and adopted pinned tools (atn/tool_store.py),
- agent shell commands run in the daemon process (atn/shell_tools.py),
- agent worker processes (atn/runtime/worker_manager.py); their own shell
  commands are descendants of the worker, so they are covered by the
  descendant walk.

Owner clients (the app, ``atn send``, the MCP proxy started by the user's
own harness) are not spawned by the daemon and are never tracked.

Known limit: a descendant that outlives its tracked ancestor (detached and
reparented) is no longer linked to it by ``parents()`` once the ancestor is
untracked or gone.

Kept dependency-free (psutil only, imported lazily) so atn/shell_tools.py
can import it without pulling in the daemon.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_PIDS: dict[int, str] = {}
# Start time per PID, so a stale entry (a run whose cleanup never ran) never
# matches an unrelated process that later reuses the PID.
_STARTED: dict[int, float] = {}
_LOCK = threading.Lock()


def _create_time(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001
        return None


def register(pid: int, kind: str = "tool") -> None:
    started = _create_time(int(pid))
    with _LOCK:
        _PIDS[int(pid)] = str(kind)
        if started is not None:
            _STARTED[int(pid)] = started


def unregister(pid: int) -> None:
    with _LOCK:
        _PIDS.pop(int(pid), None)
        _STARTED.pop(int(pid), None)


@contextmanager
def tracked(pid: int, kind: str) -> Iterator[None]:
    register(pid, kind)
    try:
        yield
    finally:
        unregister(pid)


def live() -> dict[int, str]:
    with _LOCK:
        return dict(_PIDS)


def kind_of(pid: int | None) -> str | None:
    """The kind of the tracked process ``pid`` is, or descends from; None if
    it is neither (or is gone)."""
    if pid is None:
        return None
    with _LOCK:
        pids = dict(_PIDS)
        started = dict(_STARTED)
    if not pids:
        return None

    def _same(p) -> bool:
        want = started.get(p.pid)
        if want is None:
            return True
        try:
            return abs(p.create_time() - want) < 1.0
        except Exception:  # noqa: BLE001
            return True  # unreadable: keep the match (fail closed)

    try:
        import psutil
        proc = psutil.Process(pid)
    except Exception:  # noqa: BLE001 — process gone / no access
        return pids.get(pid)
    for p in (proc, *_parents(proc)):
        if p.pid in pids and _same(p):
            return pids[p.pid]
    return None


def _parents(proc) -> list:
    try:
        return proc.parents()
    except Exception:  # noqa: BLE001
        return []


def tree_owner_of(match) -> tuple[bool, str | None]:
    """Scan the tracked processes and their current descendants for a TCP
    connection satisfying ``match(conn)``.

    Returns (complete, kind): kind is the tracked kind owning a matching
    socket, or None; complete is False if any tracked tree could not be
    fully read (the caller then fails closed). Used when the system-wide
    socket table cannot attribute a socket to a PID (macOS without root),
    since a process's own children are readable without privileges."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return False, None
    complete = True
    for pid, kind in live().items():
        try:
            root = psutil.Process(pid)
            procs = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            continue
        except Exception:  # noqa: BLE001
            complete = False
            continue
        for proc in procs:
            try:
                getter = getattr(proc, "net_connections", None) or proc.connections
                conns = getter(kind="tcp")
            except psutil.NoSuchProcess:
                continue
            except Exception:  # noqa: BLE001
                complete = False
                continue
            if any(match(c) for c in conns):
                return True, kind
    return complete, None
