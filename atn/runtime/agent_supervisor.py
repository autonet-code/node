"""AgentSupervisor — the single holder of per-agent worker PID handles.

Phase 3 of agent per-PID process isolation. This is the object that, once the
cutover (P4+) lands, OWNS the lifecycle of every worker process: the live
handle map, the kill ladder, the monitor loop (runtime / memory / wedge), the
logical delegate subtree (walked leaf-first), and orphan recovery across daemon
reboots. It is a **per-Runtime instance**, never a global singleton — each
Runtime owns exactly one supervisor so two daemons in one interpreter (tests)
don't cross-reap.

WHAT P3 IS (AND IS NOT)
-----------------------
P3 builds the supervisor and wires it beside ``control`` so it constructs, its
monitor loop can start/stop, and orphan recovery runs at boot. It does NOT move
the cognitive loop into the worker (P4) and does NOT change in-process kill
behavior while the flag is OFF: ``ExecutionControl`` keeps its exact asyncio
task-cancel path unless the flag is ON *and* a worker is actually registered.

Adapted from ``chevin.core.process_monitor`` (register / limit / SIGTERM→wait→
SIGKILL / orphan-scan) but:
  - the monitor is an ASYNCIO TASK, not a background thread (single event loop);
  - we KNOW the PID at spawn, so chevin's racy psutil-diff PID discovery is
    dropped entirely — orphan recovery keys off ``workers/*.pid.json`` written
    at spawn, verified against a ``(pid, create_time, cmdline)`` triple to
    defeat PID reuse;
  - the actual close/kill primitives live in ``WorkerManager`` (P2); the
    supervisor sequences them into the kill ladder and owns the handle map.

SECURITY HANDOFF SEAM
---------------------
Two no-op hooks, injected by Runtime so this module never imports vault code:
``_on_pid_bound(agent_id, pid, parent_agent_id)`` fired after spawn+ready but
BEFORE the worker is told "go"; ``_on_pid_revoked(pid, agent_id)`` fired in ``_reap``.
The vault/secrets track fills them later. They are pure callbacks here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

try:  # psutil is already a dependency (chevin + several nodes use it)
    import psutil  # type: ignore
except Exception:  # pragma: no cover - defensive; monitor degrades gracefully
    psutil = None  # type: ignore

log = logging.getLogger(__name__)


# Type aliases for the injected callables.
FinalizeFn = Callable[..., Awaitable[None]]     # _finalize_execution(defn, record, status, ...)
PidBoundHook = Callable[[str, int, Optional[str]], None]
# (pid, agent_id) — widened for the secret track so the revoke hook can drain the
# per-agent grant stash and drop the agent's monitor tail. ``agent_id`` may be
# None if the PID was never bound to an identity.
PidRevokedHook = Callable[[int, Optional[str]], None]


# Default limits (mirrors chevin's defaults; overridable per-register).
DEFAULT_MAX_RUNTIME_S: Optional[float] = None      # None => no runtime cap
DEFAULT_MAX_MEMORY_MB: Optional[float] = None      # None => no memory cap
DEFAULT_MAX_IDLE_S: float = 300.0                  # heartbeat gap => wedged
DEFAULT_MONITOR_INTERVAL_S: float = 5.0


@dataclass
class SupervisedWorker:
    """One tracked worker process. The supervisor is the ONLY holder of these.

    ``handle`` is the P2 ``WorkerHandle`` (proc + channel + job). ``pid`` is the
    security identity captured race-free at spawn. ``defn`` / ``record`` are the
    plain objects the on-behalf ``_finalize_execution`` needs after a hard kill.
    """
    agent_id: str
    pid: int
    handle: Any                                    # WorkerHandle (P2)
    parent_agent_id: Optional[str] = None
    execution_id: Optional[str] = None
    defn: Any = None
    record: Any = None

    started_at: float = field(default_factory=time.monotonic)
    last_heartbeat: float = field(default_factory=time.monotonic)
    create_time: Optional[float] = None            # psutil create_time() for reuse defense
    cmdline: list[str] = field(default_factory=list)

    max_runtime_s: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_idle_s: float = DEFAULT_MAX_IDLE_S

    is_root: bool = False                          # root agent auto-restarts; delegates don't
    reaped: bool = False

    def runtime_s(self) -> float:
        return time.monotonic() - self.started_at

    def idle_s(self) -> float:
        return time.monotonic() - self.last_heartbeat


class AgentSupervisor:
    """Per-Runtime holder of worker PID handles + kill/monitor/orphan logic.

    Construction is cheap and side-effect free. The monitor loop is only started
    when the flag is ON (Runtime calls ``start_monitor``); ``stop_monitor`` is
    idempotent. With the flag OFF nothing here ever runs against a real process,
    so behavior is unchanged.
    """

    def __init__(
        self,
        *,
        worker_manager: Any = None,
        finalize_fn: Optional[FinalizeFn] = None,
        on_pid_bound: Optional[PidBoundHook] = None,
        on_pid_revoked: Optional[PidRevokedHook] = None,
        workers_dir: Optional[Path] = None,
        monitor_interval_s: float = DEFAULT_MONITOR_INTERVAL_S,
    ) -> None:
        self._workers: dict[str, SupervisedWorker] = {}    # agent_id -> worker
        self._pid_index: dict[int, str] = {}               # pid -> agent_id
        # Logical delegate subtree: parent_agent_id -> set(child_agent_id).
        # NOT the process group — delegates are spawned by the daemon, so a
        # child is NOT in its parent-worker's process group. Walked LEAF-FIRST
        # on a fractal kill.
        self._children: dict[str, set[str]] = {}

        self._worker_manager = worker_manager
        self._finalize_fn = finalize_fn
        # Security seam — default to no-ops so agent_supervisor never needs to
        # import vault code. Runtime injects the real hooks (still no-op in P3).
        self._on_pid_bound: PidBoundHook = on_pid_bound or (lambda aid, pid, parent: None)
        self._on_pid_revoked: PidRevokedHook = on_pid_revoked or (lambda pid, agent_id: None)

        self._workers_dir = workers_dir
        self._monitor_interval_s = monitor_interval_s
        self._monitor_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._lock = asyncio.Lock()

    # ==================================================================
    # Registration
    # ==================================================================

    def register(
        self,
        agent_id: str,
        pid: int,
        handle: Any,
        *,
        parent_agent_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        defn: Any = None,
        record: Any = None,
        is_root: bool = False,
        max_runtime_s: Optional[float] = DEFAULT_MAX_RUNTIME_S,
        max_memory_mb: Optional[float] = DEFAULT_MAX_MEMORY_MB,
        max_idle_s: float = DEFAULT_MAX_IDLE_S,
    ) -> SupervisedWorker:
        """Record a freshly-spawned+ready worker. Captures the (pid, create_time,
        cmdline) triple for reuse-safe orphan recovery, writes the pid file, wires
        the logical child edge, and fires the ``_on_pid_bound`` seam.

        MUST be called AFTER the worker is ready but BEFORE the worker is told
        "go" (the design's PID-bound-before-go ordering) so a secret grant is in
        place before the worker can run any tool.
        """
        create_time, cmdline = self._probe_identity(pid)
        w = SupervisedWorker(
            agent_id=agent_id,
            pid=pid,
            handle=handle,
            parent_agent_id=parent_agent_id,
            execution_id=execution_id,
            defn=defn,
            record=record,
            create_time=create_time,
            cmdline=cmdline,
            is_root=is_root,
            max_runtime_s=max_runtime_s,
            max_memory_mb=max_memory_mb,
            max_idle_s=max_idle_s,
        )
        self._workers[agent_id] = w
        self._pid_index[pid] = agent_id
        if parent_agent_id:
            self._children.setdefault(parent_agent_id, set()).add(agent_id)

        self._write_pidfile(w)

        # Security seam: PID bound → grant, fired BEFORE "go". No-op in P3.
        try:
            self._on_pid_bound(agent_id, pid, parent_agent_id)
        except Exception:
            log.debug("on_pid_bound hook error for %s", agent_id, exc_info=True)

        log.info("supervisor: registered worker agent=%s pid=%s parent=%s",
                 agent_id, pid, parent_agent_id)
        return w

    def get(self, agent_id: str) -> Optional[SupervisedWorker]:
        return self._workers.get(agent_id)

    def get_by_pid(self, pid: int) -> Optional[SupervisedWorker]:
        aid = self._pid_index.get(pid)
        return self._workers.get(aid) if aid else None

    def is_supervised(self, agent_id: str) -> bool:
        """True iff there is a live worker process for this agent. This is the
        gate ExecutionControl uses to decide worker-path vs in-process path."""
        return agent_id in self._workers

    def heartbeat(self, agent_id: str) -> None:
        """Bump the wedge clock. Called on any inbound status/event from the
        worker (P4 wires the pipe reader to this)."""
        w = self._workers.get(agent_id)
        if w is not None:
            w.last_heartbeat = time.monotonic()

    # ==================================================================
    # Kill ladder
    # ==================================================================

    async def kill(self, agent_id: str, *, reason: str = "manual") -> bool:
        """Kill ONE worker (and its logical delegate subtree, leaf-first).

        Ladder per node: cooperative shutdown over IPC → grace → SIGTERM /
        Job-Object → 3s → SIGKILL. Handled by the WorkerManager primitives
        (P2). After the top process dies its ``_reap`` runs the on-behalf
        finalize (if the worker didn't get to run its own finally).
        """
        if agent_id not in self._workers:
            return False
        # LEAF-FIRST: kill the logical delegate subtree before the parent so a
        # child never outlives its parent grant. The Job Object already reaps
        # the VERTICAL tree (worker + its node child); this map covers the
        # LOGICAL subtree of daemon-spawned delegates.
        for child_id in self._subtree_leaf_first(agent_id):
            if child_id == agent_id:
                continue
            await self._kill_one(child_id, reason=reason)
        return await self._kill_one(agent_id, reason=reason)

    async def kill_agent(self, agent_id: str, *, reason: str = "manual") -> bool:
        """Alias mirroring ExecutionControl.kill_agent's per-agent semantics."""
        return await self.kill(agent_id, reason=reason)

    async def kill_all(self, *, reason: str = "kill_all") -> int:
        """Kill every supervised worker. Returns count killed. Roots are killed
        last so a delegate's parent-failure notification still finds its parent
        record while we tear down."""
        # Snapshot: killing mutates the maps.
        agent_ids = list(self._workers.keys())
        # Delegates (non-root) first, roots last.
        agent_ids.sort(key=lambda a: self._workers[a].is_root)
        killed = 0
        for aid in agent_ids:
            if aid in self._workers and await self._kill_one(aid, reason=reason):
                killed += 1
        return killed

    async def _kill_one(self, agent_id: str, *, reason: str) -> bool:
        w = self._workers.get(agent_id)
        if w is None:
            return False
        mgr = self._worker_manager
        try:
            if mgr is not None:
                # Rung 1+2: cooperative shutdown over IPC, grace wait.
                exited = await mgr.graceful_close(w.handle)
                # Rung 3+4: SIGTERM/Job-Object → 3s → SIGKILL if still alive.
                if not exited:
                    await mgr.hard_kill(w.handle)
        except Exception:
            log.warning("supervisor: kill of %s (pid=%s) raised; reaping anyway",
                        agent_id, w.pid, exc_info=True)
        await self._reap(w.pid, reason=reason)
        return True

    # ==================================================================
    # Reap + on-behalf finalize
    # ==================================================================

    async def _reap(self, pid: int, *, reason: str = "exited") -> None:
        """Remove a dead worker from tracking, fire the revoke seam, delete its
        pid file, and — if it died WITHOUT running its own finally — run the
        completion bookkeeping on its behalf via the injected ``_finalize_fn``
        (the P0 extraction). Idempotent: double-reap is a no-op."""
        agent_id = self._pid_index.get(pid)
        if agent_id is None:
            return
        w = self._workers.get(agent_id)
        if w is None or w.reaped:
            return
        w.reaped = True

        # Security seam: PID revoked → drop grant. Widened to (pid, agent_id) so
        # the secret track can release the broker session, drop the monitor
        # exposure, and drain the stale grant stash keyed by agent_id. agent_id
        # is the pipe-bound identity captured at register (may be None).
        try:
            self._on_pid_revoked(pid, agent_id)
        except Exception:
            log.debug("on_pid_revoked hook error for pid=%s", pid, exc_info=True)

        # On-behalf finalize: only when the worker did NOT get to run its own
        # finally (hard kill / crash). The flag is w.record being non-terminal.
        # In P3 no real worker runs the cognitive loop, so record is typically
        # None and this is skipped; wired for real in P4.
        try:
            await self._finalize_on_behalf(w, reason=reason)
        except Exception:
            log.warning("supervisor: on-behalf finalize failed for %s", agent_id,
                        exc_info=True)

        # Drop from all maps.
        self._workers.pop(agent_id, None)
        self._pid_index.pop(pid, None)
        self._children.pop(agent_id, None)
        for kids in self._children.values():
            kids.discard(agent_id)
        self._delete_pidfile(agent_id)
        log.info("supervisor: reaped worker agent=%s pid=%s reason=%s",
                 agent_id, pid, reason)

    async def _finalize_on_behalf(self, w: SupervisedWorker, *, reason: str) -> None:
        """Run ``_finalize_execution`` for a worker that died without its finally.

        Guarded three ways so it is a strict NO-OP in P3 (and safe in P4):
          - no ``_finalize_fn`` injected → skip;
          - no ``defn``/``record`` captured at register → skip (P3: register is
            called without them until the P4 cutover);
          - record already terminal (worker ran its own finally, then the pipe
            EOF'd) → skip, so we never double-emit a terminal event.
        """
        if self._finalize_fn is None or w.defn is None or w.record is None:
            return
        record = w.record
        status = getattr(record, "status", None)
        # Import here to avoid a hard dependency / import cycle at module load.
        try:
            from ..models import ExecutionStatus
        except Exception:
            return
        terminal = {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.KILLED,
        }
        if status in terminal:
            # Worker already finalized itself; EOF is just the process leaving.
            return
        # The worker was hard-killed mid-flight → mark KILLED and finalize.
        # recorded_inflight/accumulated_tool_calls are gone (lived in the killed
        # worker's memory) → pass None so the supervisor books the full
        # remainder from record.token_usage and degrades the run summary to
        # status-only. See P0's design-risk note.
        await self._finalize_fn(
            w.defn,
            record,
            ExecutionStatus.KILLED,
            error=f"worker killed by supervisor ({reason})",
            recorded_inflight=None,
            accumulated_tool_calls=None,
        )

    # ==================================================================
    # Logical subtree (leaf-first)
    # ==================================================================

    def _subtree_leaf_first(self, agent_id: str) -> list[str]:
        """Return [descendants..., agent_id] deepest-first via post-order DFS.

        Uses the ``_children`` map (logical delegate edges). Cycle-safe via a
        visited set. The returned order guarantees a child appears before its
        parent, so callers kill leaves first."""
        order: list[str] = []
        visited: set[str] = set()

        def _visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in sorted(self._children.get(node, set())):
                _visit(child)
            order.append(node)

        _visit(agent_id)
        return order

    # ==================================================================
    # Monitor loop (asyncio task)
    # ==================================================================

    def start_monitor(self) -> None:
        """Start the async monitor loop. Idempotent; only meaningful when the
        flag is ON (Runtime gates the call). No-op if already running."""
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._stopping = False
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="agent-supervisor-monitor")

    async def stop_monitor(self) -> None:
        """Stop the monitor loop. Idempotent."""
        self._stopping = True
        task = self._monitor_task
        self._monitor_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _monitor_loop(self) -> None:
        while not self._stopping:
            try:
                await self._monitor_pass()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("supervisor monitor pass error", exc_info=True)
            try:
                await asyncio.sleep(self._monitor_interval_s)
            except asyncio.CancelledError:
                raise

    async def _monitor_pass(self) -> None:
        """One sweep: enforce runtime / memory caps, detect wedged (no-heartbeat)
        workers and dead processes. Kills offenders via the ladder."""
        for agent_id, w in list(self._workers.items()):
            if w.reaped:
                continue
            proc = getattr(w.handle, "proc", None)
            # Process already exited on its own → reap on-behalf.
            if proc is not None and proc.poll() is not None:
                await self._reap(w.pid, reason="process_exited")
                continue
            # Runtime cap.
            if w.max_runtime_s and w.runtime_s() > w.max_runtime_s:
                log.warning("supervisor: agent %s exceeded max_runtime %.0fs — killing",
                            agent_id, w.max_runtime_s)
                await self.kill(agent_id, reason="max_runtime")
                continue
            # Wedge detection (no heartbeat).
            if w.max_idle_s and w.idle_s() > w.max_idle_s:
                log.warning("supervisor: agent %s wedged (no heartbeat %.0fs) — killing",
                            agent_id, w.idle_s())
                await self.kill(agent_id, reason="wedged")
                continue
            # Memory cap (psutil).
            if w.max_memory_mb and psutil is not None:
                try:
                    rss_mb = psutil.Process(w.pid).memory_info().rss / (1024 * 1024)
                except Exception:
                    rss_mb = 0.0
                if rss_mb > w.max_memory_mb:
                    log.warning("supervisor: agent %s exceeded memory %.0fMB (%.0fMB) — killing",
                                agent_id, w.max_memory_mb, rss_mb)
                    await self.kill(agent_id, reason="max_memory")

    # ==================================================================
    # Orphan recovery (across daemon reboots)
    # ==================================================================

    def _pidfile_path(self, agent_id: str) -> Optional[Path]:
        if self._workers_dir is None:
            return None
        # agent_id may contain path-unfriendly chars; keep it simple — agent ids
        # are already filesystem-safe elsewhere in the codebase.
        return self._workers_dir / f"{agent_id}.pid.json"

    def _write_pidfile(self, w: SupervisedWorker) -> None:
        path = self._pidfile_path(w.agent_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "agent_id": w.agent_id,
                "pid": w.pid,
                "parent_agent_id": w.parent_agent_id,
                "execution_id": w.execution_id,
                "create_time": w.create_time,
                "cmdline": w.cmdline,
                "is_root": w.is_root,
                "written_at": time.time(),
            }), encoding="utf-8")
        except Exception:
            log.debug("supervisor: failed to write pidfile for %s", w.agent_id,
                      exc_info=True)

    def _delete_pidfile(self, agent_id: str) -> None:
        path = self._pidfile_path(agent_id)
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _probe_identity(pid: int) -> tuple[Optional[float], list[str]]:
        """Best-effort (create_time, cmdline) for reuse-safe orphan matching."""
        if psutil is None:
            return None, []
        try:
            p = psutil.Process(pid)
            return p.create_time(), list(p.cmdline() or [])
        except Exception:
            return None, []

    def recover_orphans(self) -> list[str]:
        """Reap workers left behind by a previous daemon that crashed.

        Reads ``workers/*.pid.json``, and for each recorded worker verifies the
        ``(pid, create_time, cmdline)`` triple against the LIVE process:
          - if the PID is dead → stale file, just delete it (the execution is
            recovered as FAILED by ``ExecutionLog.recover_running`` via the
            ``.running.json`` path — we do NOT emit a terminal event here, to
            avoid double-emitting);
          - if the PID is alive AND the triple matches → a TRUE orphan from the
            crashed daemon → kill it out-of-band and delete the file;
          - if the PID is alive but the triple does NOT match → PID was reused by
            an unrelated process → leave it alone, delete the stale file.

        No-op (returns []) when there is no workers dir or no pid files. This
        never runs a worker's finally — the on-behalf finalize belongs to the
        LIVE supervisor's ``_reap``; recovered-crashed executions are marked
        FAILED by the existing ``recover_running`` path, and the two are
        coordinated so exactly one terminal event fires (recover_running owns
        it; this path only reaps OS processes + files).

        Returns the list of agent_ids whose orphan process was actually killed.
        """
        if self._workers_dir is None or not self._workers_dir.exists():
            return []
        killed: list[str] = []
        for path in sorted(self._workers_dir.glob("*.pid.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("supervisor: corrupt pidfile %s, removing", path)
                self._safe_unlink(path)
                continue
            pid = data.get("pid")
            agent_id = data.get("agent_id") or path.stem.replace(".pid", "")
            if not isinstance(pid, int):
                self._safe_unlink(path)
                continue
            if self._is_true_orphan(pid, data):
                if self._kill_orphan_pid(pid):
                    killed.append(agent_id)
                    log.warning("supervisor: killed orphan worker agent=%s pid=%s "
                                "from previous daemon", agent_id, pid)
            # In every branch the file is stale (this daemon didn't spawn it) →
            # remove so we don't re-process it next boot.
            self._safe_unlink(path)
        return killed

    @staticmethod
    def _is_true_orphan(pid: int, data: dict) -> bool:
        """True iff a LIVE process with this pid matches the recorded triple.

        Defeats PID reuse: a matching create_time (and, defensively, cmdline
        containing ``atn.agent_worker``) means this is the same process we
        spawned, now orphaned. A mismatch means the PID was recycled by an
        unrelated process — must NOT be killed."""
        if psutil is None:
            return False
        try:
            p = psutil.Process(pid)
        except Exception:
            return False  # dead → not an orphan to kill (stale file only)
        rec_ct = data.get("create_time")
        if rec_ct is not None:
            try:
                if abs(p.create_time() - float(rec_ct)) > 1.0:
                    return False  # PID reused
            except Exception:
                return False
        # Defensive cmdline check: must still look like our worker.
        try:
            live_cmd = " ".join(p.cmdline() or [])
        except Exception:
            live_cmd = ""
        rec_cmd = " ".join(data.get("cmdline") or [])
        if rec_cmd and "atn.agent_worker" in rec_cmd:
            if "atn.agent_worker" not in live_cmd:
                return False  # PID reused by something else
        return True

    @staticmethod
    def _kill_orphan_pid(pid: int) -> bool:
        """Out-of-band kill of a bare PID (no handle survived the reboot)."""
        if psutil is None:
            return False
        try:
            p = psutil.Process(pid)
        except Exception:
            return False
        try:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
            return True
        except Exception:
            log.debug("supervisor: failed to kill orphan pid=%s", pid, exc_info=True)
            return False

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


__all__ = ["AgentSupervisor", "SupervisedWorker"]
