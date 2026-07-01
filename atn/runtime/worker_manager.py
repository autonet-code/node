"""Spawn + handshake for per-agent worker processes (Phase 2, behind a flag).

This is the daemon-side spawner for agent per-PID process isolation. It brings
the P1 *dark library* (``atn.agent_worker`` + ``atn.runtime.worker_rpc``) to
life: given an agent_id + a non-secret manifest, it launches ONE OS process
running ``python -m atn.agent_worker`` over a single private inherited-handle
duplex pipe, completes the ready handshake, and returns a handle the caller can
gracefully close or hard-kill.

WHAT P2 IS (AND IS NOT)
-----------------------
P2 proves a WORKING spawn + handshake + kill behind ``ATN_WORKER_ISOLATION``
(default OFF). It does NOT move the cognitive loop into the worker (that is P4).
With the flag OFF the daemon never calls this module. With the flag ON,
``trigger_run`` spawns a worker, awaits ``ready``, then IMMEDIATELY shuts it back
down and falls through to the existing in-process task — a liveness proof, not a
cutover.

SECURITY INVARIANTS (locked design)
-----------------------------------
- ONE private duplex pipe per worker (multiprocessing.Pipe). The worker's end is
  made inheritable and passed by numeric handle on argv; on Windows it is placed
  on the STARTUPINFO explicit handle-inherit list. The daemon closes its copy of
  the worker end after spawn so EOF propagates.
- Identity is pipe-bound on the daemon side (WorkerChannel holds the
  authoritative agent_id). NOTHING secret ever goes into the worker's env or
  argv — only the pipe handle and a logging label. Env is a SCRUBBED copy.
- Windows: CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, and the child is assigned
  to a kill-on-close Win32 Job Object so closing the job handle reaps the whole
  tree (worker + any node child it later spawns). Optional memory cap via the
  job. The whole memory-isolation argument depends on the worker running with a
  RESTRICTED / non-elevated token — see ``scripts/check_worker_isolation.py``
  for the go/no-go elevation self-check on the real host.
- POSIX: start_new_session=True (new process group for group-kill). No Job
  Object — that gap is noted; group-kill covers the tree.

IPC uses ONLY length-prefixed JSON via send_bytes/recv_bytes (see worker_rpc);
pickle over the pipe is forbidden (RCE surface).
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .worker_rpc import (
    Envelope,
    PipeTransport,
    PROTO_VERSION,
    WorkerChannel,
)

log = logging.getLogger(__name__)

_IS_WIN = sys.platform == "win32"

# Grace windows for the kill ladder (seconds).
DEFAULT_READY_TIMEOUT = 15.0
DEFAULT_SHUTDOWN_GRACE = 3.0
DEFAULT_TERM_GRACE = 3.0


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------

@dataclass
class WorkerHandle:
    """Everything the supervisor needs to talk to / reap ONE worker.

    ``agent_id`` is the AUTHORITATIVE identity (also carried by ``channel``).
    ``pid`` is captured race-free from ``proc.pid`` at spawn — it is the future
    security identity for PID-bound secret access.
    """
    agent_id: str
    pid: int
    proc: subprocess.Popen
    channel: WorkerChannel
    transport: PipeTransport
    job: Optional[int] = None          # Win32 Job Object handle (int), else None
    ready: bool = False
    manifest: dict[str, Any] = field(default_factory=dict)
    _daemon_conn: Any = None           # daemon end of the pipe (kept for close)


# ---------------------------------------------------------------------------
# Win32 Job Object helpers (kill-on-close, optional memory cap)
# ---------------------------------------------------------------------------

if _IS_WIN:
    import ctypes.wintypes as _wt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0100
    _JobObjectExtendedLimitInformation = 9
    _HANDLE_FLAG_INHERIT = 0x1

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", _wt.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", _wt.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", _wt.DWORD),
            ("SchedulingClass", _wt.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _create_kill_on_close_job(memory_cap_mb: int = 0) -> Optional[int]:
        """Create a Job Object that kills all assigned processes when its last
        handle closes. Returns the job handle (int) or None on failure."""
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            log.warning("CreateJobObjectW failed: %s", ctypes.get_last_error())
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_cap_mb and memory_cap_mb > 0:
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_cap_mb * 1024 * 1024
        info.BasicLimitInformation.LimitFlags = flags
        ok = _kernel32.SetInformationJobObject(
            _wt.HANDLE(job),
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            log.warning("SetInformationJobObject failed: %s", ctypes.get_last_error())
            _kernel32.CloseHandle(_wt.HANDLE(job))
            return None
        return int(job)

    def _assign_process_to_job(job: int, proc_handle: int) -> bool:
        ok = _kernel32.AssignProcessToJobObject(_wt.HANDLE(job), _wt.HANDLE(proc_handle))
        if not ok:
            log.warning("AssignProcessToJobObject failed: %s", ctypes.get_last_error())
        return bool(ok)

    def _close_handle(h: int) -> None:
        try:
            _kernel32.CloseHandle(_wt.HANDLE(h))
        except Exception:
            pass

    def _make_inheritable(handle: int) -> None:
        ok = _kernel32.SetHandleInformation(
            _wt.HANDLE(handle), _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())


# ---------------------------------------------------------------------------
# Env scrub
# ---------------------------------------------------------------------------

# Substrings that mark an env var as sensitive. A worker must never inherit any
# secret in its environment — secret access (P4+) is mediated by the daemon over
# an authority RPC, gated by the recorded PID. This is a defense-in-depth scrub;
# the design already guarantees no secret is passed on argv.
_SECRET_MARKERS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "PRIVATE", "CREDENTIAL",
    "SEED", "MNEMONIC", "SIGNING", "AUTH", "SESSION", "COOKIE",
)


def _scrub_env(base: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return a copy of the environment with anything that looks secret removed.

    Keeps PATH/PYTHONPATH/SystemRoot etc. so the interpreter can start. Drops
    any var whose NAME contains a secret marker. Conservative by design: better
    to drop a benign var than leak a secret into a lower-trust process.
    """
    src = dict(base if base is not None else os.environ)
    out: dict[str, str] = {}
    for k, v in src.items():
        upper = k.upper()
        if any(marker in upper for marker in _SECRET_MARKERS):
            continue
        out[k] = v
    # The SDK nesting guard must be cleared (matches bridge.py behaviour).
    out.pop("CLAUDECODE", None)
    return out


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------

class WorkerManager:
    """Spawns + handshakes per-agent worker processes.

    Stateless w.r.t. lifecycle bookkeeping: it returns a ``WorkerHandle`` and
    the caller (P3's ``AgentSupervisor``) owns the map of live handles and the
    kill ladder policy. The manager only knows how to (a) spawn+handshake and
    (b) execute the primitive close/kill operations on one handle.
    """

    def __init__(self, memory_cap_mb: int = 0) -> None:
        self._memory_cap_mb = memory_cap_mb

    # -- spawn + handshake -------------------------------------------------
    async def ensure_worker(
        self,
        agent_id: str,
        manifest: Optional[dict[str, Any]] = None,
        *,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
        rpc_handlers: Optional[dict] = None,
        event_sink: Optional[Any] = None,
        status_sink: Optional[Any] = None,
    ) -> WorkerHandle:
        """Spawn one worker for ``agent_id``, complete the ready handshake, and
        return its handle. Raises on spawn failure or handshake timeout (and
        reaps the half-started process before raising)."""
        manifest = dict(manifest or {})
        manifest.setdefault("agent_label", agent_id)
        manifest["v"] = PROTO_VERSION

        daemon_conn, worker_conn = mp.Pipe(duplex=True)
        worker_handle = worker_conn.fileno()

        job: Optional[int] = None
        proc: Optional[subprocess.Popen] = None
        # A future set by the status sink wrapper when "ready" arrives.
        ready_evt = asyncio.Event()
        ready_payload: dict[str, Any] = {}

        async def _status_sink(aid: str, name: str, payload: dict) -> None:
            if name == "ready":
                ready_payload.update(payload or {})
                ready_evt.set()
            if status_sink is not None:
                await status_sink(aid, name, payload)

        try:
            if _IS_WIN:
                _make_inheritable(worker_handle)
                job = _create_kill_on_close_job(self._memory_cap_mb)

            kwargs: dict[str, Any] = {}
            if _IS_WIN:
                kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
                si = subprocess.STARTUPINFO()
                # Only the pipe handle is inheritable AND on the inherit list;
                # every other handle stays out of the child.
                si.lpAttributeList = {"handle_list": [worker_handle]}
                kwargs["startupinfo"] = si
            else:
                kwargs["start_new_session"] = True  # new pgid for group-kill

            argv = [
                sys.executable, "-u", "-m", "atn.agent_worker",
                "--pipe-handle", str(worker_handle),
                "--agent-label", str(agent_id),
            ]
            # close_fds=False is REQUIRED so the inherited handle/fd survives;
            # on Windows the handle_list constrains inheritance to just the pipe.
            proc = subprocess.Popen(
                argv,
                close_fds=False,
                env=_scrub_env(),
                stdin=subprocess.DEVNULL,
                **kwargs,
            )
            pid = proc.pid  # race-free capture at spawn

            if _IS_WIN and job is not None:
                # proc._handle is the process HANDLE (int) on Windows.
                phandle = int(getattr(proc, "_handle", 0) or 0)
                if phandle:
                    _assign_process_to_job(job, phandle)

            # Close our copy of the WORKER end so EOF propagates on worker exit.
            worker_conn.close()

            transport = PipeTransport(daemon_conn)
            channel = WorkerChannel(
                agent_id, transport,
                rpc_handlers=rpc_handlers,
                event_sink=event_sink,
                status_sink=_status_sink,
            )
            channel.start()

            # Handshake: manifest -> ready.
            await channel.send_cmd("manifest", manifest)
            try:
                await asyncio.wait_for(ready_evt.wait(), timeout=ready_timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"worker {agent_id} (pid={pid}) did not send ready within "
                    f"{ready_timeout}s"
                )
            if proc.poll() is not None:
                raise RuntimeError(
                    f"worker {agent_id} exited during handshake rc={proc.returncode}"
                )

            handle = WorkerHandle(
                agent_id=agent_id,
                pid=pid,
                proc=proc,
                channel=channel,
                transport=transport,
                job=job,
                ready=True,
                manifest=manifest,
                _daemon_conn=daemon_conn,
            )
            log.info("worker ready: agent=%s pid=%s", agent_id, pid)
            return handle

        except BaseException:
            # Half-started: reap everything we created before re-raising.
            try:
                worker_conn.close()
            except Exception:
                pass
            if proc is not None:
                await self._hard_kill_proc(proc, job)
            elif job is not None and _IS_WIN:
                _close_handle(job)
            try:
                daemon_conn.close()
            except Exception:
                pass
            raise

    # -- graceful close ----------------------------------------------------
    async def graceful_close(
        self, handle: WorkerHandle, *, grace: float = DEFAULT_SHUTDOWN_GRACE
    ) -> bool:
        """Ask the worker to shut down cooperatively, wait ``grace`` seconds,
        then return whether it exited on its own. Does NOT hard-kill — that is
        the caller's next rung. Returns True if the process exited."""
        try:
            await handle.channel.send_cmd("shutdown", {})
        except Exception:
            log.debug("shutdown cmd send failed for %s (pipe likely gone)", handle.agent_id)
        exited = await self._wait_exit(handle.proc, grace)
        if exited:
            await self._cleanup_handle(handle)
        return exited

    # -- hard kill ---------------------------------------------------------
    async def hard_kill(self, handle: WorkerHandle) -> None:
        """Out-of-band PID kill (a wedged worker won't read a shutdown cmd).

        Windows: closing the kill-on-close Job Object reaps the whole tree; we
        also TerminateProcess the top process. POSIX: SIGTERM the process group,
        grace, then SIGKILL the group.
        """
        await self._hard_kill_proc(handle.proc, handle.job)
        await self._cleanup_handle(handle)

    # -- primitives --------------------------------------------------------
    async def _hard_kill_proc(
        self, proc: subprocess.Popen, job: Optional[int]
    ) -> None:
        if proc.poll() is not None:
            if _IS_WIN and job is not None:
                _close_handle(job)
            return
        if _IS_WIN:
            # Terminate the top process; the Job Object reaps descendants when
            # its handle closes.
            try:
                proc.terminate()
            except Exception:
                pass
            exited = await self._wait_exit(proc, DEFAULT_TERM_GRACE)
            if not exited:
                try:
                    proc.kill()
                except Exception:
                    pass
                await self._wait_exit(proc, DEFAULT_TERM_GRACE)
            if job is not None:
                # Closing the last job handle kills anything still assigned.
                _close_handle(job)
        else:
            pgid = None
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = None
            self._posix_signal_group(proc, pgid, signal.SIGTERM)
            exited = await self._wait_exit(proc, DEFAULT_TERM_GRACE)
            if not exited:
                self._posix_signal_group(proc, pgid, signal.SIGKILL)
                await self._wait_exit(proc, DEFAULT_TERM_GRACE)

    @staticmethod
    def _posix_signal_group(proc: subprocess.Popen, pgid: Optional[int], sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError:
            pass
        except Exception:
            log.debug("posix group signal %s failed", sig, exc_info=True)

    @staticmethod
    async def _wait_exit(proc: subprocess.Popen, timeout: float) -> bool:
        """Await process exit up to ``timeout`` without blocking the loop."""
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return proc.poll() is not None
        except Exception:
            return proc.poll() is not None

    async def _cleanup_handle(self, handle: WorkerHandle) -> None:
        try:
            await handle.channel.close()
        except Exception:
            pass
        if _IS_WIN and handle.job is not None:
            _close_handle(handle.job)
            handle.job = None
        if handle._daemon_conn is not None:
            try:
                handle._daemon_conn.close()
            except Exception:
                pass


__all__ = ["WorkerManager", "WorkerHandle"]
