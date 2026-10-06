"""Containment for GUEST-authored tools (origin="integration").

A tool registered over the integration listener (docs/integration_listener.md,
"Guest tool authoring") was written by a guest harness holding nothing but a
bearer token. Its code is untrusted, so the record carries
``origin="integration"`` and EVERY execution of it, by any caller, goes
through this module instead of the authored/adopted paths in ToolStore:

1. **Separate uid** (strongest, POSIX). When the daemon was started by
   ``atn.guest_launcher`` (the Docker image does this), the run is handed to
   the root launcher over a private socketpair and executes as an
   unprivileged guest uid taken from a pool, one uid per live run: it cannot
   read the data dir (0700, daemon uid), cannot read the daemon's
   /proc/<pid>/environ, cannot signal the daemon or another run, and the
   local owner listener refuses every pool uid (ws_auth.local_peer_denied).
   After the run every process of that uid is killed before the uid is
   reused. With ``ATN_GUEST_REQUIRE_UID=1`` (set by the launcher) a missing
   launcher means the tool is REFUSED, never run weaker.
2. **Same-uid fallback** (dev boxes, Windows). OFF unless the operator sets
   ``ATN_GUEST_ALLOW_SAME_UID=1``; without the launcher and without that
   opt-in, guest tools are refused. Native code (ctypes) gets past the
   audit hook, so on this path a guest tool can act as the daemon user.
   When opted in: a fresh sandbox directory
   outside the data dir, a scrubbed environment built from scratch (no daemon
   variables, no ``capabilities.env`` pass-through, no tool secrets), rlimits
   on POSIX, its own process group / session, a wall-clock kill of the whole
   tree, ``spawn`` forced off (a child process would escape the audit hook),
   and the guard's ``deny_paths`` (data dir, keystore, ~/.atn, /proc) and
   ``deny_loopback``. The live PIDs are tracked so the local owner listener
   refuses a connection whose peer is one of them.

Both paths run the code under atn/tool_guard.py with a deny-by-default
policy from the manifest's declared ``net`` / ``fs`` / ``spawn``. That guard
is a Python audit hook: a tripwire that native code can bypass, not a wall.
The walls are the uid separation, the rlimits and the owner socket's
peer-credential refusal.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GUEST_ORIGIN = "integration"

# Capability keys a guest manifest may declare. Everything else (env,
# secrets, provides, ...) is refused at registration.
GUEST_CAPABILITY_KEYS = frozenset({"net", "fs", "spawn"})

_GUARD = Path(__file__).with_name("tool_guard.py")


class GuestIsolationUnavailable(RuntimeError):
    """uid isolation is required here but the launcher is not reachable."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_num(name: str, default: float) -> float:
    try:
        val = float(os.environ.get(name, ""))
        return val if val > 0 else default
    except ValueError:
        return default


def guest_limits() -> dict[str, Any]:
    """Per-run resource limits (env overridable)."""
    timeout = _env_num("ATN_GUEST_TOOL_TIMEOUT_S", 30.0)
    return {
        "timeout": timeout,
        "cpu_s": int(_env_num("ATN_GUEST_TOOL_CPU_S", max(1.0, timeout))),
        "mem_bytes": int(_env_num("ATN_GUEST_TOOL_MEM_MB", 512)) * 1024 * 1024,
        "fsize_bytes": int(_env_num("ATN_GUEST_TOOL_FSIZE_MB", 64)) * 1024 * 1024,
        "nofile": 256,
        "nproc": int(_env_num("ATN_GUEST_TOOL_NPROC", 64)),
    }


def uid_isolation_required() -> bool:
    return os.environ.get("ATN_GUEST_REQUIRE_UID") == "1"


def same_uid_allowed() -> bool:
    """Explicit operator opt-in to the weak same-uid fallback."""
    return os.environ.get("ATN_GUEST_ALLOW_SAME_UID") == "1"


def uid_isolation_available() -> bool:
    return os.name == "posix" and bool(os.environ.get("ATN_GUEST_LAUNCHER_FD"))


def guest_uids() -> set[int]:
    """Uids the local owner listener must never pre-auth."""
    out: set[int] = set()
    for part in (os.environ.get("ATN_GUEST_UID") or "").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
        elif "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                lo_i, hi_i = int(lo), int(hi)
                if 0 < lo_i <= hi_i and hi_i - lo_i < 4096:
                    out.update(range(lo_i, hi_i + 1))
    return out


def containment_mode() -> str:
    """'uid' | 'refused' | 'same-uid' — what a guest run gets right now."""
    if uid_isolation_available():
        return "uid"
    if uid_isolation_required() or not same_uid_allowed():
        return "refused"
    return "same-uid"


def boot_warning() -> str | None:
    """One line for the boot log when guest containment is weak or off."""
    mode = containment_mode()
    if mode == "same-uid":
        return ("ATN_GUEST_ALLOW_SAME_UID=1: guest tools run as the DAEMON "
                "user, contained only by a Python audit hook that native code "
                "(ctypes) bypasses. Any integration-token holder can then read "
                "this daemon's data dir and keystore. Use the guest launcher "
                "(the Docker image) instead.")
    if mode == "refused":
        return ("guest tools are refused: no guest launcher (uid isolation) "
                "and ATN_GUEST_ALLOW_SAME_UID is not set")
    return None


def deny_paths_for(data_dir: Path | str | None) -> list[str]:
    paths: list[str] = []
    for p in (data_dir, os.environ.get("ATN_DATA_DIR"),
              os.environ.get("KEYSTORE_DIR"), Path.home() / ".atn"):
        if p:
            try:
                paths.append(str(Path(p).resolve()))
            except OSError:
                paths.append(str(p))
    if os.name == "posix":
        paths.append("/proc")
    return sorted(set(paths))


def guest_policy(caps: dict[str, Any] | None, *, uid_isolated: bool,
                 deny_paths: list[str]) -> dict[str, Any]:
    """tool_guard policy for a guest run: deny by default, declared
    net/fs/spawn only; spawn only under uid isolation."""
    caps = caps or {}
    return {
        "net": bool(caps.get("net")),
        "fs": bool(caps.get("fs")),
        "spawn": bool(caps.get("spawn")) and uid_isolated,
        "deny_loopback": True,
        "deny_paths": list(deny_paths),
    }


# ---------------------------------------------------------------------------
# Live guest PIDs (same-uid fallback): the owner listener refuses them
# ---------------------------------------------------------------------------

_GUEST_PIDS: set[int] = set()
_PID_LOCK = threading.Lock()


def register_guest_pid(pid: int) -> None:
    with _PID_LOCK:
        _GUEST_PIDS.add(int(pid))


def unregister_guest_pid(pid: int) -> None:
    with _PID_LOCK:
        _GUEST_PIDS.discard(int(pid))


def live_guest_pids() -> frozenset[int]:
    with _PID_LOCK:
        return frozenset(_GUEST_PIDS)


def pid_is_guest(pid: int | None) -> bool:
    """True if ``pid`` is a tracked guest process or a descendant of one."""
    if pid is None:
        return False
    pids = live_guest_pids()
    if pid in pids:
        return True
    try:
        import psutil
        return any(p.pid in pids for p in psutil.Process(pid).parents())
    except Exception:  # noqa: BLE001 — process gone / no access
        return False


def _scrubbed_env(sandbox: str, policy: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {
        "PATH": os.path.dirname(sys.executable),
        "HOME": sandbox,
        "USERPROFILE": sandbox,
        "TMPDIR": sandbox,
        "TEMP": sandbox,
        "TMP": sandbox,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ATN_TOOL_POLICY": json.dumps(policy),
    }
    # Windows: the interpreter and the socket stack need these to start.
    for keep in ("SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC"):
        if keep in os.environ:
            env[keep] = os.environ[keep]
    return env


def _posix_preexec(limits: dict[str, Any]):
    def _pre() -> None:
        import resource
        for key, res in (("cpu_s", resource.RLIMIT_CPU),
                         ("mem_bytes", resource.RLIMIT_AS),
                         ("fsize_bytes", resource.RLIMIT_FSIZE),
                         ("nofile", resource.RLIMIT_NOFILE)):
            val = limits.get(key)
            if val:
                try:
                    resource.setrlimit(res, (int(val), int(val)))
                except (ValueError, OSError):
                    pass
        # NPROC is per-uid: on the same-uid fallback it would count the
        # daemon's own threads, so it is not set here (spawn is off anyway).
    return _pre


class _LocalGuestProcess:
    """asyncio Process wrapper for the same-uid fallback: tree kill,
    PID tracking and sandbox cleanup."""

    def __init__(self, proc: asyncio.subprocess.Process, sandbox: str) -> None:
        self._proc = proc
        self._sandbox = sandbox
        self.pid = proc.pid
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        self.mode = "same-uid"
        register_guest_pid(proc.pid)

    @property
    def returncode(self):
        return self._proc.returncode

    async def communicate(self, input: bytes | None = None):
        return await self._proc.communicate(input)

    async def wait(self) -> int:
        return await self._proc.wait()

    def kill(self) -> None:
        # Descendants first (snapshot before the parent dies and they are
        # reparented), then the process group, then the session: a child that
        # left the group with setsid() is still found by the snapshot.
        children = []
        try:
            import psutil
            children = psutil.Process(self._proc.pid).children(recursive=True)
        except Exception:  # noqa: BLE001
            pass
        if os.name == "posix":
            import signal
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                import psutil
                for p in psutil.process_iter(["pid"]):
                    try:
                        if p.pid != os.getpid() and os.getsid(p.pid) == self._proc.pid:
                            p.kill()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        for child in children:
            try:
                child.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass

    async def aclose(self) -> None:
        try:
            if self._proc.returncode is None:
                self.kill()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        finally:
            unregister_guest_pid(self.pid)
            shutil.rmtree(self._sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# uid isolation: client side of atn/guest_launcher.py
# ---------------------------------------------------------------------------

_CHANNEL: socket.socket | None = None
_CHANNEL_LOCK = threading.Lock()


def _channel() -> socket.socket:
    global _CHANNEL
    with _CHANNEL_LOCK:
        if _CHANNEL is None:
            fd = int(os.environ["ATN_GUEST_LAUNCHER_FD"])
            sock = socket.socket(fileno=fd)
            sock.set_inheritable(False)
            _CHANNEL = sock
        return _CHANNEL


class _LauncherProcess:
    """A guest run executing under the launcher's guest uid. Mirrors the
    asyncio Process surface ToolStore uses."""

    mode = "uid"

    def __init__(self) -> None:
        self.pid: int | None = None
        self.returncode: int | None = None
        self.stdin: asyncio.StreamWriter | None = None
        self.stdout: asyncio.StreamReader | None = None
        self.stderr: asyncio.StreamReader | None = None
        self._ctl_r: asyncio.StreamReader | None = None
        self._ctl_w: asyncio.StreamWriter | None = None
        self._exit: asyncio.Future | None = None
        self._ctl_task: asyncio.Task | None = None
        self._transports: list[Any] = []

    @classmethod
    async def start(cls, body: dict[str, Any]) -> "_LauncherProcess":
        loop = asyncio.get_running_loop()
        self = cls()
        ctl_a, ctl_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        try:
            try:
                ch = _channel()
                with _CHANNEL_LOCK:
                    socket.send_fds(ch, [b"run\n"],
                                    [ctl_b.fileno(), in_r, out_w, err_w])
            except OSError as exc:
                raise GuestIsolationUnavailable(
                    f"guest launcher unreachable: {exc}") from exc
        except BaseException:
            for fd in (in_w, out_r, err_r):
                os.close(fd)
            ctl_a.close()
            raise
        finally:
            ctl_b.close()
            for fd in (in_r, out_w, err_w):
                os.close(fd)

        self._ctl_r, self._ctl_w = await asyncio.open_unix_connection(sock=ctl_a)
        self._ctl_w.write(json.dumps(body).encode("utf-8") + b"\n")
        await self._ctl_w.drain()
        try:
            line = await asyncio.wait_for(self._ctl_r.readline(), timeout=15)
        except (asyncio.TimeoutError, OSError):
            line = b""
        try:
            hello = json.loads(line.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            hello = {}
        if "pid" not in hello:
            for fd in (in_w, out_r, err_r):
                os.close(fd)
            self._ctl_w.close()
            raise GuestIsolationUnavailable(
                f"guest launcher refused the run: {hello.get('error') or 'no reply'}")
        self.pid = int(hello["pid"])

        self.stdout = asyncio.StreamReader()
        t, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(self.stdout),
            os.fdopen(out_r, "rb", 0))
        self._transports.append(t)
        self.stderr = asyncio.StreamReader()
        t, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(self.stderr),
            os.fdopen(err_r, "rb", 0))
        self._transports.append(t)
        wt, wp = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, os.fdopen(in_w, "wb", 0))
        self._transports.append(wt)
        self.stdin = asyncio.StreamWriter(wt, wp, None, loop)

        self._exit = loop.create_future()
        self._ctl_task = asyncio.create_task(self._read_ctl())
        return self

    async def _read_ctl(self) -> None:
        rc = -9
        try:
            while True:
                line = await self._ctl_r.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "exit" in msg:
                    rc = int(msg["exit"])
                    break
        except Exception:  # noqa: BLE001
            pass
        self.returncode = rc
        if self._exit is not None and not self._exit.done():
            self._exit.set_result(rc)

    async def wait(self) -> int:
        return await asyncio.shield(self._exit)

    async def communicate(self, input: bytes | None = None):
        if input:
            self.stdin.write(input)
            try:
                await self.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            self.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        out, err, _ = await asyncio.gather(
            self.stdout.read(), self.stderr.read(), self.wait())
        return out, err

    def kill(self) -> None:
        try:
            if self._ctl_w is not None and not self._ctl_w.is_closing():
                self._ctl_w.write(b'{"kill": true}\n')
        except Exception:  # noqa: BLE001
            pass

    async def aclose(self) -> None:
        if self.returncode is None:
            self.kill()
            try:
                await asyncio.wait_for(self.wait(), timeout=15)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        for t in self._transports:
            try:
                t.close()
            except Exception:  # noqa: BLE001
                pass
        if self._ctl_w is not None:
            try:
                self._ctl_w.close()   # dropping ctl kills the run (launcher side)
            except Exception:  # noqa: BLE001
                pass
        if self._ctl_task is not None and not self._ctl_task.done():
            self._ctl_task.cancel()


# ---------------------------------------------------------------------------
# Entry point used by ToolStore
# ---------------------------------------------------------------------------

async def spawn_guest(code: bytes, caps: dict[str, Any] | None, *,
                      data_dir: Path | str | None):
    """Start one guest tool run. Returns a process-like object with
    ``stdin/stdout/stderr``, ``pid``, ``returncode``, ``wait()``,
    ``communicate()``, ``kill()`` and ``aclose()`` (always call it), and a
    ``mode`` of "uid" or "same-uid". Raises GuestIsolationUnavailable when
    uid isolation is required but unavailable."""
    limits = guest_limits()
    deny = deny_paths_for(data_dir)
    if uid_isolation_available():
        policy = guest_policy(caps, uid_isolated=True, deny_paths=deny)
        return await _LauncherProcess.start({
            "code": code.decode("utf-8", errors="replace"),
            "policy": policy,
            "limits": limits,
            "timeout": limits["timeout"],
        })
    if uid_isolation_required():
        raise GuestIsolationUnavailable(
            "guest tools require uid isolation (ATN_GUEST_REQUIRE_UID=1) and "
            "the guest launcher is not available")
    if not same_uid_allowed():
        raise GuestIsolationUnavailable(
            "guest tools need uid isolation (run the node under "
            "atn.guest_launcher, as the Docker image does); the same-uid "
            "fallback is off unless the operator sets "
            "ATN_GUEST_ALLOW_SAME_UID=1")

    policy = guest_policy(caps, uid_isolated=False, deny_paths=deny)
    sandbox = tempfile.mkdtemp(prefix="atn-guest-")
    real_sb = os.path.realpath(sandbox)
    for d in deny:
        if real_sb == d or real_sb.startswith(d.rstrip("/\\") + os.sep):
            shutil.rmtree(sandbox, ignore_errors=True)
            raise GuestIsolationUnavailable(
                "the temp dir is inside the daemon data dir; refusing to "
                "sandbox a guest tool there")
    script = os.path.join(sandbox, "tool.py")
    with open(script, "wb") as fh:
        fh.write(code)
    env = _scrubbed_env(sandbox, policy)
    kw: dict[str, Any] = {}
    if os.name == "posix":
        kw["preexec_fn"] = _posix_preexec(limits)
        kw["start_new_session"] = True
    else:
        import subprocess
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(_GUARD), script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env, cwd=sandbox, **kw)
    except Exception:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise
    return _LocalGuestProcess(proc, sandbox)
