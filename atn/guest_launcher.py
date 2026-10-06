"""Guest-tool launcher: run guest-authored tools as separate uids.

Tools a guest harness registers over the integration listener
(origin="integration", see atn/guest_sandbox.py and
docs/integration_listener.md) are untrusted code. The strongest containment
this daemon has for them is OS-level: an unprivileged uid that cannot read
the daemon's data dir (keystore, vault, tokens), cannot read the daemon's
environment (/proc/<pid>/environ is owner-only), cannot signal the daemon,
and is refused as owner by the local listener's peer-credential check.

The daemon itself runs unprivileged and cannot switch uid, so this small
stdlib-only process does it. Three modes:

  launcher (root, the container entrypoint)::

      python -m atn.guest_launcher --daemon-uid 10001 --guest-uid 10002 \\
          --guest-uid-count 16 -- atn --headless

    Creates a private AF_UNIX socketpair, starts the daemon as daemon-uid with
    one end (fd number in ATN_GUEST_LAUNCHER_FD, plus ATN_GUEST_UID as the
    pool range "10002-10017" and ATN_GUEST_REQUIRE_UID=1), and serves run
    requests on the other end. No filesystem socket exists, so nothing else
    can reach it. It needs only CAP_SETUID and CAP_SETGID. When it is not
    root it starts the daemon directly with ATN_GUEST_REQUIRE_UID=1 and no
    channel: guest tools then refuse to run (fail closed) instead of running
    unconfined.

    UID POOL: every live run gets its OWN uid (gid == uid) from the pool, so
    runs cannot read each other's sandbox, /proc entries or fds, cannot
    signal each other and do not share an RLIMIT_NPROC budget. A uid is
    handed out only after a reap as that uid finds no process left; while
    no uid is free, new runs are refused.

  stage (guest uid, one per run; started by the launcher)::

      python guest_launcher.py --stage <cfg_fd>

    Reads the run config (code, guard policy, limits) from cfg_fd, creates a
    0700 sandbox (/tmp/atn-guest-<run_id>), writes the script, applies rlimits and
    runs ``python tool_guard.py script`` on the inherited stdio, with a wall
    clock timeout.

  reap (guest uid; started by the launcher before and after each run, and
  on a kill)::

      python guest_launcher.py --reap <run_id>

    kill(-1, SIGKILL) as the run's uid: every process of that uid dies,
    whatever its environment, group or session says (a process can rewrite
    its own environ, so no marker is trusted). Repeats until /proc shows none
    left, then removes every file the uid owns at the top of the guest root
    and /dev/shm. Exit 0 = the uid is clean and may be reused. Refuses to run
    as root.

Wire protocol (daemon -> launcher), see GuestLauncherClient in
atn/guest_sandbox.py: one SCM_RIGHTS message on the master channel carrying
``b"run\\n"`` and four fds [ctl, stdin_r, stdout_w, stderr_w]. Over ``ctl``
(a fresh socketpair end) the daemon writes one JSON line ``{"code", "policy",
"limits", "timeout"}``; the launcher answers ``{"pid": n}`` (or
``{"error": ...}``, e.g. when the uid pool is exhausted) and later
``{"exit": rc}``. The daemon may write ``{"kill": true}``; closing ``ctl``
before exit also kills the run.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

GUEST_ROOT_DEFAULT = "/tmp"
GUEST_UID_COUNT_DEFAULT = 16


def sandbox_path(root: str, run_id: str) -> str:
    """Per-run sandbox. Created fresh with mkdir (fails if it exists), so a
    pre-planted directory or symlink cannot be adopted."""
    return os.path.join(root, f"atn-guest-{run_id}")
_MAX_BODY = 4 * 1024 * 1024
_HERE = os.path.abspath(__file__)
_GUARD = os.path.join(os.path.dirname(_HERE), "tool_guard.py")


def _log(msg: str) -> None:
    sys.stderr.write(f"[guest_launcher] {msg}\n")
    sys.stderr.flush()


def _guest_env(sandbox: str, run_id: str, policy: dict) -> dict[str, str]:
    """The ONLY environment a guest tool sees. Built here, never copied."""
    return {
        "PATH": os.path.dirname(sys.executable) + ":/usr/local/bin:/usr/bin:/bin",
        "HOME": sandbox,
        "TMPDIR": sandbox,
        "TEMP": sandbox,
        "TMP": sandbox,
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ATN_GUEST_RUN": run_id,
        "ATN_TOOL_POLICY": json.dumps(policy),
    }


# ---------------------------------------------------------------------------
# stage (runs as the guest uid)
# ---------------------------------------------------------------------------

def _set_rlimits(limits: dict) -> None:
    import resource
    pairs = (
        ("cpu_s", resource.RLIMIT_CPU),
        ("mem_bytes", resource.RLIMIT_AS),
        ("fsize_bytes", resource.RLIMIT_FSIZE),
        ("nofile", resource.RLIMIT_NOFILE),
        ("nproc", getattr(resource, "RLIMIT_NPROC", None)),
    )
    for key, res in pairs:
        val = limits.get(key)
        if res is None or not val:
            continue
        try:
            resource.setrlimit(res, (int(val), int(val)))
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def stage_main(cfg_fd: int) -> int:
    with os.fdopen(cfg_fd, "rb") as fh:
        cfg = json.loads(fh.read(_MAX_BODY).decode("utf-8"))
    run_id = str(cfg["run_id"])
    root = str(cfg.get("guest_root") or GUEST_ROOT_DEFAULT)
    os.umask(0o077)
    sandbox = sandbox_path(root, run_id)
    os.mkdir(sandbox, 0o700)
    script = os.path.join(sandbox, "tool.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(str(cfg.get("code") or ""))
    limits = dict(cfg.get("limits") or {})
    timeout = float(cfg.get("timeout") or 30)
    env = _guest_env(sandbox, run_id, dict(cfg.get("policy") or {}))

    def _pre() -> None:
        _set_rlimits(limits)

    proc = subprocess.Popen(
        [sys.executable, _GUARD, script],
        env=env, cwd=sandbox, preexec_fn=_pre, close_fds=True)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"guest tool timed out after {timeout:g}s\n")
        sys.stderr.flush()
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
        rc = 124
    shutil.rmtree(sandbox, ignore_errors=True)
    return rc


# ---------------------------------------------------------------------------
# reap (runs as the guest uid)
# ---------------------------------------------------------------------------

def _live_procs_of(uid: int, skip: int) -> list[int]:
    """Non-zombie processes owned by ``uid`` (from /proc), except ``skip``."""
    out: list[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == skip:
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
            with open(f"/proc/{name}/stat", encoding="ascii",
                      errors="replace") as fh:
                state = fh.read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        if state not in ("Z", "X"):
            out.append(int(name))
    return out


def _remove_owned(directory: str, uid: int) -> None:
    """Delete every top-level entry of ``directory`` owned by ``uid``, so a
    later run on the same uid finds nothing an earlier run left behind."""
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if st.st_uid != uid:
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                for dirpath, _dirs, _files in os.walk(entry.path):
                    try:
                        os.chmod(dirpath, 0o700)
                    except OSError:
                        pass
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.unlink(entry.path)
        except OSError:
            continue


def reap_main(run_id: str, root: str = GUEST_ROOT_DEFAULT) -> int:
    uid = os.getuid()
    if uid == 0 or os.geteuid() == 0:
        _log("reap refused: running as root")
        return 2
    me = os.getpid()
    for _ in range(40):
        try:
            os.kill(-1, signal.SIGKILL)     # every process of this uid but me
        except OSError:
            pass                            # ESRCH: none left
        if not _live_procs_of(uid, me):
            break
        time.sleep(0.05)
    clean = not _live_procs_of(uid, me)
    if run_id and "/" not in run_id and run_id not in (".", ".."):
        shutil.rmtree(sandbox_path(root, run_id), ignore_errors=True)
    for d in sorted({root, "/dev/shm"}):
        _remove_owned(d, uid)
    return 0 if clean else 3


# ---------------------------------------------------------------------------
# launcher (root)
# ---------------------------------------------------------------------------

class _Launcher:
    def __init__(self, guest_uids, guest_root: str = GUEST_ROOT_DEFAULT) -> None:
        if isinstance(guest_uids, int):
            guest_uids = [guest_uids]
        uids = [int(u) for u in guest_uids]
        if not uids or any(u <= 0 for u in uids):
            raise ValueError("guest uid pool must be non-empty and non-root")
        self.guest_root = guest_root
        self._free: list[int] = list(uids)
        self._seq = 0
        self._lock = threading.Lock()

    def _as_guest(self, uid: int, argv: list[str], **kw) -> subprocess.Popen:
        return subprocess.Popen(
            argv, user=uid, group=uid, extra_groups=[],
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, cwd="/",
            close_fds=True, start_new_session=True, **kw)

    def _reap(self, uid: int, run_id: str) -> bool:
        """Kill every process of ``uid`` and clear its files. True = clean."""
        try:
            return self._as_guest(uid, [sys.executable, _HERE, "--reap", run_id,
                                        "--guest-root", self.guest_root]
                                  ).wait(timeout=30) == 0
        except Exception as exc:  # noqa: BLE001
            _log(f"reap {run_id or '-'} (uid {uid}) failed: {exc}")
            return False

    def _acquire(self) -> int | None:
        """A pool uid verified to have no live process, or None."""
        with self._lock:
            tries = len(self._free)
        for _ in range(tries):
            with self._lock:
                if not self._free:
                    return None
                uid = self._free.pop(0)
            if self._reap(uid, ""):
                return uid
            with self._lock:
                self._free.append(uid)      # not clean yet; try it later
        return None

    def _release(self, uid: int) -> None:
        with self._lock:
            self._free.append(uid)

    def handle(self, ctl: socket.socket, fds: list[int]) -> None:
        stdin_r, stdout_w, stderr_w = fds
        with self._lock:
            self._seq += 1
            run_id = f"r{os.getpid()}-{self._seq}-{os.urandom(4).hex()}"
        stage = None
        uid = None
        try:
            uid = self._acquire()
            if uid is None:
                raise RuntimeError("guest uid pool exhausted; retry when a "
                                   "running guest tool finishes")
            buf = b""
            while b"\n" not in buf:
                chunk = ctl.recv(65536)
                if not chunk:
                    return
                buf += chunk
                if len(buf) > _MAX_BODY:
                    return
            line = buf.split(b"\n", 1)[0]
            req = json.loads(line.decode("utf-8"))
            cfg = {
                "run_id": run_id,
                "guest_root": self.guest_root,
                "code": str(req.get("code") or ""),
                "policy": dict(req.get("policy") or {}),
                "limits": dict(req.get("limits") or {}),
                "timeout": float(req.get("timeout") or 30),
            }
            cfg_r, cfg_w = os.pipe()
            try:
                stage = self._as_guest(
                    uid, [sys.executable, _HERE, "--stage", str(cfg_r)],
                    stdin=stdin_r, stdout=stdout_w, stderr=stderr_w,
                    pass_fds=(cfg_r,))
            finally:
                os.close(cfg_r)
            with os.fdopen(cfg_w, "wb") as fh:
                fh.write(json.dumps(cfg).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            _log(f"run request failed: {exc}")
            try:
                ctl.sendall(json.dumps({"error": str(exc)}).encode() + b"\n")
            except OSError:
                pass
            return
        finally:
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if stage is None:
                ctl.close()
                if uid is not None:
                    self._reap(uid, run_id)
                    self._release(uid)
        try:
            ctl.sendall(json.dumps({"pid": stage.pid}).encode() + b"\n")
        except OSError:
            pass

        done = threading.Event()

        def _watch_ctl() -> None:
            # A kill request or the daemon dropping ctl both kill the run.
            data = b""
            while not done.is_set():
                try:
                    chunk = ctl.recv(4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                data += chunk
                if b'"kill"' in data:
                    break
            if not done.is_set():
                self._reap(uid, run_id)

        watcher = threading.Thread(target=_watch_ctl, daemon=True)
        watcher.start()
        rc = stage.wait()
        done.set()
        self._reap(uid, run_id)     # every process of the run's uid + files
        try:
            ctl.sendall(json.dumps({"exit": rc}).encode() + b"\n")
        except OSError:
            pass
        try:
            ctl.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        # The watcher must be gone before the uid is reused: a late reap from
        # it would kill the next run on this uid.
        watcher.join(timeout=60)
        ctl.close()
        if watcher.is_alive():
            _log(f"uid {uid} withheld from the pool: ctl watcher still running")
            return
        self._release(uid)

    def serve(self, master: socket.socket) -> None:
        while True:
            try:
                msg, fds, _flags, _addr = socket.recv_fds(master, 64, 4)
            except OSError as exc:
                _log(f"channel error: {exc}")
                return
            if not msg and not fds:
                return              # daemon exited
            if msg.strip() != b"run" or len(fds) != 4:
                for fd in fds:
                    os.close(fd)
                continue
            ctl = socket.socket(fileno=fds[0])
            threading.Thread(target=self.handle, args=(ctl, fds[1:]),
                             daemon=True).start()


def launcher_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="atn.guest_launcher")
    ap.add_argument("--daemon-uid", type=int, default=10001)
    ap.add_argument("--daemon-gid", type=int, default=None)
    ap.add_argument("--guest-uid", type=int, default=10002,
                    help="first uid of the guest pool")
    ap.add_argument("--guest-uid-count", type=int,
                    default=GUEST_UID_COUNT_DEFAULT,
                    help="pool size = concurrent guest runs (one uid each)")
    ap.add_argument("--guest-gid", type=int, default=None,
                    help="ignored: each run uses gid == its uid")
    ap.add_argument("--guest-root", default=GUEST_ROOT_DEFAULT)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    cmd = [c for c in ns.cmd if c != "--"] or ["atn", "--headless"]
    daemon_gid = ns.daemon_gid if ns.daemon_gid is not None else ns.daemon_uid
    count = max(1, min(int(ns.guest_uid_count), 1024))
    pool = list(range(ns.guest_uid, ns.guest_uid + count))

    env = dict(os.environ)
    env["ATN_GUEST_REQUIRE_UID"] = "1"
    env.pop("ATN_GUEST_LAUNCHER_FD", None)

    if (os.geteuid() != 0 or ns.guest_uid <= 0 or ns.daemon_uid in pool
            or daemon_gid in pool):
        _log("not root (or guest uid pool invalid): guest tools will be refused")
        os.execvpe(cmd[0], cmd, env)

    master, daemon_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    master.set_inheritable(False)
    env["ATN_GUEST_LAUNCHER_FD"] = str(daemon_end.fileno())
    env["ATN_GUEST_UID"] = f"{pool[0]}-{pool[-1]}"
    try:
        daemon = subprocess.Popen(
            cmd, env=env, user=ns.daemon_uid, group=daemon_gid, extra_groups=[],
            umask=0o077, pass_fds=(daemon_end.fileno(),))
    except PermissionError as exc:
        # Root without CAP_SETUID/CAP_SETGID (cap_drop: ALL with no cap_add).
        # Never fall back to running the daemon as root.
        _log(f"cannot drop to the daemon uid ({exc}); add CAP_SETUID and "
             "CAP_SETGID (compose: cap_add: [SETUID, SETGID]) or run the "
             "container as the daemon user (guest tools are then refused)")
        return 1
    daemon_end.close()

    def _forward(signum, _frame):
        try:
            daemon.send_signal(signum)
        except OSError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _forward)

    launcher = _Launcher(pool, ns.guest_root)
    threading.Thread(target=launcher.serve, args=(master,), daemon=True).start()
    return daemon.wait()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--stage"]:
        return stage_main(int(argv[1]))
    if argv[:1] == ["--reap"]:
        root = GUEST_ROOT_DEFAULT
        if "--guest-root" in argv:
            root = argv[argv.index("--guest-root") + 1]
        return reap_main(argv[1], root)
    return launcher_main(argv)


if __name__ == "__main__":
    sys.exit(main())
