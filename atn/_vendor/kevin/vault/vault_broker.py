#!/usr/bin/env python3
r"""
vault_broker — the centralized credential broker. RUNS AS vault-svc, not astmo.

This is the privilege boundary. Agents (running as astmo) never hold a Vault
token. They connect to this broker's named pipe; the broker reads the caller's
PID *from the kernel* (GetNamedPipeClientProcessId), looks up that PID's granted
policy set in the in-memory session table, fetches the secret from Vault using
the broker's OWN token (which lives only in this vault-svc process), stages the
raw value to a nameless file, and returns {var_name, path} — never the value.

Why this holds where the age daemon didn't: the authority (the broker's Vault
token, the PID->policy table) lives in a process owned by vault-svc. A model-
agent running as astmo cannot ReadProcessMemory this process or write its session
table, because it's a different OS principal. The age design kept authority in
astmo-writable files; this keeps it in another user's process memory.

Session registration is itself an authenticated pipe op: cc.bat (the launcher)
calls `register` with the resolved allowlist, and the broker records
{caller_pid: {identity, policies}} in MEMORY. The launcher's authority to
register is the open question the design flagged — see SECURITY notes below.

Pipe ops (newline JSON):
  {"op":"register","pid":N,"policies":[...]}  -> {"ok"}      (launcher only)
  {"op":"list"}                               -> {"ok","services":[...]}
  {"op":"request","service":"X"}              -> {"ok","var_name","path"}
  {"op":"release","service":"X"}              -> {"ok"}
  {"op":"whoami"}                             -> {"ok","pid","policies":[...]}
  {"op":"mint_nonce","owner_secret":S,...}    -> {"ok","nonce"}  (owner only)
  {"op":"release_session","owner_secret":S,"pid":N} -> {"ok"}    (owner only)

VALUE-PUSH (owner-gated, one-way vault-svc -> daemon monitor):
  A daemon may register a monitor sink when it mints a nonce, by including
  {"push_endpoint": r"\\.\pipe\...", "push_token": "..."} in the mint_nonce
  request (both owner-authenticated by the same owner_secret). The broker binds
  that sink to the registered PID. On every successful `request` (secret staged),
  the broker connects to push_endpoint and pushes
      {"push_token": T, "value": <raw>, "staged_path": P, "secret_name": SVC,
       "pid": N, "agent_id": A?}
  so the daemon's SecurityMonitor can scan agent output for the leaked VALUE and
  PATH. The value leaves vault-svc ONLY to that daemon sink — never to the agent,
  never logged here. This is the ONLY egress of a raw value besides the staged
  file. If the push fails, staging still succeeds (fail-open on the SCAN is
  acceptable; fail-closed on the GRANT is enforced daemon-side: no monitor =>
  no grant, so a live grant implies the daemon believes its sink is up).

SECURITY notes (honest):
  * Caller PID is kernel-supplied -> unforgeable.
  * Session table is in THIS process's memory (vault-svc) -> astmo can't edit it.
  * Broker's Vault token lives only here -> astmo can't read it (different user).
  * Residual: a sibling astmo agent can't steal a token (there is none to steal),
    but CAN ask the broker for whatever ITS OWN pid was granted. Cross-agent
    theft is closed because nothing astmo-side holds authority.
  * Admin caveat: if an agent elevates (UAC), it gets SeDebugPrivilege and can
    read this process anyway. Boundary assumes agents stay non-elevated.
"""

import os
import sys
import json
import threading

_IS_WIN = os.name == "nt"
if _IS_WIN:
    import ctypes
    from ctypes import wintypes
else:
    import socket
    import struct

# Self-contained backend: secrets come from the local age-encrypted keystore
# (keystore.py at kevin root), NOT HashiCorp Vault. Ensure kevin root is
# importable whether the broker is launched from vault/ or as a vendored sibling.
_KEVIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KEVIN_ROOT not in sys.path:
    sys.path.insert(0, _KEVIN_ROOT)

# Broker endpoint: named pipe on Windows, AF_UNIX socket path on POSIX. The
# POSIX path mirrors the client's default (keystore dir / vault-broker.sock),
# env-overridable via ATN_VAULT_BROKER_SOCK.
PIPE_NAME = r"\\.\pipe\vault-broker"

# service<->policy map. Resolved from the keystore DATA dir (env-overridable),
# NOT next to this file — so the vendored/in-wheel broker finds the operator's
# map. Setup generates an identity map (policy == service) for the self-contained
# age backend.
import keystore as _ks  # importable via the _KEVIN_ROOT bootstrap above
_MAP_PATH = os.environ.get(
    "ATN_VAULT_POLICY_MAP",
    os.path.join(_ks.KEYSTORE_DIR, "service_policy_map.json"))

# POSIX AF_UNIX socket path (mirrors broker_client._default_unix_sock()).
UNIX_SOCK = os.environ.get(
    "ATN_VAULT_BROKER_SOCK",
    os.path.join(_ks.KEYSTORE_DIR, "vault-broker.sock"))

# ── Win32 named-pipe plumbing (server; Windows only) ──────────────────────────
if _IS_WIN:
    PIPE_ACCESS_DUPLEX = 0x00000003
    PIPE_TYPE_BYTE = 0x0
    PIPE_READMODE_BYTE = 0x0
    PIPE_WAIT = 0x0
    PIPE_REJECT_REMOTE_CLIENTS = 0x8
    PIPE_UNLIMITED_INSTANCES = 255
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    ERROR_PIPE_CONNECTED = 535
    k32 = ctypes.windll.kernel32


def _load_map():
    try:
        with open(_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)            # {SERVICE: policy_name}
    except (OSError, ValueError):
        return {}


SERVICE_POLICY = _load_map()
POLICY_SERVICE = {v: k for k, v in SERVICE_POLICY.items()}


# ── In-memory session table (vault-svc only) ──────────────────────────────────
# pid -> {"identity": "pid:starttime", "policies": [policy_name, ...]}
_SESSIONS = {}
# one-time registration nonces: nonce -> {"policies": [...],
#   "push_endpoint": str|None, "push_token": str|None}  (owner-minted)
_NONCES = {}
# per-session value-push sink (owner-registered via the nonce): pid ->
#   {"endpoint": r"\\.\pipe\...", "token": "..."}. The broker connects OUT to
#   this endpoint on each stage and pushes the raw value to the daemon monitor.
_PUSH_SINKS = {}
_LOCK = threading.Lock()
# Owner secret gating nonce minting. Held by vault-svc (this process env) and by
# the owner; NOT readable by astmo agents (different OS user). Set at startup.
_OWNER_SECRET = os.environ.get("BROKER_OWNER_SECRET", "")


def _identity(pid):
    """OS-read pid:starttime, used to detect PID reuse. None if dead.

    Portable create-time guard: the value need only be STABLE for the life of a
    process and DIFFERENT after PID reuse — it is never compared across hosts.
      * Windows: GetProcessTimes creation FILETIME (100ns ticks since 1601).
      * Linux:   /proc/<pid>/stat field 22 (starttime, clock ticks since boot).
      * other POSIX (macOS/BSD): psutil.create_time() (float seconds).
    """
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            creation = wintypes.FILETIME()
            a = wintypes.FILETIME(); b = wintypes.FILETIME(); c = wintypes.FILETIME()
            if not k32.GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(a),
                                       ctypes.byref(b), ctypes.byref(c)):
                return None
            return f"{pid}:{(creation.dwHighDateTime << 32) | creation.dwLowDateTime}"
        finally:
            k32.CloseHandle(h)
    # Linux fast path: read starttime straight from /proc, no psutil needed.
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{int(pid)}/stat", "rb") as f:
                data = f.read()
            # comm (field 2) is parenthesized and may contain spaces/`)`; split
            # on the LAST `)` so field indexing past it is correct. starttime is
            # field 22 (1-based); after the ')' the first token is state (field 3).
            rest = data[data.rfind(b")") + 2:].split()
            starttime = rest[19]  # field 22 == index 19 counting state as 0
            return f"{pid}:{starttime.decode('ascii', 'replace')}"
        except (OSError, IndexError, ValueError):
            return None
    # Other POSIX: fall back to psutil create_time (seconds float).
    try:
        import psutil  # type: ignore
        return f"{pid}:{psutil.Process(int(pid)).create_time()}"
    except Exception:
        return None


def _create_pipe():
    h = k32.CreateNamedPipeW(
        PIPE_NAME, PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        PIPE_UNLIMITED_INSTANCES, 65536, 65536, 0, None)
    if h == INVALID_HANDLE_VALUE:
        raise OSError(f"CreateNamedPipeW failed: {k32.GetLastError()}")
    return h


def _client_pid(handle):
    pid = wintypes.DWORD(0)
    if not k32.GetNamedPipeClientProcessId(handle, ctypes.byref(pid)):
        return None
    return int(pid.value)


def _read_msg(handle):
    buf = bytearray(); chunk = ctypes.create_string_buffer(4096)
    read = wintypes.DWORD(0)
    while True:
        if not k32.ReadFile(handle, chunk, 4096, ctypes.byref(read), None) or read.value == 0:
            break
        buf += chunk.raw[:read.value]
        if b"\n" in buf:
            break
    if not buf:
        return None
    try:
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"op": "__bad__"}


def _write_msg(handle, obj):
    """Write one newline-JSON message. Returns True ONLY if the full payload was
    written (WriteFile succeeded AND every byte went out). Callers that gate
    security on delivery (the value-push, M7) require this; reply writers that
    don't care may ignore the return."""
    data = (json.dumps(obj) + "\n").encode("utf-8")
    w = wintypes.DWORD(0)
    ok = k32.WriteFile(handle, data, len(data), ctypes.byref(w), None)
    k32.FlushFileBuffers(handle)
    return bool(ok) and w.value == len(data)


# ── Vault access (broker's own token; never leaves this process) ─────────────
def _fetch_secret(service):
    """Read the secret value from the local age-encrypted keystore (self-contained
    — no HashiCorp Vault). Broker-side only; the value goes straight to nameless
    staging + the owner-gated monitor push, never back to the agent."""
    from keystore import get_secret
    return get_secret(service)


# ── Nameless-file staging (unchanged behaviour, now written as vault-svc) ─────
import secrets as _secrets       # noqa: E402
import tempfile                  # noqa: E402

_STAGED = {}   # pid -> {service: path}


def _stage_path():
    base = tempfile.gettempdir()
    sub = os.path.join(base, _secrets.choice(("cache", "data", ".cache", "tmp")))
    os.makedirs(sub, exist_ok=True)
    # 128-bit unguessable token (bumped from 64-bit): a same-OS-user sibling that
    # never learns the path cannot enumerate it. (Learning the path is a separate,
    # accepted residual closed by the value-scan tripwire, not this token width.)
    name = _secrets.token_hex(16) + _secrets.choice((".dat", ".bin", ".cache", ""))
    return os.path.join(sub, name)


def _stage(pid, service, value):
    path = _stage_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _STAGED.setdefault(pid, {})
    old = _STAGED[pid].get(service)
    if old and old != path:
        _unlink(old)
    _STAGED[pid][service] = path
    return path


def _unlink(path):
    try:
        os.remove(path); return True
    except OSError:
        return False


def _var_name():
    return "TOK_" + _secrets.token_hex(4).upper()


# ── Value-push (owner-gated, one-way vault-svc -> daemon monitor) ─────────────
if _IS_WIN:
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3


def _push_value(pid, secret_name, staged_path, value):
    """Push the raw value to the daemon monitor sink registered for this pid.

    Connects OUT to the daemon's push endpoint (named pipe on Windows, AF_UNIX
    socket on POSIX; daemon = OWNER-run, auth is the owner-minted push_token).
    Returns True ONLY if the value was actually delivered; the ``request`` caller
    fails CLOSED (unlinks the staged file, denies) on a False/failed push so a
    readable secret is never left un-scanned (M7). The value is written and then
    this frame forgets it — nothing here logs or retains it.
    """
    with _LOCK:
        sink = _PUSH_SINKS.get(pid)
    if not sink or not sink.get("endpoint") or not sink.get("token"):
        return False
    payload = {
        "push_token": sink["token"],
        "value": value,
        "staged_path": staged_path,
        "secret_name": secret_name,
        "pid": pid,
    }
    aid = sink.get("agent_id")
    if aid:
        payload["agent_id"] = aid
    if _IS_WIN:
        return _push_value_win(sink["endpoint"], payload)
    return _push_value_unix(sink["endpoint"], payload)


def _push_value_unix(endpoint, payload):
    """AF_UNIX connect-out push. True only on full delivery (M7)."""
    s = None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        try:
            s.connect(endpoint)
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False
        data = (json.dumps(payload) + "\n").encode("utf-8")
        s.sendall(data)  # sendall raises if not all bytes are sent
        return True
    except Exception:  # noqa: BLE001 — a failed push returns False, never raises
        return False
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass


def _push_value_win(endpoint, payload):
    h = None
    try:
        h = k32.CreateFileW(endpoint, GENERIC_READ | GENERIC_WRITE, 0, None,
                            OPEN_EXISTING, 0, None)
        # CreateFileW has no restype set on k32, so a failure comes back as the
        # signed -1 rather than the unsigned INVALID_HANDLE_VALUE constant — guard
        # BOTH so a dead pipe is correctly treated as "no sink" (not a spurious ok).
        if h in (INVALID_HANDLE_VALUE, -1) or not h:
            return False
        # Gate success on the write actually landing (M7): a pipe handle can open
        # while the write fails (peer disconnects between open and write). Only a
        # fully-delivered push counts, so the caller can fail the request closed.
        return _write_msg(h, payload)
    except Exception:  # noqa: BLE001 — a failed push returns False, never raises
        return False
    finally:
        if h and h != INVALID_HANDLE_VALUE:
            try:
                k32.CloseHandle(h)
            except Exception:  # noqa: BLE001
                pass


# ── Request handling — authority derives ONLY from kernel pid ─────────────────
def _handle(req, pid):
    op = req.get("op")

    if op == "register":
        # TRUST: an agent runs as the same OS user as the launcher, so PID alone
        # can't prove "I am the launcher, not an agent self-granting." We gate
        # registration on a one-time NONCE that the broker minted and that only
        # the owner-side launch path can obtain (see `mint_nonce`, called by an
        # owner/admin step, NOT reachable by an unprivileged agent because the
        # nonce op requires the broker's admin pipe — separate from this one).
        nonce = req.get("nonce", "")
        with _LOCK:
            spec = _NONCES.pop(nonce, None)
        if spec is None:
            return {"ok": False, "error": "invalid or used registration nonce"}
        target = req.get("pid", pid)
        # policies are FIXED by the nonce the owner minted — the registrant
        # cannot choose them, only consume the pre-authorized set.
        pols = [p for p in spec["policies"] if p in POLICY_SERVICE]
        with _LOCK:
            _SESSIONS[target] = {"identity": _identity(target), "policies": pols}
            # Bind the owner-registered value-push sink (if any) to this PID.
            # The sink params came from the OWNER-authenticated mint_nonce, so a
            # registrant cannot choose or redirect them — only consume them.
            ep = spec.get("push_endpoint")
            tok = spec.get("push_token")
            if ep and tok:
                _PUSH_SINKS[target] = {
                    "endpoint": ep, "token": tok,
                    "agent_id": spec.get("agent_id"),
                }
        return {"ok": True, "pid": target, "policies": pols}

    if op == "mint_nonce":
        # OWNER-ONLY: mint a one-time nonce binding a resolved policy set, to be
        # consumed by exactly one `register`. This op is only honored when the
        # caller presents the owner secret (a value held by vault-svc / the
        # owner, never by agents). Without it, agents cannot authorize policies.
        if not _OWNER_SECRET or req.get("owner_secret", "") != _OWNER_SECRET:
            return {"ok": False, "error": "not authorized to mint"}
        pols = [p for p in req.get("policies", []) if p in POLICY_SERVICE]
        n = _secrets.token_hex(16)
        # OWNER-gated value-push sink: the daemon may register its monitor pipe +
        # a push token HERE (both under the same owner_secret). Bound to the
        # session at register-time. A missing endpoint/token simply means no push
        # (the pre-value-push behaviour).
        push_endpoint = req.get("push_endpoint") or None
        push_token = req.get("push_token") or None
        agent_id = req.get("agent_id") or None
        with _LOCK:
            _NONCES[n] = {
                "policies": pols,
                "push_endpoint": push_endpoint,
                "push_token": push_token,
                "agent_id": agent_id,
            }
        return {"ok": True, "nonce": n, "policies": pols}

    if op == "release_session":
        # OWNER-ONLY: tear down a worker session at PID revoke. Drops the PID's
        # policies + push sink and unlinks every staged file for it (so no raw
        # value outlives the worker). Gated on the owner secret (same value the
        # daemon holds; agents cannot present it — different OS user).
        if not _OWNER_SECRET or req.get("owner_secret", "") != _OWNER_SECRET:
            return {"ok": False, "error": "not authorized to release"}
        target = req.get("pid")
        if target is None:
            return {"ok": False, "error": "release_session requires pid"}
        with _LOCK:
            _SESSIONS.pop(target, None)
            _PUSH_SINKS.pop(target, None)
            staged = _STAGED.pop(target, {}) or {}
        unlinked = 0
        for path in list(staged.values()):
            if _unlink(path):
                unlinked += 1
        return {"ok": True, "pid": target, "unlinked": unlinked}

    with _LOCK:
        sess = _SESSIONS.get(pid)
        valid = sess and sess["identity"] == _identity(pid)
    if not valid:
        return {"ok": False, "error": "caller not a registered session"}
    pols = sess["policies"]
    services = [POLICY_SERVICE[p] for p in pols if p in POLICY_SERVICE]

    if op == "whoami":
        return {"ok": True, "pid": pid, "policies": pols, "services": services}
    if op == "list":
        return {"ok": True, "services": services}
    if op == "request":
        service = req.get("service", "")
        if service not in services:
            return {"ok": False, "error": f"service '{service}' not granted"}
        try:
            value = _fetch_secret(service)
        except Exception as e:  # noqa: BLE001 — never leak detail/value
            return {"ok": False, "error": "vault read failed"}
        path = _stage(pid, service, value)
        # OWNER-gated value-push to the daemon monitor — this IS the tripwire
        # feed. If it does not land, the staged secret is readable by the worker
        # while the monitor never learns the value: a blind spot where echoing the
        # secret would trip nothing. Fail CLOSED (M7): unlink the staged file and
        # deny the request rather than expose an un-scanned secret. (The grant was
        # already gated daemon-side on the monitor being up at bind time; this
        # closes the remaining PER-REQUEST gap.)
        try:
            pushed = _push_value(pid, service, path, value)
        except Exception:  # noqa: BLE001 — any push fault fails closed
            pushed = False
        if not pushed:
            _unlink(path)
            _STAGED.get(pid, {}).pop(service, None)
            return {"ok": False, "error": "tripwire unavailable; request denied"}
        return {"ok": True, "service": service, "var_name": _var_name(), "path": path}
    if op == "release":
        service = req.get("service", "")
        path = _STAGED.get(pid, {}).pop(service, None)
        return {"ok": _unlink(path) if path else False}
    return {"ok": False, "error": f"unknown op {op!r}"}


def _serve_one(handle):
    try:
        pid = _client_pid(handle)
        req = _read_msg(handle)
        if req is None:
            return
        resp = _handle(req, pid) if pid else {"ok": False, "error": "no caller pid"}
        _write_msg(handle, resp)
    finally:
        k32.FlushFileBuffers(handle)
        k32.DisconnectNamedPipe(handle)
        k32.CloseHandle(handle)


# ── POSIX AF_UNIX serving (peer PID from SO_PEERCRED) ─────────────────────────
def _peer_pid(conn):
    """Kernel-authenticated peer PID (+uid,gid) via SO_PEERCRED — the POSIX
    analog of GetNamedPipeClientProcessId. The client cannot forge these; they
    come from the kernel, not the wire. Returns pid or None."""
    try:
        creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                struct.calcsize("3i"))
        pid, _uid, _gid = struct.unpack("3i", creds)
        return int(pid) if pid > 0 else None
    except (OSError, AttributeError):
        return None


def _read_msg_sock(conn):
    conn.settimeout(5.0)
    buf = bytearray()
    while b"\n" not in buf:
        try:
            chunk = conn.recv(4096)
        except Exception:  # noqa: BLE001
            break
        if not chunk:
            break
        buf += chunk
        if len(buf) > 1 << 20:  # 1 MiB cap — a request is tiny
            break
    if not buf:
        return None
    try:
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"op": "__bad__"}


def _write_msg_sock(conn, obj):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    try:
        conn.sendall(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def _serve_one_unix(conn):
    try:
        pid = _peer_pid(conn)
        req = _read_msg_sock(conn)
        if req is None:
            return
        resp = _handle(req, pid) if pid else {"ok": False, "error": "no caller pid"}
        _write_msg_sock(conn, resp)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _harden_process():
    """Best-effort self-hardening against a same-uid attacker reading the age key
    out of this process's memory. PR_SET_DUMPABLE(0) makes the process
    non-dumpable: it disables ptrace-attach and core dumps by a non-root peer
    (the analog of denying SeDebugPrivilege on Windows). If the broker runs as a
    SEPARATE uid (the recommended P3 posture), Yama ptrace_scope already blocks
    cross-uid ptrace; this closes the same-uid case too. Never fatal."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes as _ct
        libc = _ct.CDLL("libc.so.6", use_errno=True)
        PR_SET_DUMPABLE = 4
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except Exception:  # noqa: BLE001
        print("[vault-broker] warning: PR_SET_DUMPABLE hardening unavailable",
              flush=True)


def _serve_forever_unix():
    _harden_process()
    try:
        os.unlink(UNIX_SOCK)
    except OSError:
        pass
    d = os.path.dirname(UNIX_SOCK)
    if d:
        os.makedirs(d, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(UNIX_SOCK)
    # 0600: only the broker uid may connect(). P3 sets the broker to a separate
    # uid and the containing dir 0700 so agents (a different uid) can't reach it.
    try:
        os.chmod(UNIX_SOCK, 0o600)
    except OSError:
        pass
    srv.listen(64)
    print(f"[vault-broker] listening on {UNIX_SOCK} (age keystore, local)",
          flush=True)
    print(f"[vault-broker] {len(SERVICE_POLICY)} services mapped", flush=True)
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            continue
        threading.Thread(target=_serve_one_unix, args=(conn,), daemon=True).start()


def _serve_forever_win():
    print(f"[vault-broker] listening on {PIPE_NAME} (age keystore, local)",
          flush=True)
    print(f"[vault-broker] {len(SERVICE_POLICY)} services mapped", flush=True)
    while True:
        h = _create_pipe()
        ok = k32.ConnectNamedPipe(h, None)
        if not ok and k32.GetLastError() not in (ERROR_PIPE_CONNECTED, 0):
            k32.CloseHandle(h); continue
        threading.Thread(target=_serve_one, args=(h,), daemon=True).start()


def serve_forever():
    if _IS_WIN:
        _serve_forever_win()
    else:
        _serve_forever_unix()


if __name__ == "__main__":
    serve_forever()
