#!/usr/bin/env python3
r"""
keystore — age-encrypted secret store + action-broker core.

Design: c:\code\cloud\secrets-tooling\MCP-BROKER-DESIGN.md (action-broker section).

WHAT THIS IS (and honestly is NOT)
----------------------------------
The agent IS the model, and the model drives shell/python tools. So a value the
agent's process can reach, the model can read. We do NOT pretend otherwise.

What this DOES guarantee:
  1. ENCRYPTED AT REST — secrets live in an age-encrypted vault, not plaintext.
     A casual `cat` reveals ciphertext, not values.
  2. CROSS-SERVICE BLAST RADIUS — an agent may only request services on its
     per-agent allowlist. It cannot reach another service's secret.
  3. NO ACCIDENTAL CONTEXT LEAK — a requested secret is written to a FILE; the
     agent is handed a path + variable name, never the value in its response.

What it does NOT guarantee (stated plainly): a determined model can `cat` the
exported file for a service it is ALREADY authorized for. That is accepted.
The tripwire (secret_alarm) remains the backstop for values that do leak into
the transcript.

LAYOUT (all under C:\code\secrets\keystore\, OUTSIDE any repo):
  identity.age-key   -- the broker's age PRIVATE key (decrypts the vault)
  vault.age          -- age-encrypted JSON: { "SERVICE": "secret-value", ... }
  acl.json           -- per-agent allowlist: { "agent": ["service", ...] }
  exports/           -- where decrypted single-secret files are written

The private key file is the one thing that must be protected by the OS (file
perms) — whoever can read it can decrypt the vault. age handles the crypto.
"""

import os
import json
import secrets as _secrets
import tempfile
import pyrage
from pyrage import x25519

# ── Paths (overridable via env) ───────────────────────────────────────────────
KEYSTORE_DIR = os.environ.get(
    "KEYSTORE_DIR",
    os.path.join(os.path.expanduser("~"), ".atn", "keystore"))
KEY_PATH = os.path.join(KEYSTORE_DIR, "identity.age-key")
VAULT_PATH = os.path.join(KEYSTORE_DIR, "vault.age")
BUNDLES_PATH = os.path.join(KEYSTORE_DIR, "bundles.json")
EXPORT_DIR = os.path.join(KEYSTORE_DIR, "exports")
# Session registry: one file per running MCP session, recording what it staged
# and where, so we can clean up on session end AND sweep dead sessions.
SESSIONS_DIR = os.path.join(KEYSTORE_DIR, "sessions")


class KeystoreError(Exception):
    """Base error. Messages never embed a secret value."""


# ── Identity (the broker's age keypair) ───────────────────────────────────────
def init_identity():
    """Create the keystore dir + a fresh age identity if none exists.
    Returns the public key string. Idempotent: won't overwrite an existing key."""
    os.makedirs(KEYSTORE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    if os.path.exists(KEY_PATH):
        return _load_identity().to_public()
    ident = x25519.Identity.generate()
    with open(KEY_PATH, "w", encoding="utf-8") as f:
        f.write(str(ident))
    return str(ident.to_public())


def _load_identity():
    if not os.path.exists(KEY_PATH):
        raise KeystoreError("no keystore identity; run init_identity() first")
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        return x25519.Identity.from_str(f.read().strip())


# ── Vault (age-encrypted JSON of SERVICE -> value) ────────────────────────────
def _read_vault():
    """Decrypt + parse the vault. Returns {} if vault absent."""
    if not os.path.exists(VAULT_PATH):
        return {}
    ident = _load_identity()
    with open(VAULT_PATH, "rb") as f:
        ct = f.read()
    pt = pyrage.decrypt(ct, [ident])
    return json.loads(pt.decode("utf-8"))


def _write_vault(d):
    """Encrypt + write the vault from a dict."""
    ident = _load_identity()
    pub = ident.to_public()
    pt = json.dumps(d).encode("utf-8")
    ct = pyrage.encrypt(pt, [pub])
    with open(VAULT_PATH, "wb") as f:
        f.write(ct)


def put_secret(service, value):
    """Add/update one secret in the encrypted vault. (Admin op, not agent-facing.)"""
    init_identity()
    d = _read_vault()
    d[service] = value
    _write_vault(d)


def delete_secret(service):
    """Remove one secret from the encrypted vault. (Admin op, not agent-facing.)
    Returns True if it existed. The caller regenerates the policy map; live
    staged copies are untouched (session teardown / sweep reclaims them)."""
    d = _read_vault()
    if service not in d:
        return False
    del d[service]
    _write_vault(d)
    return True


def list_services():
    """Service NAMES in the vault (no values). Admin/debug."""
    return sorted(_read_vault().keys())


def get_secret(service):
    """Return one secret's raw VALUE from the age vault, or raise KeyError.

    Broker-side only — the value goes straight to nameless staging + the
    owner-gated monitor push, never back to an agent. This is the self-contained
    replacement for a HashiCorp read: the source of truth is the local
    age-encrypted ``vault.age``, decryptable only by the broker's identity key."""
    d = _read_vault()
    if service not in d:
        raise KeyError(service)
    return d[service]


# ── Bundles (named groups of services, for `cc --secrets crypto`) ─────────────
def _read_bundles():
    if not os.path.exists(BUNDLES_PATH):
        return {}
    try:
        with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    # Each bundle MUST be a list of strings (avoid the substring type-confusion).
    return {name: [s for s in svcs if isinstance(s, str)]
            for name, svcs in data.items() if isinstance(svcs, list)}


def resolve_spec(spec):
    """Expand a launch --secrets spec into a concrete service NAME set.

    spec is a comma-separated mix of:
      'none'  -> empty set
      'all'   -> every service in the vault
      <service> -> that service
      <bundle>  -> the bundle's services (bundles.json)
    Unknown tokens are ignored (they grant nothing — fail closed).
    Returns a sorted list of real service names that exist in the vault.
    """
    if spec is None:
        return []
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        return []
    vault_services = set(_read_vault().keys())
    if any(t.lower() == "none" for t in tokens):
        return []
    if any(t.lower() == "all" for t in tokens):
        return sorted(vault_services)
    bundles = _read_bundles()
    chosen = set()
    for t in tokens:
        if t in bundles:
            chosen.update(bundles[t])
        else:
            chosen.add(t)
    # Only return services that actually exist in the vault (fail closed).
    return sorted(chosen & vault_services)


# ── Opaque staging location ───────────────────────────────────────────────────
# The staged file is a NAMELESS BLOB: random filename, no `.env`, no service
# name, dropped in a scattered-but-writable spot. Nothing on disk advertises
# itself as a secret. Only the requesting session learns where (via the MCP
# response) and only the session registry records it (for cleanup).
def _stage_path():
    """A random, non-descript file path in a writable, sweepable location."""
    base = tempfile.gettempdir()
    # innocuous-looking subdir + extension; pure obfuscation, not security.
    sub = os.path.join(base, _secrets.choice(("cache", "data", ".cache", "tmp")))
    os.makedirs(sub, exist_ok=True)
    name = _secrets.token_hex(8) + _secrets.choice((".dat", ".bin", ".cache", ""))
    return os.path.join(sub, name)


def _default_var(service):
    """Disposable, non-descript env var name. Does NOT reveal the service.
    The authorized session gets this name from the MCP response, so the value
    on disk needs no embedded key."""
    return "TOK_" + _secrets.token_hex(4).upper()


# ── Session registry (what each running session staged, for cleanup) ──────────
def _session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def register_session(session_id, pid, allowed):
    """Record a running session: its pid, the OS-read identity token
    (pid:starttime, defeats PID reuse), and its allowlist. Created at launch.
    The allowlist is fixed here by the launcher (you), never widened later."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    data = {"session_id": session_id, "pid": pid,
            "identity": identity_token(pid),
            "allowed": list(allowed), "staged": {}}
    _write_session(session_id, data)


def _read_session(session_id):
    try:
        with open(_session_file(session_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_session(session_id, data):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f)


def list_sessions():
    """All registered sessions (for the owner's 'who can see what' view)."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out = []
    for fn in os.listdir(SESSIONS_DIR):
        if fn.endswith(".json"):
            d = _read_session(fn[:-5])
            if d:
                out.append(d)
    return out


# ── Broker: the agent-facing request ──────────────────────────────────────────
def request_secret(service, allowed, session_id=None, var_name=None):
    """Stage a secret for use IF it is in this session's `allowed` list.

    `allowed` is the resolved, launch-time service list (from --secrets). The
    model cannot widen it — it's fixed by the launch args, passed in here.

    Decrypts ONLY that one secret, writes its RAW VALUE to a random, nameless
    file in a scattered location, and (if session_id given) records that path in
    the session registry so it can be cleaned up on session end / dead-pid sweep.
    Returns {ok, service, var_name, path} — the VALUE is in the file, never the
    return. The file's name/path reveal nothing; var_name is disposable.
    """
    if service not in allowed:
        raise KeystoreError(f"service '{service}' not in this session's allowlist")

    vault = _read_vault()
    if service not in vault:
        raise KeystoreError(f"service '{service}' has no secret in the vault")

    var_name = var_name or _default_var(service)
    path = _stage_path()
    # RAW value only — no KEY=, no service name embedded. Nameless blob.
    with open(path, "w", encoding="utf-8") as f:
        f.write(vault[service])
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    if session_id:
        d = _read_session(session_id) or {"session_id": session_id,
                                          "pid": None, "allowed": list(allowed),
                                          "staged": {}}
        # one live staged path per service; drop any previous one first.
        old = d["staged"].get(service)
        if old and old != path:
            _safe_unlink(old)
        d["staged"][service] = path
        _write_session(session_id, d)

    return {"ok": True, "service": service, "var_name": var_name, "path": path}


def _safe_unlink(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def revoke_export(service, session_id=None):
    """Delete a staged secret file for `service` in this session. Best-effort."""
    if not session_id:
        return False
    d = _read_session(session_id)
    if not d:
        return False
    path = d["staged"].pop(service, None)
    _write_session(session_id, d)
    return _safe_unlink(path) if path else False


def cleanup_session(session_id):
    """Delete ALL staged files for a session and drop its registry file.
    Called on session end (atexit/signal) and by the dead-pid sweep."""
    d = _read_session(session_id)
    if d:
        for path in list(d.get("staged", {}).values()):
            _safe_unlink(path)
    _safe_unlink(_session_file(session_id))


def sweep_dead_sessions():
    """Owner/maintenance: clean up any session whose pid is no longer alive.
    Catches hard-kills where atexit never ran (e.g. killed terminals)."""
    cleaned = []
    for d in list_sessions():
        pid = d.get("pid")
        if pid is not None and not _pid_alive(pid):
            cleanup_session(d["session_id"])
            cleaned.append(d["session_id"])
    return cleaned


def _pid_alive(pid):
    """True if a process with this pid is running. Cross-platform-ish."""
    if os.name == "nt":
        # Use the Win32 API directly (instant; tasklist is slow/can hang).
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            # Could not open: almost certainly no such process. (Access-denied
            # is possible in theory but our sessions are same-user.)
            return False
        try:
            code = wintypes.DWORD()
            if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True  # fail safe: couldn't determine -> assume alive
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


# ── Process identity: PID + start-time (defeats PID reuse) ────────────────────
# A PID alone is reusable: when a process dies the OS can hand its number to an
# unrelated new process, which would inherit the dead session's allowlist. The
# (pid, creation-time) PAIR is unique for the life of the machine, so we bind
# identity to both. The creation time is read from the OS, never client-claimed.
def pid_start_time(pid):
    """OS-reported process creation time as an int (100ns ticks on Windows,
    clock ticks on POSIX). None if the process is gone / unreadable."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_ = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            ok = k32.GetProcessTimes(h, ctypes.byref(creation),
                                     ctypes.byref(exit_),
                                     ctypes.byref(kernel),
                                     ctypes.byref(user))
            if not ok:
                return None
            return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        finally:
            k32.CloseHandle(h)
    # POSIX: read starttime (field 22) from /proc/<pid>/stat
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            data = f.read()
        # comm may contain spaces/parens; split after the last ')'
        rest = data[data.rfind(")") + 2:].split()
        return int(rest[19])  # starttime is field 22 -> index 19 after pid,comm
    except (OSError, ValueError, IndexError):
        return None


def identity_token(pid):
    """Stable identity for a live process: 'pid:starttime'. None if dead."""
    st = pid_start_time(pid)
    if st is None:
        return None
    return f"{int(pid)}:{st}"


def verify_identity(pid, expected_token):
    """True iff `pid` is alive AND its current start-time matches the token
    recorded at registration. Defeats PID reuse: a recycled PID has a different
    creation time, so the token won't match."""
    if not expected_token:
        return False
    cur = identity_token(pid)
    return cur is not None and cur == expected_token


# ── Daemon-side allowlist lookup (centralized broker) ─────────────────────────
def allowed_for_pid(pid):
    """The allowlist registered for this (pid, start-time). Empty if the PID is
    not registered, is dead, or its start-time no longer matches (reuse). The
    caller PID is supplied by the OS (named-pipe client id), never by the model.
    """
    for d in list_sessions():
        if d.get("pid") == pid:
            if verify_identity(pid, d.get("identity")):
                return list(d.get("allowed", []))
            return []  # registered pid but identity mismatch -> reuse, deny
    return []
