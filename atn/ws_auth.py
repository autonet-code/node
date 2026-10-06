"""Wallet-signature auth for the autonet WS server (atn/ws_server.py, :7700).

The WS server is the single transport between the Flutter frontend and the
agent runtime. Historically it had NO authentication: any client that could
reach the port got an unsolicited full-fleet snapshot and could call any tool
as any agent (caller_id was client-supplied and untrusted).

This module adds the auth primitives — kept here, isolated from WebSocketBridge,
so they are unit-testable without a live socket:

  • Loopback detection — a connection from this machine (127.0.0.0/8, ::1) is
    trusted as the owner with no wallet ("it's your machine"). Everything else
    must authenticate.
  • A nonce / challenge / signature-recovery handshake — a remote client proves
    control of a wallet by signing a server-issued challenge; the server
    recovers the signer address (eth_account) and authorizes accordingly.
  • ClientSession — per-connection auth state: who you are, what subtree you may
    see, and whether you're physically local (which gates key export).

THE PROXY TRAP (and why custody is gated on the LISTENER, not the peer address):
loopback is the TCP peer address. Behind a reverse proxy (wss://autonet.computer
-> nginx -> ws://localhost:7700) EVERY remote user arrives as loopback, so
is_loopback alone cannot tell a real local user from a proxied remote one. The
fix is structural: the daemon runs TWO listeners — a privileged loopback-only
listener (127.0.0.1) that a reverse proxy physically cannot reach from off-box,
and a separate remote listener. ``Session.local`` is set from WHICH listener
accepted the socket (the primary signal), with is_loopback only as
defense-in-depth on the local listener. Private-key export and the no-auth
bypass are permitted only when ``Session.local`` is True. There is deliberately
no ``proxy_trusted`` flag — a static flag is a fail-open escape hatch; the
two-listener split is unspoofable.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Hosts that count as loopback even when ipaddress can't classify the literal.
_LOOPBACK_LITERALS = frozenset({
    "127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost",
})

# How long a challenge is valid before it must be re-issued (replay/expiry).
CHALLENGE_TTL_SECS = 120.0


def is_loopback(remote_address) -> bool:
    """True iff the connection's TCP peer is loopback (this machine).

    ``remote_address`` is websockets' ServerConnection.remote_address: a
    (host, port[, flowinfo, scopeid]) tuple, or None on some transports.

    Fails CLOSED: None or an unparseable host returns False — a connection
    whose origin we cannot prove is treated as remote, never local. (The one
    intentional exception some servers make — None == in-process unix socket ==
    local — is NOT made here: the WS server is TCP, so None means "unknown",
    and unknown must not grant local trust.)"""
    host = None
    if isinstance(remote_address, (tuple, list)) and remote_address:
        host = remote_address[0]
    elif isinstance(remote_address, str):
        host = remote_address
    if not host:
        return False
    if host in _LOOPBACK_LITERALS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Local owner listener: peer-credential gate against guest tool processes
# ---------------------------------------------------------------------------

def _norm_ip(host) -> ipaddress._BaseAddress | None:
    """Parsed address with IPv4-mapped IPv6 folded to IPv4 (None if bad)."""
    try:
        ip = ipaddress.ip_address(str(host).split("%", 1)[0].strip("[]"))
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def _endpoint(addr) -> tuple[ipaddress._BaseAddress, int] | None:
    """(ip, port) from a socket address tuple, or None."""
    try:
        ip = _norm_ip(addr[0])
        port = int(addr[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (ip, port) if ip is not None else None


def _proc_endpoint(field: str) -> tuple[ipaddress._BaseAddress, int] | None:
    """Decode a /proc/net/tcp{,6} ``HEXADDR:HEXPORT`` column. The address is
    the kernel's raw bytes printed as host-order 32-bit words."""
    try:
        hexaddr, hexport = field.rsplit(":", 1)
        port = int(hexport, 16)
        raw = bytes.fromhex(hexaddr)
    except ValueError:
        return None
    if len(raw) not in (4, 16):
        return None
    packed = b"".join(raw[i:i + 4][::-1] for i in range(0, len(raw), 4))
    if sys.byteorder == "big":
        packed = raw
    ip = ipaddress.ip_address(packed)
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped if mapped is not None else ip), port


def _linux_peer_uid(peer, server) -> int | None:
    """Uid owning the CLIENT end of a TCP connection (from /proc/net/tcp{,6}).

    ``peer`` is the accepted connection's remote address, ``server`` its local
    address (both (host, port) tuples). The row must match the full 4-tuple
    (local == peer, remote == server). Rows whose local port is the server
    port are skipped: they are the daemon's own accepted sockets and the
    listener. More than one matching row is ambiguous and resolves to None
    (the caller refuses)."""
    want_local = _endpoint(peer)
    want_remote = _endpoint(server)
    if want_local is None or want_remote is None:
        return None
    found: list[int] = []
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as fh:
                next(fh, None)
                for line in fh:
                    cols = line.split()
                    if len(cols) < 8:
                        continue
                    local = _proc_endpoint(cols[1])
                    remote = _proc_endpoint(cols[2])
                    if local is None or remote is None:
                        continue
                    if local[1] == want_remote[1]:
                        continue
                    if local == want_local and remote == want_remote:
                        try:
                            found.append(int(cols[7]))
                        except ValueError:
                            return None
        except OSError:
            continue
    return found[0] if len(found) == 1 else None


def _peer_pid(peer, server) -> int | None:
    """Pid owning the client end of a TCP connection (psutil), matched on the
    full 4-tuple like _linux_peer_uid. None when it cannot be resolved (another
    uid's socket, no permission) or when the match is ambiguous."""
    want_local = _endpoint(peer)
    want_remote = _endpoint(server)
    if want_local is None or want_remote is None:
        return None
    try:
        import psutil
        found: list[int | None] = []
        for c in psutil.net_connections(kind="tcp"):
            if not c.laddr or not c.raddr:
                continue
            local = _endpoint((c.laddr.ip, c.laddr.port))
            remote = _endpoint((c.raddr.ip, c.raddr.port))
            if local is None or remote is None or local[1] == want_remote[1]:
                continue
            if local == want_local and remote == want_remote:
                found.append(c.pid)
    except Exception:  # noqa: BLE001
        return None
    return found[0] if len(found) == 1 else None


def local_peer_denied(remote_address, local_address) -> str | None:
    """Reason to refuse a connection on the privileged local listener, or
    None to allow it.

    ``remote_address`` / ``local_address`` are the accepted socket's peer and
    local (host, port) tuples. A bare int ``local_address`` is read as the
    port on 127.0.0.1 (older callers).

    The local listener pre-auths every connection as the OWNER, so a guest
    tool process (atn/guest_sandbox.py) must never be served there. Checked by
    peer credential, never by address (guests connect from loopback too). The
    client socket is found by the full 4-tuple, never by ports alone:

    - separate-uid guests (ATN_GUEST_UID, set by atn.guest_launcher): the
      client socket's owning uid, from /proc/net/tcp. Unresolvable or
      ambiguous while uid isolation is configured => refused (fail closed).
    - same-uid fallback guests: the client socket's owning pid (psutil) is a
      tracked guest PID or a descendant of one. Checked only while a guest run
      is live; unresolvable during one => refused.

    With no guest uid configured and no guest run live, this is a no-op and
    the listener behaves exactly as before."""
    from .guest_sandbox import guest_uids, live_guest_pids
    uids = guest_uids()
    live = live_guest_pids()
    if not uids and not live:
        return None
    if isinstance(local_address, int):
        local_address = ("127.0.0.1", local_address)
    if _endpoint(remote_address) is None:
        return "peer address unavailable"
    if _endpoint(local_address) is None:
        return "listener address unavailable"
    if uids:
        if not sys.platform.startswith("linux"):
            return None if not live else _pid_gate(remote_address, local_address)
        uid = _linux_peer_uid(remote_address, local_address)
        if uid is None:
            return "peer credentials unavailable while guest uid isolation is on"
        if uid in uids:
            return f"peer uid {uid} is a guest tool uid"
    if live:
        return _pid_gate(remote_address, local_address)
    return None


def _pid_gate(peer, server) -> str | None:
    from .guest_sandbox import pid_is_guest
    pid = _peer_pid(peer, server)
    if pid is None:
        return "peer process unresolved while a guest tool is running"
    if pid_is_guest(pid):
        return f"peer pid {pid} is a guest tool process"
    return None


def new_nonce() -> str:
    """A fresh, unguessable challenge nonce."""
    return secrets.token_hex(16)


DAEMON_ID_FILE = "daemon_id"


def load_or_create_daemon_id(data_dir: Path) -> str:
    """The daemon's stable per-install identity (random, persisted in the data
    dir on first use). Independent of agents, load order and the owner wallet,
    so the auth challenge's domain separation never shifts when the fleet
    changes. Falls back to an in-memory id if the data dir is unwritable."""
    path = Path(data_dir) / DAEMON_ID_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    new_id = "atn-daemon-" + secrets.token_hex(16)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id, encoding="utf-8")
    except OSError:
        pass
    return new_id


def build_challenge_text(nonce: str, *, daemon_id: str, chain_id: int,
                         owner_wallet: str, conn_id: str,
                         issued_at: float | None = None) -> str:
    """The exact human-readable string the wallet signs (EIP-4361-style).

    Sent verbatim to the client; the client signs the literal text it received
    (no client-side reconstruction). The server re-derives the same text from
    its OWN stored nonce + conn_id to recover the signer — it never trusts a
    client-echoed nonce.

    Domain separation (finding 6 of the security review): a bare host:port is
    identical across every Autonet daemon, so a signature for daemon A would
    replay on daemon B. We bind the signature to:
      - daemon_id: the daemon's persisted per-install id (load_or_create_daemon_id),
      - chain_id: so a testnet signature can't authorize a mainnet daemon,
      - owner_wallet: the address this challenge expects to recover,
      - conn_id: a server-random per-connection id, so a signature lifted from
        one socket cannot be replayed on another.
    """
    issued = issued_at if issued_at is not None else time.time()
    return (
        "Autonet daemon authentication\n"
        "version: 1\n"
        f"daemon: {daemon_id}\n"
        f"chain: {chain_id}\n"
        f"owner: {owner_wallet}\n"
        f"connection: {conn_id}\n"
        f"nonce: {nonce}\n"
        f"issued: {int(issued)}"
    )


def recover_signer(challenge_text: str, signature: str) -> str | None:
    """Recover the wallet address that signed ``challenge_text``.

    Returns the checksummed address, or None on any failure. Fails CLOSED if
    eth_account is unavailable — unlike nodes/common/crypto.verify_signature,
    which returns True in mock mode (a verification bypass we must NOT copy for
    an access-control gate)."""
    if not signature:
        return None
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        return None
    try:
        msg = encode_defunct(text=challenge_text)
        return Account.recover_message(msg, signature=signature)
    except Exception:
        return None


@dataclass
class ClientSession:
    """Per-connection auth + scope state. One per WS connection.

    A connection on the privileged LOCAL listener is constructed pre-authed as
    the owner with full-fleet scope (``local=True, authed=True``) —
    preserving the localhost-is-full-control behavior. A connection on the
    remote listener starts unauthed and must complete the challenge handshake.

    ``local`` is set from the accepting listener (primary custody signal), not
    from remote_address. scope_ids: None means "full fleet" (root scope);
    a set restricts the snapshot + event stream + tool targets to that subtree.
    """

    local: bool = False                    # accepted on the privileged listener
    is_loopback: bool = False              # TCP peer is loopback (defense-in-depth)
    authed: bool = False
    owner: bool = False
    # The agent this session is rooted at; None = unscoped (full-fleet) owner.
    root_agent_id: str | None = None
    scope_ids: set[str] | None = None      # None = full fleet
    wallet_address: str = ""
    nonce: str | None = None
    nonce_issued_at: float = 0.0
    conn_id: str = ""                      # server-random, binds a sig to this socket
    auth_failures: int = 0                 # rate-limit / lockout counter
    # Outbound event delivery (set by the WS server on connect). Events are
    # queued and written by a per-connection writer task so a slow client's
    # TCP backpressure can never stall the runtime's EventBus.
    event_queue: object | None = None      # asyncio.Queue[str]
    dropped_events: int = 0                # count of events dropped on overflow

    def challenge_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.nonce_issued_at) > CHALLENGE_TTL_SECS

    def clear_nonce(self) -> None:
        """Consume the nonce so a signature can't be replayed on this
        connection (single-use challenge)."""
        self.nonce = None


def assert_local_key_access(session: "ClientSession") -> bool:
    """The localhost-only gate for private-key export (locked decision #3).

    True only if this connection arrived on the privileged LOCAL listener.
    Keys off ``session.local`` (set from the accepting listener), NOT off
    auth/owner status — even a proven owner over the remote listener cannot
    extract keys. Gating on the listener (not remote_address) is what makes
    this unspoofable behind a reverse proxy."""
    return session.local


# ---------------------------------------------------------------------------
# Integration listener: per-agent bearer tokens (docs/integration_listener.md)
# ---------------------------------------------------------------------------
#
# A THIRD listener (default 127.0.0.1:7710) for a guest harness (e.g. the
# Odysseus sidecar) that must act as exactly ONE agent without holding that
# agent's private key or the owner wallet. The credential is an opaque bearer
# token minted by the owner on the daemon host (``atn integration-token
# create``). Only its sha256 is stored; the plaintext is shown once at mint.
# Each token is bound to exactly one agent id, and the server forces
# caller_id to that id on every call: the token is never the owner and never
# any other agent. It is NOT a money key: it reaches no wallet, transfer,
# signing or key-export path (see INTEGRATION_ALLOWED_MESSAGES in
# ws_server.py).

INTEGRATION_TOKENS_FILE = "integration_tokens.json"
INTEGRATION_TOKEN_PREFIX = "atn_it_"
DEFAULT_INTEGRATION_PORT = 7710
DEFAULT_INTEGRATION_HOST = "127.0.0.1"

# Agent ids an integration token may NEVER be bound to: every string that
# atn.agent_tools.is_owner_caller() treats as the owner ("", "user"; None is
# excluded by the str check). Literals rather than an
# import so this module stays import-light; tests/test_integration_listener.py
# asserts the two agree.
_OWNER_SENTINELS = frozenset({"", "user"})


def hash_integration_token(token: str) -> str:
    """sha256 hex of a presented token (what is stored and compared)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_bindable_agent_id(agent_id: object) -> bool:
    """True iff ``agent_id`` is a non-empty, whitespace-free-at-the-edges
    string that is not an owner sentinel. The integration clamp relies on
    this: a token bound to a sentinel would make every call owner-trusted
    downstream (is_owner_caller)."""
    if not isinstance(agent_id, str):
        return False
    aid = agent_id.strip()
    return bool(aid) and aid == agent_id and aid not in _OWNER_SENTINELS


def parse_bearer(header_value: str | None) -> str:
    """Extract the token from an ``Authorization: Bearer <token>`` header.
    Returns "" when absent or malformed (fails closed)."""
    if not header_value or not isinstance(header_value, str):
        return ""
    parts = header_value.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    token = parts[1].strip()
    if not token.startswith(INTEGRATION_TOKEN_PREFIX):
        return ""
    return token


@dataclass
class IntegrationToken:
    """One stored token record. ``token_hash`` is sha256(plaintext); the
    plaintext is never persisted. ``token_id`` is a short public handle (a
    hash prefix) used by list/revoke and in logs."""

    token_id: str
    token_hash: str
    agent_id: str
    label: str = ""
    created_at: float = 0.0
    revoked_at: float | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def to_dict(self) -> dict:
        return {
            "id": self.token_id,
            "hash": self.token_hash,
            "agent_id": self.agent_id,
            "label": self.label,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
        }

    def public_view(self) -> dict:
        """Listing view: everything but the hash."""
        return {
            "id": self.token_id,
            "agent_id": self.agent_id,
            "label": self.label,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
            "active": self.active,
        }


class IntegrationTokenStore:
    """Hashed bearer-token store at ``<data_dir>/integration_tokens.json``.

    The CLI writes it; the daemon reads it. The daemon re-reads the file
    whenever its mtime/size changes and re-validates on EVERY message, so a
    ``revoke`` from the CLI takes effect on live connections without a
    restart. Writes are atomic (temp file + os.replace) and 0600 on POSIX.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.path = Path(data_dir) / INTEGRATION_TOKENS_FILE
        self._lock = threading.Lock()
        self._records: dict[str, IntegrationToken] = {}   # keyed by token_hash
        self._stamp: tuple[int, int] | None = None
        self._loaded = False

    # -- persistence -------------------------------------------------------

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_if_changed(self) -> None:
        stamp = self._file_stamp()
        if self._loaded and stamp == self._stamp:
            return
        records: dict[str, IntegrationToken] = {}
        if stamp is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # Unreadable/corrupt store: fail CLOSED (no token valid)
                # rather than keep serving a stale in-memory copy.
                raw = {}
            rows = raw.get("tokens") if isinstance(raw, dict) else None
            for row in rows if isinstance(rows, list) else []:
                try:
                    rec = IntegrationToken(
                        token_id=str(row["id"]),
                        token_hash=str(row["hash"]),
                        agent_id=str(row["agent_id"]),
                        label=str(row.get("label") or ""),
                        created_at=float(row.get("created_at") or 0.0),
                        revoked_at=(float(row["revoked_at"])
                                    if row.get("revoked_at") is not None else None),
                    )
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
                records[rec.token_hash] = rec
        self._records = records
        self._stamp = stamp
        self._loaded = True

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        body = json.dumps(
            {"version": 1,
             "tokens": [r.to_dict() for r in self._records.values()]},
            indent=2)
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._stamp = self._file_stamp()

    # -- operations --------------------------------------------------------

    def create(self, agent_id: str, label: str = "") -> tuple[str, IntegrationToken]:
        """Mint a token bound to ``agent_id``. Returns (plaintext, record);
        the plaintext is not recoverable afterwards."""
        if not is_bindable_agent_id(agent_id):
            raise ValueError(
                f"refusing to bind an integration token to {agent_id!r}: "
                "owner sentinels (empty, 'user') are never "
                "a valid integration identity")
        with self._lock:
            self._reload_if_changed()
            token = INTEGRATION_TOKEN_PREFIX + secrets.token_urlsafe(32)
            h = hash_integration_token(token)
            rec = IntegrationToken(token_id=h[:12], token_hash=h,
                                   agent_id=agent_id, label=label,
                                   created_at=time.time())
            self._records[h] = rec
            self._write()
            return token, rec

    def revoke(self, ident: str) -> list[IntegrationToken]:
        """Revoke by token id (hash prefix) or exact label. Returns the
        records newly revoked (empty if nothing matched)."""
        if not ident:
            return []
        with self._lock:
            self._reload_if_changed()
            hit = [r for r in self._records.values()
                   if r.active and (r.token_id == ident or r.label == ident)]
            now = time.time()
            for r in hit:
                r.revoked_at = now
            if hit:
                self._write()
            return hit

    def list(self) -> list[IntegrationToken]:
        with self._lock:
            self._reload_if_changed()
            return sorted(self._records.values(), key=lambda r: r.created_at)

    def verify(self, token: str) -> IntegrationToken | None:
        """The active record for a presented plaintext token, else None.
        Re-reads the store if it changed on disk (CLI revocation)."""
        if not token or not token.startswith(INTEGRATION_TOKEN_PREFIX):
            return None
        h = hash_integration_token(token)
        with self._lock:
            self._reload_if_changed()
            rec = self._records.get(h)
        if rec is None or not rec.active:
            return None
        if not is_bindable_agent_id(rec.agent_id):
            return None     # hand-edited store binding a sentinel: refuse
        return rec

    def lookup_active(self, token_hash: str) -> IntegrationToken | None:
        """The active record for a stored hash (per-message re-check)."""
        with self._lock:
            self._reload_if_changed()
            rec = self._records.get(token_hash)
        if rec is None or not rec.active:
            return None
        return rec


class TokenBucket:
    """Per-token rate limiter, shared across that token's connections.
    Refills ``rate`` units/sec up to ``burst``."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = float(rate)
        self.burst = float(burst)
        self._level = float(burst)
        self._last = time.monotonic()

    def allow(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._level = min(self.burst, self._level + (now - self._last) * self.rate)
        self._last = now
        if self._level >= 1.0:
            self._level -= 1.0
            return True
        return False


@dataclass
class IntegrationSession:
    """Per-connection state on the integration listener. ``agent_id`` is
    fixed at the handshake from the token record and is the ONLY caller_id
    this connection ever dispatches with."""

    token_id: str
    token_hash: str
    agent_id: str
    label: str = ""
    conn_id: str = ""
