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

import ipaddress
import secrets
import time
from dataclasses import dataclass
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
# Local owner listener: peer-credential gate against daemon-spawned processes
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


def _peer_pid(peer, server) -> int | None:
    """Pid owning the client end of a TCP connection (psutil), matched on the
    full 4-tuple (rows on the server port are the daemon's own sockets and
    are skipped). None when it cannot be resolved (another
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

    The local listener pre-auths every connection as the OWNER, so no
    process the daemon spawned to run lower-trust code may be served there:
    authored and adopted tools, agent shell commands and agent workers
    (atn/contained_pids.py). Checked by peer credential, never by address
    (they all connect from loopback too). The client socket is found by the
    full 4-tuple, never by ports alone: its owning pid (psutil) is a tracked
    PID or a descendant of one. Checked only while such a run is live. When
    the system socket table cannot name the pid (macOS without root), the
    tracked process trees' own sockets are scanned instead; a scan that
    cannot complete => refused.

    Owner clients (app, ``atn send``, the user's MCP proxy) are not spawned
    by the daemon, so they resolve to an untracked pid and are accepted.
    With no tracked run live, this is a no-op."""
    from . import contained_pids
    if not contained_pids.live():
        return None
    if isinstance(local_address, int):
        local_address = ("127.0.0.1", local_address)
    if _endpoint(remote_address) is None:
        return "peer address unavailable"
    if _endpoint(local_address) is None:
        return "listener address unavailable"
    return _pid_gate(remote_address, local_address)


def _pid_gate(peer, server) -> str | None:
    from . import contained_pids
    pid = _peer_pid(peer, server)
    if pid is not None:
        kind = contained_pids.kind_of(pid)
        if kind:
            return f"peer pid {pid} is a daemon-spawned {kind} process"
        return None
    want_local = _endpoint(peer)
    want_remote = _endpoint(server)

    def _match(c) -> bool:
        if not c.laddr or not c.raddr:
            return False
        return (_endpoint((c.laddr.ip, c.laddr.port)) == want_local
                and _endpoint((c.raddr.ip, c.raddr.port)) == want_remote)

    complete, kind = contained_pids.tree_owner_of(_match)
    if kind:
        return f"peer socket belongs to a daemon-spawned {kind} process"
    if not complete:
        return "peer process unresolved while a daemon-spawned run is live"
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
