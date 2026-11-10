"""WebSocket server — the single transport for ATN's frontend.

One WebSocket connection handles everything:
  - Request-response: client sends {type, msg_id, ...}, server responds {msg_id, result}
  - Event streaming: server pushes {type: "event", data: ...} from EventBus

Protocol:
  Client -> Server (requests):
    {"type": "list_agents", "msg_id": "1"}
    {"type": "trigger_run", "msg_id": "2", "agent_id": "echo01"}
    {"type": "create_agent", "msg_id": "3", "id": "myagent", "name": "...", "steps": [...]}

  Server -> Client (responses):
    {"msg_id": "1", "ok": true, "result": {...}}
    {"msg_id": "2", "ok": false, "error": "Agent not found"}

  Server -> Client (events, no msg_id):
    {"type": "event", "event_type": "execution.started", "source": "echo01", "data": {...}}

Note: `msg_id` is the protocol correlation field.  `id` is free for tool arguments
(e.g. agent id in create_agent).

Run standalone:  python -m atn.ws_server
Or start from Runtime:  await ws_server.start(runtime, port=7700)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import Server as WSServer, ServerConnection

from . import ws_auth
from .agent_tools import execute_tool
from .events import Event, EventBus, EventType
from .runtime import Runtime
from .runtime.provider_manager import get_model_tier, get_tier_label
from .ws_auth import ClientSession
from nodes.common.tx_nonce import next_nonce

log = logging.getLogger(__name__)

# Suppress noisy websockets handshake errors (e.g. bare TCP probes from
# port-in-use checks, or clients connecting before the upgrade completes).
# These are non-fatal — the library handles them — but the full tracebacks
# confuse users.
logging.getLogger("websockets").setLevel(logging.CRITICAL)

# Default port
DEFAULT_PORT = 7700

# Messages that export or carry a raw private key over the wire. Refused unless
# the connection arrived on the privileged LOCAL listener (session.local). The
# inbound-private-key handlers are here too: over a proxied/remote link the WS
# hop is plain ws:// (TLS terminates at the proxy), so a private key in the
# payload would cross the wire in plaintext. Checked at the top of
# _handle_message, before any dispatch — so the execute_tool fallthrough cannot
# bypass it.
# RepToken (OpenZeppelin ERC20Votes + ERC20Permit) surface the operator
# flow uses: delegation state and delegateBySig.
_REP_DELEGATION_ABI = [
    {"name": "delegates", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "address"}]},
    {"name": "name", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string"}]},
    {"name": "nonces", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "delegateBySig", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "delegatee", "type": "address"},
                {"name": "nonce", "type": "uint256"},
                {"name": "expiry", "type": "uint256"},
                {"name": "v", "type": "uint8"},
                {"name": "r", "type": "bytes32"},
                {"name": "s", "type": "bytes32"}],
     "outputs": []},
]

# Gas budget per operator transaction (claimFor / castVote / registerOption)
# for the operator's low-gas estimate; generous on purpose.
_OPERATOR_TX_GAS = 500_000

KEY_LOCAL_ONLY_MESSAGES = frozenset({
    "export_agent_key",
    "export_operator_key",
    # DEPRECATED (pre-substrate Autonet.sol, removed Phase 5.6a): the
    # handlers now return a gone-marker. Kept in the set so an old client
    # that still sends a private key in the payload is refused on the remote
    # listener before the payload is read at all.
    "autonet_publish_standards",
    "autonet_claim_reward",
    "register_agent_on_chain",
    # Vault mutations carry a raw secret VALUE in the payload; over a
    # proxied/remote link the WS hop is plain ws://, so it would cross the
    # wire in plaintext. Reads (secrets_status/usage/alarms/probe) are
    # names-only and stay owner-gated but remotely reachable.
    "secrets_put",
    "secrets_delete",
    # secrets_import reads a raw VALUE out of the host environment and writes
    # it into the vault — same custody profile as secrets_put (the value never
    # leaves the daemon, but the mutation must be owner+local).
    "secrets_import",
    # secrets_config MUTATES daemon config (worker isolation, root allowance).
    # Same custody tier as the vault mutations: owner + local listener only.
    "secrets_config",
    # set_owner_wallet REASSIGNS fleet ownership: the address it writes both
    # owns all future earnings and is the one a remote connection may sign
    # with to be rooted at the full fleet. Local listener only, so it can
    # never be handed away over the wire.
    "set_owner_wallet",
})

# Messages that hand over ownership or a key. The local listener trusts the
# machine, and an external agent runs on the same machine with a shell, so
# these also need the owner wallet's signature over a one-use challenge
# (``owner_challenge``): the agent has no access to the owner's wallet.
OWNER_SIGNED_MESSAGES = frozenset({
    "export_agent_key",
    "export_operator_key",
    "set_owner_wallet",
})
_OWNER_CHALLENGE_TTL_S = 300


def owner_challenge_text(action: str, nonce: str) -> str:
    """What the owner's wallet signs (EIP-191 personal_sign) to authorize
    one ``action``."""
    return f"Autonet owner action: {action}\nnonce: {nonce}"


# Messages a STRANGER may send on the remote (auth-required) listener without
# completing the owner/agent-self handshake, because they carry their OWN
# authorization and are useless without it.
#
# ``service_request`` is the only one: a paying counterparty on another machine
# is by construction not the daemon's owner and holds no agent key in this
# daemon's fleet, so there is no handshake it could ever pass. What authorizes
# it is the PAYMENT — ``_validate_service_payment`` fetches the on-chain
# ServicePayment receipt, checks recipient == the serving agent and amount >=
# the ask, and consumes the request_id against a persisted replay set. That is a
# strictly stronger credential than a session login: it costs ATN per call.
# See docs/services_market.md §3 ("authenticates the channel, validates
# vouchers"). Nothing else is reachable from such a session.
PAYMENT_AUTHORIZED_MESSAGES = frozenset({
    "service_request",
})

# The secrets vault surface (owner-only, HUMAN-ONLY by construction: these
# exist solely as WS handlers and are never registered in any agent tool
# surface). Values are WRITE-ONLY — no handler ever returns one.
_SECRETS_MESSAGES = frozenset({
    "secrets_status",
    "secrets_put",
    "secrets_delete",
    "secrets_usage_log",
    "secrets_alarms",
    # Host-discovery surface: secrets_probe reads names-only host-scan results
    # (read-gated, same tier as secrets_status); secrets_import imports named
    # host env vars into the vault (mutating, forced onto the local listener
    # via KEY_LOCAL_ONLY_MESSAGES).
    "secrets_probe",
    "secrets_import",
    # secrets_config: owner-gated GET/SET of daemon security settings (worker
    # isolation, default root allowance). Empty payload = read-only GET;
    # mutations are forced onto the local listener via KEY_LOCAL_ONLY_MESSAGES.
    "secrets_config",
    # Authorized-egress-host binding (plaintext metadata; hostnames are NOT
    # secret). secrets_set_hosts mutates the sidecar; secrets_suggest_hosts is
    # a read-only curated lookup.
    "secrets_set_hosts",
    "secrets_suggest_hosts",
})

# Tools an authed-but-SCOPED (non-full-fleet) session may NOT call: they read
# or mutate owner-global state (budgets, profile/PII, providers, governance).
# A full-fleet owner session (local or remote) is unaffected. Default-deny
# still applies on top: see _authorize_tool_call.
# Chat attachments (clipboard paste / attach button): size ceiling for one
# upload. Generous for screenshots and documents; a session that needs more
# should reference the file by path instead of uploading it.
_ATTACHMENT_MAX_BYTES = 15 * 1024 * 1024

OWNER_ONLY_TOOLS = frozenset({
    "set_credit_budget", "get_credit_budget", "get_user_profile",
    "update_user_profile",
    "set_agent_model",
    # Group-thread administration (docs/group_chat.md): membership, seats,
    # full transcripts and deletion are the owner's. group_post/list/read/
    # create stay agent-callable and membership-checked.
    "group_get", "group_add_member", "group_remove_member",
    "group_take_seat", "group_release_seat", "group_delete",
})

# Inline handlers a SCOPED session (a delegated, non-full-fleet remote
# session, root_agent_id set) may reach (survey 2026-10-26). Every other
# message handled inline in _handle_message is refused for such a session:
# providers, OAuth, budgets, sponsors, wallets and ownership, chain
# registration, services that spend the owner's wallet, robots, sanctions,
# Claude sessions, voice, daemon control. Agent tools still fall through to
# execute_tool, where _authorize_tool_call clamps them to the subtree. The
# set of inline handlers is read from _handle_message's own source
# (_inline_message_types), so a handler added later is denied by default
# until it is listed here.
SCOPED_INLINE_ALLOWED = frozenset({
    # Handshake, the scoped snapshot, the mic.
    "auth_response", "auth_status", "snapshot", "input_request",
    "input_status", "service_request",
    # The subtree's own agents: chat, delegates, processes, context, prompt.
    "send_agent_message", "delegate_message", "interrupt_delegate",
    "get_delegates", "get_delegate_output", "reset_agent_conversation",
    "agent_processes_list", "agent_process_output", "agent_process_stop",
    "session_stats", "session_context", "context_breakdown",
    "agent_system_prompt", "attachment_upload", "post_message",
    "agent_activity",
    # Public network and library reads (no owner state, no authority).
    "probe_tools", "tool_reviews", "tool_surface", "list_registered_tools",
    "search_model_status",
    "autonet_status", "epoch_status", "epoch_history", "economy_graph",
    "market_services", "list_services", "service_topics",
    "rpb_state", "network_status", "rpb_agent_training",
})

# Allowed inline messages that act on ONE agent: a scoped session must name
# it (and it must be in the subtree). Without one they act fleet-wide.
SCOPED_REQUIRES_AGENT = frozenset({
    "send_agent_message", "delegate_message", "interrupt_delegate",
    "get_delegates", "get_delegate_output", "reset_agent_conversation",
    "agent_processes_list", "agent_process_output", "agent_process_stop",
    "session_stats", "session_context", "context_breakdown",
    "agent_system_prompt", "agent_activity",
})

# Not logged to an external agent's Activity: reading the log itself and the
# connection handshake.
_UNLOGGED = frozenset({"agent_activity", "auth_response", "auth_status",
                       "snapshot"})

# For scoped sessions, the message keys that name a target agent. Every one
# present must resolve to an agent inside the session's subtree, else the call
# is refused — clamping caller_id alone is insufficient because these tools read
# the target straight from the message and ignore caller_id.
_TARGET_ARG_KEYS = ("agent_id", "target", "parent_id", "id")

# Chain ids network_status names. Anything else is reported by number
# only; the label must never imply value that isn't there.
_CHAIN_LABELS = {
    127823: "Etherlink Shadownet (testnet)",
}


from dataclasses import dataclass


@dataclass(frozen=True)
class WSAuthor:
    """Platform-neutral author for a WS-originated inbound message.

    Adapts a ClientSession into the `author` an InputPolicy.evaluate() expects
    (exposes at least ``id`` and ``is_bot``). The extra fields (local / owner /
    conn_id / root_agent_id) are what a future InputArbiter needs to build a
    SurfaceId and reason about ownership — threaded now so P3 can gate without
    re-plumbing. A WS client is always a human driver, never a bot."""

    id: str                          # the session's root agent id ("" = unscoped owner)
    conn_id: str                     # server-random per-connection id (SurfaceId.instance)
    local: bool = False              # arrived on the privileged loopback listener
    owner: bool = False              # authed as the daemon owner
    root_agent_id: str | None = None # None = unscoped (full-fleet) owner session
    is_bot: bool = False             # a WS client is a human driver, never a bot

    @classmethod
    def from_session(cls, session: ClientSession) -> "WSAuthor":
        return cls(
            id=session.root_agent_id or "",
            conn_id=session.conn_id,
            local=session.local,
            owner=session.owner,
            root_agent_id=session.root_agent_id,
        )


class WebSocketBridge:
    """Bridges the ATN Runtime to WebSocket clients.

    Handles:
      - Routing incoming JSON messages to agent tools
      - Broadcasting EventBus events to all connected clients
    """

    def __init__(self, runtime: Runtime, host: str = "localhost", port: int = DEFAULT_PORT,
                 *, remote_host: str = "", remote_port: int = 0,
                 owner_wallet: str = "") -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        # The REMOTE (auth-required) listener. Empty remote_host => disabled
        # (local-only daemon, the default). Custody is gated on WHICH listener
        # accepted a connection — never on remote_address — because a reverse
        # proxy makes every remote peer look like loopback. The privileged
        # local listener a proxy physically cannot reach from off-box is the
        # only place keys export or the no-auth bypass apply.
        self.remote_host = remote_host
        self.remote_port = remote_port or (port + 1)
        self.owner_wallet = owner_wallet
        # One-use owner challenges: nonce -> (action, expiry).
        self._owner_challenges: dict[str, tuple[str, float]] = {}
        self._server: WSServer | None = None          # local (privileged) listener
        self._remote_server: WSServer | None = None    # remote (auth) listener
        # Per-connection auth/scope state (replaces the old flat client set).
        self._sessions: dict[ServerConnection, ClientSession] = {}
        self._event_handler_registered = False
        # Input-seam policy (Surface contract). AllowAll by default; the runtime
        # overrides it via register_surface() -> _make_ws_input_policy(). The
        # single-writer decision itself lives in the runtime-owned InputArbiter
        # (a gate ABOVE this policy), not here — this stays the per-surface seam.
        from .surface import AllowAll, InputPolicy  # local import avoids cycle
        self.policy: InputPolicy = AllowAll()
        # Host-discovery runner (OS probe). Lazily built on first secrets_probe
        # so a daemon that never opens the Secrets tab pays no boot cost. Loads
        # the cached host_scan.json at construction.
        self._host_discovery: Any = None

    def _get_host_discovery(self) -> Any:
        """Lazily construct the HostDiscovery runner (cache-backed)."""
        if self._host_discovery is None:
            from .runtime.host_discovery import HostDiscovery
            self._host_discovery = HostDiscovery(self.runtime._config.data_dir)
        return self._host_discovery

    @property
    def _clients(self):
        """Back-compat read accessor: the set of live connections. Internal
        code now uses self._sessions; this keeps any external reader working."""
        return set(self._sessions.keys())

    async def start(self) -> None:
        """Start the WebSocket server(s): always the privileged local listener;
        the remote listener too if configured (and owner_wallet is set)."""
        # Subscribe to all EventBus events (once; both listeners share it).
        if not self._event_handler_registered:
            self.runtime.events.subscribe(None, self._on_event)
            self._event_handler_registered = True

        # §12: give the InputArbiter a liveness view over ws: tokens so hand-off
        # skips dead ws surfaces. Backward-compat: only if the arbiter supports
        # the setter (older runtimes constructed it without one).
        arbiter = getattr(self.runtime, "input_arbiter", None)
        setter = getattr(arbiter, "set_liveness_predicate", None)
        if callable(setter):
            setter(self._ws_token_live)

        # Paid service calls over libp2p (/autonet/service/1.0.0) land on the
        # same handler as a WS `service_request` from a stranger.
        autonet = getattr(self.runtime, "autonet", None)
        setter = getattr(autonet, "set_service_request_handler", None)
        if callable(setter):
            setter(self._serve_p2p_service_request, asyncio.get_running_loop())

        # Privileged local listener — full control, key export, no handshake.
        self._server = await websockets.serve(
            lambda ws: self._handle_client(ws, local=True),
            self.host,
            self.port,
        )
        log.info("WebSocket server (local) listening on ws://%s:%d", self.host, self.port)

        # The agent API (atn/agent_api.py): plain HTTP on loopback for agents
        # in other harnesses. ATN_API_PORT overrides; default port + 2.
        self.agent_api = None
        try:
            from .agent_api import AgentAPI
            api_port = int(os.environ.get("ATN_API_PORT") or (self.port + 2))
            api = AgentAPI(self, "127.0.0.1", api_port)
            await api.start()
            self.agent_api = api
        except Exception as exc:                          # noqa: BLE001
            log.warning("Agent API did not start: %s", exc)

        # Remote listener — auth-required, export-denied. Refuses to start
        # without a pre-configured owner_wallet (no trust-on-first-use on a
        # network-reachable socket).
        # The remote listener is safe to start whenever a bind host is given:
        # BOTH auth paths fail closed — a signer that is neither the configured
        # owner nor a known agent address is denied. (There is no
        # trust-on-first-use anywhere.) With no owner_wallet, only agent
        # self-auth works — a holder of an agent's own key signs in as that
        # agent, scoped to its subtree — which needs no owner configured.
        if self.remote_host:
            self._remote_server = await websockets.serve(
                lambda ws: self._handle_client(ws, local=False),
                self.remote_host,
                self.remote_port,
            )
            auth_modes = []
            if self.owner_wallet:
                auth_modes.append("owner")
            auth_modes.append("agent-self")
            log.info(
                "WebSocket server (remote, auth-required: %s) listening on "
                "ws://%s:%d", "+".join(auth_modes), self.remote_host, self.remote_port)

    async def stop(self) -> None:
        """Stop the WebSocket server(s)."""
        for srv_attr in ("_server", "_remote_server"):
            srv = getattr(self, srv_attr)
            if srv:
                srv.close()
                await srv.wait_closed()
                setattr(self, srv_attr, None)
        api = getattr(self, "agent_api", None)
        if api is not None:
            await api.stop()
            self.agent_api = None
        if self._event_handler_registered:
            self.runtime.events.unsubscribe(None, self._on_event)
            self._event_handler_registered = False
        log.info("WebSocket server stopped")

    # ------------------------------------------------------------------
    # Surface contract (see atn/surface.py)
    #
    # The WS bridge is a first-class registered Surface: a long-lived,
    # bidirectional bridge with an input seam (self.policy) and a lifecycle
    # (start/stop). Output already flows via the shared EventBus, so the bridge
    # contributes no agent-facing tools of its own — the browser IS the tool
    # surface. agent_tools/call_surface_tool exist to structurally satisfy the
    # Surface protocol; both are inert.
    # ------------------------------------------------------------------

    def _surface_id_for(self, session: "ClientSession") -> "SurfaceId":
        """Build the InputArbiter SurfaceId for a WS connection. Each connection
        is its own surface: kind 'ws', instance = the per-connection conn_id
        (its token). in_process mirrors session.local (a loopback client shares
        the daemon's box; a remote peer does not)."""
        from .input_arbiter import SurfaceId
        label = "Local app" if session.local else (
            f"Remote {session.root_agent_id}" if session.root_agent_id else "Remote app")
        return SurfaceId(kind="ws", instance=session.conn_id,
                         label=label, in_process=session.local)

    def _register_input_surface(self, session: "ClientSession") -> None:
        """Register (or re-register, to refresh the label) this connection with
        the InputArbiter. Idempotent on the conn_id token."""
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is not None:
            arbiter.register(self._surface_id_for(session))

    def _arbiter_gate(self, session: "ClientSession") -> dict | None:
        """Single-writer gate for the WS sibling input path (delegate_message)
        that bypasses send_agent_message. Returns None if
        this surface may write (holds the mic, or auto-acquired a free mic),
        else a deny payload to merge into the error response. Same is_active
        semantics as the send_agent_message chokepoint, so the three paths
        acquire/deny identically."""
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is None:
            return None
        sid = self._surface_id_for(session)
        if arbiter.is_active(sid):
            return None
        # §12 ghost-holder recovery: the denial may be because a dead ws:
        # surface is still holding the mic (its session closed but release_for
        # never fired, or it was captured before disconnect). Validate the
        # holder; if it's a ws: token with no live session, release it (auto-
        # hands to a live surface / frees the mic) and re-run the gate ONCE.
        holder = arbiter.holder_token()
        if holder and holder.startswith("ws:") and not self._ws_token_live(holder):
            log.info("[arbiter] gate denied to %s; holder %s is a dead ws surface — releasing",
                     sid.token, holder)
            arbiter.release_for(holder)
            if arbiter.is_active(sid):
                return None
        return {"error": "not the active input surface",
                "code": "input_not_active",
                "holder": arbiter.holder_token()}

    def _ws_token_live(self, token: str) -> bool:
        """Liveness predicate for the InputArbiter: is a ``ws:<conn_id>`` token
        backed by a still-connected authed session? Only ws tokens are judged;
        the arbiter treats every other kind as live. Passed to the arbiter at
        start() and reused by _arbiter_gate."""
        if not token.startswith("ws:"):
            return True
        conn_id = token[len("ws:"):]
        for sess in self._sessions.values():
            if sess.conn_id == conn_id:
                return True
        return False

    def agent_tools(self, agent_id: str) -> list[dict]:
        """The WS bridge contributes no agent-callable tools; the browser is
        the interface, not something an agent acts back on. Return []."""
        return []

    async def call_surface_tool(self, name: str, tool_input: dict, agent_id: str) -> dict:
        """No surface tools are offered (see agent_tools); reachable only via a
        misrouted `surface_`-prefixed call."""
        return {"error": f"surface tool {name} not handled"}

    # ------------------------------------------------------------------
    # Client handling
    # ------------------------------------------------------------------

    async def _handle_client(self, ws: ServerConnection, *, local: bool) -> None:
        """Handle a single WebSocket client connection.

        ``local`` is set by which listener accepted the socket (the privileged
        loopback listener => True). A local session is pre-authed as the owner
        rooted at the full fleet (today's behavior). A remote session starts
        unauthed and is issued an auth_challenge instead of a snapshot."""
        remote = ws.remote_address
        if local:
            # The local listener pre-auths as OWNER, so no process the daemon
            # spawned to run lower-trust code (a tracked tool, shell command
            # or worker PID and its descendants, see atn/contained_pids.py)
            # may get a session here. Peer credential,
            # not address: they all connect from loopback too.
            local_addr = getattr(ws, "local_address", None) or ()
            if len(local_addr) < 2:
                local_addr = ("127.0.0.1", self.port)
            denied = await asyncio.to_thread(
                ws_auth.local_peer_denied, remote, tuple(local_addr[:2]))
            if denied:
                log.warning("Refused local owner connection from %s: %s",
                            remote, denied)
                try:
                    await ws.close(code=4403, reason="forbidden")
                except Exception:                          # noqa: BLE001
                    pass
                return
        session = ClientSession(
            local=local,
            is_loopback=ws_auth.is_loopback(remote),
            conn_id=ws_auth.new_nonce(),
        )
        if local:
            # Privileged local listener: full control, no handshake.
            session.authed = True
            session.owner = True
            session.root_agent_id = None      # unscoped owner session
            session.scope_ids = None          # full fleet
        self._sessions[ws] = session
        # Register this connection as an input surface with the single-writer
        # arbiter — but ONLY once authorized (M6). A remote session is unauthed at
        # connect; registering it here would let an unauthenticated peer sit in
        # the arbiter's connected set and passively capture a freed mic
        # (release_for auto-hands to the most-recently-registered surface),
        # silently dropping the real user's input. Local sessions are pre-authed
        # as owner and register now; remote sessions register at auth success
        # (see _handle_auth_response). Registration does NOT grab the mic — the
        # first genuine inbound auto-acquires when the mic is free.
        if session.authed:
            self._register_input_surface(session)
        log.info("Client connected: %s (local=%s)", remote, local)

        # Per-connection outbound event queue + writer task. _on_event enqueues
        # (non-blocking); the writer drains to the socket. A slow client fills
        # its own queue and drops its own events — it cannot backpressure the
        # EventBus and stall every agent's streaming output.
        session.event_queue = asyncio.Queue(maxsize=2000)
        writer_task = asyncio.create_task(self._event_writer(ws, session))

        try:
            if session.authed:
                # Authorized at connect (local): send the snapshot immediately,
                # scoped to the session (None => full fleet, byte-for-byte
                # identical to the pre-auth behavior).
                await self._send_snapshot(ws, session)
            else:
                # Remote, unauthed: withhold the snapshot, issue a challenge.
                await self._send_auth_challenge(ws, session)

            snapshot_sent = session.authed
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"ok": False, "error": "Invalid JSON"}))
                    continue

                response = await self._handle_message(msg, session)
                await ws.send(json.dumps(response, default=str))
                if session.authed and not snapshot_sent:
                    # Remote sign-in just succeeded: send the (scoped) fleet
                    # now, as a local connection gets at connect. Without it
                    # a remote session showed an empty fleet until something
                    # asked for a snapshot.
                    snapshot_sent = True
                    await self._send_snapshot(ws, session)

        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("Client handler error")
        finally:
            writer_task.cancel()
            self._sessions.pop(ws, None)
            # Release this connection's input surface. If it held the mic, the
            # arbiter auto-hands it to the most-recently-registered surface.
            arbiter = getattr(self.runtime, "input_arbiter", None)
            if arbiter is not None:
                arbiter.release_for(f"ws:{session.conn_id}")
            if session.dropped_events:
                log.warning("Client %s: %d event(s) dropped (slow consumer)",
                            remote, session.dropped_events)
            log.info("Client disconnected: %s", remote)

    @staticmethod
    async def _event_writer(ws: ServerConnection, session: "ClientSession") -> None:
        """Drain a session's event queue to its socket until cancelled."""
        queue = session.event_queue
        try:
            while True:
                payload = await queue.get()
                await ws.send(payload)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception:
            log.debug("Event writer error", exc_info=True)

    async def _send_snapshot(self, ws: ServerConnection, session: ClientSession) -> None:
        """Send the (scope-aware, redacted-for-non-local) initial snapshot."""
        try:
            snapshot = self.runtime.snapshot(session.scope_ids)
            snapshot = self._redact_snapshot(snapshot, session)
            await ws.send(json.dumps({"type": "snapshot", "data": snapshot}, default=str))
        except Exception:
            log.exception("Failed to send initial snapshot")

    async def _send_auth_challenge(self, ws: ServerConnection, session: ClientSession) -> None:
        """Issue a fresh single-use challenge for a remote connection."""
        import time as _time
        session.nonce = ws_auth.new_nonce()
        session.nonce_issued_at = _time.time()
        challenge = ws_auth.build_challenge_text(
            session.nonce,
            daemon_id=self._daemon_id(),
            chain_id=int(getattr(self.runtime._config.autonet, "chain_id", 0) or 0),
            owner_wallet=self.owner_wallet,
            conn_id=session.conn_id,
            issued_at=session.nonce_issued_at,
        )
        await ws.send(json.dumps({
            "type": "auth_challenge",
            "nonce": session.nonce,
            "challenge": challenge,
            "conn_id": session.conn_id,
        }))

    def _daemon_id(self) -> str:
        """This daemon's stable, per-install identity: domain-separates the
        auth challenge so a signature can't replay across daemons. Persisted
        in the data dir on first use; never derived from the agent fleet."""
        if not getattr(self, "_daemon_id_cache", ""):
            try:
                self._daemon_id_cache = ws_auth.load_or_create_daemon_id(
                    self.runtime._config.data_dir)
            except Exception:                              # noqa: BLE001
                # No usable data dir (bare/test construction): a per-process
                # id still domain-separates this daemon's challenges.
                import secrets as _secrets
                self._daemon_id_cache = "atn-daemon-" + _secrets.token_hex(16)
        return self._daemon_id_cache

    # ------------------------------------------------------------------
    # Auth handshake
    # ------------------------------------------------------------------

    # Max failed auth attempts before the challenge is no longer re-issued on
    # this connection (online-guessing / nonce-grinding brake).
    _MAX_AUTH_FAILURES = 5

    async def _handle_auth_response(self, msg: dict, session: ClientSession) -> dict:
        """Verify a signed challenge and, on success, authorize + scope the
        session. Owner must match the PRE-CONFIGURED owner_wallet (no TOFU on a
        network listener). The recovered owner may then root at any agent in
        the fleet by naming it in ``root``."""
        import time as _time
        msg_id = msg.get("msg_id")

        if session.authed:
            return {"msg_id": msg_id, "type": "auth_ok", "ok": True,
                    "owner": session.owner, "root": session.root_agent_id,
                    "wallet": session.wallet_address}
        if session.auth_failures >= self._MAX_AUTH_FAILURES:
            return {"msg_id": msg_id, "type": "auth_denied", "ok": False,
                    "error": "too many attempts"}
        if not session.nonce:
            return {"msg_id": msg_id, "type": "auth_denied", "ok": False,
                    "error": "no challenge issued"}
        if session.challenge_expired(_time.time()):
            session.clear_nonce()
            session.auth_failures += 1
            return {"msg_id": msg_id, "type": "auth_denied", "ok": False,
                    "error": "challenge expired"}

        # Re-derive the challenge text from the SERVER-stored nonce + conn_id
        # (never a client-echoed nonce), then recover the signer. Consume the
        # nonce immediately so a captured signature can't be replayed.
        challenge = ws_auth.build_challenge_text(
            session.nonce,
            daemon_id=self._daemon_id(),
            chain_id=int(getattr(self.runtime._config.autonet, "chain_id", 0) or 0),
            owner_wallet=self.owner_wallet,
            conn_id=session.conn_id,
            issued_at=session.nonce_issued_at,
        )
        signature = msg.get("signature", "")
        signer = ws_auth.recover_signer(challenge, signature)
        session.clear_nonce()

        if not signer:
            session.auth_failures += 1
            return {"msg_id": msg_id, "type": "auth_denied", "ok": False,
                    "error": "could not recover a signer from the signature"}

        is_owner = bool(self.owner_wallet) and signer.lower() == self.owner_wallet.lower()

        resolved: str | None
        if is_owner:
            # OWNER: full control. No root (omitted/null/"") is the unscoped
            # full-fleet session; naming an agent scopes it to that subtree.
            root = str(msg.get("root") or "").strip()
            resolved, err = self._resolve_root(root)
            if err:
                return {"msg_id": msg_id, "type": "auth_denied", "ok": False, "error": err}
            session.owner = True
        else:
            # AGENT SELF-AUTH: the signer holds an agent's own key (exported
            # from this daemon and imported into their wallet). They become
            # THAT agent, rooted at its own subtree — they cannot name an
            # arbitrary root the way the owner can. Address-as-credential, the
            # same per-agent identity model the rest of ATN uses.
            aid = self.runtime.registry.get_agent_id_by_address(signer)
            if aid is None:
                session.auth_failures += 1
                return {"msg_id": msg_id, "type": "auth_denied", "ok": False,
                        "error": "signer is neither the daemon owner nor a known agent"}
            resolved = aid
            session.owner = False

        session.authed = True
        session.wallet_address = signer
        session.root_agent_id = resolved
        session.scope_ids = (None if resolved is None
                             else self.runtime.registry.get_subtree_ids(resolved))
        log.info("Remote auth: signer=%s owner=%s root=%s scoped=%s",
                 signer, session.owner, resolved, session.scope_ids is not None)
        # Refresh the input-surface label now that the root agent is known
        # (re-register is idempotent on the conn_id token).
        self._register_input_surface(session)
        return {"msg_id": msg_id, "type": "auth_ok", "ok": True,
                "owner": session.owner, "root": resolved, "wallet": signer}

    def _resolve_root(self, root: str) -> tuple[str | None, str | None]:
        """Map a client-named root (agent_id or address) to a canonical
        agent_id. Returns (agent_id, error); (None, None) when no root is
        named, meaning the unscoped full-fleet session. Ambiguous address or
        unknown root is refused."""
        if not root:
            return None, None
        reg = self.runtime.registry
        # Direct agent_id?
        if root in reg._agents:
            return root, None
        # Address? (unique match required)
        if root.startswith("0x"):
            matches = [aid for aid, d in reg._agents.items()
                       if d.identity and d.identity.address
                       and d.identity.address.lower() == root.lower()]
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                return "", "ambiguous root address"
        return "", f"unknown agent root: {root}"

    def _library(self) -> Any:
        autonet = getattr(self.runtime, "autonet", None)
        service = getattr(autonet, "_service", None) if autonet else None
        return getattr(service, "_library", None) if service else None

    async def _anchored_epochs(self, last_n: int = 20) -> dict[str, Any]:
        """The jurisdiction's anchored epochs from Substrate (newest last):
        {epochs, total_closed, running, source: "anchored", reason?}. Never
        the local projection ticks: when the anchors are unreadable the list
        is empty and ``reason`` says why."""
        cfg = getattr(self.runtime._config, "rpb", None)
        if not getattr(cfg, "network_joined", False):
            return {"epochs": [], "total_closed": 0, "running": False,
                    "source": "anchored", "reason": "not joined"}
        substrate = getattr(cfg, "substrate_address", "") or ""
        rpc_url = getattr(cfg, "rpc_url", "") or ""
        if not substrate or not rpc_url:
            return {"epochs": [], "total_closed": 0, "running": True,
                    "source": "anchored",
                    "reason": "no Substrate address or RPC configured"}
        try:
            from nodes.common.voice_state import read_recent_anchors
            got = await asyncio.to_thread(read_recent_anchors, substrate,
                                          rpc_url, last_n)
        except Exception as exc:                           # noqa: BLE001
            return {"epochs": [], "total_closed": 0, "running": True,
                    "source": "anchored",
                    "reason": f"anchors unreadable: {exc}"[:200]}
        return {"epochs": got["anchors"], "total_closed": got["anchor_count"],
                "running": True, "source": "anchored"}

    def _tool_reviews_payload(self, digest: str, limit: Any = None, *,
                              include_local: bool = True) -> dict[str, Any]:
        """Reviews, ToolBond rating and recent mint for one tool digest.
        Backs the owner `tool_reviews` handler."""
        try:
            limit = int(limit or 50)
        except (TypeError, ValueError):
            limit = 50
        reviews = (self.runtime.tool_store.recent_attestations(digest, limit=limit)
                   if include_local else [])
        library = self._library()
        rating: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        if library is not None:
            eg = library.read_economy_graph(last_n_epochs=10)
            rating = dict(eg.get("ratings", {}).get(digest) or {})
            # Mint for the tool window: recent mint over the window + the
            # last close's per-digest entry (the review mint's split).
            usage = {
                "recent_mint": float(
                    (eg.get("recent_tool_mint") or {}).get(digest)
                    or 0.0),
                "epochs_considered": eg.get("epochs_considered", 0),
            }
            last = eg.get("last_epoch") or {}
            entry = (last.get("tool_mint") or {}).get(digest)
            if isinstance(entry, dict):
                usage["last_epoch"] = {
                    k: entry.get(k, 0)
                    for k in ("mint", "reviewer_mint", "author_mint",
                              "unpaid", "reviewers", "score",
                              "novelty", "malware", "weight")
                }
        # What a malware verdict left unpaid, per recipient, for a proposal
        # that clears it (Decision 2026-11-10); None when nothing was.
        autonet = getattr(self.runtime, "autonet", None)
        service = getattr(autonet, "_service", None) if autonet else None
        drv = getattr(service, "_federated_close_driver", None) if service else None
        unpaid_of = getattr(drv, "tool_unpaid", None)
        return {
            "digest": digest,
            "reviews": reviews,
            "rating": rating,
            "usage": usage,
            "malware_unpaid": unpaid_of(digest) if unpaid_of is not None else None,
            "pruned": digest in set(getattr(library, "pruned_tool_digests", lambda: [])()),
        }

    def _network_status_payload(self, anchored: dict[str, Any] | None = None
                                ) -> dict[str, Any]:
        """Node, chain, epoch and peer summary. ``epochs_closed`` is the
        jurisdiction's anchored epochs (``anchored``, from _anchored_epochs),
        never the local projection ticks. Public facts
        only: no RPC URL (it can embed a provider key), no wallet, no keys.
        Before the daemon joined the network (a chosen jurisdiction, first registration or
        autonet.enabled) it is local-only: the payload says "not joined"
        instead of reporting chain data."""
        cfg = getattr(self.runtime._config, "rpb", None)
        if not getattr(cfg, "network_joined", False):
            from .config import not_joined_state
            return {**not_joined_state(), "autonet": "disabled",
                    "chain": None, "p2p": {"running": False, "peers": None,
                                           "gossip": False},
                    "epoch": {"running": False, "epoch_id": None},
                    "epochs_closed": 0, "service_market": False}
        autonet = getattr(self.runtime, "autonet", None)
        state: dict[str, Any] = {}
        if autonet is not None:
            try:
                state = autonet.get_status() or {}
            except Exception:                              # noqa: BLE001
                state = {}
        chain_id = int(state.get("chain_id") or getattr(cfg, "chain_id", 0) or 0)
        substrate = (state.get("substrate_address")
                     or getattr(cfg, "substrate_address", "") or "")
        peers: int | None = None
        relays: int | None = None
        addrs: list[str] = []
        host = getattr(autonet, "_p2p_host", None) if autonet else None
        if host is not None:
            try:
                peers = len(host.get_connected_peers())
                relays = len(host.connected_relays())
            except Exception:                              # noqa: BLE001
                peers = None
            try:
                # What peers dial (with /p2p/<id>): the address this daemon
                # would be listed under as a seed in the Registry.
                addrs = list(host.addrs)
            except Exception:                              # noqa: BLE001
                addrs = []
        service = getattr(autonet, "_service", None) if autonet else None
        epoch: dict[str, Any] = {"running": False, "epoch_id": None}
        anchored = anchored or {}
        library = self._library()
        if library is not None:
            try:
                epoch = dict(library.epoch_status())
                scheduler = getattr(service, "_epoch_scheduler", None)
                if scheduler is not None:
                    sched = scheduler.status()
                    sched.pop("opened_at", None)
                    epoch.update(sched)
            except Exception:                              # noqa: BLE001
                pass
        return {
            "joined": True,
            "status": "joined",
            "autonet": state.get("status", "disabled"),
            "chain": {
                "chain_id": chain_id,
                "label": _CHAIN_LABELS.get(chain_id,
                                           f"chain {chain_id}" if chain_id
                                           else "not configured"),
                "testnet": chain_id in _CHAIN_LABELS,
                "substrate_address": substrate,
            },
            "p2p": {
                "running": host is not None,
                "peers": peers,
                "relays": relays,
                "gossip": getattr(service, "_event_gossip", None) is not None,
                "addrs": addrs,
            },
            "epoch": epoch,
            "epochs_closed": int(anchored.get("total_closed") or 0),
            "epochs_closed_source": "anchored",
            **({"epochs_closed_reason": anchored["reason"]}
               if anchored.get("reason") else {}),
            "service_market": bool(getattr(cfg, "service_registry_address", "")),
        }

    # ------------------------------------------------------------------
    # Snapshot redaction (non-local sessions never see owner-global secrets)
    # ------------------------------------------------------------------

    # Snapshot sections that carry owner-global secrets / PII. Dropped for any
    # scoped session; for a remote full-fleet owner they are kept (the owner
    # legitimately manages them) but credential VALUES are never in the
    # snapshot anyway (providers exposes booleans; see snapshot.py).
    _SECRET_SECTIONS = ("providers", "budget", "user", "connectors", "autonet")

    def _redact_snapshot(self, snap: dict, session: ClientSession) -> dict:
        """Strip owner-global sections from a SCOPED session's snapshot. A
        full-fleet session (local owner, or remote owner rooted at the full
        fleet) keeps them. Every snapshot says whether native agents are on
        (atn/native_mode.py), which is what the app shows or hides."""
        try:
            from .native_mode import native_agents_enabled
            snap = {**snap, "native_agents": native_agents_enabled(self.runtime)}
        except Exception:  # noqa: BLE001
            log.debug("native_agents flag unavailable", exc_info=True)
        if session.scope_ids is None:
            return snap            # full fleet => unchanged
        return {k: v for k, v in snap.items() if k not in self._SECRET_SECTIONS}

    # ------------------------------------------------------------------
    # Tool-call authorization (default-deny at the target for scoped sessions)
    # ------------------------------------------------------------------

    def _authorize_tool_call(self, msg_type: str, msg: dict,
                             session: ClientSession) -> str | None:
        """Return an error string if this tool call is not allowed for the
        session, else None. A full-fleet session (scope_ids is None) is
        unrestricted (preserves today's localhost 'act-as any agent')."""
        if session.scope_ids is None:
            return None            # owner / full fleet
        # Scoped session: owner-only tools are off-limits.
        if msg_type in OWNER_ONLY_TOOLS:
            return f"'{msg_type}' is owner-only and not available to a scoped session"
        # Every named target must be inside the subtree.
        scope = session.scope_ids
        for key in _TARGET_ARG_KEYS:
            val = msg.get(key)
            if not val or not isinstance(val, str):
                continue
            target = val if val in self.runtime.registry._agents else \
                self.runtime.registry.get_agent_id_by_address(val)
            if target is not None and target not in scope:
                return f"target '{val}' is outside the authorized subtree"
        return None

    @staticmethod
    def _inline_message_types() -> frozenset:
        """Every message type _handle_message answers inline, before the
        execute_tool fall-through: read once from its source, so the
        scoped-session gate denies a new handler until it is allowlisted."""
        cached = getattr(WebSocketBridge, "_INLINE_TYPES", None)
        if cached is not None:
            return cached
        import inspect
        import re

        src = inspect.getsource(WebSocketBridge._handle_message)
        src = src.split("# --- Gate 3", 1)[0]
        types = set(re.findall(r'msg_type == "([a-z_0-9]+)"', src))
        for grp in re.findall(r'msg_type in \(([^)]*)\)', src):
            types |= set(re.findall(r'"([a-z_0-9]+)"', grp))
        for grp in re.findall(r'msg_type in \{([^}]*)\}', src):
            types |= set(re.findall(r'"([a-z_0-9]+)"', grp))
        g = globals()
        for name in set(re.findall(r'msg_type in ([A-Z_][A-Z_0-9]*)', src)):
            if isinstance(g.get(name), (set, frozenset)):
                types |= set(g[name])
        WebSocketBridge._INLINE_TYPES = frozenset(types)
        return WebSocketBridge._INLINE_TYPES

    def _authorize_scoped_inline(self, msg_type: str, msg: dict,
                                 session: ClientSession) -> str | None:
        """Gate 2b: a scoped session reaches only SCOPED_INLINE_ALLOWED
        among the inline handlers, each clamped to its subtree."""
        if session.scope_ids is None:
            return None
        if msg_type in SCOPED_INLINE_ALLOWED:
            if msg_type in SCOPED_REQUIRES_AGENT and not msg.get("agent_id"):
                return (f"'{msg_type}' needs an agent_id inside the "
                        "authorized subtree")
            return self._authorize_tool_call(msg_type, msg, session)
        if msg_type in self._inline_message_types():
            return (f"'{msg_type}' is owner-only: not available to a scoped "
                    "session (it needs the owner's full-fleet session)")
        return None   # an agent tool: gate 3 clamps it in the fall-through

    async def _handle_message(self, msg: dict[str, Any],
                              session: ClientSession) -> dict[str, Any]:
        """Route an incoming message to the appropriate handler.

        Three gates run before any normal dispatch, in order:
          1. KEY CUSTODY — messages that export or carry a raw private key are
             refused unless the connection arrived on the privileged LOCAL
             listener (session.local). Runs first so the execute_tool
             fallthrough can never bypass it.
          2. AUTH — a remote (non-local) session must complete the handshake
             before anything but the auth/ping messages is accepted.
          3. SCOPE/OWNER (in the execute_tool wrapper) — a scoped session may
             only target its own subtree and may not call owner-only tools.
        """
        msg_id = msg.get("msg_id")
        msg_type = msg.get("type", "")

        if not msg_type:
            return {"msg_id": msg_id, "ok": False, "error": "Missing 'type' field"}

        # An external agent's calls (agent API, atn send) go to
        # its Activity log, whichever path they take.
        if not msg.get("_activity_logged") and msg_type not in _UNLOGGED:
            scoped = getattr(session, "scope_ids", None) is not None
            actor = (getattr(session, "root_agent_id", None) or "") if scoped \
                else (msg.get("caller_id") or "")
            if self._is_external_agent(actor):
                return await self._handle_logged(msg, session, actor)

        # --- Gate 1: key custody (localhost-only) --------------------------
        if msg_type in KEY_LOCAL_ONLY_MESSAGES and not ws_auth.assert_local_key_access(session):
            return {"msg_id": msg_id, "ok": False,
                    "error": "Key operations are localhost-only and were "
                             "refused (connection is not on the local listener)."}

        # --- Gate 1b: ownership and keys need the owner's wallet -----------
        if msg_type == "owner_challenge":
            return self._issue_owner_challenge(msg)
        if msg_type in OWNER_SIGNED_MESSAGES:
            refusal = await self._owner_proof_refusal(msg, msg_type)
            if refusal:
                return {"msg_id": msg_id, "ok": False, "error": refusal,
                        "owner_signature_required": True}

        # --- Auth handshake messages (the only ones allowed pre-auth) ------
        if msg_type == "auth_response":
            return await self._handle_auth_response(msg, session)
        if msg_type == "auth_status":
            return {"msg_id": msg_id, "ok": True, "result": {
                "authed": session.authed, "owner": session.owner,
                "root": session.root_agent_id,
                "requires_auth": not session.local,
            }}

        # --- Payment-authorized stranger surface ---------------------------
        # A cross-machine buyer cannot complete the handshake (not the owner,
        # holds no agent key here) — its credential is the on-chain payment the
        # gate verifies. Dispatch it before gate 2. See
        # PAYMENT_AUTHORIZED_MESSAGES.
        if msg_type in PAYMENT_AUTHORIZED_MESSAGES and not session.authed:
            return await self._handle_service_request(
                msg, msg_id, local=bool(session.local))

        # --- Gate 2: everything else requires an authed session ------------
        if not session.authed:
            return {"msg_id": msg_id, "ok": False, "error": "unauthenticated"}

        # --- Gate 2b: scoped sessions are default-deny on the inline surface
        scoped_err = self._authorize_scoped_inline(msg_type, msg, session)
        if scoped_err:
            return {"msg_id": msg_id, "ok": False, "error": scoped_err}

        # Special case: snapshot request (scoped + redacted per session)
        if msg_type == "snapshot":
            snap = self.runtime.snapshot(session.scope_ids)
            return {
                "msg_id": msg_id,
                "ok": True,
                "result": self._redact_snapshot(snap, session),
            }

        # Single-writer input arbitration (P3). Additive messages:
        #   input_request — explicitly ask for the mic for THIS connection.
        #   input_status  — read the arbiter state (holder + connected surfaces).
        if msg_type == "input_request":
            arbiter = getattr(self.runtime, "input_arbiter", None)
            if arbiter is None:
                return {"msg_id": msg_id, "ok": False, "error": "input arbiter unavailable"}
            # credential is threaded to the switch-authorizer seam but UNUSED
            # today (mic-preemption policy is deferred).
            result = await arbiter.request_input(
                self._surface_id_for(session),
                requester_is_owner=bool(session.owner),
                credential=msg.get("credential"))
            return {"msg_id": msg_id, "ok": bool(result.get("granted")), "result": result}

        if msg_type == "input_status":
            arbiter = getattr(self.runtime, "input_arbiter", None)
            if arbiter is None:
                return {"msg_id": msg_id, "ok": False, "error": "input arbiter unavailable"}
            return {"msg_id": msg_id, "ok": True, "result": arbiter.state()}

        # Daemon restart: signal the CLI driver to re-exec the process after
        # shutdown completes. Reply first so the client sees an ack before
        # the WebSocket connection drops.
        if msg_type == "daemon_restart":
            try:
                from . import cli as _cli_module
                _cli_module._restart_requested = True
                # Schedule the runtime shutdown after the reply is sent
                # so the client gets the ack. The shutdown event is what
                # the input loop watches; setting it here breaks the loop.
                self.runtime._shutdown_event.set()
                return {"msg_id": msg_id, "ok": True, "result": {"restarting": True}}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": f"restart failed: {exc}"}

        # OAuth flow: start authorization for a connector
        if msg_type == "oauth_start":
            connector_id = msg.get("connector_id", "")
            if not connector_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'connector_id' field"}
            from .oauth import build_auth_url, requires_oauth, run_oauth_callback
            if not requires_oauth(connector_id):
                return {"msg_id": msg_id, "ok": False, "error": f"Connector '{connector_id}' does not use OAuth"}
            try:
                auth_url = build_auth_url(connector_id)
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

            # Start the callback server in background — it will wait for
            # the redirect and exchange the code for tokens.
            async def _do_oauth() -> None:
                try:
                    tokens = await run_oauth_callback(connector_id, timeout=120)
                    # Store tokens
                    self.runtime.credential_store.save(connector_id, tokens)
                    # Inject into connector and restart if running
                    self.runtime._inject_connector_credentials()
                    if connector_id in self.runtime.connectors._sessions:
                        await self.runtime.connectors.stop(connector_id)
                        await self.runtime.connectors.start(connector_id)
                    # Notify all clients via event
                    await self.runtime.events.emit(Event(
                        type=EventType.AGENT_REGISTERED,  # reuse as generic notification
                        source="oauth",
                        data={"connector_id": connector_id, "status": "connected"},
                    ))
                    log.info("OAuth complete for connector '%s'", connector_id)
                except Exception as exc:
                    log.warning("OAuth flow failed for '%s': %s", connector_id, exc)

            asyncio.create_task(_do_oauth())
            return {"msg_id": msg_id, "ok": True, "result": {
                "auth_url": auth_url,
                "connector_id": connector_id,
                "status": "Navigate to auth_url to authorize",
            }}

        # OAuth status: check if a connector has stored credentials
        if msg_type == "oauth_status":
            connector_id = msg.get("connector_id", "")
            if not connector_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'connector_id' field"}
            from .oauth import requires_oauth
            has_creds = self.runtime.credential_store.exists(connector_id)
            return {"msg_id": msg_id, "ok": True, "result": {
                "connector_id": connector_id,
                "requires_oauth": requires_oauth(connector_id),
                "authenticated": has_creds,
            }}

        # Provider management
        if msg_type == "provider_list":
            providers = await self.runtime.provider_list()
            return {"msg_id": msg_id, "ok": True, "result": providers}

        if msg_type == "provider_configure":
            provider_id = msg.get("provider_id", "")
            api_key = msg.get("api_key", "")
            if not provider_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'provider_id' field"}
            try:
                result = await self.runtime.configure_provider(provider_id, api_key)
                return {"msg_id": msg_id, "ok": True, "result": result}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "provider_bridge_update":
            # Manual trigger for the bridge SDK updater: check the registry
            # now (past the 24h gate and the opt-out) and swap in a newer
            # in-range install if it passes the ping probe. {"rollback": true}
            # points the bridge back at the previous install instead.
            try:
                result = await self.runtime.providers.check_bridge_update(
                    force=True, rollback=bool(msg.get("rollback")), reason="manual")
                result["status_detail"] = self.runtime.providers.bridge_update_status()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "provider_rate_limits":
            # Serve the SDK-event-fed rate-limit cache (what /usage shows),
            # merged across active bridges. NO network call: this is the
            # cache that stream events and throttled bridge get_usage reads
            # (the SDK's /usage data) fill. Empty result = nothing seen yet.
            try:
                # The daemon-global shared cache, NOT the active-provider
                # aggregate: between tasks there are often zero active bridge
                # instances, and a quota gate polling this surface must not
                # go blind at exactly those moments.
                from .providers.bridge import BridgeProvider
                merged = dict(BridgeProvider._shared_rate_limits)
            except Exception:
                merged = {}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"rate_limits": merged}}

        if msg_type == "provider_refresh_usage":
            provider_id = str(msg.get("provider_id") or "")
            if not provider_id:
                return {"msg_id": msg_id, "ok": False,
                        "error": "provider_refresh_usage needs a provider_id"}
            # _active_providers is keyed by agent_id, not provider_id. Find any
            # active provider instance whose .name matches the requested provider.
            prov = None
            for candidate in self.runtime.providers._active_providers.values():
                if getattr(candidate, "name", "") == provider_id and hasattr(candidate, "refresh_usage"):
                    prov = candidate
                    break
            if prov is None:
                # Rootless fleet / no run yet: no per-agent instance exists.
                # Fall back to the boot-time executor registry (refresh_usage
                # asks the bridge's get_usage; the CLI reads its own login).
                registered = self.runtime.providers.get_registered_provider(provider_id)
                if registered is not None and hasattr(registered, "refresh_usage"):
                    prov = registered
            if prov is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Provider '{provider_id}' is not active or does not support usage refresh"}
            try:
                rate_limits = await prov.refresh_usage()
                return {"msg_id": msg_id, "ok": True,
                        "result": {"provider_id": provider_id, "rate_limits": rate_limits}}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "provider_remove":
            provider_id = msg.get("provider_id", "")
            if not provider_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'provider_id' field"}
            result = await self.runtime.remove_provider(provider_id)
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Custom provider management
        if msg_type == "custom_provider_add":
            provider_id = msg.get("provider_id", "")
            # The frontend has always sent the label as "display_name";
            # accept both spellings so an older or newer client both land.
            name = msg.get("name") or msg.get("display_name") or provider_id
            base_url = msg.get("base_url", "")
            api_key = msg.get("api_key", "")
            default_model = msg.get("default_model", "")
            models = msg.get("models", [])
            if not provider_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'provider_id' field"}
            if not base_url:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'base_url' field"}
            try:
                result = await self.runtime.add_custom_provider(
                    provider_id=provider_id,
                    name=name,
                    base_url=base_url,
                    api_key=api_key,
                    default_model=default_model,
                    models=models if isinstance(models, list) else [],
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "custom_provider_remove":
            provider_id = msg.get("provider_id", "")
            if not provider_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'provider_id' field"}
            try:
                result = await self.runtime.remove_custom_provider(provider_id)
                return {"msg_id": msg_id, "ok": True, "result": result}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Delegate sub-agent tree
        if msg_type == "get_delegates":
            return {
                "msg_id": msg_id,
                "ok": True,
                "result": self.runtime.delegate_registry.get_tree(),
            }

        # Inject message into a running delegate's session
        if msg_type == "delegate_message":
            agent_id = msg.get("agent_id", "")
            content = msg.get("content", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if not content:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'content' field"}
            # G1: delegate_message bypasses send_agent_message, so gate the
            # single-writer arbiter here at the handler (interrupts stay ungated
            # — stopping a turn is not input).
            gate = self._arbiter_gate(session)
            if gate is not None:
                return {"msg_id": msg_id, "ok": False, **gate}
            delivered = await self.runtime.send_delegate_message(agent_id, content)
            if not delivered:
                return {"msg_id": msg_id, "ok": False, "error": f"Delegate '{agent_id}' is not running"}
            # Reflect reality (finding 8): "injected" only when the running loop
            # actually consumed it; "inbox_fallback" when it raced turn end and
            # was re-posted to the inbox instead. Legacy truthy (True) → injected.
            status = delivered if isinstance(delivered, str) else "injected"
            return {"msg_id": msg_id, "ok": True, "result": {"status": status, "agent_id": agent_id}}

        # Send message to a cognitive agent (universal chat)
        if msg_type == "send_agent_message":
            agent_id = msg.get("agent_id", "")
            content = msg.get("content", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if not content:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'content' field"}
            # Gate through the single-writer arbiter as this WS surface.
            sid = self._surface_id_for(session)
            result = await self.runtime.send_agent_message(
                agent_id, content, surface=sid)
            # "Take the mic": when another surface holds it, an owner client
            # that passed force=true claims it through the arbiter (which
            # emits input.granted / input.active_changed so the losing
            # surface learns it lost) and the send is retried once.
            if (result.get("code") == "input_not_active"
                    and msg.get("force") is True):
                arbiter = getattr(self.runtime, "input_arbiter", None)
                if arbiter is not None:
                    grant = await arbiter.request_input(
                        sid, requester_is_owner=bool(session.owner))
                    if grant.get("granted"):
                        result = await self.runtime.send_agent_message(
                            agent_id, content, surface=sid)
            if result.get("error"):
                return {"msg_id": msg_id, "ok": False, "error": result["error"],
                        "code": result.get("code"), "holder": result.get("holder")}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Background processes (shells, subagents, monitors) agents left running.
        if msg_type == "agent_processes_list":
            procs = self.runtime.agent_processes.list(msg.get("agent_id") or None)
            return {"msg_id": msg_id, "ok": True, "result": {"processes": procs}}
        if msg_type in ("agent_process_output", "agent_process_stop"):
            agent_id = msg.get("agent_id") or ""
            task_id = msg.get("task_id") or ""
            if not agent_id or not task_id:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'agent_id' or 'task_id'"}
            if msg_type == "agent_process_output":
                result = self.runtime.agent_processes.output(
                    agent_id, task_id, int(msg.get("max_bytes") or 0))
            else:
                result = await self.runtime.agent_processes.stop(agent_id, task_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Interrupt: gracefully stop a running agent's LLM session mid-turn.
        if msg_type == "interrupt_delegate":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            sent = await self.runtime.interrupt_delegate(agent_id)
            if not sent:
                return {"msg_id": msg_id, "ok": False, "error": f"Delegate '{agent_id}' is not running"}
            return {"msg_id": msg_id, "ok": True, "result": {"status": "interrupted", "agent_id": agent_id}}

        # Context inspection — session stats and conversation history
        if msg_type == "session_stats":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            result = self.runtime.get_session_stats(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "get_delegate_output":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            text = self.runtime.get_delegate_output(agent_id)
            return {"msg_id": msg_id, "ok": True, "result": {"agent_id": agent_id, "text": text}}

        if msg_type == "session_context":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            result = await self.runtime.get_session_context(agent_id)
            if result.get("no_session"):
                # Idle agent: no live bridge session is normal, not a fault.
                return {"msg_id": msg_id, "ok": True, "result": {
                    "agent_id": agent_id, "active": False,
                    "messages": [], "turns": [], "stats": {}}}
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "context_breakdown":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            result = await self.runtime.get_context_breakdown(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # The system prompt an agent runs with (atn/system_prompt.py), so the
        # app shows the real one before the owner changes it. With no
        # agent_id: what a new agent granted ``tools`` would get.
        if msg_type == "agent_system_prompt":
            from .system_prompt import system_prompt_parts
            agent_id = msg.get("agent_id") or ""
            defn = None
            if agent_id:
                defn = self.runtime.get_agent(agent_id)
                if defn is None:
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"Agent not found: {agent_id}"}
            tools = msg.get("tools") if isinstance(msg.get("tools"), list) else None
            return {"msg_id": msg_id, "ok": True,
                    "result": system_prompt_parts(self.runtime, defn, tools=tools)}

        # The canonical agent tool surface — everything an agent can be
        # granted, grouped by the same bundle ids the create-agent flow
        # uses, with each tool's endpoints (name / description / params).
        if msg_type == "tool_surface":
            from .agent_tools import _TOOL_CATEGORIES, _TOOLS
            from .shell_tools import SHELL_TOOLS
            from .web_tools import WEB_TOOLS

            def _entry(name, description, schema):
                props = list(((schema or {}).get("properties")) or {})
                return {"name": name,
                        "description": (description or "").strip().split("\n")[0][:220],
                        "params": props}

            tool_index = {t.name: t for t in _TOOLS}
            from .runtime.shell_provider import swap_enabled as _swap_on
            _swap_live = _swap_on()
            # Which bundles are DAEMON-COUPLED vs genuinely externalizable.
            # Every atn_* bundle's executors take the live Runtime (registry,
            # providers, supervisor), so their manifests are IDENTITY records
            # over daemon internals — they can never be a subprocess without
            # handing it daemon authority. The shell bundle is the exception:
            # its executors take only an input dict, so its code blob is a
            # real runnable program (atn/shell_tools.py __main__).
            #
            # The frontend must not present these as the same kind of thing.
            # 'kind' says what a bundle IS; 'swappable' says whether a
            # substrate tool could replace it. swap_enabled=True (2026-10-12)
            # means the swap path is live, but each swap still needs the
            # core-replacement approval, a ToolBond bond and a per-agent
            # loadout binding (atn/runtime/shell_provider.py).
            bundles = []
            for cat in sorted(_TOOL_CATEGORIES):
                if cat == "shell":
                    tools = [_entry(t["name"], t.get("description", ""),
                                    t.get("input_schema")) for t in SHELL_TOOLS]
                elif cat == "web":
                    tools = [_entry(t["name"], t.get("description", ""),
                                    t.get("input_schema")) for t in WEB_TOOLS]
                else:
                    tools = [
                        _entry(n, tool_index[n].description, tool_index[n].input_schema)
                        for n in sorted(_TOOL_CATEGORIES[cat]) if n in tool_index
                    ]
                bundles.append({
                    "id": cat,
                    "tools": tools,
                    "kind": "external" if cat == "shell" else "daemon",
                    "swappable": cat == "shell",
                    "swap_enabled": cat == "shell" and _swap_live,
                })
            # Bridge-native (Claude SDK) built-ins — granted via "sdk_builtin".
            # Schemas live in the SDK; names are the stable contract.
            bundles.append({
                "id": "sdk_builtin",
                "kind": "provider",   # lives in the Claude SDK, not this daemon
                "swappable": False,
                "swap_enabled": False,
                "tools": [
                    {"name": n, "description": d, "params": []} for n, d in [
                        ("Bash", "Run shell commands"),
                        ("Read", "Read files"), ("Write", "Write files"),
                        ("Edit", "Edit files"),
                        ("Glob", "Find files by pattern"),
                        ("Grep", "Search file contents"),
                        ("WebSearch", "Search the web"),
                        ("WebFetch", "Fetch a URL"),
                        ("Task", "Spawn a sub-task"),
                    ]]})
            return {"msg_id": msg_id, "ok": True,
                    "result": {"bundles": bundles}}

        # Agent cloning — HUMAN-ONLY by construction: these exist solely as
        # WS handlers and are never registered in any agent tool surface, so
        # no agent can invoke (or even name) the capability.
        if msg_type == "clone_agent":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            result = await self.runtime.clone_agent(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "merge_clone":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            result = await self.runtime.merge_clone(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Registered-tool grants — HUMAN-ONLY by construction (same pattern
        # as clone_agent): granting a tool outside its author lineage is the
        # owner's call, so grant/revoke exist only as WS handlers and are
        # never part of any agent tool surface. docs/tool_substrate.md.
        if msg_type == "list_registered_tools":
            store = self.runtime.tool_store
            from .tool_security import run_optins
            try:
                _optins = run_optins(store)
            except Exception:  # noqa: BLE001 — a listing never fails on it
                _optins = None
            tools = []
            for record in store.visible_to(None):  # owner sees everything
                m = record.manifest
                tools.append({
                    "digest": record.digest,
                    "name": record.name,
                    "description": m.get("description", ""),
                    "trust_class": m.get("trust_class", ""),
                    "author": m.get("author", ""),
                    # The daemon's own resident-module manifests (atn_*,
                    # harness_distro.py). The UI shows these as the one
                    # ATN Harness card, not as ~14 loose tool cards. The
                    # atn_ prefix is reserved to the daemon by
                    # _register_tool, so it identifies them regardless of
                    # whether an owner wallet has restamped the author.
                    "resident": record.name.startswith("atn_"),
                    "fee_atn": m.get("fee_atn", 0),
                    "grants": sorted(record.grants),
                    "enabled": record.enabled,
                    "published": record.published,
                    "origin": record.origin,
                    "capabilities": m.get("capabilities", {}),
                    "version_of": m.get("version_of"),
                    "dependencies": m.get("dependencies") or [],
                    "input_schema": m.get("input_schema", {}),
                    # Decision 2026-10-13: the publish-time novelty claim,
                    # and for network tools the owner's run opt-in.
                    "novelty": m.get("novelty"),
                    **({"run_optin": bool(_optins and _optins.allows(record.digest))}
                       if record.origin == "adopted" else {}),
                })
            return {"msg_id": msg_id, "ok": True, "result": {"tools": tools}}

        if msg_type == "grant_tool":
            digest = msg.get("digest", "")
            agent_id = msg.get("agent_id", "")
            if not digest or not agent_id:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'digest' or 'agent_id' field"}
            if not self.runtime.tool_store.grant(digest, agent_id):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown tool digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "granted_to": agent_id}}

        if msg_type == "revoke_tool":
            digest = msg.get("digest", "")
            agent_id = msg.get("agent_id", "")
            if not digest or not agent_id:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'digest' or 'agent_id' field"}
            if not self.runtime.tool_store.revoke(digest, agent_id):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown tool digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "revoked_from": agent_id}}

        # Secrets vault surface — owner/full-fleet only (same HUMAN-ONLY
        # construction as clone_agent: WS handlers only, no agent tool ever
        # names these). Gate 1 already forced secrets_put/secrets_delete onto
        # the local listener.
        if msg_type in _SECRETS_MESSAGES:
            if session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "secrets management is owner-only and not "
                                 "available to a scoped session"}
            return await self._handle_secrets_message(msg_type, msg, msg_id)

        if msg_type == "tool_earnings":
            # Off-chain fee ledger derived from usage receipts. On-chain
            # settlement (labeled ATN transfer via payForService) is not
            # wired yet (docs/tool_substrate.md open knobs).
            return {"msg_id": msg_id, "ok": True,
                    "result": self.runtime.tool_store.balances()}

        # Owner claim status (docs/tool_substrate.md, Decision 2026-10-27):
        # the owner wallet is the household's claimant. The daemon operator
        # claims for it after every anchored epoch (claimFor), so this
        # reports what the household has received and which tools earn for
        # it directly.
        if msg_type == "owner_claim_status":
            rpb_cfg = self.runtime._config.autonet
            wallet = (getattr(rpb_cfg, "owner_wallet", "") or "").strip()
            store = self.runtime.tool_store

            # Tools whose CONSENSUS author is the owner wallet (or which
            # would be, once a wallet exists). These are the ones whose
            # mint accrues to the owner rather than to an agent.
            from .tool_store import _is_claimable_identity
            owner_tools: list[dict[str, Any]] = []
            orphan_tools: list[dict[str, Any]] = []
            for rec in list(store._records.values()):
                consensus = rec.author
                if wallet and consensus.lower() == wallet.lower():
                    owner_tools.append({
                        "digest": rec.digest,
                        "name": rec.name,
                        "published": bool(getattr(rec, "published", False)),
                    })
                elif not _is_claimable_identity(consensus):
                    # Authored under a local id with no wallet configured:
                    # private-plane only, cannot enter consensus at all.
                    orphan_tools.append({
                        "digest": rec.digest,
                        "name": rec.name,
                        "author": consensus,
                    })

            mint_total = "0"
            if wallet:
                try:
                    from .on_chain import OnChainService
                    svc = OnChainService(rpb_cfg)

                    async def _read(_svc=svc, _w=wallet) -> int:
                        w3 = _svc._get_web3()
                        return int(_svc._get_contract(w3).functions.agentMintTotal(
                            w3.to_checksum_address(_w)).call())
                    mint_total = str(await self._offload(_read))
                except Exception as exc:                   # noqa: BLE001
                    log.debug("owner_claim_status chain read failed: %s", exc)

            return {"msg_id": msg_id, "ok": True, "result": {
                "owner_wallet": wallet,
                # Always receivable now: the operator claims for the owner.
                "wallet_registered": bool(wallet),
                "owner_authored_count": len(owner_tools),
                "owner_authored": owner_tools[:50],
                # Tools that cannot enter consensus at all for want of a
                # wallet. Distinct from "earned but unclaimable".
                "orphan_count": len(orphan_tools),
                "orphans": orphan_tools[:50],
                "claimed_mint_total": mint_total,
            }}

        if msg_type == "set_tool_published":
            # Owner surface: flip publish state directly (agents use the
            # separately-granted publish_tool capability).
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            published = bool(msg.get("published", True))
            if published:
                # Decision 2026-10-13: publishing needs a novelty claim and
                # an on-chain stake signed by the AUTHORING agent's key
                # (owner-authored tools are refused: no owner key here).
                # Optional "novelty" {statement, prior_art} stamps the claim
                # in; the published digest may then differ from "digest".
                from .tool_security import publish_with_stake
                record = self.runtime.tool_store.get(digest)
                if record is None:
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"Unknown tool digest: {digest[:16]}"}
                result = await publish_with_stake(
                    self.runtime, record, novelty=msg.get("novelty"))
                if "error" in result:
                    return {"msg_id": msg_id, "ok": False,
                            "error": result["error"], "result": result}
                return {"msg_id": msg_id, "ok": True, "result": result}
            try:
                ok = self.runtime.tool_store.set_published(digest, published)
            except ValueError as exc:  # orphan-author publish gate
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            if not ok:
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown tool digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "published": published}}

        # Adoption rail (docs/tool_substrate.md — Adoption): the approval
        # queue is HUMAN-ONLY, like grants. Agents can only propose
        # (adopt_tool, case-by-case granted); installing foreign code is
        # the owner's call, always per-tool.
        #
        # "By construction" USED TO BE FALSE. These handlers sat behind
        # Gate 2 (authenticated) and nothing else — no scope check, absent
        # from OWNER_ONLY_TOOLS — so a SCOPED session (a delegated remote
        # link, which by design does not act for the owner) could install
        # foreign pinned code. That was a strictly weaker gate than
        # reading secret NAMES (_SECRETS_MESSAGES above rejects scoped
        # sessions). Approving an adoption is the single highest-privilege
        # act on this surface: it installs code that then runs on this
        # host, so it gets the same owner-only treatment.
        if msg_type in ("list_adoption_proposals", "approve_adoption",
                        "reject_adoption", "list_distros",
                        "core_replacement_evidence"):
            if session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "adoption approval is owner-only and not "
                                 "available to a scoped session"}

        # The owner's per-tool notes (owner-only like adoption approval)
        # and each tool's bond with its rating as of the latest close. No
        # run gate (Decision 2026-11-09): nothing here decides what runs.
        if msg_type in ("list_tool_run_optins", "set_tool_run_optin",
                        "tool_bond_status"):
            if session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "tool run opt-in is owner-only and not "
                                 "available to a scoped session"}
            from .tool_security import rating_for, run_optins
            optins = run_optins(self.runtime.tool_store)
            if msg_type == "list_tool_run_optins":
                return {"msg_id": msg_id, "ok": True,
                        "result": {"optins": optins.list()}}
            from .tool_security import client_for
            client = client_for(self.runtime)

            async def _bond_view(d: str) -> dict:
                rec = self.runtime.tool_store.resolve(d)
                return {"digest": d, "status": await client.status(d),
                        "rating": rating_for(self.runtime, d),
                        "optin": optins.allows(d),
                        "network_tool": bool(rec is not None and rec.origin == "adopted")}

            if msg_type == "tool_bond_status" and isinstance(msg.get("digests"), list):
                # Bulk form for the Tools page: one call instead of one per card.
                digests = [str(d or "").strip().lower().removeprefix("0x")
                           for d in msg["digests"]][:100]
                sem = asyncio.Semaphore(6)

                async def _one(d: str) -> tuple:
                    async with sem:
                        return d, await _bond_view(d)
                views = dict(await asyncio.gather(*(_one(d) for d in digests if d)))
                min_stake = getattr(client, "min_stake", None)
                return {"msg_id": msg_id, "ok": True,
                        "result": {"statuses": views,
                                   "min_stake": await min_stake() if min_stake else None}}
            digest = str(msg.get("digest") or "").strip().lower().removeprefix("0x")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            record = self.runtime.tool_store.resolve(digest)
            if msg_type == "set_tool_run_optin":
                allowed = msg.get("allowed")
                if not isinstance(allowed, bool):
                    return {"msg_id": msg_id, "ok": False,
                            "error": "'allowed' must be true or false"}
                try:
                    optins.set(digest, allowed,
                               name=record.name if record is not None else "")
                except ValueError as exc:
                    return {"msg_id": msg_id, "ok": False, "error": str(exc)}
                out = {"digest": digest, "allowed": allowed}
                return {"msg_id": msg_id, "ok": True, "result": out}
            # tool_bond_status: chain state and the rating, plus the current
            # minStake so a publish dialog can show the stake up front.
            out = await _bond_view(digest)
            min_stake = getattr(client, "min_stake", None)
            out["min_stake"] = await min_stake() if min_stake else None
            return {"msg_id": msg_id, "ok": True, "result": out}

        if msg_type == "list_adoption_proposals":
            status = msg.get("status") or None
            rows = self.runtime.tool_store.list_adoption_proposals(status)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"proposals": rows}}

        if msg_type == "approve_adoption":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            # core_replacement_ack: the owner's distinct, explicit approval of
            # a tool that replaces core built-ins (Decision 2026-10-12). The
            # store rejects such a tool without it.
            _ack = ({"core_replacement_ack": True}
                    if msg.get("core_replacement_ack") is True else {})
            result = await self.runtime.tool_store.approve_adoption(
                digest, **_ack)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"],
                        **({"core_replacement": result["core_replacement"]}
                           if result.get("core_replacement") else {})}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "list_distros":
            # Harness distro picker (agent settings). Owner-only: binding is
            # an owner action (update_agent {loadout}).
            return {"msg_id": msg_id, "ok": True,
                    "result": {"distros": self.runtime.tool_store.list_distros()}}

        if msg_type == "core_replacement_evidence":
            # Signed evidence traces of core-replacing tool calls; the
            # evidence_digest is the evidenceCid for ToolBond.accuse.
            from .runtime.shell_provider import list_evidence
            rows = list_evidence(self.runtime.tool_store,
                                 str(msg.get("tool_digest") or ""),
                                 int(msg.get("limit") or 50))
            return {"msg_id": msg_id, "ok": True,
                    "result": {"evidence": rows}}

        # Staked reviews (docs/multi_jurisdiction.md decision 10).
        # Owner-only: ``review_batches`` lists this daemon's agents' ToolBond
        # review batches with status and the sealed reveal (salt included),
        # so the owner can back them up; a reveal is relayable, so whoever
        # holds a copy can reveal if this daemon dies. ``tool_security_status``
        # is one tool's bond plus its rating as of the latest close.
        if msg_type in ("review_batches", "tool_security_status"):
            if session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "review batches and tool security status are "
                                 "owner-only and not available to a scoped "
                                 "session"}
            if msg_type == "review_batches":
                from .review_driver import batch_rows
                return {"msg_id": msg_id, "ok": True,
                        "result": {"batches": batch_rows(self.runtime)}}
            d = str(msg.get("digest") or "").strip().lower().removeprefix("0x")
            if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
                return {"msg_id": msg_id, "ok": False,
                        "error": "digest must be a 64-hex manifest digest"}
            from .tool_security import client_for, rating_for as _rating
            client = client_for(self.runtime)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": d, "status": await client.status(d),
                               "rating": _rating(self.runtime, d)}}

        if msg_type == "reject_adoption":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            result = self.runtime.tool_store.reject_adoption(
                digest, reason=str(msg.get("reason") or ""))
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "probe_tools":
            # The Tools screen's library search: the same search the agents
            # use (atn/tool_search.py), keyword + semantic, relevance order,
            # metadata on each match. The owner sees every record; a scoped
            # session (an agent, e.g. through the agent API) only what its
            # root agent may call.
            query = str(msg.get("query") or "").strip()
            if not query:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'query' field"}
            try:
                k = max(1, min(int(msg.get("k") or 12), 50))
            except (TypeError, ValueError):
                k = 12
            caller = (getattr(session, "root_agent_id", None)
                      if getattr(session, "scope_ids", None) is not None else None)
            from .tool_search import search_tools
            result = await search_tools(self.runtime, query, k, caller_id=caller)
            return {"msg_id": msg_id, "ok": True, "result": result}

        # The local search model's download / check / load
        # (atn/search_model.py); `search.model_status` events carry the same
        # dict as it changes.
        if msg_type == "search_model_status":
            from .search_model import status_for
            return {"msg_id": msg_id, "ok": True,
                    "result": status_for(self.runtime)}

        if msg_type == "set_tool_enabled":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            enabled = bool(msg.get("enabled", True))
            if not self.runtime.tool_store.set_enabled(digest, enabled):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown tool digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "enabled": enabled}}

        # ---- Services market (docs/services_market.md) --------------------
        # Provider-side rail: publish a monetizable remote API and serve
        # its requests. register/retire are OWNER surface (selling a tool is
        # the owner's call — same doctrine as tool grants). service_request
        # is the PROVIDER-side entry: a paid client's work item, dispatched
        # to the backing local tool. Payment/voucher validation is the
        # contracts workstream's job — see _validate_service_payment.
        if msg_type == "register_service":
            name = msg.get("name", "")
            description = msg.get("description", "")
            input_schema = msg.get("input_schema") or {}
            # The price goes on chain only (Decision 2026-11-10): the spec
            # carries none. The app sends {"ask": {"amount"}}; agents send
            # "ask_amount".
            ask = msg.get("ask")
            ask_raw = (ask.get("amount") if isinstance(ask, dict)
                       else msg.get("ask_amount"))
            try:
                ask_amount = int(str(ask_raw if ask_raw is not None else "0")
                                 or "0")
            except (TypeError, ValueError):
                return {"msg_id": msg_id, "ok": False,
                        "error": "ask.amount must be an integer (ATN base units)"}
            if ask_amount < 0:
                return {"msg_id": msg_id, "ok": False,
                        "error": "ask.amount must be non-negative"}
            author = msg.get("agent_id") or "user"
            # The Flutter client sends `tool_digest`; agent-side callers send
            # `backing_tool`. Accept both — a service whose backing tool is
            # empty can never be fulfilled (service_request dispatches to it),
            # so silently dropping the key published dead listings.
            backing_tool = (msg.get("backing_tool")
                            or msg.get("tool_digest") or "")
            # The other backing kind (decision 2026-07-26): sell cognition
            # off this daemon's own provider stack. {provider, model,
            # max_tokens_cap?, max_input_chars?}: the upstream provider and
            # model are named, never inferred. The store refuses a block
            # without them, one naming a provider this daemon does not
            # know, and one alongside a backing_tool.
            inference = msg.get("inference")
            if not isinstance(inference, dict):
                inference = None
            if not name:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'name' field"}
            if not input_schema:
                # The app's Publish dialog names a backing, not a schema:
                # the backing already defines what a buyer sends. Without
                # this the spec validator refused every app-published
                # service ("missing required field 'input_schema'").
                input_schema = self._default_service_input_schema(
                    backing_tool, inference)
            try:
                result = self.runtime.service_store.register(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    author=author,
                    backing_tool=backing_tool,
                    output_schema=msg.get("output_schema"),
                    endpoint_hint=msg.get("endpoint_hint", ""),
                    # Display-plane only: a listing-card banner. Advisory
                    # like endpoint_hint, so no validation beyond str
                    # coercion — an unreachable URL degrades to the UI's
                    # default colour banner.
                    image_uri=str(msg.get("image_uri") or "").strip(),
                    inference=inference,
                    version_of=msg.get("version_of"),
                )
            except (ValueError, RuntimeError) as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            # Chain leg: without it the listing exists only in this
            # daemon's store, so no other daemon's market can see it and no
            # buyer can pay for it. Same half agent_tools._register_service
            # already does; a failure here downgrades to local-only rather
            # than discarding the persisted spec.
            chain = await self._register_service_on_chain(
                author, result["digest"], ask_amount)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": result["digest"],
                               "spec": result["spec"],
                               **chain}}

        if msg_type == "list_services":
            include_retired = bool(msg.get("include_retired", False))
            services = []
            for record in self.runtime.service_store.list(
                    include_retired=include_retired):
                s = record.spec
                chain = await self._chain_service(record)
                services.append({
                    "digest": record.digest,
                    "name": record.name,
                    "description": s.get("description", ""),
                    "author": record.author,
                    "author_pubkey": s.get("author_pubkey", ""),
                    # From the ServiceRegistry entry, the only reference for
                    # them (Decision 2026-11-10): the provider is the address
                    # paid, the one the gate verifies against.
                    "service_id": chain.get("service_id"),
                    "provider": chain.get("provider", ""),
                    "ask": ({"amount": str(chain["ask_amount"]),
                             "unit": "per_item"}
                            if "ask_amount" in chain else None),
                    "active": chain.get("active"),
                    "endpoint_hint": s.get("endpoint_hint", ""),
                    "image_uri": s.get("image_uri", ""),
                    "backing_tool": record.backing_tool,
                    # {} for a tool-backed service; {provider, model,
                    # max_tokens_cap, max_input_chars} when the daemon sells
                    # its own inference.
                    "inference": record.inference,
                    "retired": record.retired,
                    "version_of": s.get("version_of"),
                    "input_schema": s.get("input_schema", {}),
                    # The service store is daemon-local: everything it holds
                    # was published from HERE, so ownership is a fact, not an
                    # inference. The UI used to guess by resolving `author`
                    # as an agent id, which never matched the "user" default
                    # and hid Retire on the owner's own listings.
                    "owned": True,
                    "registered_ts": record.registered_ts,
                })
            summary = self.runtime.service_store.summary()
            return {"msg_id": msg_id, "ok": True,
                    "result": {"services": services, "summary": summary}}

        # The MARKET view: every service on-chain, not just what this daemon
        # published. `list_services` reads the daemon-local store, so before
        # this the app could only ever see its own listings — there was no
        # way for a human to browse the marketplace at all. Mirrors the
        # agent-side `find_services` tool (agent_tools.py).
        if msg_type == "market_services":
            query = str(msg.get("query") or "").strip()
            try:
                limit = max(1, min(int(msg.get("limit") or 100), 200))
            except (TypeError, ValueError):
                limit = 100
            if not getattr(self.runtime._config.rpb, "network_joined", False):
                from .config import not_joined_state
                return {"msg_id": msg_id, "ok": True,
                        "result": not_joined_state()}
            # The daemon's catalog (Decision 2026-11-10): every service in
            # the jurisdiction, specs gossiped or fetched, searched locally
            # with the tools' ranking. No registry scan per search.
            from . import service_catalog as sc
            catalog = sc.catalog_for(self.runtime)
            synced = await sc.ensure_synced(catalog)
            if synced.get("error") and not catalog.synced:
                return {"msg_id": msg_id, "ok": False, "error": synced["error"]}
            store = getattr(self.runtime, "service_store", None)
            if store is not None:
                await sc.fetch_missing(catalog, store.spec_for, limit=limit)
            rows = await asyncio.to_thread(
                catalog.search, query, limit, include_inactive=True)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"services": rows, "count": len(rows)}}

        # Semantic search + clustering over service specs.
        #
        # `service_embedding_text` (service_spec.py) has been written and
        # unused since the rail landed: name + description + interface
        # vocabulary. The tools rail gets semantic discovery through the
        # ArtifactIndex, but that only ingests blob-store artifacts and
        # requires a running library service, while service specs live in the
        # daemon-local store. So embed here with the same underlying
        # embedder — services stay searchable whether or not autonet is on.
        #
        # Categories are NOT declared: the Service struct has no such
        # field, and asking authors to self-classify invites the same
        # keyword-stuffing the topic embedding already resists. Clusters
        # are DERIVED, and named after the fact from their members.
        if msg_type == "service_topics":
            try:
                from nodes.common.library.usefulness_coords import (
                    coords_for_query, usefulness_embedder_if_ready,
                )
                import numpy as _np
            except Exception as exc:                       # noqa: BLE001
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

            store = self.runtime.service_store
            records = list(store.list(include_retired=False))
            if not records:
                return {"msg_id": msg_id, "ok": True,
                        "result": {"hits": [], "clusters": []}}

            # The semantic embedder, or "not yet": while the search model
            # is still being fetched (atn/search_model.py) the answer is
            # `pending` and the frontend asks again shortly.
            _pending = {"msg_id": msg_id, "ok": True,
                        "result": {"hits": [], "clusters": [],
                                   "indexed": 0, "pending": True}}
            from .service_topics import DIM as _st_dim
            try:
                embedder = await asyncio.to_thread(
                    usefulness_embedder_if_ready, _st_dim, 1.0)
            except Exception:                              # noqa: BLE001
                log.debug("service_topics: embedder unavailable", exc_info=True)
                embedder = None
            if embedder is None:
                return _pending

            # Vectors come from the process-wide cache in service_topics
            # (pre-filled by warm_service_vectors once the model loads);
            # only what is still missing is embedded here, on a thread.
            from . import service_topics as _st
            query = str(msg.get("query") or "").strip()

            def _embed_all() -> tuple[list[Any], list[str], Any] | None:
                # None = the boot warm-up is embedding the catalogue right
                # now; do not push it through the worker a second time.
                if not _st.EMBED_LOCK.acquire(blocking=False):
                    return None
                try:
                    vecs_, digests_ = _st.embed_services(records, embedder)
                finally:
                    _st.EMBED_LOCK.release()
                q_ = None
                if query:
                    q_ = _np.asarray(coords_for_query(query, embedder),
                                     dtype=_np.float64)
                return vecs_, digests_, q_

            try:
                embedded = await asyncio.to_thread(_embed_all)
            except Exception:                              # noqa: BLE001
                # A worker that died mid-batch respawns on the next call;
                # tell the frontend to come back rather than failing loud.
                log.warning("service_topics: embedding failed", exc_info=True)
                return _pending
            if embedded is None:
                return _pending
            vecs, digests, q = embedded
            if not vecs:
                return {"msg_id": msg_id, "ok": True,
                        "result": {"hits": [], "clusters": []}}
            matrix = _np.vstack(vecs)
            norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = matrix / norms

            hits: list[dict[str, Any]] = []
            if query:
                qn = _np.linalg.norm(q) or 1.0
                sims = unit @ (q / qn)
                order = _np.argsort(-sims)
                for i in order[:int(msg.get("k") or 25)]:
                    hits.append({"digest": digests[int(i)],
                                 "score": round(float(sims[int(i)]), 6)})

            # Derived clusters: greedy cosine agglomeration. Deliberately
            # simple — this is a browsing aid, not consensus, so it does
            # not need to be stable across daemons or reproducible.
            try:
                # 0.18 tuned against a 27-service catalogue: it yields ~9
                # rails topping out at 4-6 members with few strays. Higher
                # thresholds fragment into pairs, and a rail of two is not
                # worth scrolling.
                threshold = float(msg.get("threshold") or 0.18)
            except (TypeError, ValueError):
                threshold = 0.18
            # Seed each cluster from the DENSEST remaining service (the one
            # with the most neighbours above threshold) rather than from an
            # arbitrary index. Seeding arbitrarily makes the first service
            # in the list a magnet and produces junk-drawer clusters whose
            # members relate to the seed but not to each other.
            sim = unit @ unit.T
            unassigned = set(range(len(digests)))
            clusters: list[dict[str, Any]] = []
            while unassigned:
                idx = sorted(unassigned)
                # Density = neighbour count within the unassigned set.
                seed = max(idx, key=lambda i: (
                    sum(1 for j in idx if j != i and float(sim[i, j]) >= threshold),
                    -i,
                ))
                unassigned.discard(seed)
                members = [seed]
                for j in sorted(unassigned):
                    # Require the candidate to be close to the seed AND to
                    # every member already admitted, so a cluster stays
                    # internally coherent instead of chaining outward.
                    if all(float(sim[m, j]) >= threshold for m in members):
                        members.append(j)
                        unassigned.discard(j)
                clusters.append({
                    "members": [digests[m] for m in members],
                    "size": len(members),
                })
            clusters.sort(key=lambda c: -c["size"])
            return {"msg_id": msg_id, "ok": True,
                    "result": {"hits": hits, "clusters": clusters,
                               "indexed": len(digests)}}

        # A service provider's VERIFIABLE track record. Services have no
        # reputation and no quality score by design (remote execution is
        # unknowable), so the honest thing to show a buyer is what the
        # chain already proves about the counterparty: what they have been
        # PAID, what they have authored, and how long they have existed.
        # Costly to inflate, not impossible: a provider can pay itself
        # through payForService, losing the fee on every payment.
        if msg_type == "provider_record":
            addresses = msg.get("addresses")
            if isinstance(addresses, str):
                addresses = [addresses]
            if not isinstance(addresses, list) or not addresses:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'addresses' (list of 0x)"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
            except Exception as exc:                       # noqa: BLE001
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            records: dict[str, Any] = {}
            for raw in addresses[:50]:
                addr = str(raw or "").strip()
                if not addr.startswith("0x") or len(addr) != 42:
                    continue
                try:
                    rec = await svc.get_provider_record(addr)
                except Exception as exc:                   # noqa: BLE001
                    log.debug("provider_record %s failed: %s", addr, exc)
                    rec = None
                if rec is not None:
                    records[addr.lower()] = rec
            return {"msg_id": msg_id, "ok": True,
                    "result": {"records": records}}

        if msg_type == "retire_service":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            record = self.runtime.service_store.get(digest)
            if record is None or not self.runtime.service_store.retire(digest):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown service digest: {digest[:16]}"}
            # Chain leg: a local retire alone leaves the registry row active,
            # and buyers pay before dispatch — so the listing keeps taking
            # money for a service that then answers "Service retired".
            chain = await self._retire_service_on_chain(record)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "retired": True, **chain}}

        if msg_type == "service_request":
            return await self._handle_service_request(
                msg, msg_id, local=bool(session.local))

        # The CONSUMER side of the rail, for a human: buy ONE work item.
        # `service_request` is what a paying counterparty sends US;
        # `invoke_service` is what the owner's app sends to buy something —
        # pay, dispatch, hand back the result plus a receipt.
        if msg_type == "invoke_service":
            return await self._handle_invoke_service(msg, msg_id)

        # Voice service control
        if msg_type == "voice_start":
            result = await self.runtime.start_voice()
            # Only a service that is actually up counts as ok. "unavailable"
            # (extras missing) used to report ok=true, which swallowed the
            # install hint before it reached the user.
            started = result.get("status") in ("started", "already_running")
            return {"msg_id": msg_id, "ok": started, "result": result}

        if msg_type == "voice_stop":
            result = await self.runtime.stop_voice()
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "voice_status":
            if self.runtime.voice:
                return {"msg_id": msg_id, "ok": True, "result": self.runtime.voice.get_status()}
            return {"msg_id": msg_id, "ok": True, "result": {"running": False}}

        # TTS transport controls (P4). Additive; guarded by a running voice
        # service so the WS contract is stable whether or not voice is up.
        if msg_type == "voice_pause":
            if self.runtime.voice:
                self.runtime.voice.pause()
                return {"msg_id": msg_id, "ok": True, "result": {"paused": True}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_resume":
            if self.runtime.voice:
                self.runtime.voice.resume()
                return {"msg_id": msg_id, "ok": True, "result": {"paused": False}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_skip":
            if self.runtime.voice:
                skipped = self.runtime.voice.skip_current()
                return {"msg_id": msg_id, "ok": True, "result": {"skipped": bool(skipped)}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_focus":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if self.runtime.voice:
                self.runtime.voice.set_focus(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"focused_agent": agent_id}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_enabled":
            enabled = msg.get("enabled", True)
            if self.runtime.voice:
                self.runtime.voice.set_voice_enabled(enabled)
                return {"msg_id": msg_id, "ok": True, "result": {"voice_enabled": enabled}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_backend":
            backend = msg.get("backend", "")
            if not backend:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'backend' field"}
            if self.runtime.voice:
                self.runtime.voice.config.backend = backend
                return {"msg_id": msg_id, "ok": True, "result": {"backend": backend}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_voice_focus":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if self.runtime.voice:
                self.runtime.voice.set_voice_focus(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"voice_focus": agent_id}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_tools_focus":
            agent_id = msg.get("agent_id") or ""
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if self.runtime.voice:
                self.runtime.voice.set_tools_focus(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"tools_focus": agent_id}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_narrate_tools":
            enabled = bool(msg.get("enabled", True))
            if self.runtime.voice:
                self.runtime.voice.set_narrate_tools(enabled)
                return {"msg_id": msg_id, "ok": True,
                        "result": {"narrate_tools": enabled}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_announcements":
            categories = msg.get("categories", [])
            if not isinstance(categories, list):
                return {"msg_id": msg_id, "ok": False, "error": "'categories' must be a list"}
            if self.runtime.voice:
                self.runtime.voice.set_announcements(categories)
                return {"msg_id": msg_id, "ok": True, "result": {"announcements": categories}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_devices":
            try:
                from .voice_service import get_device_list
                outputs, inputs = get_device_list()
                return {"msg_id": msg_id, "ok": True, "result": {
                    "outputs": outputs, "inputs": inputs,
                }}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Onboarding
        if msg_type == "get_onboarding_status":
            profile = self.runtime.user_profile.get_profile()
            return {"msg_id": msg_id, "ok": True, "result": {
                "status": profile.onboarding_status.value,
                "has_profile": profile.onboarding_status.value == "completed",
            }}

        if msg_type == "skip_onboarding":
            self.runtime.user_profile.skip_onboarding()
            return {"msg_id": msg_id, "ok": True, "result": {"status": "skipped"}}

        # User profile
        if msg_type == "get_profile":
            p = self.runtime.user_profile.get_profile()
            # Goals are now agents — build from registry
            agent_goals = []
            for defn, status in self.runtime.list_agents():
                agent_goals.append({
                    "id": defn.id,
                    "title": defn.name,
                    "description": defn.task_prompt or defn.description,
                    "status": status.value,
                })
            return {"msg_id": msg_id, "ok": True, "result": {
                "onboarding_status": p.onboarding_status.value,
                "summary": p.summary,
                "standards": p.standards,
                "goals": agent_goals,
                "projects": p.projects,
                "strengths": p.strengths,
                "weaknesses": p.weaknesses,
                "jurisdiction_id": p.jurisdiction_id,
            }}

        # Credit budget
        if msg_type == "get_budget":
            return {"msg_id": msg_id, "ok": True, "result": self.runtime.credit_budget.to_summary_dict()}

        if msg_type == "set_budget":
            provider = msg.get("provider", "")
            token_limit = msg.get("token_limit")
            if not provider or token_limit is None:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'provider' or 'token_limit'"}
            self.runtime.credit_budget.set_budget(
                provider=provider,
                token_limit=int(token_limit),
                period=msg.get("period", "monthly"),
                auto_allocate=msg.get("auto_allocate", True),
            )
            return {"msg_id": msg_id, "ok": True, "result": {"status": "configured"}}

        # ---------------------------------------------------------------
        # Autonet network service
        # ---------------------------------------------------------------
        if msg_type == "autonet_status":
            return {"msg_id": msg_id, "ok": True, "result": self.runtime.autonet.get_status()}

        # Claude Code sessions on this machine (atn/claude_sessions.py): list
        # a folder's sessions, and import one as an agent whose runs resume
        # it in place. Owner-only: an import gives the daemon the session.
        if msg_type == "claude_sessions":
            folder = str(msg.get("folder") or "").strip()
            if not folder:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'folder'"}
            from .claude_sessions import list_sessions
            listing = list_sessions(folder)
            imported = self._imported_sessions()
            for row in listing["sessions"]:
                row["imported_as"] = imported.get(row["session_id"], "")
            return {"msg_id": msg_id, "ok": True, "result": listing}

        if msg_type == "claude_session_import":
            return {"msg_id": msg_id, **(await self._import_claude_session(msg))}

        # Robot side (docs/robots.md). A challenge answer is signed by the
        # robot agent's key over its recent worklog chunks.
        if msg_type in ("robot_status", "robot_export_packs", "robot_attest"):
            robot = getattr(self.runtime, "robot", None)
            if robot is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "this daemon is not a robot (robot.enabled is off)"}
            try:
                if msg_type == "robot_status":
                    result = robot.status()
                elif msg_type == "robot_export_packs":
                    result = robot.export_packs()
                else:
                    result = robot.attest(str(msg.get("challenge") or ""))
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "autonet_jurisdictions":
            return {"msg_id": msg_id, "ok": True,
                    "result": self._jurisdictions_view()}

        if msg_type == "sanctions":
            # The jurisdiction's sanctions in force (docs/accountability.md):
            # household -> {until?, reason?, ...}, for the app's badges.
            import asyncio as _asyncio
            entries = await _asyncio.to_thread(
                self.runtime.autonet.sanctions.entries)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"sanctions": entries}}

        if msg_type == "autonet_switch":
            # Switch the daemon to another jurisdiction (docs/
            # multi_jurisdiction.md decision 1): check it can be joined,
            # save the selection, restart. Substrate data of the current
            # jurisdiction stays parked under its key; agents keep their
            # identity, registration is per jurisdiction.
            dao_registry = str(msg.get("dao_registry") or "").strip()
            rpc_url = str(msg.get("rpc_url") or "").strip()
            if msg.get("recommended"):
                # The recommended jurisdiction (the repo's registry.json),
                # pinned: it is joined like any chosen DAO, by its Registry.
                import asyncio as _asyncio
                from .config import recommended_jurisdiction
                rec = await _asyncio.to_thread(recommended_jurisdiction)
                if rec is None:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "no recommended jurisdiction found"}
                dao_registry, rpc_url = rec["dao_registry"], rec["rpc_url"]
                msg = {**msg, "name": msg.get("name") or rec.get("name") or ""}
            elif msg.get("packaged"):
                # The jurisdiction this release ships (issue #50): its
                # registry.json is the source, nothing to read on chain.
                from .config import packaged_jurisdiction, save_selected_jurisdiction
                pk = packaged_jurisdiction()
                if pk is None:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "this release ships no jurisdiction"}
                key = f"{pk['chain_id']}:{pk['dao'].lower()}"
                if key == self.runtime.jurisdiction:
                    return {"msg_id": msg_id, "ok": True,
                            "result": {**pk, "key": key, "switched": False,
                                       "note": "already in this jurisdiction"}}
                save_selected_jurisdiction({"packaged": True},
                                           getattr(self.runtime._config, "data_dir", None))
                if msg.get("restart", True):
                    from . import cli as _cli_module
                    _cli_module._restart_requested = True
                    self.runtime._shutdown_event.set()
                return {"msg_id": msg_id, "ok": True,
                        "result": {**pk, "key": key, "switched": True,
                                   "restarting": bool(msg.get("restart", True))}}
            if not dao_registry or not rpc_url:
                return {"msg_id": msg_id, "ok": False,
                        "error": "autonet_switch needs 'dao_registry' and 'rpc_url'"}
            import asyncio as _asyncio
            from .config import check_jurisdiction, save_selected_jurisdiction
            check = await _asyncio.to_thread(check_jurisdiction, dao_registry, rpc_url)
            if not check["ok"]:
                return {"msg_id": msg_id, "ok": False,
                        "error": "; ".join(check["errors"]), "result": check}
            if check["key"] == self.runtime.jurisdiction:
                return {"msg_id": msg_id, "ok": True,
                        "result": {**check, "switched": False,
                                   "note": "already in this jurisdiction"}}
            save_selected_jurisdiction(
                {**check, "name": str(msg.get("name") or check.get("name") or "")},
                getattr(self.runtime._config, "data_dir", None))
            if msg.get("restart", True):
                from . import cli as _cli_module
                _cli_module._restart_requested = True
                self.runtime._shutdown_event.set()
            return {"msg_id": msg_id, "ok": True,
                    "result": {**check, "switched": True,
                               "restarting": bool(msg.get("restart", True))}}

        if msg_type == "autonet_start":
            result = await self.runtime.autonet.start()
            return {"msg_id": msg_id, "ok": result.get("status") != "error", "result": result}

        if msg_type == "autonet_stop":
            result = await self.runtime.autonet.stop()
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "autonet_wallet_connect":
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address' field"}
            result = self.runtime.autonet.connect_wallet(address)
            await self.runtime.autonet._emit("AUTONET_WALLET", {"action": "connected", "address": address})
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "autonet_wallet_disconnect":
            result = self.runtime.autonet.disconnect_wallet()
            await self.runtime.autonet._emit("AUTONET_WALLET", {"action": "disconnected"})
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "autonet_set_chain":
            rpc_url = msg.get("rpc_url", "")
            chain_id = msg.get("chain_id", 0)
            if not rpc_url:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'rpc_url' field"}
            result = self.runtime.autonet.set_chain(rpc_url, chain_id)
            await self.runtime.autonet._emit("AUTONET_STATUS", {"action": "chain_changed", **result})
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Pre-substrate earnings/governance handlers (Stories 3.5 / 3.6 / 3.7).
        # They read Autonet.sol (claimParticipantReward, totalInferenceBurned,
        # EvolutionProposal, ...), which was deleted wholesale in Phase 5.6a;
        # there is no such contract to call. Same treatment as the
        # _DEPRECATED_RPB set below: an explicit gone-marker so the UI can hide
        # the panel instead of showing a chain error it cannot act on.
        # The live rails are Substrate.claimFor, surfaced by
        # 'tool_earnings' and 'owner_claim_status'.
        # Retired with them: standards publication (Autonet.sol
        # createUserContract), the alignment dashboard and the JEPA-era
        # screen/browser capture settings. No app screen calls these.
        _DEPRECATED_AUTONET = {
            "autonet_earnings", "autonet_claim_reward",
            "autonet_proposals", "autonet_governance",
            "autonet_standards", "autonet_publish_standards",
            "autonet_alignment", "autonet_capture_config",
            "autonet_set_capture_config", "autonet_enumerate_sources",
        }
        if msg_type in _DEPRECATED_AUTONET:
            return {
                "msg_id": msg_id, "ok": False,
                "error": f"'{msg_type}' belonged to the pre-substrate design "
                         "(Autonet.sol, removed in Phase 5.6a, and JEPA "
                         "capture) and is gone. Use 'tool_earnings' / "
                         "'owner_claim_status' for the Substrate.sol mint rail.",
            }

        # ---------------------------------------------------------------
        # Sponsor / dependent inference ("work AI")
        # ---------------------------------------------------------------
        # A sponsor agent supplies LLM inference to a dependent agent on a
        # different daemon, capped by a per-dependent token budget. The
        # sponsor is the resource owner and the authority: bindings live in
        # sponsor-local state (runtime.sponsor_bindings), keyed by the
        # dependent's on-chain address.
        if msg_type == "create_sponsor_agent":
            dependent_address = msg.get("dependent_address", "")
            budget_tokens = msg.get("budget_tokens", 0)
            label = msg.get("label", "")
            provider = msg.get("provider", "")
            model = msg.get("model", "")
            if not dependent_address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'dependent_address'"}
            try:
                budget_tokens = int(budget_tokens)
            except (TypeError, ValueError):
                return {"msg_id": msg_id, "ok": False, "error": "'budget_tokens' must be an integer"}
            try:
                binding = self.runtime.sponsor_bindings.add(
                    dependent_address, budget_tokens=budget_tokens, label=label,
                )
                enable = self.runtime.autonet.enable_sponsor_inference(
                    provider=provider, model=model,
                )
                return {"msg_id": msg_id, "ok": True, "result": {
                    "binding": binding.to_dict(),
                    "sponsor": enable,
                }}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Dependent side: THIS daemon's sponsor. Daemon-level, keyed on the
        # owner wallet — there is no per-agent sponsor setting
        # (docs/sponsored_inference.md).
        if msg_type == "get_my_sponsor":
            def _rpb_last_budget() -> int:
                try:
                    from .providers.rpb import last_remaining_budget_tokens
                    return int(last_remaining_budget_tokens())
                except Exception:                          # noqa: BLE001
                    return -1
            rpb_cfg = self.runtime._config.autonet
            return {"msg_id": msg_id, "ok": True, "result": {
                "sponsor_address": getattr(rpb_cfg, "sponsor_address", "") or "",
                # The dependent identity we present to a sponsor. Empty here
                # means this daemon cannot be sponsored at all: no wallet, no
                # address to bind.
                "dependent_address": getattr(rpb_cfg, "owner_wallet", "") or "",
                # How much of the sponsor's grant is left, as of the last
                # answered request. The sponsor reports this on every
                # response; it used to be captured only to log a warning,
                # so a dependent could not see its own remaining budget.
                # -1 = unlimited, or nothing observed yet.
                "remaining_budget_tokens": _rpb_last_budget(),
            }}

        if msg_type == "set_owner_wallet":
            # The owner wallet is the identity that owns this fleet's earnings
            # (tool_store's author fallback, service_store's provider identity)
            # and the address a remote connection must sign with to be rooted
            # at the full fleet. Local listener only (KEY_LOCAL_ONLY_MESSAGES):
            # setting it from a remote session would let a remote caller hand
            # ownership of the fleet to an address of its choosing.
            address = (msg.get("address") or "").strip()
            if address and not (address.startswith("0x") and len(address) == 42):
                return {"msg_id": msg_id, "ok": False,
                        "error": "address must be a 0x address (42 chars)"}
            try:
                from .config import save_owner_wallet_to_config
                save_owner_wallet_to_config(address)
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            # Mirror onto the live config and the running WS server so the
            # change takes effect without a restart.
            self.runtime._config.autonet.owner_wallet = address
            self.owner_wallet = address
            log.info("Owner wallet set to %s", address or "(cleared)")
            return {"msg_id": msg_id, "ok": True, "result": {
                "owner_wallet": address,
            }}

        if msg_type == "set_my_sponsor":
            sponsor_address = (msg.get("sponsor_address") or "").strip()
            if sponsor_address and not (
                    sponsor_address.startswith("0x") and len(sponsor_address) == 42):
                return {"msg_id": msg_id, "ok": False,
                        "error": "sponsor_address must be a 0x address (42 chars)"}
            rpb_cfg = self.runtime._config.autonet
            if sponsor_address and not (getattr(rpb_cfg, "owner_wallet", "") or ""):
                return {"msg_id": msg_id, "ok": False,
                        "error": "configure an owner wallet first — it is the "
                                 "identity a sponsor binds"}
            try:
                from .config import save_sponsor_address_to_config
                save_sponsor_address_to_config(sponsor_address)
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            # Mirror onto the live config so the next run picks it up without a
            # restart: resolve_provider_with_fallback reads the config each
            # time it builds an rpb provider.
            rpb_cfg.sponsor_address = sponsor_address
            return {"msg_id": msg_id, "ok": True, "result": {
                "sponsor_address": sponsor_address,
            }}

        if msg_type == "list_sponsor_bindings":
            # Also report sponsor MODE, so the panel can show whether this
            # daemon is actually serving: bindings on disk do not imply the
            # handler is wired (the mode is a separate, persisted flag).
            sponsor_cfg = self.runtime._config.autonet
            return {"msg_id": msg_id, "ok": True, "result": {
                "bindings": self.runtime.sponsor_bindings.to_summary_list(),
                "sponsor_inference": bool(
                    getattr(sponsor_cfg, "sponsor_inference", False)),
                "sponsor_provider": getattr(sponsor_cfg, "sponsor_provider", "") or "",
                "sponsor_model": getattr(sponsor_cfg, "sponsor_model", "") or "",
            }}

        if msg_type == "disable_sponsor_inference":
            try:
                result = self.runtime.autonet.disable_sponsor_inference()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "update_sponsor_budget":
            dependent_address = msg.get("dependent_address", "")
            budget_tokens = msg.get("budget_tokens")
            if not dependent_address or budget_tokens is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'dependent_address' or 'budget_tokens'"}
            try:
                budget_tokens = int(budget_tokens)
            except (TypeError, ValueError):
                return {"msg_id": msg_id, "ok": False, "error": "'budget_tokens' must be an integer"}
            binding = self.runtime.sponsor_bindings.update_budget(dependent_address, budget_tokens)
            if binding is None:
                return {"msg_id": msg_id, "ok": False, "error": "No binding for that address"}
            return {"msg_id": msg_id, "ok": True, "result": {"binding": binding.to_dict()}}

        if msg_type == "remove_sponsor_binding":
            dependent_address = msg.get("dependent_address", "")
            if not dependent_address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'dependent_address'"}
            removed = self.runtime.sponsor_bindings.remove(dependent_address)
            return {"msg_id": msg_id, "ok": True, "result": {"removed": removed}}

        # The daemon operator key (Decision 2026-10-27): registered bound to
        # the owner, it anchors, proposes, votes with delegated REP and
        # submits the household's claims.
        if msg_type in ("operator_status", "register_operator") and (
                not session.owner or session.scope_ids is not None):
            return {"msg_id": msg_id, "ok": False,
                    "error": f"'{msg_type}' is for the daemon owner's full-fleet "
                             "session only"}
        if msg_type == "sweep_to_owner":
            if not session.owner or session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "'sweep_to_owner' is for the daemon owner's "
                                 "full-fleet session only"}
            try:
                return {"msg_id": msg_id, "ok": True,
                        "result": await self._sweep_to_owner(msg.get("agent_ids"))}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}
        if msg_type == "submit_delegation":
            # The owner's signed EIP-712 Delegation to the operator; the
            # operator submits delegateBySig and pays the gas.
            if not session.owner or session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "'submit_delegation' is for the daemon owner's "
                                 "full-fleet session only"}
            try:
                result = await self._submit_delegation(
                    int(msg.get("nonce", 0)), int(msg.get("expiry", 0)),
                    str(msg.get("signature", "")))
                return {"msg_id": msg_id, "ok": bool(result.get("success")),
                        "result": result, "error": result.get("error", "")}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}
        if msg_type == "operator_status":
            try:
                return {"msg_id": msg_id, "ok": True,
                        "result": await self._operator_status()}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}
        if msg_type == "register_operator":
            try:
                result = await self._handle_register_operator(
                    owner=msg.get("owner", ""),
                    owner_sig=msg.get("owner_sig", ""),
                    self_owned=bool(msg.get("self_owned", False)),
                )
                return {"msg_id": msg_id, "ok": bool(result.get("success")),
                        "result": result, "error": result.get("error", "")}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # On-chain agent registration
        if msg_type == "register_agent_on_chain":
            agent_id = msg.get("agent_id", "")
            is_root = msg.get("is_root", False)
            private_key = msg.get("private_key", "")
            sponsor_address = msg.get("sponsor_address", "")
            # Owner-rooted registration (optional). When owner_sig is present
            # the agent is registered via registerAgentBound(owner, parent,
            # ownerSig); absent → plain registerAgent exactly as before.
            owner = msg.get("owner", "")
            parent = msg.get("parent", "")
            owner_sig = msg.get("owner_sig", "")
            self_owned = bool(msg.get("self_owned", False))
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            try:
                result = await self._handle_register_on_chain(
                    self_owned=self_owned,
                    agent_id=agent_id,
                    is_root=is_root,
                    private_key=private_key,
                    sponsor_address=sponsor_address,
                    owner=owner,
                    parent=parent,
                    owner_sig=owner_sig,
                )
                # Surface the agent's private key on a SUCCESSFUL registration
                # so the (local) user can import it into MetaMask and later sign
                # in remotely AS this agent (address-as-credential). Localhost
                # only — this whole message type is in KEY_LOCAL_ONLY_MESSAGES,
                # and we re-assert session.local as defense-in-depth. The key is
                # never returned over the remote listener.
                # Never for an external agent: its key is the operator's, which
                # only the owner-signed export_operator_key hands out.
                if result.get("success") and session.local \
                        and not self.runtime.registry.acts_as_operator(agent_id):
                    key = self.runtime.registry.get_agent_key(agent_id)
                    if key:
                        result["private_key"] = key
                        log.warning("Surfaced private key for agent '%s' on "
                                    "registration (local connection)", agent_id)
                # Notify all connected clients so any open view (Network tab,
                # agent list) re-pulls and shows the now-registered address,
                # instead of sitting on a stale snapshot ("Address: pending…").
                if result.get("success"):
                    await self.runtime.events.emit(Event(
                        type=EventType.AGENT_REGISTERED,
                        source=agent_id,
                        data={"agent_id": agent_id,
                              "agent_address": result.get("agent_address", ""),
                              "registered_on_chain": True},
                    ))
                return {"msg_id": msg_id, "ok": result.get("success", False), "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "check_agent_registration":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            try:
                result = await self._handle_check_registration(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Phase 12: registration is daemon-signed in one shot, so the
        # frontend no longer needs a "confirm" round-trip. The handler
        # is kept for back-compat with older clients but never overwrites
        # identity.address — the agent's on-chain address is its own
        # daemon-held key, never the user's wallet.
        if msg_type == "confirm_agent_registration":
            agent_id = msg.get("agent_id", "")
            tx_hash = msg.get("tx_hash", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            agent_def = self.runtime.registry.get_agent(agent_id)
            # An external agent's registration is the operator's, read from
            # chain (register_operator, operator_status), never asserted.
            if agent_def and agent_def.identity \
                    and not self.runtime.registry.is_operator_identity(agent_def.identity):
                agent_def.identity.registered_on_chain = True
                if tx_hash:
                    agent_def.identity.registration_tx = tx_hash
                self.runtime.registry.persist_identity(agent_id)
                log.info("Agent %s confirmed on-chain: tx=%s", agent_id, tx_hash)
            return {"msg_id": msg_id, "ok": True, "result": {"confirmed": True}}

        # Substrate on-chain state queries (rpb_* names retained for
        # frontend back-compat; map onto Substrate.sol reads).
        if msg_type == "rpb_state":
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "Substrate not configured"}
                result = await svc.get_substrate_state()
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "network_status":
            # Public summary; the owner's dashboards and the WAN run's
            # telemetry read it here.
            anchored = await self._anchored_epochs(last_n=0)
            return {"msg_id": msg_id, "ok": True,
                    "result": self._network_status_payload(anchored)}

        # Live status of the open epoch tick: the library's half (epoch id,
        # buffered events) with the EpochScheduler's (mode, candle window,
        # T_max, seed source).
        if msg_type == "epoch_status":
            autonet = getattr(self.runtime, "autonet", None)
            service = getattr(autonet, "_service", None) if autonet else None
            library = getattr(service, "_library", None) if service else None
            scheduler = getattr(service, "_epoch_scheduler", None) if service else None
            if library is None:
                # Degrade like economy_graph: a daemon with autonet off is a
                # normal state, not a fault. Neutral body, ok:true.
                return {"msg_id": msg_id, "ok": True,
                        "epoch": {"running": False, "epoch_id": None}}
            try:
                epoch = library.epoch_status()
                if scheduler is not None:
                    sched = scheduler.status()
                    # The library's opened_at wins when both exist (same
                    # value in practice).
                    sched.pop("opened_at", None)
                    epoch.update(sched)
                return {"msg_id": msg_id, "ok": True, "epoch": epoch}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # The jurisdiction's anchored epochs (Substrate anchors), newest
        # last: {epochs: [{number, epoch_id, cutoff, closed_at, anchored_at,
        # ...}], total_closed, running, source, reason?}. Not the local
        # epoch ticks (LibraryService.epoch_history), which close every
        # minute and say nothing about settlement.
        if msg_type == "epoch_history":
            try:
                last_n = max(1, min(int(msg.get("last_n", 20)), 200))
            except (TypeError, ValueError):
                last_n = 20
            return {"msg_id": msg_id, "ok": True,
                    "result": await self._anchored_epochs(last_n)}

        # Daemon auto-update status: current/available/staged version + state.
        # The updater stages verified releases; they apply on the next restart.
        if msg_type == "update_status":
            updater = getattr(self.runtime, "_updater", None)
            if updater is None:
                # Auto-update disabled or poll task not running.
                from .runtime.snapshot import _daemon_version
                return {"msg_id": msg_id, "ok": True, "status": {
                    "state": "disabled",
                    "current_version": _daemon_version(),
                    "available_version": "",
                    "staged_version": "",
                    "pending": False,
                }}
            try:
                return {"msg_id": msg_id, "ok": True, "status": updater.get_status()}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Tool-economy graph — registrations + declared-dep DAG
        # + recent per-digest mint (world consensus state), merged with the
        # local tool store (display names) and the service store (asks).
        # Powers the economy visualization. Empty result when the world
        # service isn't up yet.
        if msg_type == "economy_graph":
            try:
                last_n = int(msg.get("last_n_epochs", 10))
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                library = getattr(service, "_library", None) if service else None

                tools = []
                store = getattr(self.runtime, "tool_store", None)
                if store is not None:
                    for record in store.visible_to(None):
                        m = record.manifest
                        tools.append({
                            "digest": record.digest,
                            "name": record.name,
                            "description": m.get("description", ""),
                            "author": m.get("author", ""),
                            "trust_class": m.get("trust_class", ""),
                            "origin": record.origin,
                            "published": record.published,
                            "enabled": record.enabled,
                            "fee_atn": m.get("fee_atn", 0),
                            "version_of": m.get("version_of"),
                            "dependencies": m.get("dependencies") or [],
                            "local": True,
                        })

                services = []
                svc_store = getattr(self.runtime, "service_store", None)
                if svc_store is not None:
                    for rec in svc_store.list(include_retired=True):
                        services.append({
                            "digest": rec.digest,
                            "name": rec.name,
                            "description": str(rec.spec.get("description") or ""),
                            "author": rec.author,
                            "service_id": rec.service_id,
                            "backing_tool": rec.backing_tool,
                            "retired": rec.retired,
                        })

                economy = (library.read_economy_graph(last_n_epochs=last_n)
                           if library is not None else {
                               "registrations": {}, "ratings": {},
                               "recent_tool_mint": {}, "last_epoch": None,
                               "epochs_considered": 0,
                           })
                return {"msg_id": msg_id, "ok": True, "result": {
                    "tools": tools,
                    "services": services,
                    **economy,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Recent reviews for one tool (v3) — the Substrate view's review
        # drawer. Local attestation rows (axes, score, note text) plus
        # the tool's drifted position from the world service.
        if msg_type == "tool_reviews":
            digest = str(msg.get("digest") or "").strip().lower()
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            try:
                return {"msg_id": msg_id, "ok": True,
                        "result": self._tool_reviews_payload(
                            digest, msg.get("limit"))}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Every household's on-chain reviews of one tool (Decision
        # 2026-11-09), from the ToolBond logs the close's snapshot is built
        # from: reviewer, household, the four fields, charter scores,
        # weight, accuracy once settled and evidence_hash (pass it to
        # fetch_evidence). Public chain data: any session may read it.
        if msg_type == "tool_round_reviews":
            d = str(msg.get("digest") or "").strip().lower().removeprefix("0x")
            if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
                return {"msg_id": msg_id, "ok": False,
                        "error": "digest must be a 64-hex manifest digest"}
            from .runtime.tool_bond import ToolBondClient
            client = ToolBondClient(self.runtime)
            if not client.available():
                return {"msg_id": msg_id, "ok": False, "error": "no ToolBond configured"}
            bond_addr = client._addresses()[0]
            start = int(getattr(self.runtime._config.rpb, "deploy_block", 0) or 0) or None
            try:
                from nodes.common.voice_state import read_tool_round_reviews
                reviews = await asyncio.to_thread(
                    lambda: read_tool_round_reviews(client._w3(), bond_addr, d,
                                                    start_block=start))
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": f"ToolBond read failed: {e}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": d, "reviews": reviews}}

        if msg_type == "rpb_agent_training":
            # Substrate-native: returns mint_total (cumulative ATN pool
            # earnings, MONEY) + ATN balance + native gas for the agent.
            # The pre-substrate "training tokens / unclaimed rewards" split
            # is gone. Governance REP is NOT here: it is claimed DAO-side
            # (RepToken), 1:1 against these ratified earnings.
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "Substrate not configured"}
                result = await svc.get_agent_balances(address)
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "fleet_voice":
            # Owner section (ratified 2026-07-08: ATN = money, reputation
            # = voice): the owner's fleet — wallet balance + every bound
            # agent's ATN/reputation — and the derived voice weight the
            # close prices the household's reviews and usage with. The
            # voice weight is now REPUTATION-based (fleet reputation / rep
            # supply); ATN figures are still reported as money info.
            owner = msg.get("owner", "")
            if not owner:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'owner'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": True, "result": {
                        "owner": owner, "available": False,
                        "reason": "Substrate not configured on this daemon",
                    }}
                raw = await svc.get_fleet_voice(owner)
                if raw is None:
                    return {"msg_id": msg_id, "ok": True, "result": {
                        "owner": owner, "available": False,
                        "reason": "Chain read failed — is the RPC reachable?",
                    }}
                try:
                    from nodes.common.federated_reconcile import (
                        VOICE_EPSILON as _eps,
                    )
                except Exception:
                    _eps = 0.05
                _scale = 1_000_000.0  # chain units are ATN x 1e6
                local_names: dict[str, str] = {}
                try:
                    reg = self.runtime.registry
                    for defn, _status in self.runtime.list_agents():
                        ident = getattr(defn, "identity", None)
                        addr = (getattr(ident, "address", "") or "").lower()
                        if not addr:
                            continue
                        if reg.is_operator_identity(ident):
                            # External sessions share the operator's address
                            # (Decision 2026-11-08): one row, named for it.
                            local_names[addr] = "Operator"
                        elif addr not in local_names:
                            local_names[addr] = (
                                getattr(defn, "name", "") or defn.id)
                except Exception:
                    pass
                agents = []
                for a in raw["agents"]:
                    addr = str(a["agent_id"])
                    agents.append({
                        "agent_id": addr,
                        "name": local_names.get(addr.lower(), ""),
                        "local": addr.lower() in local_names,
                        "balance": a["balance_raw"] / _scale,
                        # Cumulative ATN pool earnings (agentMintTotal), NOT
                        # governance REP. "reputation" is the LEGACY-WIRE
                        # alias kept for one release.
                        "mint_total": a["reputation_raw"] / _scale,
                        "reputation": a["reputation_raw"] / _scale,
                    })
                # MONEY: ATN supply + fleet ATN total (reported, not voted).
                supply_raw = int(raw["supply_raw"])
                # WEIGHT: derived from cumulative ATN EARNINGS (networkMintTotal
                # / agentMintTotal), which is what an earned voice is priced
                # against here. Governance REP itself is DAO-side (RepToken).
                rep_supply_raw = int(raw.get("rep_supply_raw", 0))
                fleet_rep_raw = int(raw.get("fleet_reputation_raw", 0))
                # The weight the close uses: household REP / REP supply.
                # Without a RepToken read, fall back to the earnings share.
                h_rep = raw.get("household_rep_raw")
                rep_total = raw.get("rep_total_supply_raw")
                if h_rep is not None and rep_total:
                    share = h_rep / rep_total
                else:
                    share = ((fleet_rep_raw / rep_supply_raw)
                             if rep_supply_raw > 0 else 0.0)
                return {"msg_id": msg_id, "ok": True, "result": {
                    "owner": raw["owner"],
                    "available": True,
                    "reason": "",
                    # Money panel (unchanged fields).
                    "owner_balance": raw["owner_balance_raw"] / _scale,
                    "fleet_total": raw["fleet_total_raw"] / _scale,
                    "supply": supply_raw / _scale,
                    # Voice panel: weight derives from cumulative ATN
                    # EARNINGS. fleet_mint_total / mint_supply are the honest
                    # names; fleet_reputation / rep_supply are the LEGACY-WIRE
                    # aliases, kept for one release.
                    "fleet_mint_total": fleet_rep_raw / _scale,
                    "mint_supply": rep_supply_raw / _scale,
                    "fleet_reputation": fleet_rep_raw / _scale,
                    "rep_supply": rep_supply_raw / _scale,
                    # LEGACY-WIRE version marker: the literal "reputation"
                    # tells a client this daemon derives the weight from
                    # agentMintTotal/networkMintTotal rather than from raw
                    # ATN supply. It is a protocol version tag, not a label.
                    "weight_source": "reputation",
                    "voice_weight": round(_eps + share, 6),
                    # REP (18 decimals) the weight is read from.
                    "household_rep": (h_rep / 1e18) if h_rep is not None else None,
                    "rep_total_supply": (rep_total / 1e18) if rep_total else None,
                    "epsilon": _eps,
                    "agents": agents,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type in ("owner_binding_status", "rotate_owner") and (
                not session.owner or session.scope_ids is not None):
            # Owner, full fleet only. rotate_owner signs rotateOwner with each
            # named agent's daemon-held key, and Substrate needs nothing more
            # than that plus the new owner's own signature: a scoped session
            # (an agent's exported key, signed in remotely) could otherwise
            # move every agent on the daemon to a wallet of its choosing. The
            # status call hands out the nonces those signatures need.
            return {"msg_id": msg_id, "ok": False,
                    "error": f"'{msg_type}' is for the daemon owner's full-fleet "
                             "session only"}

        if msg_type == "owner_binding_status":
            # Report on-chain owner-binding state for every LOCAL agent, so
            # the Owner page can show which agents are bound / to whom / at
            # what nonce and drive registerAgentBound / rotateOwner. Chain
            # reads are batched politely (one web3 per call); a chain-
            # unreachable daemon returns ok:false with a clear reason.
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "Substrate not configured on this daemon"}
                # Local agents (defn.identity.address, like fleet_voice).
                local: list[tuple[str, str, str]] = []  # (agent_id, name, addr)
                try:
                    for defn, _status in self.runtime.list_agents():
                        ident = getattr(defn, "identity", None)
                        addr = (getattr(ident, "address", "") or "")
                        # External agents act as the operator, whose binding
                        # is operator_status's: no row of their own.
                        if self.runtime.registry.is_operator_identity(ident):
                            continue
                        if addr:
                            local.append((
                                defn.id,
                                getattr(defn, "name", "") or defn.id,
                                addr,
                            ))
                except Exception as e:
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"Could not enumerate local agents: {e}"}

                chain_id = getattr(self.runtime._config.rpb, "chain_id", 0)
                sub_addr = svc._substrate_address()
                _zero = "0x0000000000000000000000000000000000000000"
                owner_filter = (msg.get("owner", "") or "").lower()

                # Chain reads run in worker threads (they're synchronous
                # web3 HTTP calls — on the event loop a 15-agent fleet
                # starves the websocket keepalive and the client sees a
                # 1011 disconnect). Bounded concurrency keeps the RPC
                # polite; a failure anywhere surfaces as an error rather
                # than a misleading "unbound" state.
                _sem = asyncio.Semaphore(5)

                async def _read(agent_id: str, name: str, addr: str):
                    async with _sem:
                        status = await asyncio.to_thread(
                            svc.read_binding_status, addr)
                    return agent_id, name, addr, status

                try:
                    rows = await asyncio.gather(
                        *(_read(a, n, ad) for a, n, ad in local))
                except Exception as e:
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"Chain read failed: {e}"}

                agents_out = []
                for agent_id, name, addr, status in rows:
                    bound = status.get("owner", "")
                    owner_str = ("" if not bound or str(bound).lower() == _zero
                                 else str(bound))
                    if owner_filter and owner_str.lower() != owner_filter:
                        continue
                    agents_out.append({
                        "agent_id": agent_id,
                        "name": name,
                        "address": addr,
                        "registered": bool(status.get("registered")),
                        "owner": owner_str,
                        "binding_nonce": int(status.get("nonce", 0)),
                    })
                return {"msg_id": msg_id, "ok": True, "result": {
                    "substrate_address": sub_addr,
                    "chain_id": chain_id,
                    "agents": agents_out,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rotate_owner":
            # Rotate (or first-set) the owner+parent of one or more LOCAL
            # agents. Each binding carries the NEW owner's EIP-712 signature
            # at the agent's current nonce; the daemon submits rotateOwner
            # signed by THAT agent's own daemon-held key. No raw keys cross
            # the wire (only signatures), so this stays owner-authed but
            # remotely reachable — unlike register_agent_on_chain. Per-agent
            # failures don't abort the batch.
            new_owner = msg.get("new_owner", "")
            parent = msg.get("parent", "")
            bindings = msg.get("bindings", [])
            if not new_owner:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'new_owner'"}
            if not isinstance(bindings, list) or not bindings:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing 'bindings' (list of {address, sig})"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "Substrate not configured on this daemon"}
                # address (lowercased) -> local agent_id, for key lookup.
                by_addr: dict[str, str] = {}
                for defn, _status in self.runtime.list_agents():
                    ident = getattr(defn, "identity", None)
                    a = (getattr(ident, "address", "") or "").lower()
                    if a:
                        by_addr[a] = defn.id
                results = []
                for b in bindings:
                    addr = str(b.get("address", "") or "")
                    sig = str(b.get("sig", "") or "")
                    if not addr or not sig:
                        results.append({"address": addr, "ok": False,
                                        "error": "missing address or sig"})
                        continue
                    agent_id = by_addr.get(addr.lower())
                    if not agent_id:
                        results.append({"address": addr, "ok": False,
                                        "error": "no local agent with that address"})
                        continue
                    key = self.runtime.registry.get_agent_key(agent_id)
                    if not key:
                        results.append({"address": addr, "ok": False,
                                        "error": "no daemon-held key for that agent"})
                        continue
                    try:
                        # Off-loop: the call waits up to 120 s for a receipt.
                        r = await self._offload(
                            lambda key=key, new_owner=new_owner, sig=sig,
                            parent=parent: svc.rotate_owner(
                                private_key=key,
                                new_owner=new_owner,
                                new_owner_sig=sig,
                                new_parent=parent,
                            ))
                    except Exception as e:
                        results.append({"address": addr, "ok": False, "error": str(e)})
                        continue
                    if r.get("success"):
                        results.append({"address": addr, "ok": True,
                                        "tx_hash": r.get("tx_hash", "")})
                    else:
                        results.append({"address": addr, "ok": False,
                                        "error": r.get("error", "rotateOwner reverted")})
                return {"msg_id": msg_id, "ok": True, "result": {"results": results}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_agent_record":
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "Substrate not configured"}
                result = await svc.get_agent_record(address)
                # None means cleanly not registered; an RPC/ABI failure
                # raises and falls through to the ok:false below.
                if result is None:
                    return {"msg_id": msg_id, "ok": True,
                            "result": {"registered": False,
                                       "agent_address": address}}
                return {"msg_id": msg_id, "ok": True,
                        "result": {**result, "registered": True}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Pre-substrate handlers — investment/share/dividend/sponsor surface
        # was deleted with the contract nuke. These return an explicit
        # deprecation error so the UI can hide the affected panels rather
        # than silently fail. Substrate-native equivalents (if any) will
        # land alongside their respective product features.
        _DEPRECATED_RPB = {
            "rpb_investor_info", "rpb_accepted_tokens",
            "rpb_purchase_shares", "rpb_fund_training",
            "rpb_claim_training_reward", "rpb_claim_dividends",
            "rpb_record_inference", "rpb_sponsor_budget",
        }
        if msg_type in _DEPRECATED_RPB:
            return {
                "msg_id": msg_id, "ok": False,
                "error": f"'{msg_type}' was removed with the pre-substrate contract nuke "
                         "and has no Substrate.sol equivalent yet.",
            }

        if msg_type == "rpb_registered_agents":
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "Substrate not configured"}
                agents_list = await svc.get_all_registered_agents()

                # Build lookup: on-chain address → local agent metadata.
                # Phase 12: agents are always identified by their own
                # daemon-held address — no special-case for the root
                # agent or the connected wallet.
                local_meta = {}
                reg = self.runtime.registry
                sessions: list[str] = []
                for defn in reg._agents.values():
                    if defn.identity and defn.identity.address:
                        addr = defn.identity.address.lower()
                        if reg.is_operator_identity(defn.identity):
                            # External sessions share the operator's address
                            # (Decision 2026-11-08): one row, named for it.
                            sessions.append(defn.name or defn.id)
                            local_meta[addr] = {
                                "display_name": "Operator",
                                "display_description": "acts for: " + ", ".join(sessions),
                                "agent_type": "operator",
                                "model": "",
                                "is_online": True,
                            }
                            continue
                        local_meta[addr] = {
                            "display_name": defn.name,
                            "display_description": defn.description,
                            "agent_type": getattr(defn, "agent_type", ""),
                            "model": defn.model or "",
                            "is_online": True,
                        }

                # Enrich on-chain records with local metadata
                for agent in agents_list:
                    addr = agent.get("agent_address", "").lower()
                    if addr in local_meta:
                        agent.update(local_meta[addr])

                return {"msg_id": msg_id, "ok": True, "result": {"agents": agents_list}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Phase 13.3: re-verify cached registered_on_chain flags against
        # Substrate.areRegistered. Manual trigger because the contract is
        # only redeployed during dev — boot-time auto-reconcile would burn
        # an RPC every start for the vast majority of runs where nothing
        # has changed.
        if msg_type == "rpb_reconcile_registrations":
            try:
                report = self.runtime.reconcile_chain_registrations()
                return {"msg_id": msg_id, "ok": True, "result": report}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Local model selection (LLM backbone for JEPA training)
        if msg_type == "local_models":
            try:
                from nodes.common.local_models import host_status
                return {"msg_id": msg_id, "ok": True, "result": host_status()}
            except ImportError:
                return {"msg_id": msg_id, "ok": False, "error": "Network package not installed"}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "set_local_model":
            model_id = msg.get("model_id", "")
            try:
                from nodes.common.local_models import get_model_spec
                if model_id and not get_model_spec(model_id):
                    return {"msg_id": msg_id, "ok": False, "error": f"Unknown model: {model_id}"}
                # Update the autonet config
                if hasattr(self.runtime, "autonet") and self.runtime.autonet:
                    bridge = self.runtime.autonet
                    if hasattr(bridge, "_autonet_config") and bridge._autonet_config:
                        bridge._autonet_config.model.backbone_model_id = model_id
                return {
                    "msg_id": msg_id, "ok": True,
                    "result": {"model_id": model_id, "status": "set" if model_id else "cleared"},
                }
            except ImportError:
                return {"msg_id": msg_id, "ok": False, "error": "Network package not installed"}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Generic agent operations: reset conversation, change model, remove
        if msg_type == "reset_agent_conversation":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            try:
                await self.runtime.reset_agent_conversation(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"status": "reset", "agent_id": agent_id}}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "set_agent_model":
            agent_id = msg.get("agent_id", "")
            model = msg.get("model", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if not model:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'model' field"}
            try:
                await self.runtime.set_agent_model(agent_id, model)
                tier = get_model_tier(model)
                return {"msg_id": msg_id, "ok": True, "result": {
                    "agent_id": agent_id, "model": model, "status": "Model changed",
                    "capability_tier": tier, "tier_label": get_tier_label(tier),
                }}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        if msg_type == "remove_agent":
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            try:
                await self.runtime.unregister_agent(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"status": "removed", "agent_id": agent_id}}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Chat attachment upload (clipboard paste / attach button). The
        # frontend sends bytes; we persist under data_dir/attachments/ and
        # return the PATH — the chat message then references the path as
        # plain text and any agent with a file-read tool views it (Claude
        # bridge Read renders images; codex has view_image). Content-addressed
        # filename => repeat pastes of the same image dedupe for free.
        if msg_type == "attachment_upload":
            data_b64 = msg.get("data_b64", "")
            if not data_b64:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'data_b64' field"}
            import base64 as _b64
            import hashlib as _hashlib
            try:
                blob = _b64.b64decode(data_b64, validate=True)
            except Exception:
                return {"msg_id": msg_id, "ok": False, "error": "Invalid base64 payload"}
            if len(blob) > _ATTACHMENT_MAX_BYTES:
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Attachment too large ({len(blob)} bytes; "
                                 f"limit {_ATTACHMENT_MAX_BYTES})"}
            # Extension from the client filename, sanitized to a short
            # alnum token; the basename itself is the content hash.
            raw_name = str(msg.get("filename", "") or "")
            ext = Path(raw_name).suffix.lower().lstrip(".")
            if not ext.isalnum() or len(ext) > 8:
                ext = "png"
            digest = _hashlib.sha256(blob).hexdigest()[:16]
            att_dir = self.runtime._config.data_dir / "attachments"
            try:
                att_dir.mkdir(parents=True, exist_ok=True)
                dest = att_dir / f"{digest}.{ext}"
                if not dest.exists():
                    dest.write_bytes(blob)
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": f"Failed to store attachment: {exc}"}
            return {"msg_id": msg_id, "ok": True, "result": {
                "path": str(dest), "size": len(blob), "filename": raw_name or dest.name,
            }}

        # Export an agent's private key. The ONLY message that returns a
        # private key in its body. Gated localhost-only at the top of
        # _handle_message (KEY_LOCAL_ONLY_MESSAGES); re-asserted here as
        # defense-in-depth so the handler stays safe if dispatch is ever moved.
        if msg_type == "export_operator_key":
            # The owner imports the operator into their wallet to vote on
            # ordinary proposals with the household's delegated REP (the
            # daemon only votes on epoch elections). Decision 2026-10-27.
            if not ws_auth.assert_local_key_access(session) or not session.owner                     or session.scope_ids is not None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Operator key export is for the owner, localhost-only."}
            from eth_account import Account
            key = self._operator_key()
            log.warning("Operator key exported (local connection)")
            return {"msg_id": msg_id, "ok": True,
                    "result": {"address": Account.from_key(key).address,
                               "private_key": key}}

        if msg_type == "set_native_agents":
            # The owner's switch for the native agent framework
            # (atn/native_mode.py). Takes effect at once; existing native
            # agents are kept either way.
            if not session.owner:
                return {"msg_id": msg_id, "ok": False,
                        "error": "only the owner switches native agents"}
            from .native_mode import set_native_agents
            enabled = set_native_agents(self.runtime, bool(msg.get("enabled")))
            return {"msg_id": msg_id, "ok": True,
                    "result": {"native_agents": enabled}}

        if msg_type == "agent_api_url":
            # Where an agent on this machine starts (atn/agent_api.py); the
            # app shows it on the empty Agents screen.
            api = getattr(self, "agent_api", None)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"url": api.url if api is not None else ""}}

        if msg_type == "agent_activity":
            # The calls an external agent made through the MCP proxy
            # (atn/agent_activity.py), newest first.
            agent_id = str(msg.get("agent_id") or "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            limit = int(msg.get("limit") or 200)
            return {"msg_id": msg_id, "ok": True, "result": {
                "agent_id": agent_id,
                "entries": self._activity_log().read(agent_id, limit)}}

        if msg_type == "export_agent_key":
            if not ws_auth.assert_local_key_access(session):
                return {"msg_id": msg_id, "ok": False,
                        "error": "Private key export is localhost-only."}
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if self.runtime.registry.acts_as_operator(agent_id):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"'{agent_id}' is an external agent: it acts as "
                                 "the operator and has no key of its own "
                                 "(export_operator_key exports the operator's)"}
            key = self.runtime.registry.get_agent_key(agent_id)
            if not key:
                return {"msg_id": msg_id, "ok": False,
                        "error": f"No key stored for agent '{agent_id}'"}
            defn = self.runtime.registry._agents.get(agent_id)
            address = defn.identity.address if (defn and defn.identity) else ""
            # Log that an export happened — never the key itself.
            log.warning("Private key exported for agent '%s' (local connection)", agent_id)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"agent_id": agent_id, "address": address, "private_key": key}}

        # Strip protocol fields, pass only tool arguments.
        # Note: "type" is the routing field but also a valid arg for some tools
        # (e.g. post_message).  We strip it since the JSON object can't have two
        # "type" keys anyway — clients should use "message_type" for post_message.
        args = {k: v for k, v in msg.items()
                if k not in ("msg_id", "type", "_activity_logged", "_note")}
        # Tag client-initiated post_message with source="user" so the target
        # agent can distinguish user messages from agent messages.
        if msg_type == "post_message" and "source" not in args:
            args["source"] = "user"

        # --- Gate 3: scope/owner authorization (default-deny at the target) -
        # A scoped session may only call non-owner tools that target its own
        # subtree. Clamping caller_id is NOT enough — the dangerous tools read
        # an explicit agent_id/target and ignore caller_id — so we validate
        # every named target against the subtree here, before dispatch.
        authz_err = self._authorize_tool_call(msg_type, msg, session)
        if authz_err:
            return {"msg_id": msg_id, "ok": False, "error": authz_err}

        # caller_id: a full-fleet session may act as any agent (today's UI
        # behavior). A scoped session is clamped to its own root. When no
        # caller is named, the call is the OWNER's ("" → owner-trusted,
        # parentless creates).
        if session.scope_ids is None:
            caller_id = msg.get("caller_id") or ""
        else:
            caller_id = session.root_agent_id
        result = await execute_tool(msg_type, args, self.runtime, caller_id=caller_id)

        # Tools signal errors by returning {"error": "..."} with a truthy value.
        # Some tools include "error": None as a data field — that's not an error.
        if result.get("error"):
            return {"msg_id": msg_id, "ok": False, "error": result["error"]}
        return {"msg_id": msg_id, "ok": True, "result": result}

    def _activity_log(self):
        log_ = getattr(self, "_activity", None)
        if log_ is None:
            from .agent_activity import ActivityLog
            log_ = self._activity = ActivityLog(self.runtime._config.data_dir)
        return log_

    def _is_external_agent(self, agent_id: str) -> bool:
        if not agent_id:
            return False
        from .provider_identity import is_external
        defn = self.runtime.registry._agents.get(agent_id)
        return defn is not None and is_external(defn)

    async def _handle_logged(self, msg: dict, session: ClientSession,
                             agent_id: str) -> dict:
        """Handle an external agent's call and keep it in its activity log,
        with the same tool-use events a native agent's loop emits, so an
        open window updates live."""
        import uuid
        from .agent_activity import now
        tool = msg.get("type", "")
        use_id = uuid.uuid4().hex
        shown = {k: v for k, v in msg.items()
                 if k not in ("type", "msg_id", "caller_id", "_note")}
        note = str(msg.get("_note") or "")
        await self.runtime.events.emit(Event(
            type=EventType.AGENT_TOOL_USE_START, source=agent_id,
            data={"agent_id": agent_id, "tool_use_id": use_id,
                  "tool_name": tool, "input": shown, "note": note}))
        started = now()
        resp = await self._handle_message({**msg, "_activity_logged": True}, session)
        ok = bool(resp.get("ok"))
        try:
            entry = await asyncio.to_thread(
                self._activity_log().record, agent_id, tool, shown, ok,
                resp.get("result") if ok else resp.get("error"), started, now(),
                note)
        except Exception as exc:
            log.debug("activity log write failed for %s: %s", agent_id, exc)
            entry = {"result": ""}
        await self.runtime.events.emit(Event(
            type=EventType.AGENT_TOOL_USE_RESULT, source=agent_id,
            data={"agent_id": agent_id, "tool_use_id": use_id,
                  "tool_name": tool, "is_error": not ok,
                  "result_preview": entry.get("result", "")}))
        return resp

    # ------------------------------------------------------------------
    # Secrets vault surface (owner-only; values are WRITE-ONLY)
    # ------------------------------------------------------------------

    def _live_secret_holders(self, service: str) -> list[str]:
        """Agents whose LIVE broker session includes ``service`` — they may
        hold a staged copy of the OLD value until release/exit."""
        holders = {
            str(g.get("agent_id"))
            for g in self.runtime._grants.values()
            if g.get("agent_id") and service in (g.get("services") or [])
        }
        return sorted(holders)

    async def _handle_secrets_message(self, msg_type: str, msg: dict[str, Any],
                                      msg_id: Any) -> dict[str, Any]:
        """The five secrets_* handlers. Caller (_handle_message) has already
        enforced owner/full-fleet scope and the local-listener custody gate on
        the mutating types. NO handler ever returns a secret VALUE."""
        # Log/alarm reads need no keystore — handle them first so they work
        # even when pyrage/the vault is not provisioned.
        if msg_type == "secrets_usage_log":
            audit = getattr(self.runtime, "secret_audit", None)
            if audit is None:
                return {"msg_id": msg_id, "ok": True, "result": {"entries": []}}
            entries = audit.tail(
                limit=msg.get("limit", 200),
                agent_id=msg.get("agent_id") or None,
                service=msg.get("service") or None,
            )
            return {"msg_id": msg_id, "ok": True, "result": {"entries": entries}}

        if msg_type == "secrets_alarms":
            path = Path(self.runtime._config.data_dir) / "security" / "secret_alarms.json"
            alarms: list[dict[str, Any]] = []
            try:
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        alarms = data
            except (OSError, ValueError):
                log.debug("secrets_alarms: store unreadable", exc_info=True)
            return {"msg_id": msg_id, "ok": True, "result": {"alarms": alarms}}

        if msg_type == "secrets_probe":
            # Host-discovery (OS probe). Read-only, names-only. refresh falsy =>
            # return cached results (loaded from host_scan.json at construction);
            # refresh true => run a fresh concurrent scan (awaitable, capped).
            hd = self._get_host_discovery()
            if msg.get("refresh"):
                try:
                    await hd.scan()
                except Exception as exc:  # noqa: BLE001 — a scan failure is not fatal
                    log.debug("secrets_probe: scan failed", exc_info=True)
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"host scan failed: {exc}"}
            return {"msg_id": msg_id, "ok": True, "result": {
                "last_scan": hd.last_scan or None,
                "scanning": hd.scanning,
                "results": hd.to_list(),
            }}

        if msg_type == "secrets_config":
            # Config surface (worker isolation + root allowance). Needs no vault
            # keystore — handle before the keystore acquisition so a read-only
            # GET works even when the vault is unprovisioned.
            return await self._handle_secrets_config(msg, msg_id)

        from .vault_setup import _keystore, write_policy_map
        try:
            ks = _keystore()
        except Exception as exc:  # noqa: BLE001 — pyrage missing / vault absent
            return {"msg_id": msg_id, "ok": False,
                    "error": f"vault keystore unavailable: {exc}"}

        if msg_type == "secrets_status":
            try:
                # Only FLAT names are agent-grantable / user-managed. Dotted
                # names are daemon-plane (CredentialStore app.*, agent-key.*):
                # excluded from ``services`` (the grantable plane) but listed
                # names-only in ``daemon_services`` for display.
                all_names = ks.list_services()
                services = [s for s in all_names if "." not in s]
                daemon_services = [s for s in all_names if "." in s]
                # Per-secret authorized egress hosts (plaintext sidecar). Empty
                # list = not host-bound (value-in-hand mode until the owner sets
                # hosts to enable the forwarding proxy).
                secret_hosts = {
                    s: ks.get_authorized_hosts(s) for s in services
                }
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False,
                        "error": f"vault unreadable: {exc}"}

            # Live grants: pid -> {agent_id, services} inverted to per-agent.
            live: dict[str, dict[str, Any]] = {}
            for pid, g in self.runtime._grants.items():
                aid = g.get("agent_id")
                if aid:
                    live[str(aid)] = {"pid": pid,
                                      "services": list(g.get("services") or [])}

            def _resolve_wish(wish: str | None) -> list[str]:
                # Mirrors worker_host._resolve_parent_allowance's mapping.
                if not wish:
                    return []
                tokens = [t.strip().lower() for t in str(wish).split(",") if t.strip()]
                if not tokens or any(t == "none" for t in tokens):
                    return []
                if any(t == "all" for t in tokens):
                    # ``services`` is already dot-filtered above.
                    return list(services)
                try:
                    # Filter dotted (daemon-plane) names from any resolved
                    # allowance: a spec that literally names a dotted service or
                    # a bundle containing one must NOT resolve it.
                    return [s for s in ks.resolve_spec(wish) if "." not in s]
                except Exception:  # noqa: BLE001 — fail closed
                    return []

            # ``resolved`` is the agent's own authored wish expanded; it is NOT
            # what the daemon will hand over. ``effective`` is: the same wish
            # clamped by the parent's allowance (or by
            # secrets.default_root_allowance for a parent-less agent) and by the
            # isolation flag, computed by the very resolver the grant path uses.
            from .runtime.worker_host import resolve_effective_grant

            assignments: dict[str, dict[str, Any]] = {}
            for agent_id, defn in self.runtime.registry._agents.items():
                wish = getattr(defn, "secrets_allowance", None)
                lg = live.get(agent_id)
                pending = self.runtime._pending_grants.get(agent_id)
                if not wish and lg is None and not pending:
                    continue  # nothing secret-related about this agent
                parent_id = getattr(defn, "parent_id", None)
                if parent_id:
                    parent_defn = self.runtime.registry._agents.get(parent_id)
                    parent_spec = getattr(parent_defn, "secrets_allowance", None)
                else:
                    parent_spec = \
                        self.runtime._config.secrets.default_root_allowance
                try:
                    effective = resolve_effective_grant(agent_id, self.runtime)
                except Exception:  # noqa: BLE001 — display path, fail closed
                    effective = []
                assignments[agent_id] = {
                    "name": getattr(defn, "name", agent_id),
                    "allowance_spec": wish,
                    "resolved": _resolve_wish(wish),
                    "effective": effective,
                    "parent_allowance_spec": parent_spec,
                    # The ceiling itself, expanded: the widest set this agent
                    # could ever be granted whatever it asks for. The allowance
                    # picker greys out everything outside it.
                    "ceiling": _resolve_wish(parent_spec),
                    "pending": list(pending or []),
                    "granted": list(lg["services"]) if lg else [],
                    "pid": lg["pid"] if lg else None,
                    "live": lg is not None,
                }

            # Plane B — connector/provider credentials (presence only; managed
            # through their existing flows, surfaced here for completeness).
            connectors = {
                cid: self.runtime.credential_store.exists(cid)
                for cid in self.runtime.connectors.list_available()
            }
            providers: list[dict[str, Any]] = []
            try:
                # probe=False: this only needs id + configured. The probing
                # path costs seconds when ollama is down, and requests on a
                # connection are serialized, so it was timing out the page.
                for p in await self.runtime.provider_list(probe=False):
                    providers.append({"id": p.get("id"),
                                      "configured": bool(p.get("configured"))})
            except Exception:  # noqa: BLE001 — display-only, degrade quietly
                log.debug("secrets_status: provider_list failed", exc_info=True)

            monitor = getattr(self.runtime, "security_monitor", None)
            try:
                push_armed = bool(self.runtime._broker_client.value_push_armed)
            except Exception:  # noqa: BLE001
                push_armed = False
            return {"msg_id": msg_id, "ok": True, "result": {
                "services": services,
                "isolation_enabled": bool(
                    self.runtime._config.worker_isolation.enabled),
                "monitor_healthy": bool(monitor.is_healthy()) if monitor else False,
                "push_armed": push_armed,
                "default_root_allowance":
                    self.runtime._config.secrets.default_root_allowance,
                "assignments": assignments,
                "connectors": connectors,
                "providers": providers,
                "daemon_services": daemon_services,
                "secret_hosts": secret_hosts,
            }}

        if msg_type == "secrets_put":
            service = str(msg.get("service", "")).strip()
            value = msg.get("value", "")
            if not service:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'service' field"}
            if "," in service or any(c.isspace() for c in service):
                return {"msg_id": msg_id, "ok": False,
                        "error": "Service names must not contain commas or "
                                 "whitespace (they double as allowance-spec tokens)"}
            if "." in service:
                return {"msg_id": msg_id, "ok": False,
                        "error": "dotted names are reserved for daemon-internal "
                                 "credentials"}
            if service.lower() in ("none", "all"):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"'{service}' is a reserved allowance-spec keyword"}
            if not isinstance(value, str) or not value:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'value' field"}
            # Authorized egress hosts (plaintext metadata, never secret). If the
            # client omits the field on a NEW secret, prefill from the curated
            # name→host table so common providers get bound automatically; the
            # owner can edit later. An explicit [] clears binding (value-in-hand).
            raw_hosts = msg.get("authorized_hosts")
            hosts: list[str] | None = None
            if isinstance(raw_hosts, list):
                hosts = [str(h) for h in raw_hosts if isinstance(h, str)]
            try:
                existed = service in ks.list_services()
                ks.put_secret(service, value)
                if hosts is not None:
                    ks.set_authorized_hosts(service, hosts)
                elif not existed:
                    from .runtime.host_discovery.host_hints import suggest_hosts
                    suggested = suggest_hosts(service)
                    if suggested:
                        ks.set_authorized_hosts(service, suggested)
                write_policy_map(ks)
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False,
                        "error": f"vault write failed: {exc}"}
            del value  # drop our reference to the raw value promptly
            live_holders = self._live_secret_holders(service)
            audit = getattr(self.runtime, "secret_audit", None)
            if audit is not None:
                audit.record("rotated" if existed else "added",
                             agent_id="owner", services=[service])
            return {"msg_id": msg_id, "ok": True, "result": {
                "service": service,
                "status": "rotated" if existed else "created",
                # The broker caches the policy map at import; a NEW service
                # needs a broker restart before it is grantable. Rotation of
                # an existing name works live.
                "broker_restart_required": not existed,
                # Rotation does NOT re-stage: these agents may keep the old
                # value until their session ends.
                "live_holders": live_holders,
                "authorized_hosts": ks.get_authorized_hosts(service),
            }}

        if msg_type == "secrets_set_hosts":
            # Edit a secret's authorized egress hosts WITHOUT re-entering the
            # value. Plaintext metadata; no vault decryption involved.
            service = str(msg.get("service", "")).strip()
            if not service:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'service' field"}
            if "." in service:
                return {"msg_id": msg_id, "ok": False,
                        "error": "dotted names are reserved for daemon-internal "
                                 "credentials"}
            if service not in ks.list_services():
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown service: '{service}'"}
            raw_hosts = msg.get("authorized_hosts")
            if not isinstance(raw_hosts, list):
                return {"msg_id": msg_id, "ok": False,
                        "error": "'authorized_hosts' must be a list of hostnames"}
            hosts = [str(h) for h in raw_hosts if isinstance(h, str)]
            try:
                applied = ks.set_authorized_hosts(service, hosts)
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False,
                        "error": f"host update failed: {exc}"}
            return {"msg_id": msg_id, "ok": True, "result": {
                "service": service,
                "authorized_hosts": applied,
            }}

        if msg_type == "secrets_suggest_hosts":
            # Read-only: curated host suggestion for a prospective secret name.
            from .runtime.host_discovery.host_hints import suggest_hosts
            name = str(msg.get("service", "")).strip()
            return {"msg_id": msg_id, "ok": True, "result": {
                "service": name,
                "suggested_hosts": suggest_hosts(name),
            }}

        if msg_type == "secrets_delete":
            service = str(msg.get("service", "")).strip()
            if not service:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'service' field"}
            if "." in service:
                return {"msg_id": msg_id, "ok": False,
                        "error": "dotted names are reserved for daemon-internal "
                                 "credentials"}
            try:
                removed = ks.delete_secret(service)
                ks.delete_secret_meta(service)
                write_policy_map(ks)
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False,
                        "error": f"vault write failed: {exc}"}
            if not removed:
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown service: '{service}'"}
            live_holders = self._live_secret_holders(service)
            audit = getattr(self.runtime, "secret_audit", None)
            if audit is not None:
                audit.record("deleted", agent_id="owner", services=[service])
            return {"msg_id": msg_id, "ok": True, "result": {
                "service": service,
                "status": "deleted",
                "live_holders": live_holders,
            }}

        if msg_type == "secrets_import":
            # Import named host env vars into the vault. The raw VALUE is read
            # from os.environ and handed straight to ks.put_secret — it NEVER
            # appears in any response, log line, or error message. Same name
            # validation as secrets_put.
            names = msg.get("names") or []
            if not isinstance(names, list):
                return {"msg_id": msg_id, "ok": False,
                        "error": "'names' must be a list of env var names"}
            from .runtime.host_discovery import validate_secret_name
            existing = set(ks.list_services())
            imported: list[str] = []
            rotated: list[str] = []
            missing: list[str] = []
            invalid: list[dict[str, str]] = []
            new_service = False
            audit = getattr(self.runtime, "secret_audit", None)
            for raw in names:
                name = str(raw).strip()
                # Name doubles as an allowance-spec token + vault service key:
                # apply the identical validation secrets_put enforces.
                reason = validate_secret_name(name)
                if reason is not None:
                    invalid.append({"name": name or str(raw), "reason": reason})
                    continue
                value = os.environ.get(name)
                if not isinstance(value, str) or not value:
                    missing.append(name)
                    continue
                try:
                    existed = name in existing
                    ks.put_secret(name, value)
                except Exception:  # noqa: BLE001 — never leak the value in the error
                    log.debug("secrets_import: put_secret failed for a name",
                              exc_info=True)
                    invalid.append({"name": name, "reason": "vault write failed"})
                    continue
                finally:
                    value = None  # drop our reference to the raw value promptly
                if existed:
                    rotated.append(name)
                else:
                    imported.append(name)
                    existing.add(name)
                    new_service = True
                if audit is not None:
                    audit.record("rotated" if existed else "added",
                                 agent_id="owner", services=[name])
            # Regenerate the service->policy map once, after all writes.
            if imported or rotated:
                try:
                    write_policy_map(ks)
                except Exception as exc:  # noqa: BLE001
                    return {"msg_id": msg_id, "ok": False,
                            "error": f"vault policy map refresh failed: {exc}"}
            return {"msg_id": msg_id, "ok": True, "result": {
                "imported": imported,
                "rotated": rotated,
                "missing": missing,
                "invalid": invalid,
                # The broker caches the policy map at import; a NEW service
                # needs a broker restart before it is grantable.
                "broker_restart_required": new_service,
            }}

        return {"msg_id": msg_id, "ok": False,
                "error": f"Unknown secrets message: {msg_type}"}

    async def _handle_secrets_config(self, msg: dict[str, Any],
                                     msg_id: Any) -> dict[str, Any]:
        """GET/SET daemon security settings: worker isolation + default root
        allowance. Owner-only + local-listener (enforced by the caller via
        _SECRETS_MESSAGES scope check + KEY_LOCAL_ONLY_MESSAGES custody gate).

        Empty payload (neither ``worker_isolation`` nor ``default_root_allowance``)
        => read-only GET of the current values.

        Applies changes BOTH in-memory (``runtime._config``, live for NEW runs)
        and persisted (config.yaml). See the response ``restart_required`` /
        ``isolation_live`` fields for what a live in-memory flip does and does
        not activate.
        """
        cfg = self.runtime._config
        has_iso = "worker_isolation" in msg
        has_allow = "default_root_allowance" in msg

        # --- validation -----------------------------------------------------
        new_iso: bool | None = None
        if has_iso:
            new_iso = bool(msg.get("worker_isolation"))

        new_allow: str | None = None
        if has_allow:
            raw_allow = msg.get("default_root_allowance")
            if not isinstance(raw_allow, str):
                return {"msg_id": msg_id, "ok": False,
                        "error": "'default_root_allowance' must be a string spec "
                                 "('none', 'all', or a comma-separated list)"}
            spec = raw_allow.strip()
            if not spec:
                spec = "none"
            tokens = [t.strip() for t in spec.split(",") if t.strip()]
            if not tokens:
                spec = "none"
                tokens = ["none"]
            # Reject dotted names anywhere in the spec — dotted names are
            # daemon-plane (CredentialStore app.*, agent-key.*) and are never
            # grantable. Mirrors the secrets_put / validate_secret_name idiom.
            for t in tokens:
                if "." in t:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "dotted names are reserved for daemon-internal "
                                     "credentials and are not allowance-spec tokens"}
            new_allow = spec

        # --- read-only GET (empty payload) ----------------------------------
        write = has_iso or has_allow

        # --- apply in-memory ------------------------------------------------
        if new_iso is not None:
            cfg.worker_isolation.enabled = new_iso
        if new_allow is not None:
            cfg.secrets.default_root_allowance = new_allow

        # --- persist --------------------------------------------------------
        if write:
            try:
                from .config import save_secrets_config_to_yaml
                save_secrets_config_to_yaml(
                    worker_isolation=new_iso,
                    default_root_allowance=new_allow,
                )
            except Exception as exc:  # noqa: BLE001
                return {"msg_id": msg_id, "ok": False,
                        "error": f"failed to persist security settings: {exc}"}

        # --- effective-state + warnings -------------------------------------
        iso_now = bool(cfg.worker_isolation.enabled)
        allow_now = cfg.secrets.default_root_allowance

        # Isolation is read live from runtime._config on every spawn/execution
        # (_worker_isolation_enabled / _worker_eligible / worker_host.spawn_child),
        # so the in-memory flip is effective for NEW runs immediately. The one
        # boot-time consumer is the orphan-reaper monitor loop, started once at
        # runtime start under the flag; enabling isolation live does not start
        # that monitor until a daemon restart.
        isolation_live = True
        restart_required = False
        warnings: list[str] = []

        if new_iso is True:
            restart_required = True  # orphan-monitor loop only starts at boot
            warnings.append(
                "Worker isolation is live for new runs, but the orphan-reaper "
                "monitor loop only starts on daemon restart.")
            # Warn if the broker value-push is not armed: with isolation ON,
            # secret grants stage through the broker; a down broker fails closed.
            try:
                push_armed = bool(self.runtime._broker_client.value_push_armed)
            except Exception:  # noqa: BLE001
                push_armed = False
            if not push_armed:
                warnings.append(
                    "The secret broker value-push is not armed: grants will fail "
                    "closed until the broker is provisioned and running "
                    "(run `atn-vault-setup`).")

        # Env-var precedence: ATN_WORKER_ISOLATION overrides config at boot. If
        # it is set, a config change to isolation won't stick past what the env
        # dictates on the next restart.
        env_iso = os.environ.get("ATN_WORKER_ISOLATION")
        if has_iso and env_iso is not None:
            warnings.append(
                f"ATN_WORKER_ISOLATION={env_iso!r} is set in the environment and "
                "overrides config at boot: this change will not survive a daemon "
                "restart unless the env var is changed too.")

        warning = " ".join(warnings) if warnings else None
        return {"msg_id": msg_id, "ok": True, "result": {
            "worker_isolation": iso_now,
            "default_root_allowance": allow_now,
            "isolation_live": isolation_live,
            "restart_required": restart_required,
            "warning": warning,
        }}

    # ------------------------------------------------------------------
    # On-chain registration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_libp2p_peer_id(autonet: Any) -> str:
        """Best-effort read of the autonet bridge's libp2p PeerId.

        Returns an empty string when the host isn't running yet.
        """
        host = getattr(autonet, "_p2p_host", None)
        if host is None:
            return ""
        try:
            return getattr(host, "peer_id", "") or ""
        except Exception:
            return ""

    async def _serve_p2p_service_request(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """A ``service_request`` that arrived over libp2p
        (``/autonet/service/1.0.0``, direct or through a relay). Same code
        path as the WS stranger surface: the caller is never local, so the
        payment gate and payer-signature check apply in full."""
        msg_id = msg.get("msg_id")
        if msg.get("type", "service_request") != "service_request":
            return {"msg_id": msg_id, "ok": False,
                    "error": "only service_request is served over libp2p"}
        return await self._handle_service_request(msg, msg_id, local=False)

    async def _handle_service_request(
        self, msg: dict[str, Any], msg_id: Any, *, local: bool = False
    ) -> dict[str, Any]:
        """Provider-side service entry (docs/services_market.md §3).

        ``local`` says the frame arrived on the privileged LOCAL listener
        (or was synthesized by the owner's own `invoke_service`). It only
        widens the no-chain-config degrade: a stranger on the remote
        listener is refused rather than served for free. Defaults False so
        an un-threaded caller gets the safe answer.

        Wire frame: {spec_digest, request_id, args} → look up spec →
        validate payment (seam) → dispatch to the backing local tool →
        record the request in the provider's log → return the result.

        The backing implementation of a v1 service is a registered tool
        the OWNER chose to sell, so dispatch goes through
        ``runtime.tool_store.call`` with ``caller_id=None`` — the sale is
        the owner's sanction, not a lineage-scoped agent call.

        A service whose spec declares an ``inference`` block is served
        instead by this daemon's OWN provider stack (decision 2026-07-26)
        — no backing tool required or consulted. See
        ``_serve_inference_request``.

        Every reply about one of this daemon's services is signed
        (``receipt_sig``, ``service_client.sign_reply``) with the key that
        registered the service, the address the buyer paid: the buyer takes
        nothing a relay or a wss impostor says as served.

        Not served = not paid: a voucher the gate accepted is kept only when
        the dispatch succeeds and released otherwise. Every reply says so in
        ``charged`` (whether this request's payment was kept, signed with
        the rest), which is how the buyer knows whether to count its voucher.
        """
        reply = await self._service_request_reply(msg, msg_id, local=local)
        return self._sign_service_reply(msg, reply)

    def _sign_service_reply(self, msg: dict[str, Any],
                            reply: dict[str, Any]) -> dict[str, Any]:
        """``reply`` signed by the service's provider address, or unchanged
        when the digest is not ours or no key is held for its author (the
        buyer then treats it as not served)."""
        from . import service_client

        digest = str(msg.get("spec_digest") or "")
        request_id = str(msg.get("request_id") or "")
        record = self.runtime.service_store.get(digest) if digest else None
        if record is None or not request_id:
            return reply
        key = self._service_signing_key(record.author)
        if not key:
            return reply
        try:
            return service_client.sign_reply(key, request_id, digest, reply)
        except Exception as exc:                            # noqa: BLE001
            log.warning("could not sign the reply for %s: %s",
                        request_id[:18], exc)
            return reply

    def _verified_payer(self, msg: dict[str, Any], record: Any,
                        gate: dict[str, Any]) -> str:
        """The address that paid for this request, as the payment gate
        verified it, or "" when nothing was paid (a free service, the
        no-chain dev degrade). The request's own ``client`` field is never
        used: anyone can write any address there."""
        if gate.get("payer"):
            return str(gate["payer"])
        try:
            if int(gate.get("ask_amount") or 0) <= 0:
                return ""
        except (TypeError, ValueError):
            return ""
        # The gate passed a paid request only if its payer_signature
        # recovers to the payer (the payment's sender, or the channel's
        # client) under the service's provider address, the one
        # `_chain_service` holds the registry entry to.
        provider = self._service_provider_address(record.author)
        sig = str(msg.get("payer_signature") or "").strip()
        if not provider or not sig:
            return ""
        from .service_client import recover_request_signer
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        try:
            return str(recover_request_signer(
                sig, str(msg.get("request_id") or ""),
                str(msg.get("spec_digest") or record.digest), args, provider))
        except Exception:                                   # noqa: BLE001
            return ""

    async def _service_request_reply(
        self, msg: dict[str, Any], msg_id: Any, *, local: bool = False
    ) -> dict[str, Any]:
        """The unsigned reply to a ``service_request`` (see
        ``_handle_service_request``).

        Everything a request can be refused for without its payment is
        checked BEFORE the payment gate (unknown or retired service, no
        backing, malformed or oversized inference input, an upstream model
        provider this daemon cannot build), so a refused request leaves its
        payment unspent: the buyer can send it again, fixed, under the same
        request id. Attempts the gate refuses are counted on the provider's
        log but not attributed to anyone."""
        spec_digest = msg.get("spec_digest", "")
        request_id = msg.get("request_id", "")
        args = msg.get("args")
        if args is None:
            args = {}
        if not spec_digest or not request_id:
            return {"msg_id": msg_id, "ok": False, "charged": False,
                    "error": "Missing 'spec_digest' or 'request_id' field"}

        store = self.runtime.service_store
        record = store.get(spec_digest)
        if record is None:
            return {"msg_id": msg_id, "ok": False, "charged": False,
                    "error": f"Unknown service digest: {spec_digest[:16]}"}
        if record.retired:
            store.count_refused(spec_digest)
            return {"msg_id": msg_id, "ok": False, "charged": False,
                    "error": "Service retired"}
        if not isinstance(args, dict):
            store.count_refused(spec_digest)
            return {"msg_id": msg_id, "ok": False, "charged": False,
                    "error": "'args' must be an object"}

        upstream: Any = None
        tool_record: Any = None
        if record.inference:
            problem = self._inference_request_problem(record, args)
            if problem is None:
                upstream = self._inference_upstream(record)
                if isinstance(upstream, str):
                    problem = upstream
            if problem is not None:
                store.count_refused(spec_digest)
                return {"msg_id": msg_id, "ok": False, "charged": False,
                        "error": f"{problem}; the payment was not taken"}
        else:
            backing = record.backing_tool
            if not backing:
                store.count_refused(spec_digest)
                return {"msg_id": msg_id, "ok": False, "charged": False,
                        "error": "Service has no backing implementation"}
            tool_record = self.runtime.tool_store.get(backing)
            if tool_record is None:
                tool_record = self.runtime.tool_store.resolve(backing)
            if tool_record is None:
                store.count_refused(spec_digest)
                return {"msg_id": msg_id, "ok": False, "charged": False,
                        "error": f"Backing tool {backing[:16]} not found"}

        # Payment/voucher validation gate (Stage B — real on-chain verify).
        gate = await self._validate_service_payment(msg, record, local=local)
        amount = str(gate.get("ask_amount", ""))
        if not gate.get("ok"):
            store.count_refused(spec_digest)
            return {"msg_id": msg_id, "ok": False, "charged": False,
                    "error": f"Payment validation failed: "
                             f"{gate.get('reason', 'unknown')}"}
        payer = self._verified_payer(msg, record, gate)
        reservation = gate.get("reservation")
        try:
            # Inference-backed: the work item is a chat completion served
            # off this daemon's provider stack. Payment already cleared
            # above; the gate is backing-agnostic on purpose.
            if record.inference:
                result = await self._serve_inference_request(
                    record, args, upstream)
            else:
                # Owner-sanctioned dispatch: selling a tool IS the owner's
                # choice.
                result = await self.runtime.tool_store.call(
                    tool_record, args, caller_id=None)
        except BaseException:
            # Not served (a crash in the backing, or the call was cancelled).
            if reservation is not None:
                store.release_channel_voucher(reservation)
            raise
        ok = "error" not in result
        if reservation is not None:
            if ok:
                store.keep_channel_voucher(reservation)
            else:
                store.release_channel_voucher(reservation)
        # The log records the price only for a payment kept.
        proof = gate.get("proof")
        charged = ok if proof == "voucher" else proof == "direct"
        store.record_request(spec_digest, request_id, payer, ok=ok,
                             amount=amount if charged else "0")
        # A direct payment is on chain already, whatever the dispatch did.
        return {"msg_id": msg_id, "ok": ok, "charged": charged,
                "result": {"request_id": request_id, **result}}

    # ------------------------------------------------------------------
    # Consumer side: the owner buys one work item (invoke_service)
    # ------------------------------------------------------------------

    async def _handle_invoke_service(
        self, msg: dict[str, Any], msg_id: Any
    ) -> dict[str, Any]:
        """OWNER surface: purchase ONE work item from a service.

        Wire contract (the app is built against exactly this):

            request  {type: "invoke_service", digest, args}
            success  {ok: true, result: {request_id, receipt, output}}
            failure  {ok: false, error: "<human-readable>"}

        ``receipt`` is ``{paid, degraded, tx_hash, amount, token,
        recipient}`` — the buyer's proof of what was spent, and honest
        about the local-dev case where nothing was (``paid=false,
        degraded=true``). ``token`` is the constant ``"ATN"``: settlement
        is ATN-only, the key is kept for wire compatibility.

        Two resolution paths:

          - **Local digest** (in ``service_store``): pay, then dispatch
            through the SAME ``_handle_service_request`` a remote buyer
            would hit, carrying the real payment proof. The gate then
            verifies the payment we just made, so the owner buying from
            their own daemon exercises the production path rather than a
            private shortcut. Works for tool- and inference-backed alike.
          - **Foreign digest**: scan the on-chain registry for the
            digest's provider + ask, resolve their endpoint, pay, and
            call ``service_client.request_service`` cross-daemon. With no
            chain configured a foreign digest is unresolvable — there is
            nowhere to look up who serves it — so that fails loudly.

        Signing: WHOEVER PURCHASES SIGNS. The app may attach a payment
        it already made with the HUMAN's wallet — ``tx_hash`` +
        ``request_id`` in the message (both or neither) — and the
        handler then pays nothing itself: the proof rides to the
        provider gate, which verifies recipient/amount/replay on-chain
        exactly as it would a stranger's. Without a prepaid proof the
        daemon falls back to signing with the OWNER key
        (``autonet.private_key``) when one is configured — the headless
        setup — same doctrine as the ``service`` provider type.
        """
        digest = str(msg.get("digest") or "").strip()
        args = msg.get("args")
        if args is None:
            args = {}
        if not digest:
            return {"msg_id": msg_id, "ok": False,
                    "error": "Missing 'digest' field"}
        if not isinstance(args, dict):
            return {"msg_id": msg_id, "ok": False,
                    "error": "'args' must be an object"}

        prepaid_tx = str(msg.get("tx_hash") or "").strip()
        prepaid_rid = str(msg.get("request_id") or "").strip()
        prepaid: dict[str, str] | None = None
        if prepaid_tx and prepaid_rid:
            prepaid = {"tx_hash": prepaid_tx, "request_id": prepaid_rid}
        elif prepaid_tx or prepaid_rid:
            return {"msg_id": msg_id, "ok": False,
                    "error": "A prepaid proof needs BOTH 'tx_hash' and "
                             "'request_id' (got one without the other)."}

        record = self.runtime.service_store.get(digest)
        if record is not None and record.retired:
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Service {record.name or digest[:16]} is retired "
                             "and no longer offered."}
        if record is None:
            return await self._invoke_foreign_service(
                digest, args, msg_id, prepaid=prepaid)
        return await self._invoke_local_service(
            record, args, msg_id, prepaid=prepaid)

    @staticmethod
    async def _offload(factory: Any) -> Any:
        """Run a chain coroutine on a worker thread.

        ``on_chain.py``'s methods are ``async def`` but synchronous inside
        — web3 HTTP, ``estimate_gas``, and a
        ``wait_for_transaction_receipt(timeout=120)``. Awaited directly
        they occupy the event loop for the whole tx and the websocket
        keepalive starves (the same failure mode the
        ``owner_binding_status`` chain reads were moved off the loop to
        fix). ``factory`` is a zero-arg callable returning the coroutine,
        called INSIDE the thread so the coroutine is never bound to the
        wrong loop.
        """
        def _run() -> Any:
            return asyncio.run(factory())
        return await asyncio.to_thread(_run)

    def _owner_signing_key(self) -> str:
        autonet_cfg = getattr(self.runtime._config, "rpb", None)
        return str(getattr(autonet_cfg, "private_key", "") or "").strip()

    def _owner_client_address(self) -> str:
        autonet_cfg = getattr(self.runtime._config, "rpb", None)
        return str(getattr(autonet_cfg, "owner_wallet", "") or "").strip()

    # ---- ServiceRegistry (the chain half of publish/retire) -------------
    #
    # The owner surface publishes through the WS handlers, the agent surface
    # through agent_tools._register_service. Both must reach the SAME
    # on-chain registry: `market_services` / `find_services` read the chain,
    # not the daemon-local store, so a listing that never registers is
    # invisible to every buyer and `_invoke_foreign_service` refuses it.

    def _service_signing_key(self, author: str) -> str:
        """The key that signs ServiceRegistry writes for ``author``.

        "user" (the owner surface) signs with the operator key, the address
        the owner's services are registered and paid under; an agent author
        signs with its own stored key (an external agent's is the
        operator's). Empty when none is held."""
        if not author or author == "user":
            try:
                return str(self._operator_key() or "").strip()
            except Exception:                               # noqa: BLE001
                return ""
        registry = getattr(self.runtime, "registry", None)
        if registry is not None and hasattr(registry, "get_agent_key"):
            return str(registry.get_agent_key(author) or "").strip()
        return ""

    def _service_provider_address(self, author: str) -> str:
        """The address ``author``'s services are registered under: the
        address of the key that signs its ServiceRegistry writes."""
        key = self._service_signing_key(author)
        if not key:
            return ""
        try:
            from eth_account import Account
            return str(Account.from_key(key).address)
        except Exception:                                   # noqa: BLE001
            return ""

    async def _chain_service(self, record: Any) -> dict[str, Any]:
        """The ServiceRegistry entry of one of this daemon's services.

        The chain is the only reference for a service's price, whether it
        is active and who is paid (Decision 2026-11-10); the spec carries
        none of them. The entry is the number this daemon's registration
        returned; a record without one is matched by digest among the
        entries of the address that registers ``record.author``'s
        services, never another address (anyone can register a copy of a
        public spec). Returns ``{service_id, provider, ask_amount, active}``
        or ``{"error": ...}``; never raises."""
        from . import service_client
        from .on_chain import ServiceMarketClient

        config = self.runtime._config.rpb
        try:
            smc = ServiceMarketClient(config)
            if not smc.registry_available:
                return {"error": "ServiceRegistry not configured (missing "
                                 "service_registry_address or rpc_url)"}
            expected = self._service_provider_address(record.author)
            if record.service_id is not None:
                sid = int(record.service_id)
                entry = await self._offload(lambda: smc.get_service(sid))
                if entry is None:
                    return {"error": f"no on-chain service #{sid}"}
                if (service_client.normalize_digest(
                        str(entry.get("spec_digest") or ""))
                        != service_client.normalize_digest(record.digest)):
                    return {"error": f"on-chain service #{sid} names another "
                                     "spec"}
            else:
                if not expected:
                    return {"error": f"no signing key held for "
                                     f"'{record.author}', so its on-chain "
                                     "listing cannot be identified"}
                found = await self._offload(
                    lambda: service_client.lookup_service_ask(
                        config, expected, record.digest))
                if found.get("error"):
                    return {"error": str(found["error"])}
                entry = found
                if found.get("service_id") is not None:
                    self.runtime.service_store.set_service_id(
                        record.digest, int(found["service_id"]))
            provider = str(entry.get("provider") or "")
            if expected and provider.lower() != expected.lower():
                return {"error": "the on-chain listing is registered under "
                                 "another address"}
            return {"service_id": entry.get("service_id"),
                    "provider": provider,
                    "ask_amount": int(str(entry.get("ask_amount") or "0")
                                      or "0"),
                    "active": bool(entry.get("active"))}
        except Exception as exc:                            # noqa: BLE001
            log.warning("ServiceRegistry read for %s failed: %s",
                        record.digest[:16], exc)
            return {"error": f"ServiceRegistry read failed: {exc}"}

    # A buyer of an inference-backed service sends a chat request (see
    # the inference fulfilment path: args {messages, max_tokens?}).
    _INFERENCE_INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "messages": {"type": "array", "items": {"type": "object"}},
            "max_tokens": {"type": "integer"},
        },
        "required": ["messages"],
    }

    def _default_service_input_schema(
        self, backing_tool: str, inference: dict | None,
    ) -> dict:
        """The input schema a service inherits from its backing when the
        caller names none: the backing tool's manifest schema, or the chat
        request shape for an inference backing. ``{}`` when neither is
        known (the spec validator then refuses, as before)."""
        if inference is not None:
            return dict(self._INFERENCE_INPUT_SCHEMA)
        if backing_tool:
            store = getattr(self.runtime, "tool_store", None)
            record = store.get(backing_tool) if store is not None else None
            schema = (record.manifest.get("input_schema")
                      if record is not None else None)
            if isinstance(schema, dict) and schema:
                return dict(schema)
        return {}

    async def _register_service_on_chain(
        self, author: str, digest: str, ask_amount: int,
    ) -> dict[str, Any]:
        """Mirror a freshly-published spec into the on-chain ServiceRegistry.

        Returns ``{on_chain: bool, service_id?, tx_hash?, reason?}``. Never
        raises: the local spec is already persisted, so a chain failure
        downgrades the listing to local-only rather than losing it — the
        caller surfaces ``on_chain`` so the app can say so plainly."""
        from .on_chain import ServiceMarketClient

        try:
            smc = ServiceMarketClient(self.runtime._config.rpb)
            if not smc.registry_available:
                return {"on_chain": False,
                        "reason": "ServiceRegistry not configured (missing "
                                  "service_registry_address or rpc_url), so "
                                  "the service is published locally only and "
                                  "will not appear on the market."}
            key = self._service_signing_key(author)
            if not key:
                return {"on_chain": False,
                        "reason": f"No signing key held for '{author}', so the "
                                  "service is published locally only and will "
                                  "not appear on the market."}
            res = await self._offload(
                lambda: smc.register_service(key, digest, int(ask_amount)))
        except Exception as exc:                            # noqa: BLE001
            log.warning("register_service on-chain leg failed: %s", exc)
            return {"on_chain": False, "reason": str(exc)}
        if not res.get("success"):
            return {"on_chain": False,
                    "reason": f"On-chain registration failed: "
                              f"{res.get('error')}"}
        if res.get("service_id") is not None:
            self.runtime.service_store.set_service_id(
                digest, int(res["service_id"]))
        return {"on_chain": True,
                "service_id": res.get("service_id"),
                "tx_hash": res.get("tx_hash")}

    async def _retire_service_on_chain(
        self, record: Any,
    ) -> dict[str, Any]:
        """Flip the on-chain listing inactive for a locally-retired service.

        A local retire alone leaves the registry row `active`, and every
        buyer path pays BEFORE dispatch — so a stale row keeps taking money
        for a service that then answers "Service retired". Returns
        ``{on_chain: bool, service_id?, tx_hash?, reason?}``; never raises."""
        from .on_chain import ServiceMarketClient

        try:
            smc = ServiceMarketClient(self.runtime._config.rpb)
            if not smc.registry_available:
                return {"on_chain": False,
                        "reason": "ServiceRegistry not configured, so the "
                                  "service is retired locally only."}
            found = await self._chain_service(record)
            if found.get("error"):
                return {"on_chain": False, "reason": str(found["error"])}
            service_id = found.get("service_id")
            if service_id is None:
                return {"on_chain": False,
                        "reason": "no matching on-chain listing"}
            if not found.get("active"):
                return {"on_chain": True, "service_id": service_id,
                        "reason": "already inactive on chain"}
            key = self._service_signing_key(record.author)
            if not key:
                return {"on_chain": False,
                        "reason": f"No signing key held for "
                                  f"'{record.author}', so it is retired "
                                  "locally only: the market listing is still "
                                  "active."}
            res = await self._offload(
                lambda: smc.retire_service(key, int(service_id)))
        except Exception as exc:                            # noqa: BLE001
            log.warning("retire_service on-chain leg failed: %s", exc)
            return {"on_chain": False, "reason": str(exc)}
        if not res.get("success"):
            return {"on_chain": False,
                    "reason": f"On-chain retire failed: {res.get('error')}"}
        return {"on_chain": True, "service_id": service_id,
                "tx_hash": res.get("tx_hash")}

    @staticmethod
    def _receipt(
        *, paid: bool, degraded: bool, tx_hash: str | None,
        amount: Any, recipient: str, token: str = "ATN",
    ) -> dict[str, Any]:
        """The buyer's receipt, in the exact shape the app parses.

        ``token`` is a constant "ATN" label: settlement is ATN-only
        (ratified 2026-07-10) and the ask no longer names a token, but
        the receipt key stays on the wire so the Flutter Services page
        keeps parsing unchanged."""
        return {
            "paid": bool(paid),
            "degraded": bool(degraded),
            "tx_hash": tx_hash or None,
            "amount": str(amount if amount is not None else ""),
            "token": str(token or "ATN"),
            "recipient": str(recipient or ""),
        }

    async def _pay_for_invoke(
        self, *, recipient: str, amount: int, request_id: str,
    ) -> dict[str, Any]:
        """Pay one work item's ask with the owner key.

        Three outcomes, all of which the receipt distinguishes:

          - ``{"degraded": True}`` — no chain configured, so nothing was
            paid (LOUD warning; the provider-side gate degrades open in
            exactly the same condition, which is what makes a local demo
            possible).
          - ``{"free": True}`` — chain is up and the ask is zero, so
            nothing was OWED. Not a degrade: the price was zero.
          - ``{"tx_hash": ..., "request_id": ...}`` — really paid.

        Or ``{"error": ...}``, which the caller turns into an abort.
        """
        from . import service_client
        from .on_chain import OnChainService

        config = self.runtime._config.rpb
        try:
            chain_up = bool(OnChainService(config).available)
        except Exception:                                   # noqa: BLE001
            chain_up = False
        if not chain_up:
            log.warning(
                "INVOKE_SERVICE PAYMENT SKIPPED: no chain config "
                "(substrate_address/rpc_url unset) — buying a work item from "
                "%s WITHOUT paying. The provider gate degrades open in the "
                "same condition; local-dev only, never against a real "
                "counterparty.", (recipient or "(unknown)")[:10])
            return {"degraded": True}

        if amount <= 0:
            # A zero ask is a giveaway; payForService rejects zero anyway.
            log.info("invoke_service: ask is 0 — no payment needed")
            return {"free": True}

        key = self._owner_signing_key()
        if not key:
            return {"error": "Payment failed: chain is configured but this "
                             "daemon holds no owner signing key "
                             "(autonet.private_key) — buying a service is an "
                             "owner-level act."}
        if not recipient:
            return {"error": "Payment failed: could not resolve the service "
                             "provider's payment address."}
        paid = await self._offload(lambda: service_client.pay_for_service(
            config, key, recipient, amount, request_id=request_id))
        if paid.get("error"):
            return {"error": f"Payment failed: {paid['error']}"}
        return {"tx_hash": str(paid.get("tx_hash") or ""),
                "request_id": str(paid.get("request_id") or request_id)}

    async def _invoke_local_service(
        self, record: Any, args: dict[str, Any], msg_id: Any,
        prepaid: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Buy from a service this daemon publishes.

        Pay first, then re-enter ``_handle_service_request`` with a real
        ``service_request`` frame. Nothing about the dispatch is special-
        cased for the owner: the payment gate runs, the request lands on
        the provider log, and a tool- or inference-backing is chosen by
        the same code a remote buyer drives.
        """
        from . import service_client

        # Same price and payee the gate verifies against: the service's
        # ServiceRegistry entry (Decision 2026-11-10). Without a registry
        # (local dev) there is nothing to pay; the gate degrades the same.
        chain = await self._chain_service(record)
        if chain.get("error") is None and not chain.get("active"):
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Service {record.name or record.digest[:16]} is "
                             "retired on chain: nothing to buy."}
        amount = int(chain.get("ask_amount") or 0)
        amount_raw = str(amount)
        recipient = str(chain.get("provider") or "")

        if not record.inference and not record.backing_tool:
            # Answer before spending: an unfulfillable listing must not
            # take the buyer's money and then fail at dispatch.
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Service {record.name or record.digest[:16]} has "
                             "no backing implementation — nothing to buy."}

        if prepaid is not None:
            # The buyer already paid with their own wallet; the proof
            # rides to the gate, which verifies it on-chain. A bogus
            # proof fails at dispatch, not here.
            tx_hash = prepaid["tx_hash"]
            request_id = prepaid["request_id"]
            receipt = self._receipt(
                paid=True, degraded=False, tx_hash=tx_hash,
                amount=amount_raw, recipient=recipient)
        else:
            request_id = service_client.new_request_id()
            payment = await self._pay_for_invoke(
                recipient=recipient, amount=amount, request_id=request_id)
            if payment.get("error"):
                return {"msg_id": msg_id, "ok": False,
                        "error": payment["error"]}
            tx_hash = str(payment.get("tx_hash") or "")
            request_id = str(payment.get("request_id") or request_id)
            receipt = self._receipt(
                paid=bool(tx_hash), degraded=bool(payment.get("degraded")),
                tx_hash=tx_hash, amount=amount_raw,
                recipient=recipient)

        frame: dict[str, Any] = {
            "spec_digest": record.digest,
            "request_id": request_id,
            "args": args,
            "client": self._owner_client_address() or "owner",
        }
        if tx_hash:
            frame["tx_hash"] = tx_hash
            key = self._owner_signing_key()
            if key:
                frame["payer_signature"] = service_client.sign_request(
                    key, request_id, record.digest, args, recipient)
        # The owner buying from their own daemon: `_pay_for_invoke` already
        # degraded open in the same no-chain condition, so the gate must
        # match or the owner pays nothing and is then refused.
        reply = await self._handle_service_request(frame, msg_id, local=True)
        return self._invoke_result(reply, request_id, receipt, msg_id)

    async def _invoke_foreign_service(
        self, digest: str, args: dict[str, Any], msg_id: Any,
        prepaid: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Buy from a service published by ANOTHER daemon.

        The digest alone does not say who serves it: the catalog names the
        registry entry, read fresh from the chain for its provider, price
        and active flag (Decision 2026-11-10). Without chain config
        there is no registry to scan and no endpoint to dial, which is a
        hard failure rather than a degrade — unlike the local path there
        is no counterparty to reach at all.
        """
        from . import service_client
        from .on_chain import ServiceMarketClient

        config = self.runtime._config.rpb
        smc = ServiceMarketClient(config)
        if not smc.registry_available:
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Unknown service digest: {digest[:16]} — it is "
                             "not published by this daemon, and the "
                             "ServiceRegistry is not configured here, so a "
                             "foreign provider cannot be resolved."}
        from . import service_catalog as sc
        catalog = sc.catalog_for(self.runtime)
        await sc.ensure_synced(catalog)
        entry = catalog.entry_for(digest)
        if entry is None:
            # The spec may not be here yet: fetch it, then look again.
            store = getattr(self.runtime, "service_store", None)
            if store is not None:
                catalog.note_spec(digest, await store.spec_for(digest))
                entry = catalog.entry_for(digest)
        match = None
        if entry is not None:
            sid = int(entry["service_id"])
            fresh = await self._offload(
                lambda: service_client.read_service(config, sid))
            if not fresh.get("error"):
                match = fresh
        if match is None:
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Unknown service digest: {digest[:16]} — not in "
                             "this daemon's store and not registered on chain."}
        if not match.get("active"):
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Service {digest[:16]} is retired on chain — "
                             "nothing to buy."}

        provider_address = str(match.get("provider") or "")
        try:
            amount = int(str(match.get("ask_amount") or "0") or "0")
        except (TypeError, ValueError):
            amount = 0

        if prepaid is not None:
            tx_hash = prepaid["tx_hash"]
            request_id = prepaid["request_id"]
            receipt = self._receipt(
                paid=True, degraded=False, tx_hash=tx_hash,
                amount=amount, recipient=provider_address)
        else:
            request_id = service_client.new_request_id()
            payment = await self._pay_for_invoke(
                recipient=provider_address, amount=amount,
                request_id=request_id)
            if payment.get("error"):
                return {"msg_id": msg_id, "ok": False,
                        "error": payment["error"]}
            tx_hash = str(payment.get("tx_hash") or "")
            request_id = str(payment.get("request_id") or request_id)
            receipt = self._receipt(
                paid=bool(tx_hash), degraded=bool(payment.get("degraded")),
                tx_hash=tx_hash, amount=amount,
                recipient=provider_address)

        # Resolve the endpoint off the loop (blocking web3 read) and hand it
        # to request_service so it doesn't read the chain itself; the dial
        # and the wait are genuine async I/O and stay on the loop.
        from .on_chain import OnChainService
        try:
            endpoint = await self._offload(
                lambda: OnChainService(config).get_agent_endpoint(
                    provider_address))
        except Exception as exc:                            # noqa: BLE001
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Could not resolve the provider's endpoint: {exc}",
                    "result": {"request_id": request_id, "receipt": receipt}}

        reply = await service_client.request_service(
            config, provider_address, digest, args,
            tx_hash=tx_hash, request_id=request_id,
            client_address=self._owner_client_address(),
            endpoint=str(endpoint or ""),
            payer_key=self._owner_signing_key() or "",
        )
        return self._invoke_result(
            {"ok": bool(reply.get("ok")), "result": reply.get("result"),
             "error": reply.get("error")},
            request_id, receipt, msg_id)

    def _invoke_result(
        self, reply: dict[str, Any], request_id: str,
        receipt: dict[str, Any], msg_id: Any,
    ) -> dict[str, Any]:
        """Fold a dispatch reply into the invoke_service envelope.

        The receipt travels on the FAILURE path too: money may already
        have moved, and hiding that from the buyer would be the one
        genuinely dishonest thing this handler could do.
        """
        inner = reply.get("result")
        inner = inner if isinstance(inner, dict) else {}
        if not reply.get("ok"):
            detail = (reply.get("error")
                      or inner.get("error")
                      or "the provider returned not-ok")
            return {"msg_id": msg_id, "ok": False,
                    "error": str(detail),
                    "result": {"request_id": request_id, "receipt": receipt}}
        # Strip the transport's echoed request_id out of the payload so
        # `output` is just the service's own result object.
        output = {k: v for k, v in inner.items() if k != "request_id"}
        return {"msg_id": msg_id, "ok": True, "result": {
            "request_id": request_id,
            "receipt": receipt,
            "output": output,
        }}

    @staticmethod
    def _inference_request_problem(record: Any,
                                   args: dict[str, Any]) -> str | None:
        """Why an inference work item cannot be served as sent, or None.
        Checked before the payment gate."""
        from .service_spec import inference_input_chars

        messages = args.get("messages")
        if not isinstance(messages, list) or not messages:
            return ("inference service requires a non-empty 'messages' list "
                    "in args")
        system = args.get("system")
        if system is not None and not isinstance(system, str):
            return "'system' must be a string"
        size = inference_input_chars(messages, system or "")
        limit = record.max_input_chars
        if size > limit:
            return (f"request too large: its messages and system come to "
                    f"{size} characters, and this service takes at most "
                    f"{limit} (inference.max_input_chars)")
        return None

    def _inference_upstream(self, record: Any) -> Any:
        """The model provider this service is served on: exactly the one
        its spec names (``inference.provider``, with ``inference.model``),
        built by the daemon's provider manager. Nothing is inferred from the
        model's name and nothing falls back (no default provider, decision
        2026-10-06). Returns the provider, or a string saying why it cannot
        be built."""
        block = record.inference
        name = str(block.get("provider") or "").strip()
        model = str(block.get("model") or "").strip()
        if not name or not model:
            return ("this service's spec names no upstream provider and model "
                    "(inference.provider, inference.model)")
        manager = getattr(self.runtime, "providers", None)
        if manager is None:
            return "this daemon has no provider stack wired"
        from .models import AgentDefinition, AgentMode
        defn = AgentDefinition(
            id=f"service-{record.digest[:12]}", name="service",
            mode=AgentMode.COGNITIVE, provider=name, cognitive_model=model,
            tools=[])
        try:
            provider = manager.resolve_provider_with_fallback(defn)
        except Exception as exc:                            # noqa: BLE001
            log.warning("inference service %s: provider %r unavailable: %s",
                        record.digest[:16], name, exc)
            return f"upstream provider {name!r} is not available here: {exc}"
        if provider is None:
            return f"upstream provider {name!r} is not available here"
        return provider

    async def _serve_inference_request(
        self, record: Any, args: dict[str, Any], provider: Any = None,
    ) -> dict[str, Any]:
        """Serve a paid work item off this daemon's own provider stack.

        Decision 2026-07-26 (docs/services_market.md): a service may
        declare an inference backing instead of a backing tool. The work
        item is ``{messages, max_tokens?, system?, temperature?}``; the
        reply is ``{content, model, usage}``.

        The upstream is the provider the spec names, with the model it
        names (``_inference_upstream``); ``provider`` is that one, already
        built before the payment gate ran. No budget metering: the ask is
        the price, the token cap and the input bound are the ceiling.
        """
        problem = self._inference_request_problem(record, args)
        if problem is not None:
            return {"error": problem}
        if provider is None:
            provider = self._inference_upstream(record)
            if isinstance(provider, str):
                return {"error": provider}
        messages = args.get("messages")

        cap = record.max_tokens_cap
        requested = args.get("max_tokens")
        try:
            max_tokens = int(requested) if requested not in (None, "") else cap
        except (TypeError, ValueError):
            max_tokens = cap
        # Clamp: the cap is what the ask was priced against, and a buyer
        # asking for more than they paid for doesn't get it.
        max_tokens = max(1, min(max_tokens, cap))

        model = str(record.inference.get("model") or "").strip()
        try:
            temperature = float(args.get("temperature") or 0.0)
        except (TypeError, ValueError):
            temperature = 0.0
        try:
            resp = await provider.send(
                messages=messages,
                system=str(args.get("system") or ""),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:                            # noqa: BLE001
            log.warning("inference service dispatch failed: %s", exc)
            return {"error": f"inference failed: {exc}"}

        usage = getattr(resp, "usage", None)
        return {
            "content": getattr(resp, "text", "") or "",
            "model": getattr(resp, "model", "") or model,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            },
            "stop_reason": getattr(resp, "stop_reason", "") or "end_turn",
            "max_tokens": max_tokens,
        }

    async def _validate_service_payment(
        self, request: dict[str, Any], record: Any, *, local: bool = False
    ) -> dict[str, Any]:
        """Verify a service request's payment proof (Stage B).

        The request must carry ONE of two proofs:

          - Direct: ``{tx_hash, request_id}`` — a Substrate.payForService
            transfer. We fetch that receipt, decode the ServicePayment event,
            and check recipient == the serving agent's address and amount >=
            the service ask. Replay is closed by a persisted seen-request-id
            set (a request_id already served is rejected).

          - Voucher: ``{channel_id, cumulative_amount, signature}`` — a
            PaymentChannel voucher. We recover+verify the signature against
            on-chain channel state (client + Open) and require the increment
            (cumulative - last_seen) >= ask. The voucher is reserved (it
            counts toward last_seen at once) and returned as
            ``reservation``: the caller keeps it, to settle later via
            close_channel, only when the request is served, and releases
            it otherwise. The request id is consumed either way.

        Config escape hatch: when the daemon has NO chain config, a request
        that arrived on the LOCAL listener (the two-daemon dev demo) is
        allowed with a LOUD warning. A request from the REMOTE listener is
        REFUSED: `service_request` is dispatched pre-auth precisely because
        the payment IS the credential, so degrading open there would turn a
        transient registry outage into unlimited free work (including
        inference off the owner's provider stack) for any stranger who
        knows the digest. When chain IS configured, the gate is enforced
        on both listeners.

        Returns ``{ok: bool, reason: str}``; an accepted request also names
        its ``proof`` ("direct", "voucher", "free" or "degraded").
        """
        from .on_chain import OnChainService, ServiceMarketClient

        config = self.runtime._config.rpb
        oc = OnChainService(config)
        # --- Config escape hatch: no chain ----------------------------------
        # Local listener: degrade loud-and-open (the dev demo).
        # Remote listener: fail CLOSED — see the docstring.
        if not oc.available:
            if not local:
                log.warning(
                    "SERVICE PAYMENT GATE: no chain config "
                    "(substrate_address/rpc_url unset) — REFUSING remote "
                    "service request for %s; payment cannot be verified.",
                    record.name or record.digest[:16])
                return {"ok": False,
                        "reason": "this daemon has no chain configuration, so "
                                  "it cannot verify payment and will not serve "
                                  "paid requests"}
            log.warning(
                "SERVICE PAYMENT GATE DEGRADED: no chain config "
                "(substrate_address/rpc_url unset) — ALLOWING service request "
                "for %s WITHOUT payment verification (local listener only).",
                record.name or record.digest[:16])
            return {"ok": True, "reason": "chain not configured (dev degrade)",
                    "proof": "degraded"}

        # The price, the active flag and the payee come from the service's
        # ServiceRegistry entry, the only reference for them (Decision
        # 2026-11-10): a price changed by updateServiceAsk applies at once,
        # and a service retired on chain is refused.
        chain = await self._chain_service(record)
        if chain.get("error"):
            return {"ok": False,
                    "reason": f"cannot read this service's on-chain listing: "
                              f"{chain['error']}"}
        if not chain.get("active"):
            return {"ok": False, "reason": "service retired on chain"}
        recipient = str(chain.get("provider") or "")
        ask_amount = int(chain.get("ask_amount") or 0)

        # A free service needs no payment proof (payForService accepts zero,
        # but a zero ask has nothing to pay; the client sends no proof).
        if ask_amount <= 0:
            return {"ok": True, "reason": "free service", "ask_amount": 0,
                    "proof": "free"}

        store = self.runtime.service_store
        tx_hash = str(request.get("tx_hash") or "").strip()
        request_id = str(request.get("request_id") or "").strip()
        channel_id = request.get("channel_id")
        signature = request.get("signature")
        cumulative = request.get("cumulative_amount")

        # --- Direct payForService path --------------------------------------
        if tx_hash and request_id:
            # One spelling for the replay key AND the on-chain match: the
            # chain pads or cuts any hex to 32 bytes, so "R", "R0", "R00" all
            # matched one payment while each counted as a new request id.
            rid = request_id.lower().removeprefix("0x")
            if len(rid) != 64 or any(c not in "0123456789abcdef" for c in rid):
                return {"ok": False,
                        "reason": "request_id must be 32 bytes of hex (64 hex digits)"}
            request_id = "0x" + rid
            inflight = self.__dict__.setdefault("_payments_in_flight", set())
            if store.has_served_request(request_id) or request_id in inflight:
                return {"ok": False,
                        "reason": f"request_id {request_id[:18]} already served "
                                  "(replay)"}
            if not recipient:
                return {"ok": False,
                        "reason": "service has no on-chain provider address "
                                  "to verify payment against"}
            # Only the payer may redeem its payment: the request must be
            # signed by the key that paid (service_client.sign_request). A
            # request id copied off the chain carries no such signature.
            from .service_client import recover_request_signer
            sig = str(request.get("payer_signature") or "").strip()
            if not sig:
                return {"ok": False,
                        "reason": "unsigned request: a paid request must carry "
                                  "the payer's signature (payer_signature)"}
            try:
                payer = recover_request_signer(
                    sig, request_id,
                    str(request.get("spec_digest") or record.digest),
                    request.get("args") if isinstance(request.get("args"), dict)
                    else {}, recipient)
            except Exception:
                return {"ok": False, "reason": "payer_signature is malformed"}
            # Reserved while the chain is read, so two copies of one request
            # sent at once cannot both pass the replay check.
            inflight.add(request_id)
            try:
                # Off-loop: verify_service_payment is `async def` but
                # synchronous inside (web3 HTTP receipt fetch + log decode).
                # Awaited directly it occupies the event loop for the whole
                # read, and a buyer's connection — plus every other client of
                # this daemon — starves on the websocket keepalive and gets
                # dropped mid-request. Same fix `invoke_service` and
                # `owner_binding_status` already carry.
                res = await self._offload(lambda: oc.verify_service_payment(
                    request_id=request_id,
                    recipient=recipient,
                    min_amount=ask_amount,
                    tx_hash=tx_hash,
                    payer=payer,
                ))
                if not res.get("verified"):
                    return {"ok": False, "reason": res.get("reason", "unverified")}
                # Consume the request_id AFTER a positive verify (replay guard).
                store.mark_request_served(request_id)
            finally:
                inflight.discard(request_id)
            return {"ok": True, "reason": "ok", "ask_amount": ask_amount,
                    "proof": "direct"}

        # --- Voucher (PaymentChannel) path ----------------------------------
        if channel_id is not None and signature and cumulative is not None:
            try:
                cid = int(channel_id)
                cum = int(cumulative)
            except (TypeError, ValueError):
                return {"ok": False,
                        "reason": "channel_id and cumulative_amount must be integers"}
            smc = ServiceMarketClient(config)
            if not smc.channel_available:
                return {"ok": False,
                        "reason": "PaymentChannel not configured on this daemon"}
            # The request id is single use here too: a voucher is released
            # when its request fails, and a copy of that request sent later
            # must not be served on it.
            rid = store.normalize_request_id(request_id)
            if not rid or not str(request.get("payer_signature") or "").strip():
                return {"ok": False,
                        "reason": "a voucher request must be signed by the "
                                  "channel's client (payer_signature over "
                                  "request_id and args)"}
            if len(rid) != 64 or any(c not in "0123456789abcdef" for c in rid):
                return {"ok": False,
                        "reason": "request_id must be 32 bytes of hex (64 hex digits)"}
            # One voucher check per channel at a time: two vouchers raced
            # through the cumulative check before either was recorded.
            locks = self.__dict__.setdefault("_channel_locks", {})
            async with locks.setdefault(cid, asyncio.Lock()):
                if store.has_served_request(rid):
                    return {"ok": False,
                            "reason": f"request_id 0x{rid[:16]} already used (replay)"}
                # Off-loop for the same reason as the direct path above: the
                # voucher check reads on-chain channel state synchronously.
                res = await self._offload(
                    lambda: smc.verify_voucher(cid, cum, signature))
                if not res.get("valid"):
                    return {"ok": False, "reason": res.get("reason", "invalid voucher")}
                # The voucher pays for these args only: the channel's client
                # signs the request (as on the direct path), so a relay that
                # sees the voucher cannot spend it on a request of its own.
                from .service_client import recover_request_signer
                sig = str(request.get("payer_signature") or "").strip()
                try:
                    payer = recover_request_signer(
                        sig, request_id,
                        str(request.get("spec_digest") or record.digest),
                        request.get("args") if isinstance(request.get("args"), dict)
                        else {}, recipient) if sig and request_id else ""
                except Exception:
                    payer = ""
                if not payer or payer.lower() != str(res.get("client") or "").lower():
                    return {"ok": False,
                            "reason": "a voucher request must be signed by the "
                                      "channel's client (payer_signature over "
                                      "request_id and args)"}
                # The channel's provider must be the serving agent.
                if recipient and str(res.get("provider", "")).lower() != recipient.lower():
                    return {"ok": False,
                            "reason": "voucher channel provider is not this service's "
                                      "provider"}
                # Settlement pays at most the deposit: a voucher above it is
                # work the provider can never collect.
                if res.get("exceeds_deposit"):
                    return {"ok": False,
                            "reason": f"voucher cumulative {cum} exceeds the channel "
                                      f"deposit {res.get('deposit')}"}
                last = store.last_channel_cumulative(cid)
                if cum - last < ask_amount:
                    return {"ok": False,
                            "reason": f"voucher increment {cum - last} < ask "
                                      f"{ask_amount}"}
                # Accept, as a reservation: it counts at once (a copy sent
                # meanwhile is refused), but _handle_service_request keeps it
                # only if the request is served and otherwise releases it, so
                # a buyer never pays for a failed request (atomic tasks).
                reservation = store.reserve_channel_voucher(
                    cid, cum, str(signature),
                    provider=str(res.get("provider", "")),
                    service_id=chain.get("service_id"), ask=ask_amount)
                store.mark_request_served(rid)
            return {"ok": True, "reason": "ok", "ask_amount": ask_amount,
                    "proof": "voucher", "reservation": reservation}

        return {"ok": False,
                "reason": "no payment proof: provide {tx_hash, request_id} or "
                          "{channel_id, cumulative_amount, signature}"}

    async def _handle_register_on_chain(
        self,
        agent_id: str,
        is_root: bool = False,
        private_key: str = "",
        sponsor_address: str = "",
        owner: str = "",
        parent: str = "",
        owner_sig: str = "",
        self_owned: bool = False,
    ) -> dict[str, Any]:
        """Handle agent registration on the RPB contract.

        Two modes:
        - Root agent (is_root=True): requires private_key from the frontend
          wallet (custodial) or returns call_data for the frontend to sign
          (non-custodial / WalletConnect).
        - Child agent: daemon holds the key and signs directly.

        If private_key is empty and is_root=True, returns unsigned call_data
        for the frontend wallet to submit via sendTransaction.
        """
        from .on_chain import OnChainService

        # Registering is joining the network: resolve the packaged registry
        # into the config now (the daemon is local-only until this point).
        self.runtime.autonet.ensure_network_config()
        config = self.runtime.autonet.config
        svc = OnChainService(config)

        if not svc.available:
            return {"success": False,
                    "error": "On-chain not configured (missing substrate_address or rpc_url)"}

        # Look up the agent.
        agent_def = self.runtime.registry.get_agent(agent_id)
        if not agent_def:
            return {"success": False, "error": f"Agent '{agent_id}' not found"}
        if not agent_def.identity:
            return {"success": False, "error": f"Agent '{agent_id}' has no identity"}

        identity = agent_def.identity

        if self.runtime.registry.is_operator_identity(identity):
            # An external agent acts as the operator (Decision 2026-11-08):
            # registering it is registering the operator, once per household.
            if identity.registered_on_chain:
                return {"success": True, "acts_as_operator": True,
                        "already_registered": True,
                        "agent_address": identity.address,
                        "operator": identity.address}
            result = await self._handle_register_operator(
                owner=owner, owner_sig=owner_sig, self_owned=self_owned)
            result["acts_as_operator"] = True
            result["agent_address"] = identity.address
            return result

        # Substrate-native registration flow (Phase 12):
        #
        # The agent has its own keypair generated at agent creation time
        # (held by the daemon, not the user's wallet). Registration is
        # ALWAYS daemon-signed from the agent's own key — this includes
        # root agents. The user's wallet only funds the
        # agent's address; it never signs the registerAgent tx itself.
        #
        # This means consensus operations (anchor submission, training
        # records) sign autonomously from the agent's key without
        # involving the user's wallet on every epoch close.
        key = self.runtime.registry.get_agent_key(agent_id)
        if not key:
            return {"success": False,
                    "error": f"No private key stored for agent '{agent_id}' "
                             "(daemon should have generated one at agent creation time)"}

        # Pre-flight: agent address needs gas to pay for registerAgent.
        try:
            w3 = svc._get_web3()
            # Off the loop: an RPC round trip must not stall the WS server.
            balance = await asyncio.to_thread(
                w3.eth.get_balance, w3.to_checksum_address(identity.address))
            if balance == 0:
                return {
                    "success": False,
                    "mode": "needs_funding",
                    "error": "Agent address has no gas balance — fund it first.",
                    "agent_address": identity.address,
                    "chain_id": config.chain_id,
                }
        except Exception as e:
            return {"success": False, "error": f"Could not check agent balance: {e}"}

        # Resolve the daemon's libp2p PeerId. Required by Substrate.sol
        # so other daemons can DHT-resolve this agent's reachability.
        #
        # Phase 12: autonet is registration-driven. If the libp2p host
        # isn't up yet, this is the moment to start it — clicking
        # "register on chain" is the user signal that they want the
        # daemon to participate in the wider consensus network.
        autonet = getattr(self.runtime, "autonet", None)
        if autonet is None:
            return {"success": False, "error": "Autonet bridge not available"}

        peer_id = self._read_libp2p_peer_id(autonet)
        if not peer_id:
            log.info("Starting autonet just-in-time for agent registration (%s)",
                     agent_id)
            try:
                start_result = await autonet.start()
                if start_result.get("status") == "error":
                    return {
                        "success": False,
                        "error": f"Could not start autonet to register: "
                                 f"{start_result.get('error', 'unknown')}",
                    }
            except Exception as e:
                return {"success": False,
                        "error": f"Failed to start autonet: {e}"}

            # Poll briefly for the libp2p host to come up. AutonetHost
            # publishes peer_id as soon as its ready_event fires.
            import asyncio as _asyncio
            for _ in range(60):  # ~30s
                peer_id = self._read_libp2p_peer_id(autonet)
                if peer_id:
                    break
                await _asyncio.sleep(0.5)

        if not peer_id:
            return {
                "success": False,
                "error": "libp2p host did not come up in time — try again "
                         "in a moment.",
            }

        # Every registration is owner-bound (docs/multi_jurisdiction.md
        # decision 7). With the owner wallet's signature (owner_sig) the agent
        # binds to that wallet + optional parent. Without one, the agent's
        # registered parent signs the binding instead (same household); a root
        # agent with no signature cannot register. The contract verifies the
        # signature on-chain.
        if self_owned and not owner_sig:
            # The daemon's own owner key signs (atn/external_agents.md): a
            # signature is all the contract needs, whoever holds the key.
            try:
                owner_key = self.runtime.registry.local_owner_key()
            except Exception as e:
                return {"success": False, "error": str(e)}
            from eth_account import Account
            owner = Account.from_key(owner_key).address
            parent = ""
            nonce = await self._offload(lambda: svc.binding_nonce(identity.address))
            owner_sig = svc.sign_binding(owner_key, identity.address, "", nonce)
        if owner_sig:
            if not owner:
                return {"success": False,
                        "error": "owner_sig present but 'owner' address missing"}
            # Off-loop: registration waits for its receipt (up to 120 s), which
            # froze every connection and every agent's streaming meanwhile.
            result = await self._offload(lambda: svc.register_agent_bound(
                identity=identity,
                private_key=key,
                peer_id=peer_id,
                owner=owner,
                owner_sig=owner_sig,
                parent=parent,
            ))
        else:
            from .agent_tools import register_parent_signed
            result = await register_parent_signed(
                self.runtime, svc, agent_def, key, peer_id)
        if result.get("success"):
            identity.registered_on_chain = True
            identity.registration_tx = result.get("tx_hash")
            self.runtime.registry.persist_identity(agent_id)
        return result

    def _issue_owner_challenge(self, msg: dict[str, Any]) -> dict[str, Any]:
        """A one-use challenge for one owner-signed action."""
        import secrets
        import time
        action = str(msg.get("action") or "")
        if action not in OWNER_SIGNED_MESSAGES:
            return {"msg_id": msg.get("msg_id"), "ok": False,
                    "error": f"no owner signature is needed for {action!r}"}
        now = time.time()
        self._owner_challenges = {n: v for n, v in self._owner_challenges.items()
                                  if v[1] > now}
        if len(self._owner_challenges) >= 64:
            return {"msg_id": msg.get("msg_id"), "ok": False,
                    "error": "too many open owner challenges; try again shortly"}
        nonce = secrets.token_hex(16)
        self._owner_challenges[nonce] = (action, now + _OWNER_CHALLENGE_TTL_S)
        return {"msg_id": msg.get("msg_id"), "ok": True, "result": {
            "action": action, "nonce": nonce,
            "challenge": owner_challenge_text(action, nonce)}}

    async def _owner_of_record(self) -> str:
        """The household's owner: the wallet the operator is bound to on
        chain, else the configured owner wallet."""
        configured = (getattr(self.runtime._config.autonet, "owner_wallet", "")
                      or self.owner_wallet or "").strip()
        try:
            status = await self._operator_status()
            bound = str(status.get("bound_owner") or "")
            if bound and int(bound, 16):
                return bound
        except Exception:  # noqa: BLE001  (no chain: the configured wallet)
            pass
        return configured

    async def _owner_proof_refusal(self, msg: dict[str, Any], action: str) -> str:
        """Why ``msg`` lacks a valid owner signature for ``action``, or ""."""
        import time
        nonce = str(msg.get("owner_nonce") or "")
        sig = msg.get("owner_sig") or ""
        entry = self._owner_challenges.pop(nonce, None) if nonce else None
        if not entry or not sig:
            return ("this needs the owner's wallet: request owner_challenge "
                    f"with action {action!r}, sign its challenge in the owner "
                    "wallet, and send owner_nonce and owner_sig with the call")
        issued_for, expiry = entry
        if issued_for != action or time.time() > expiry:
            return "the owner challenge expired or was issued for another action"
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            signer = Account.recover_message(
                encode_defunct(text=owner_challenge_text(action, nonce)),
                signature=sig)
        except Exception:  # noqa: BLE001
            return "the owner signature is not valid"
        owner = await self._owner_of_record()
        if not owner:
            if action != "set_owner_wallet":
                return "this daemon has no owner wallet yet"
            # First owner: whoever proves control of the address they set.
            owner = str(msg.get("address") or "")
        if signer.lower() != owner.lower():
            return f"signed by {signer}, which is not this household's owner"
        return ""

    def _operator_key(self) -> str:
        """The key the daemon anchors, votes and claims with: the configured
        daemon key, else the vault-held operator key. External agents sign
        with the same key (AgentRegistry.daemon_operator_key)."""
        configured = getattr(self.runtime.autonet.config, "private_key", "") or ""
        return configured or self.runtime.registry.daemon_operator_key()

    async def _operator_status(self) -> dict[str, Any]:
        """Operator address, whether it is registered and bound to which
        owner, its gas balance, the nonce the owner signs the binding at,
        and whether the owner's REP votes are delegated to it."""
        from eth_account import Account
        from .on_chain import OnChainService
        config = self.runtime.autonet.config
        key = self._operator_key()
        address = Account.from_key(key).address
        out: dict[str, Any] = {
            "operator": address,
            "owner_wallet": getattr(config, "owner_wallet", "") or "",
            "chain_id": getattr(config, "chain_id", 0),
        }
        svc = OnChainService(config)
        if not svc.available:
            out["error"] = "Substrate not configured on this daemon"
            return out
        out["substrate_address"] = svc._substrate_address()

        async def _reads() -> dict[str, Any]:
            w3 = svc._get_web3()
            sub = svc._get_contract(w3)
            addr = w3.to_checksum_address(address)
            r: dict[str, Any] = {
                "registered": int(sub.functions.agents(addr).call()[0]) != 0,
                "live": bool(sub.functions.isLive(addr).call()),
                "bound_owner": sub.functions.agentOwner(addr).call(),
                "binding_nonce": int(sub.functions.bindingNonce(addr).call()),
                "gas_balance_wei": int(w3.eth.get_balance(addr)),
                # External agents stake and bond from it (Decision 2026-11-08).
                "atn_balance": int(sub.functions.balanceOf(addr).call()),
            }
            # How many claim/vote transactions the balance still covers at
            # today's gas price (a claim costs well under this per-tx
            # budget), so the app warns relative to the chain, not a sum.
            gas_price = int(w3.eth.gas_price or 0)
            r["txs_affordable"] = (r["gas_balance_wei"] // (gas_price * _OPERATOR_TX_GAS)
                                   if gas_price else None)
            rep_addr = getattr(config, "rep_token_address", "") or ""
            owner = r["bound_owner"] if int(r["bound_owner"], 16) else out["owner_wallet"]
            if rep_addr and owner:
                rep = w3.eth.contract(address=w3.to_checksum_address(rep_addr),
                                      abi=_REP_DELEGATION_ABI)
                owner_cs = w3.to_checksum_address(owner)
                delegatee = rep.functions.delegates(owner_cs).call()
                r["owner_delegatee"] = delegatee
                r["delegated_to_operator"] = delegatee.lower() == address.lower()
                # What the owner signs for delegateBySig (EIP-712 domain of
                # the token: ERC20Permit(name), version "1").
                r["rep_token_address"] = rep.address
                r["rep_token_name"] = rep.functions.name().call()
                r["owner_rep_nonce"] = int(rep.functions.nonces(owner_cs).call())
            return r

        out.update(await self._offload(_reads))
        # Every external agent acts as the operator: its registration is theirs.
        from .provider_identity import is_external
        out["external_agents"] = sorted(
            d.id for d, _s in self.runtime.list_agents() if is_external(d))
        try:
            self.runtime.registry.set_operator_registered(bool(out.get("registered")))
        except Exception as exc:  # noqa: BLE001
            log.debug("operator flag not synced: %s", exc)
        return out

    async def _submit_delegation(self, nonce: int, expiry: int,
                                 signature: str) -> dict[str, Any]:
        """Submit the owner's delegateBySig (delegatee = the operator)."""
        from eth_account import Account
        from .on_chain import OnChainService
        config = self.runtime.autonet.config
        rep_addr = getattr(config, "rep_token_address", "") or ""
        svc = OnChainService(config)
        if not svc.available or not rep_addr:
            return {"success": False, "error": "RepToken not configured on this daemon"}
        sig = bytes.fromhex(signature.removeprefix("0x"))
        if len(sig) != 65:
            return {"success": False, "error": "signature must be 65 bytes"}
        r, s_, v = sig[:32], sig[32:64], sig[64]
        if v < 27:
            v += 27
        op_key = self._operator_key()
        op = Account.from_key(op_key)

        async def _send() -> dict[str, Any]:
            w3 = svc._get_web3()
            rep = w3.eth.contract(address=w3.to_checksum_address(rep_addr),
                                  abi=_REP_DELEGATION_ABI)
            call = rep.functions.delegateBySig(op.address, nonce, expiry, v, r, s_)
            gas = int(call.estimate_gas({"from": op.address}) * 12 // 10)
            tx = call.build_transaction({
                "from": op.address, "gas": gas, "gasPrice": int(w3.eth.gas_price),
                "chainId": int(w3.eth.chain_id),
                "nonce": next_nonce(w3, op.address)})
            signed = op.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            h = w3.eth.send_raw_transaction(raw)
            if w3.eth.wait_for_transaction_receipt(h, timeout=120).status != 1:
                return {"success": False, "error": "delegateBySig reverted",
                        "tx_hash": h.hex()}
            return {"success": True, "tx_hash": h.hex(), "delegatee": op.address}

        return await self._offload(lambda: _send())

    async def _sweep_to_owner(self, agent_ids: Any = None) -> dict[str, Any]:
        """Move local agents' ATN to their on-chain owner (the configured
        owner wallet when unbound). The owner asks for it explicitly; the
        operator first sends each agent just enough gas for the transfer,
        since ATN has no permit. Substrate never moves agent funds itself."""
        from eth_account import Account
        from .on_chain import OnChainService
        config = self.runtime.autonet.config
        svc = OnChainService(config)
        if not svc.available:
            return {"swept": [], "error": "Substrate not configured on this daemon"}
        op_key = self._operator_key()
        wanted = set(agent_ids) if isinstance(agent_ids, list) and agent_ids else None
        agents = []
        operator_rows: list[dict[str, Any]] = []
        for defn, _status in self.runtime.list_agents():
            ident = getattr(defn, "identity", None)
            addr = getattr(ident, "address", "") or ""
            key = self.runtime.registry.get_agent_key(defn.id)
            if not (addr and key and (wanted is None or defn.id in wanted)):
                continue
            if self.runtime.registry.is_operator_identity(ident):
                # An external agent's balance is the operator's, which pays
                # the household's stakes, bonds and gas: not swept.
                operator_rows.append({"agent_id": defn.id, "address": addr,
                                      "skipped": "acts as the operator; its "
                                                 "balance funds stakes and gas"})
                continue
            agents.append((defn.id, addr, key))
        fallback_owner = getattr(config, "owner_wallet", "") or ""

        async def _sweep() -> list[dict[str, Any]]:
            w3 = svc._get_web3()
            sub = svc._get_contract(w3)
            transfer_abi = [{"name": "transfer", "type": "function",
                             "stateMutability": "nonpayable",
                             "inputs": [{"name": "to", "type": "address"},
                                        {"name": "amount", "type": "uint256"}],
                             "outputs": [{"name": "", "type": "bool"}]}]
            atn = w3.eth.contract(address=sub.address, abi=transfer_abi)
            chain_id = int(w3.eth.chain_id)
            op = Account.from_key(op_key)

            def _send(tx: dict, key: str) -> str:
                signed = Account.from_key(key).sign_transaction(tx)
                raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
                h = w3.eth.send_raw_transaction(raw)
                rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=120)
                if rcpt.status != 1:
                    raise RuntimeError(f"tx reverted: {h.hex()}")
                return h.hex()

            out: list[dict[str, Any]] = []
            for agent_id, addr, key in agents:
                a = w3.to_checksum_address(addr)
                row: dict[str, Any] = {"agent_id": agent_id, "address": a}
                try:
                    bal = int(sub.functions.balanceOf(a).call())
                    if bal == 0:
                        row["skipped"] = "no ATN"
                        out.append(row)
                        continue
                    owner = sub.functions.agentOwner(a).call()
                    if not int(owner, 16):
                        owner = fallback_owner
                    if not owner:
                        row["error"] = "agent has no owner to sweep to"
                        out.append(row)
                        continue
                    owner = w3.to_checksum_address(owner)
                    gas_price = int(w3.eth.gas_price)
                    call = atn.functions.transfer(owner, bal)
                    gas = int(call.estimate_gas({"from": a}) * 12 // 10)
                    need = gas * gas_price - int(w3.eth.get_balance(a))
                    if need > 0:
                        row["gas_tx"] = _send({
                            "to": a, "value": need, "gas": 21_000,
                            "gasPrice": gas_price, "chainId": chain_id,
                            "nonce": next_nonce(w3, op.address)}, op_key)
                    row["tx"] = _send(call.build_transaction({
                        "from": a, "gas": gas, "gasPrice": gas_price,
                        "chainId": chain_id,
                        "nonce": next_nonce(w3, a)}), key)
                    row.update(amount=bal, owner=owner)
                except Exception as e:
                    row["error"] = str(e)
                out.append(row)
            return out

        return {"swept": await self._offload(lambda: _sweep()) + operator_rows}

    async def _handle_register_operator(self, owner: str = "", owner_sig: str = "",
                                        self_owned: bool = False) -> dict[str, Any]:
        """Register the operator key on Substrate, bound to the owner.

        The owner signs the EIP-712 OwnerBinding for the operator address
        (the same popup as binding an agent); ``self_owned`` signs with the
        daemon's own owner key instead (server setups). The operator pays
        the registration gas, so it must be funded first.
        """
        from eth_account import Account
        from .models import AgentIdentity
        from .on_chain import OnChainService
        self.runtime.autonet.ensure_network_config()
        config = self.runtime.autonet.config
        svc = OnChainService(config)
        if not svc.available:
            return {"success": False,
                    "error": "On-chain not configured (missing substrate_address or rpc_url)"}
        key = self._operator_key()
        address = Account.from_key(key).address
        try:
            w3 = svc._get_web3()
            if await asyncio.to_thread(
                    w3.eth.get_balance, w3.to_checksum_address(address)) == 0:
                return {"success": False, "mode": "needs_funding",
                        "error": "Operator address has no gas balance; fund it first.",
                        "operator": address, "chain_id": config.chain_id}
        except Exception as e:
            return {"success": False, "error": f"Could not check operator balance: {e}"}
        autonet = getattr(self.runtime, "autonet", None)
        peer_id = self._read_libp2p_peer_id(autonet) if autonet is not None else ""
        if not peer_id:
            return {"success": False,
                    "error": "libp2p host is not up yet; start autonet and retry."}
        if self_owned and not owner_sig:
            owner_key = self.runtime.registry.local_owner_key()
            owner = Account.from_key(owner_key).address
            nonce = await self._offload(lambda: svc.binding_nonce(address))
            owner_sig = svc.sign_binding(owner_key, address, "", nonce)
        if not owner or not owner_sig:
            return {"success": False,
                    "error": "registering the operator needs the owner's binding "
                             "signature (owner + owner_sig) or self_owned"}
        result = await self._offload(lambda: svc.register_agent_bound(
            identity=AgentIdentity(address=address),
            private_key=key,
            peer_id=peer_id,
            owner=owner,
            owner_sig=owner_sig,
            parent="",
        ))
        if result.get("success"):
            result["operator"] = address
            # Every external agent acts as the operator: now registered too.
            try:
                self.runtime.registry.set_operator_registered(
                    True, result.get("tx_hash"))
            except Exception as exc:  # noqa: BLE001
                log.debug("operator flag not recorded: %s", exc)
        return result

    def _imported_sessions(self) -> dict[str, str]:
        """session id -> the agent that imported it."""
        out: dict[str, str] = {}
        for defn, _status in self.runtime.list_agents():
            cs = getattr(defn, "claude_session", None)
            if cs and cs.get("session_id"):
                out[cs["session_id"]] = defn.id
        return out

    async def _import_claude_session(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Make an agent that IS a Claude Code session. Refused while a
        Claude process may hold the session, or if it is imported already.
        The model is the one the session last ran on unless one is given."""
        import re as _re
        from pathlib import Path
        from .claude_sessions import import_check
        from .loader import save_agent
        from .models import AgentDefinition, AgentMode

        folder = str(msg.get("folder") or "").strip()
        sid = str(msg.get("session_id") or "").strip().lower()
        if not folder or not sid:
            return {"ok": False, "error": "Missing 'folder' or 'session_id'"}
        owner = self._imported_sessions().get(sid)
        if owner:
            return {"ok": False, "error": f"session already imported as '{owner}'"}
        info, why = import_check(folder, sid)
        if why:
            return {"ok": False, "error": why}
        model = str(msg.get("model") or info.get("model") or "")
        if not model:
            return {"ok": False, "error": "the session names no model; pass 'model'"}
        name = str(msg.get("name") or info.get("title") or Path(folder).name)
        agent_id = str(msg.get("id") or "").strip() or (
            _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "session")
        base, n = agent_id, 2
        while self.runtime.get_agent(agent_id) is not None:
            agent_id, n = f"{base}-{n}", n + 1
        defn = AgentDefinition(
            id=agent_id, name=name, mode=AgentMode.COGNITIVE,
            provider="claude_max", cognitive_model=model,
            # Claude Code's own tools come with the session; ATN adds the
            # tool library on top.
            tools=list(msg.get("tools") or ["unified_tools"]),
            description=info.get("first_prompt", "")[:200],
            claude_session={"folder": str(Path(folder).resolve()), "session_id": sid},
        )
        try:
            await self.runtime.register_agent(defn)
            save_agent(defn, self.runtime._config.agents_dir)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        # Show the session's conversation so far, as Claude Code does. Display
        # only: the run resumes Claude Code's own session, never this store.
        try:
            from .claude_sessions import project_dir, transcript_turns
            store = self.runtime.get_agent_conversation_store(agent_id)
            if store.turn_count() == 0:
                for role, text in transcript_turns(project_dir(folder) / f"{sid}.jsonl"):
                    if role == "user":
                        store.add_user_turn(text)
                    else:
                        store.add_assistant_turn(text)
        except Exception:
            log.warning("could not load the history of session %s", sid, exc_info=True)
        return {"ok": True, "result": {
            "agent_id": agent_id, "name": name, "model": model,
            "claude_session": defn.claude_session}}

    def _jurisdictions_view(self) -> dict[str, Any]:
        """The current jurisdiction plus every jurisdiction this daemon's
        agents are registered in (their substrate data is parked locally).
        The frontend merges this with the DAO directory it reads itself."""
        from .config import load_selected_jurisdiction
        current = self.runtime.jurisdiction
        st = self.runtime.autonet.state
        known: dict[str, dict[str, Any]] = {}
        for defn, _status in self.runtime.registry.list_agents():
            ident = getattr(defn, "identity", None)
            for key in (getattr(ident, "registrations", None) or {}):
                row = known.setdefault(key, {"key": key, "agents": []})
                row["agents"].append(defn.id)
        from .config import NO_JURISDICTION_MESSAGE, packaged_jurisdiction
        sel = load_selected_jurisdiction(getattr(self.runtime._config, "data_dir", None))
        configured = bool(st.jurisdiction_configured)
        return {
            # False: no jurisdiction chosen yet (issue #50); the app offers
            # ``packaged`` as the one-click choice.
            "configured": configured,
            "message": "" if configured else NO_JURISDICTION_MESSAGE,
            "packaged": packaged_jurisdiction(),
            "current": {
                "key": current,
                "selection": sel,          # None = no jurisdiction
                "dao_registry": st.dao_registry_address,
                "dao": st.dao_address,
                "chain_id": st.chain_id,
                "substrate": st.substrate_address,
                "errors": list(st.jurisdiction_errors),
            },
            "registered_in": sorted(known.values(), key=lambda r: r["key"]),
        }

    async def _handle_check_registration(self, agent_id: str) -> dict[str, Any]:
        """Check if an agent is registered on-chain."""
        from .on_chain import OnChainService

        config = self.runtime.autonet.config
        svc = OnChainService(config)

        if not svc.available:
            return {"registered": False, "available": False}

        agent_def = self.runtime.registry.get_agent(agent_id)
        if not agent_def or not agent_def.identity:
            return {"registered": False, "error": "Agent has no identity"}

        # Phase 12: agents are always identified by their own
        # daemon-held keypair (root agents included). Check that
        # one address only — never fall back to the user's wallet.
        registered = False
        if agent_def.identity.address:
            registered = await svc.is_registered(agent_def.identity.address)

        result: dict[str, Any] = {
            "registered": registered,
            "available": True,
            "agent_address": agent_def.identity.address,
        }

        # Sync daemon's in-memory flag with on-chain truth so a stale
        # local flag (e.g. after a contract redeploy) gets reconciled.
        if registered != agent_def.identity.registered_on_chain:
            agent_def.identity.registered_on_chain = registered
            self.runtime.registry.persist_identity(agent_id)
            log.info("Agent %s on-chain status synced to %s (address: %s)",
                     agent_id, registered, agent_def.identity.address)

        if registered:
            # Detail enrichment only — a read failure must not fail the
            # registration check itself (get_agent_record now raises).
            try:
                record = await svc.get_agent_record(agent_def.identity.address)
            except Exception as exc:                       # noqa: BLE001
                log.debug("check_registration record read failed: %s", exc)
                record = None
            if record:
                result["record"] = record

        return result

    # ------------------------------------------------------------------
    # Event broadcasting
    # ------------------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        """Broadcast an EventBus event to connected clients, scoped per session.

        A scoped session only receives events originating inside its subtree.
        Subtree membership is re-derived at send time against the LIVE subtree
        (get_subtree_ids) rather than a cached set, so a child spawned after the
        connection authed still has its events delivered (no stale window)."""
        if not self._sessions:
            return

        payload = json.dumps({
            "type": "event",
            "event_type": event.type.value,
            "source": event.source,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
        }, default=str)

        origin = self._event_origin(event)
        for ws, session in list(self._sessions.items()):
            # Unauthed sessions get nothing until they complete the handshake.
            if not session.authed:
                continue
            # Scoped session: deliver only events from inside its subtree.
            if session.scope_ids is not None:
                live = self.runtime.registry.get_subtree_ids(session.root_agent_id)
                if origin is not None and origin not in live:
                    continue
            queue = session.event_queue
            if queue is None:
                continue
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow consumer: drop ITS event rather than stall the bus.
                session.dropped_events += 1
                if session.dropped_events in (1, 100, 1000):
                    log.warning(
                        "WS client queue full (%d dropped so far) — slow consumer",
                        session.dropped_events,
                    )

    @staticmethod
    def _event_origin(event: Event) -> str | None:
        """The agent id an event is 'about', for subtree scoping. Prefers an
        explicit agent_id in the payload (delegate/execution events carry the
        real subject there; event.source is often 'runtime'), else source."""
        data = event.data or {}
        for key in ("agent_id", "source", "delegate_id"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        return event.source or None




# Old pidfile-based lock removed — see atn/lock_manager.py for the
# OS-level file-locking singleton used by cli.py and run_standalone().


# ---------------------------------------------------------------------------
# Stdin command reader (background thread)
# ---------------------------------------------------------------------------

_CMD_RESTART = "restart"
_CMD_QUIT = "quit"


def _stdin_reader(loop: asyncio.AbstractEventLoop, cmd_event: asyncio.Event, cmd_box: list[str]) -> None:
    """Read stdin in a background thread.  Sets cmd_event when a command arrives.

    Runs as a daemon thread so it dies when the process exits.
    """
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, OSError):
            break
        if not line:
            break  # stdin closed (e.g. piped input exhausted)

        cmd = line.strip().lower()
        if cmd in ("r", "restart"):
            cmd_box.clear()
            cmd_box.append(_CMD_RESTART)
            loop.call_soon_threadsafe(cmd_event.set)
        elif cmd in ("q", "quit", "exit"):
            cmd_box.clear()
            cmd_box.append(_CMD_QUIT)
            loop.call_soon_threadsafe(cmd_event.set)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _init_and_serve(
    host: str, port: int,
) -> tuple["Runtime", "WebSocketBridge"]:
    """Load config, create Runtime + WebSocketBridge, start both."""
    from .config import load_config
    from .loader import load_agents_dir

    config = load_config()

    bus = EventBus()
    rt = Runtime(bus, data_dir=config.data_dir, config=config)
    await rt.start()

    # Load agents
    if config.agents_dir.exists():
        agents, errors = load_agents_dir(config.agents_dir)
        # One-time pinning of pre-2026-10-06 agents (see cli._load_agents).
        from .legacy_routing_migration import (
            pin_unpinned_agents_from_legacy_defaults,
        )
        pin_unpinned_agents_from_legacy_defaults(agents, config, rt.providers)
        for defn in agents:
            await rt.register_agent(defn)
            if defn.schedule or defn.heartbeat:
                await rt.activate_agent(defn.id)
        # Heal parent pointers naming an agent that is no longer on disk (see
        # AgentRegistry.reconcile_dangling_parents).
        from .loader import save_agent
        for healed_id in rt.registry.reconcile_dangling_parents():
            healed = rt.get_agent(healed_id)
            if healed is None:
                continue
            try:
                save_agent(healed, config.agents_dir)
            except Exception:
                log.warning("Promoted orphan %s but YAML save failed",
                            healed_id, exc_info=True)
            log.info("Promoted orphaned agent to top level: %s", healed_id)
        # Note: execution history is hydrated automatically in register_agent()
        if agents:
            log.info("Loaded %d agent(s)", len(agents))

    # Phase 12: auto-start autonet for already-registered agents.
    try:
        if await rt.maybe_autostart_autonet_for_registered_agents():
            log.info("Autonet auto-started for registered agent(s)")
    except Exception as exc:
        log.warning("Could not auto-start autonet: %s", exc)

    bridge = WebSocketBridge(rt, host=host, port=port)
    await bridge.start()
    return rt, bridge


async def run_standalone(host: str = "localhost", port: int = DEFAULT_PORT) -> None:
    """Start ATN with WebSocket server (standalone mode).

    Supports interactive commands on stdin:
      r / restart  — reload config + agents, restart the server
      q / quit     — clean shutdown
    """
    from .config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()

    # Singleton lock — only one ws_server per data_dir
    from .lock_manager import LockManager
    lock = LockManager()
    lock.set_data_dir(config.data_dir)
    existing = lock.is_daemon_running()
    if existing:
        log.error(
            "ATN daemon is already running (PID %s, started %s)",
            existing["pid"], existing.get("started_at", "?"),
        )
        sys.exit(1)
    if not lock.acquire_lock():
        log.error("Failed to acquire daemon lock")
        sys.exit(1)

    # Start stdin reader thread
    loop = asyncio.get_running_loop()
    cmd_event = asyncio.Event()
    cmd_box: list[str] = []  # holds the latest command string

    if sys.stdin and sys.stdin.readable():
        reader = threading.Thread(
            target=_stdin_reader, args=(loop, cmd_event, cmd_box), daemon=True,
        )
        reader.start()

    rt: Runtime | None = None
    bridge: WebSocketBridge | None = None

    try:
        rt, bridge = await _init_and_serve(host, port)
        log.info("ATN running — ws://%s:%d  (r=restart, q=quit)", host, port)

        while True:
            cmd_event.clear()
            await cmd_event.wait()

            cmd = cmd_box[-1] if cmd_box else ""

            if cmd == _CMD_QUIT:
                log.info("Quit requested")
                break

            if cmd == _CMD_RESTART:
                log.info("Restarting...")
                await bridge.stop()
                await rt.stop()
                rt, bridge = await _init_and_serve(host, port)
                log.info("ATN restarted — ws://%s:%d  (r=restart, q=quit)", host, port)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if bridge:
            await bridge.stop()
        if rt:
            await rt.stop()
        lock.release_lock()


def main() -> None:
    """Entry point for python -m atn.ws_server."""
    asyncio.run(run_standalone())


if __name__ == "__main__":
    main()
