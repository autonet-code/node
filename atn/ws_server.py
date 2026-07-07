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
from .events import Event, EventBus, EventType
from .orchestrator import ORCHESTRATOR_ID
from .orchestrator.tools import execute_tool
from .runtime import Runtime
from .runtime.provider_manager import get_model_tier, get_tier_label
from .ws_auth import ClientSession

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
KEY_LOCAL_ONLY_MESSAGES = frozenset({
    "export_agent_key",
    "autonet_publish_standards",
    "autonet_claim_reward",
    "register_agent_on_chain",
    # Vault mutations carry a raw secret VALUE in the payload; over a
    # proxied/remote link the WS hop is plain ws://, so it would cross the
    # wire in plaintext. Reads (secrets_status/usage/alarms) are names-only
    # and stay owner-gated but remotely reachable.
    "secrets_put",
    "secrets_delete",
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
})

# Tools an authed-but-SCOPED (non-full-fleet) session may NOT call: they read
# or mutate owner-global state (budgets, profile/PII, providers, governance).
# A full-fleet owner session (local, or remote-owner-at-orchestrator) is
# unaffected. Default-deny still applies on top: see _authorize_tool_call.
OWNER_ONLY_TOOLS = frozenset({
    "set_credit_budget", "get_credit_budget", "get_user_profile",
    "set_orchestrator_model", "set_agent_model",
})

# For scoped sessions, the message keys that name a target agent. Every one
# present must resolve to an agent inside the session's subtree, else the call
# is refused — clamping caller_id alone is insufficient because these tools read
# the target straight from the message and ignore caller_id.
_TARGET_ARG_KEYS = ("agent_id", "target", "parent_id", "id")


from dataclasses import dataclass


@dataclass(frozen=True)
class WSAuthor:
    """Platform-neutral author for a WS-originated inbound message.

    Adapts a ClientSession into the `author` an InputPolicy.evaluate() expects
    (exposes at least ``id`` and ``is_bot``). The extra fields (local / owner /
    conn_id / root_agent_id) are what a future InputArbiter needs to build a
    SurfaceId and reason about ownership — threaded now so P3 can gate without
    re-plumbing. A WS client is always a human driver, never a bot."""

    id: str                          # the session's root agent id (its identity)
    conn_id: str                     # server-random per-connection id (SurfaceId.instance)
    local: bool = False              # arrived on the privileged loopback listener
    owner: bool = False              # authed as the daemon owner
    root_agent_id: str = ORCHESTRATOR_ID
    is_bot: bool = False             # a WS client is a human driver, never a bot

    @classmethod
    def from_session(cls, session: ClientSession) -> "WSAuthor":
        return cls(
            id=session.root_agent_id,
            conn_id=session.conn_id,
            local=session.local,
            owner=session.owner,
            root_agent_id=session.root_agent_id,
        )


class WebSocketBridge:
    """Bridges the ATN Runtime to WebSocket clients.

    Handles:
      - Routing incoming JSON messages to orchestrator tools
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

        # Privileged local listener — full control, key export, no handshake.
        self._server = await websockets.serve(
            lambda ws: self._handle_client(ws, local=True),
            self.host,
            self.port,
        )
        log.info("WebSocket server (local) listening on ws://%s:%d", self.host, self.port)

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
        """Single-writer gate for the WS sibling input paths (delegate_message,
        orchestrator_message) that bypass send_agent_message. Returns None if
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
        rooted at the orchestrator (today's behavior). A remote session starts
        unauthed and is issued an auth_challenge instead of a snapshot."""
        remote = ws.remote_address
        session = ClientSession(
            local=local,
            is_loopback=ws_auth.is_loopback(remote),
            conn_id=ws_auth.new_nonce(),
        )
        if local:
            # Privileged local listener: full control, no handshake.
            session.authed = True
            session.owner = True
            session.root_agent_id = ORCHESTRATOR_ID
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

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"ok": False, "error": "Invalid JSON"}))
                    continue

                response = await self._handle_message(msg, session)
                await ws.send(json.dumps(response, default=str))

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
        """The orchestrator's identity address — the per-daemon unique id that
        domain-separates the auth challenge (so a signature can't replay across
        daemons). Falls back to the configured owner wallet, then a constant."""
        try:
            orch = self.runtime.registry._agents.get(ORCHESTRATOR_ID)
            if orch and orch.identity and orch.identity.address:
                return orch.identity.address
        except Exception:
            pass
        return self.owner_wallet or "autonet-daemon"

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

        if is_owner:
            # OWNER: full control. May root at the orchestrator (full fleet) or
            # name ANY agent subtree to render.
            root = (msg.get("root") or ORCHESTRATOR_ID).strip()
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
        session.scope_ids = (None if resolved == ORCHESTRATOR_ID
                             else self.runtime.registry.get_subtree_ids(resolved))
        log.info("Remote auth: signer=%s owner=%s root=%s scoped=%s",
                 signer, session.owner, resolved, session.scope_ids is not None)
        # Refresh the input-surface label now that the root agent is known
        # (re-register is idempotent on the conn_id token).
        self._register_input_surface(session)
        return {"msg_id": msg_id, "type": "auth_ok", "ok": True,
                "owner": session.owner, "root": resolved, "wallet": signer}

    def _resolve_root(self, root: str) -> tuple[str, str | None]:
        """Map a client-named root (agent_id, address, or 'orchestrator') to a
        canonical agent_id. Returns (agent_id, error). Ambiguous address or
        unknown root is refused."""
        if not root or root == ORCHESTRATOR_ID:
            return ORCHESTRATOR_ID, None
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
        full-fleet session (local owner, or remote owner rooted at the
        orchestrator) keeps them."""
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

        # --- Gate 1: key custody (localhost-only) --------------------------
        if msg_type in KEY_LOCAL_ONLY_MESSAGES and not ws_auth.assert_local_key_access(session):
            return {"msg_id": msg_id, "ok": False,
                    "error": "Key operations are localhost-only and were "
                             "refused (connection is not on the local listener)."}

        # --- Auth handshake messages (the only ones allowed pre-auth) ------
        if msg_type == "auth_response":
            return await self._handle_auth_response(msg, session)
        if msg_type == "auth_status":
            return {"msg_id": msg_id, "ok": True, "result": {
                "authed": session.authed, "owner": session.owner,
                "root": session.root_agent_id,
                "requires_auth": not session.local,
            }}

        # --- Gate 2: everything else requires an authed session ------------
        if not session.authed:
            return {"msg_id": msg_id, "ok": False, "error": "unauthenticated"}

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

        # Model selection: change orchestrator model and start new conversation
        if msg_type == "set_orchestrator_model":
            model = msg.get("model", "")
            if not model:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'model' field"}
            try:
                await self.runtime.set_orchestrator_model(model)
                tier = get_model_tier(model)
                return {"msg_id": msg_id, "ok": True, "result": {
                    "model": model,
                    "status": "Model changed",
                    "capability_tier": tier,
                    "tier_label": get_tier_label(tier),
                }}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

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

        if msg_type == "provider_refresh_usage":
            provider_id = msg.get("provider_id", "claude_max")
            # _active_providers is keyed by agent_id, not provider_id. Find any
            # active provider instance whose .name matches the requested provider.
            prov = None
            for candidate in self.runtime.providers._active_providers.values():
                if getattr(candidate, "name", "") == provider_id and hasattr(candidate, "refresh_usage"):
                    prov = candidate
                    break
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
            name = msg.get("name", provider_id)
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
            result = await self.runtime.send_agent_message(
                agent_id, content, surface=self._surface_id_for(session))
            if result.get("error"):
                return {"msg_id": msg_id, "ok": False, "error": result["error"],
                        "code": result.get("code"), "holder": result.get("holder")}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # Interrupt — gracefully stop a running LLM session mid-turn
        if msg_type == "interrupt_orchestrator":
            sent = await self.runtime.interrupt_orchestrator()
            if not sent:
                return {"msg_id": msg_id, "ok": False, "error": "Orchestrator is not running"}
            return {"msg_id": msg_id, "ok": True, "result": {"status": "interrupted"}}

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
            agent_id = msg.get("agent_id")  # None = orchestrator
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
            agent_id = msg.get("agent_id")  # None = orchestrator
            result = await self.runtime.get_session_context(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        if msg_type == "context_breakdown":
            agent_id = msg.get("agent_id")  # None = orchestrator
            result = await self.runtime.get_context_breakdown(agent_id)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

        # The canonical agent tool surface — everything an agent can be
        # granted, grouped by the same bundle ids the create-agent flow
        # uses, with each tool's endpoints (name / description / params).
        if msg_type == "tool_surface":
            from .orchestrator.tools import _TOOL_CATEGORIES, _TOOLS
            from .shell_tools import SHELL_TOOLS

            def _entry(name, description, schema):
                props = list(((schema or {}).get("properties")) or {})
                return {"name": name,
                        "description": (description or "").strip().split("\n")[0][:220],
                        "params": props}

            tool_index = {t.name: t for t in _TOOLS}
            bundles = []
            for cat in sorted(_TOOL_CATEGORIES):
                if cat == "shell":
                    tools = [_entry(t["name"], t.get("description", ""),
                                    t.get("input_schema")) for t in SHELL_TOOLS]
                else:
                    tools = [
                        _entry(n, tool_index[n].description, tool_index[n].input_schema)
                        for n in sorted(_TOOL_CATEGORIES[cat]) if n in tool_index
                    ]
                bundles.append({"id": cat, "tools": tools})
            # Bridge-native (Claude SDK) built-ins — granted via "sdk_builtin".
            # Schemas live in the SDK; names are the stable contract.
            bundles.append({"id": "sdk_builtin", "tools": [
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
            tools = []
            for record in store.visible_to(None):  # owner sees everything
                m = record.manifest
                tools.append({
                    "digest": record.digest,
                    "name": record.name,
                    "description": m.get("description", ""),
                    "trust_class": m.get("trust_class", ""),
                    "author": m.get("author", ""),
                    "fee_atn": m.get("fee_atn", 0),
                    "grants": sorted(record.grants),
                    "enabled": record.enabled,
                    "published": record.published,
                    "origin": record.origin,
                    "capabilities": m.get("capabilities", {}),
                    "version_of": m.get("version_of"),
                    "dependencies": m.get("dependencies") or [],
                    "input_schema": m.get("input_schema", {}),
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
            # settlement lands once payForService vs payForInference is
            # decided (docs/tool_substrate.md open knobs).
            return {"msg_id": msg_id, "ok": True,
                    "result": self.runtime.tool_store.balances()}

        if msg_type == "set_tool_published":
            # Owner surface: flip publish state directly (agents use the
            # separately-granted publish_tool capability).
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            published = bool(msg.get("published", True))
            if not self.runtime.tool_store.set_published(digest, published):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown tool digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "published": published}}

        # Adoption rail (docs/tool_substrate.md — Adoption): the approval
        # queue is HUMAN-ONLY by construction, like grants. Agents can
        # only propose (adopt_tool, case-by-case granted); installing
        # foreign code is the owner's call, always per-tool.
        if msg_type == "list_adoption_proposals":
            status = msg.get("status") or None
            rows = self.runtime.tool_store.list_adoption_proposals(status)
            return {"msg_id": msg_id, "ok": True,
                    "result": {"proposals": rows}}

        if msg_type == "approve_adoption":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            result = await self.runtime.tool_store.approve_adoption(digest)
            if "error" in result:
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
            return {"msg_id": msg_id, "ok": True, "result": result}

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
            # The inference probe as the Tools screen's search: semantic
            # retrieval over manifests (embedding + standing + coverage)
            # when the substrate is up; graceful degradation to a plain
            # substring match over the local store when it isn't.
            query = str(msg.get("query") or "").strip()
            k = int(msg.get("k") or 12)
            if not query:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'query' field"}
            matches = []
            source = "local"
            try:
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None
                if world_service is not None:
                    from nodes.common.world_model_substrate.tool_manifest import (
                        is_tool_manifest,
                    )
                    result = world_service.infer_artifacts(query, k=max(k * 3, 12))
                    for art in result.get("artifacts", []):
                        payload = art.get("payload")
                        if not is_tool_manifest(payload):
                            continue
                        matches.append({
                            "digest": art.get("digest", ""),
                            "name": payload.get("name", ""),
                            "description": payload.get("description", ""),
                            "author": payload.get("author", ""),
                            "trust_class": payload.get("trust_class", ""),
                            "score": art.get("final", 0.0),
                            # v3: review-drifted rating + charter head
                            # replace debate standing.
                            "rating": art.get("rating", 0.0),
                            "axes": art.get("axes", []),
                        })
                        if len(matches) >= k:
                            break
                    source = "substrate"
            except Exception as exc:
                log.debug("probe_tools substrate path failed: %s", exc)
            if not matches:
                needle = query.lower()
                for record in self.runtime.tool_store.visible_to(None):
                    hay = f"{record.name} {record.manifest.get('description', '')}".lower()
                    if all(w in hay for w in needle.split()):
                        matches.append({
                            "digest": record.digest,
                            "name": record.name,
                            "description": record.manifest.get("description", ""),
                            "author": record.author,
                            "trust_class": record.trust_class,
                            "score": 0.0,
                            "rating": 0.0,
                            "axes": [],
                        })
                        if len(matches) >= k:
                            break
                source = "local"
            return {"msg_id": msg_id, "ok": True,
                    "result": {"matches": matches, "source": source}}

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
            ask = msg.get("ask") or {}
            author = msg.get("agent_id") or "user"
            backing_tool = msg.get("backing_tool", "")
            if not name:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'name' field"}
            try:
                result = self.runtime.service_store.register(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    author=author,
                    ask=ask,
                    backing_tool=backing_tool,
                    output_schema=msg.get("output_schema"),
                    endpoint_hint=msg.get("endpoint_hint", ""),
                    version_of=msg.get("version_of"),
                )
            except (ValueError, RuntimeError) as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": result["digest"],
                               "spec": result["spec"]}}

        if msg_type == "list_services":
            include_retired = bool(msg.get("include_retired", False))
            services = []
            for record in self.runtime.service_store.list(
                    include_retired=include_retired):
                s = record.spec
                services.append({
                    "digest": record.digest,
                    "name": record.name,
                    "description": s.get("description", ""),
                    "author": record.author,
                    "ask": record.ask,
                    "endpoint_hint": s.get("endpoint_hint", ""),
                    "backing_tool": record.backing_tool,
                    "retired": record.retired,
                    "version_of": s.get("version_of"),
                    "input_schema": s.get("input_schema", {}),
                })
            summary = self.runtime.service_store.summary()
            return {"msg_id": msg_id, "ok": True,
                    "result": {"services": services, "summary": summary}}

        if msg_type == "retire_service":
            digest = msg.get("digest", "")
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            if not self.runtime.service_store.retire(digest):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"Unknown service digest: {digest[:16]}"}
            return {"msg_id": msg_id, "ok": True,
                    "result": {"digest": digest, "retired": True}}

        if msg_type == "service_request":
            return await self._handle_service_request(msg, msg_id)

        # Voice service control
        if msg_type == "voice_start":
            result = await self.runtime.start_voice()
            return {"msg_id": msg_id, "ok": result.get("status") != "failed", "result": result}

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
            agent_id = msg.get("agent_id", "orchestrator")
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
            agent_id = msg.get("agent_id", "orchestrator")
            if self.runtime.voice:
                self.runtime.voice.set_voice_focus(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"voice_focus": agent_id}}
            return {"msg_id": msg_id, "ok": False, "error": "Voice service not running"}

        if msg_type == "voice_set_tools_focus":
            agent_id = msg.get("agent_id", "orchestrator")
            if self.runtime.voice:
                self.runtime.voice.set_tools_focus(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"tools_focus": agent_id}}
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
            from .orchestrator import ORCHESTRATOR_ID as _ORCH_ID
            p = self.runtime.user_profile.get_profile()
            # Goals are now agents — build from registry
            agent_goals = []
            for defn, status in self.runtime.list_agents():
                if defn.id == _ORCH_ID:
                    continue
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
            # Lazily load constitution CID from chain on first status request
            await self.runtime.autonet.load_constitution()
            return {"msg_id": msg_id, "ok": True, "result": self.runtime.autonet.get_status()}

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

        # Standards publication (Story 3.2)
        if msg_type == "autonet_standards":
            try:
                result = self.runtime.autonet.get_standards(
                    user_profile=self.runtime.user_profile,
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "autonet_publish_standards":
            private_key = msg.get("private_key", "")
            if not private_key:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'private_key' field"}
            try:
                result = await self.runtime.autonet.publish_standards(
                    user_profile=self.runtime.user_profile,
                    private_key=private_key,
                )
                return {"msg_id": msg_id, "ok": result.get("status") != "error", "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Earnings dashboard (Story 3.5)
        if msg_type == "autonet_earnings":
            try:
                result = await self.runtime.autonet.get_earnings()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "autonet_claim_reward":
            epoch_id = msg.get("epoch_id")
            service_id = msg.get("service_id", "")
            private_key = msg.get("private_key", "")
            if epoch_id is None or not service_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'epoch_id' or 'service_id'"}
            if not private_key:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'private_key'"}
            try:
                result = await self.runtime.autonet.claim_reward(
                    epoch_id=int(epoch_id),
                    service_id=service_id,
                    private_key=private_key,
                )
                return {"msg_id": msg_id, "ok": result.get("status") != "error", "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Proposals & governance (Stories 3.6 / 3.7)
        if msg_type == "autonet_proposals":
            try:
                result = await self.runtime.autonet.get_proposals()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "autonet_governance":
            try:
                result = await self.runtime.autonet.get_governance()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Alignment dashboard (Story 3.8)
        if msg_type == "autonet_alignment":
            try:
                result = self.runtime.autonet.get_alignment(
                    user_profile=self.runtime.user_profile,
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Data capture & privacy (Story 3.4)
        if msg_type == "autonet_capture_config":
            try:
                result = self.runtime.autonet.get_capture_config()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "autonet_set_capture_config":
            try:
                result = self.runtime.autonet.set_capture_config(
                    capture=msg.get("capture"),
                    privacy=msg.get("privacy"),
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "autonet_enumerate_sources":
            try:
                result = self.runtime.autonet.enumerate_sources()
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

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

        if msg_type == "list_sponsor_bindings":
            return {"msg_id": msg_id, "ok": True, "result": {
                "bindings": self.runtime.sponsor_bindings.to_summary_list(),
            }}

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

        # On-chain agent registration
        if msg_type == "register_agent_on_chain":
            agent_id = msg.get("agent_id", "")
            is_root = msg.get("is_root", False)
            private_key = msg.get("private_key", "")
            sponsor_address = msg.get("sponsor_address", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            try:
                result = await self._handle_register_on_chain(
                    agent_id=agent_id,
                    is_root=is_root,
                    private_key=private_key,
                    sponsor_address=sponsor_address,
                )
                # Surface the agent's private key on a SUCCESSFUL registration
                # so the (local) user can import it into MetaMask and later sign
                # in remotely AS this agent (address-as-credential). Localhost
                # only — this whole message type is in KEY_LOCAL_ONLY_MESSAGES,
                # and we re-assert session.local as defense-in-depth. The key is
                # never returned over the remote listener.
                if result.get("success") and session.local:
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
            if agent_def and agent_def.identity:
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

        # Phase 12.20: substrate nodes — list visualization-ready nodes
        # (coords, recent_mint, score, parent_id) with optional spatial
        # binning for low-LOD views. Powers the substrate visualization.
        # External capsule writers (e.g. the substrate MCP server used by
        # Claude Code) drop conversation files directly into the agents dir.
        # This nudge bumps the substrate feed's pending counter so the
        # capsule counts toward the next ingestion cycle instead of waiting
        # for unrelated agent executions to cross the threshold.
        if msg_type == "substrate_feed_nudge":
            agent_id = str(msg.get("agent_id", "") or "external")
            autonet = getattr(self.runtime, "autonet", None)
            service = getattr(autonet, "_service", None) if autonet else None
            if service is None or not hasattr(service, "notify_execution"):
                return {"msg_id": msg_id, "ok": False,
                        "error": "autonet service not running"}
            try:
                service.notify_execution(agent_id, "capsule", "completed")
                return {"msg_id": msg_id, "ok": True, "result": {"nudged": True}}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Epoch mechanics observability (EPOCH_OBSERVABILITY_SPEC.md in
        # atn_web): live status of the open epoch. Combines the
        # WorldService half (epoch id, buffered events, emission) with
        # the EpochScheduler half (mode, candle window, T_max, seed
        # source).
        if msg_type == "epoch_status":
            autonet = getattr(self.runtime, "autonet", None)
            service = getattr(autonet, "_service", None) if autonet else None
            world_service = getattr(service, "_world_service", None) if service else None
            scheduler = getattr(service, "_epoch_scheduler", None) if service else None
            if world_service is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "autonet world service not running"}
            try:
                epoch = world_service.epoch_status()
                if scheduler is not None:
                    sched = scheduler.status()
                    # World's opened_at wins when both exist (same value
                    # in practice; the world is the source of truth).
                    sched.pop("opened_at", None)
                    epoch.update(sched)
                return {"msg_id": msg_id, "ok": True, "epoch": epoch}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Closed-epoch records, newest last. Full close-record shape
        # (cutoff_ts, events_rolled_forward, emission_pool, per-agent
        # and per-node mint) — the slim WORLD_EPOCH_CLOSED push event
        # is the refresh trigger; this is the detail read.
        if msg_type == "epoch_history":
            autonet = getattr(self.runtime, "autonet", None)
            service = getattr(autonet, "_service", None) if autonet else None
            world_service = getattr(service, "_world_service", None) if service else None
            if world_service is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "autonet world service not running"}
            try:
                last_n = max(1, min(int(msg.get("last_n", 20)), 200))
                history = world_service.epoch_history
                return {"msg_id": msg_id, "ok": True,
                        "epochs": history[-last_n:],
                        "total_closed": len(history)}
            except Exception as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

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

        # substrate_post_con RETIRED (v3, docs/tool_substrate.md Decision
        # 2026-07-08): debates left the live path — reviews (attest_tools
        # per-axis scores) are how the network judges tools now. The
        # WorldService submit_con/submit_support methods remain only as
        # the evidence-recording rail behind check_evidence.
        if msg_type == "substrate_post_con":
            return {"msg_id": msg_id, "ok": False,
                    "error": "substrate_post_con was retired (substrate v3): "
                             "review tools via attest_tools axis scores"}

        if msg_type == "rpb_substrate_nodes":
            try:
                max_nodes = int(msg.get("max_nodes", 200))
                bin_size_raw = msg.get("bin_size")
                bin_size = float(bin_size_raw) if bin_size_raw is not None else None
                recent_n = int(msg.get("recent_mint_epochs", 5))
                # Phase 12.26b — viewport-scoped lazy load. The frontend
                # passes (region_xy, region_radius) for the visible disk
                # and/or parent_id when drilled into a subtree.
                region_xy_raw = msg.get("region_xy")
                region_xy = None
                if isinstance(region_xy_raw, (list, tuple)) and len(region_xy_raw) == 2:
                    try:
                        region_xy = (float(region_xy_raw[0]), float(region_xy_raw[1]))
                    except (TypeError, ValueError):
                        region_xy = None
                region_radius_raw = msg.get("region_radius")
                region_radius = (
                    float(region_radius_raw)
                    if region_radius_raw is not None else None
                )
                parent_id_raw = msg.get("parent_id")
                parent_id = (
                    str(parent_id_raw)
                    if isinstance(parent_id_raw, str) and parent_id_raw else None
                )
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None
                if world_service is None:
                    return {"msg_id": msg_id, "ok": True, "result": {
                        "mode": "nodes", "items": [], "epochs_considered": 0,
                    }}
                result = world_service.list_nodes_for_visualization(
                    max_nodes=max_nodes,
                    recent_mint_epochs=recent_n,
                    bin_size=bin_size,
                    region_xy=region_xy,
                    region_radius=region_radius,
                    parent_id=parent_id,
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Per-subtree PCA projection — returns the same shape as
        # rpb_substrate_nodes (mode="nodes") but restricted to the
        # transitive descendants of parent_id and projected onto axes
        # fit on only that subset. Used by zoom-driven exploration:
        # as the user dives into a cluster, the frontend swaps the
        # network-level projection for this one.
        if msg_type == "rpb_substrate_subtree_projection":
            try:
                parent_id_raw = msg.get("parent_id")
                if not isinstance(parent_id_raw, str) or not parent_id_raw:
                    return {"msg_id": msg_id, "ok": False, "error": "parent_id required"}
                max_nodes = int(msg.get("max_nodes", 200))
                recent_n = int(msg.get("recent_mint_epochs", 5))
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None
                if world_service is None:
                    return {"msg_id": msg_id, "ok": True, "result": {
                        "mode": "nodes", "items": [], "epochs_considered": 0,
                        "parent_id": parent_id_raw,
                    }}
                result = world_service.compute_subtree_projection(
                    parent_id_raw,
                    max_nodes=max_nodes,
                    recent_mint_epochs=recent_n,
                )
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Phase 12.19: substrate distribution — current root scores +
        # per-root mint over recent epochs. Powers the constellation
        # visualization. Empty result is fine if autonet isn't running
        # yet or no epochs have closed.
        if msg_type == "rpb_substrate_distribution":
            try:
                last_n = int(msg.get("last_n_epochs", 10))
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None
                if world_service is None:
                    return {"msg_id": msg_id, "ok": True, "result": {
                        "root_scores": {},
                        "root_mint_recent": {},
                        "epochs_considered": 0,
                        "total_mint_recent": 0.0,
                    }}
                result = world_service.read_substrate_distribution(last_n_epochs=last_n)
                return {"msg_id": msg_id, "ok": True, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Tool-economy graph — registrations + declared-dep DAG + vetting
        # + recent per-digest mint (world consensus state), merged with the
        # local tool store (display names) and the service store (asks).
        # Powers the economy visualization. Empty result when the world
        # service isn't up yet.
        if msg_type == "economy_graph":
            try:
                last_n = int(msg.get("last_n_epochs", 10))
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None

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
                            "ask": rec.ask,
                            "backing_tool": rec.backing_tool,
                            "retired": rec.retired,
                        })

                world = (world_service.read_economy_graph(last_n_epochs=last_n)
                         if world_service is not None else {
                             "registrations": {}, "vetting": {},
                             "recent_tool_mint": {}, "last_epoch": None,
                             "epochs_considered": 0,
                         })
                return {"msg_id": msg_id, "ok": True, "result": {
                    "tools": tools,
                    "services": services,
                    **world,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # Recent reviews for one tool (v3) — the Substrate view's review
        # drawer. Local attestation rows (axes, score, note text) plus
        # the tool's drifted position/vetting from the world service.
        if msg_type == "tool_reviews":
            digest = str(msg.get("digest") or "").strip().lower()
            if not digest:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'digest' field"}
            try:
                reviews = self.runtime.tool_store.recent_attestations(
                    digest, limit=int(msg.get("limit") or 50))
                autonet = getattr(self.runtime, "autonet", None)
                service = getattr(autonet, "_service", None) if autonet else None
                world_service = getattr(service, "_world_service", None) if service else None
                position: dict[str, Any] = {}
                vetting: dict[str, Any] = {}
                usage: dict[str, Any] = {}
                if world_service is not None:
                    eg = world_service.read_economy_graph(last_n_epochs=10)
                    position = dict(eg.get("positions", {}).get(digest) or {})
                    vetting = dict(eg.get("vetting", {}).get(digest) or {})
                    # Usage/mint economics for the tool window: recent mint
                    # over the window + the last close's per-digest entry
                    # (ok_count, attesters, usage_term, mint).
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
                            "ok_count": entry.get("ok_count", 0),
                            "attesters": entry.get("attesters", 0),
                            "usage_term": entry.get("usage_term", 0.0),
                            "mint": entry.get("mint", 0.0),
                        }
                return {"msg_id": msg_id, "ok": True, "result": {
                    "digest": digest,
                    "reviews": reviews,
                    "position": position,
                    "vetting": vetting,
                    "usage": usage,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_agent_training":
            # Substrate-native: returns reputation + ATN balance for the
            # agent. The pre-substrate "training tokens / unclaimed rewards"
            # split is gone — training mints both ledgers directly.
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
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
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
                for defn in self.runtime.registry._agents.values():
                    if defn.identity and defn.identity.address:
                        addr = defn.identity.address.lower()
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
                    # Update active training feed config
                    if bridge._service and bridge._service._training_feed:
                        bridge._service._training_feed.config.backbone_model_id = model_id
                return {
                    "msg_id": msg_id, "ok": True,
                    "result": {"model_id": model_id, "status": "set" if model_id else "cleared"},
                }
            except ImportError:
                return {"msg_id": msg_id, "ok": False, "error": "Network package not installed"}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # New conversation: reset conversation history without changing model
        if msg_type == "new_conversation":
            await self.runtime.new_conversation()
            return {"msg_id": msg_id, "ok": True, "result": {"status": "Conversation reset"}}

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
            from .orchestrator import ORCHESTRATOR_ID
            # Legacy mode only: the provisioned root agent stays protected.
            # Rootless fleets treat it as a normal, removable agent (the
            # registry enforces the same rule — this is just a friendlier
            # message).
            _orch_cfg = getattr(self.runtime._config, "orchestrator", None)
            if agent_id == ORCHESTRATOR_ID and getattr(_orch_cfg, "enabled", False):
                return {"msg_id": msg_id, "ok": False, "error": "The root agent cannot be removed"}
            try:
                await self.runtime.unregister_agent(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"status": "removed", "agent_id": agent_id}}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Export an agent's private key. The ONLY message that returns a
        # private key in its body. Gated localhost-only at the top of
        # _handle_message (KEY_LOCAL_ONLY_MESSAGES); re-asserted here as
        # defense-in-depth so the handler stays safe if dispatch is ever moved.
        if msg_type == "export_agent_key":
            if not ws_auth.assert_local_key_access(session):
                return {"msg_id": msg_id, "ok": False,
                        "error": "Private key export is localhost-only."}
            agent_id = msg.get("agent_id", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
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

        # Special case: inject user message into running orchestrator session.
        # If the bridge process isn't running (e.g. after a daemon restart),
        # fall through to the normal post_message path which will trigger
        # a new execution.
        if msg_type == "orchestrator_message":
            content = msg.get("content", "")
            if not content:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'content' field"}
            # G1: orchestrator_message also bypasses send_agent_message (it
            # injects into the running bridge, or falls through to post_message
            # which triggers a run). Both are input — gate the arbiter here.
            gate = self._arbiter_gate(session)
            if gate is not None:
                return {"msg_id": msg_id, "ok": False, **gate}
            from .orchestrator import ORCHESTRATOR_ID
            from .providers.bridge import BridgeProvider
            provider = self.runtime._active_providers.get(ORCHESTRATOR_ID)
            orch_is_running = self.runtime._running_count.get(ORCHESTRATOR_ID, 0) > 0
            if orch_is_running and isinstance(provider, BridgeProvider) and provider._process and provider._process.returncode is None:
                await provider.send_user_message(content)
                return {"msg_id": msg_id, "ok": True, "result": {"status": "injected"}}
            # Bridge not running — convert to post_message so it triggers an execution
            msg_type = "post_message"
            msg = {
                "msg_id": msg_id,
                "type": "post_message",
                "target": "orchestrator",
                "message_type": "work",
                "priority": "high",
                "data": {"instruction": content},
                "source": "user",
            }

        # Strip protocol fields, pass only tool arguments.
        # Note: "type" is the routing field but also a valid arg for some tools
        # (e.g. post_message).  We strip it since the JSON object can't have two
        # "type" keys anyway — clients should use "message_type" for post_message.
        args = {k: v for k, v in msg.items() if k not in ("msg_id", "type")}
        # Tag client-initiated post_message with source="user" so the
        # orchestrator can distinguish user messages from agent messages.
        if msg_type == "post_message" and "source" not in args:
            args["source"] = "user"

        # Record user turn in conversation history early so the UI can fetch
        # it immediately via get_conversation (the frontend does an optimistic
        # add, but a history reload would lose it without this).
        # The execution engine skips re-adding it (dedup check at line ~624).
        if (msg_type == "post_message"
                and args.get("target") == "orchestrator"
                and args.get("source") == "user"):
            instruction = ""
            data = args.get("data", {})
            if isinstance(data, dict):
                instruction = data.get("instruction", "")
            if instruction:
                self.runtime.conversation.add_user_turn(instruction)

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
        # caller is named, a legacy install with a provisioned root agent
        # keeps acting as the orchestrator; a rootless fleet acts as the
        # OWNER ("" → owner-trusted, parentless creates).
        if session.scope_ids is None:
            caller_id = msg.get("caller_id")
            if caller_id is None:
                from .orchestrator import ORCHESTRATOR_ID
                caller_id = (
                    "orchestrator"
                    if self.runtime.get_agent(ORCHESTRATOR_ID) is not None
                    else ""
                )
        else:
            caller_id = session.root_agent_id
        result = await execute_tool(msg_type, args, self.runtime, caller_id=caller_id)

        # Tools signal errors by returning {"error": "..."} with a truthy value.
        # Some tools include "error": None as a data field — that's not an error.
        if result.get("error"):
            return {"msg_id": msg_id, "ok": False, "error": result["error"]}
        return {"msg_id": msg_id, "ok": True, "result": result}

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

        from .vault_setup import _keystore, write_policy_map
        try:
            ks = _keystore()
        except Exception as exc:  # noqa: BLE001 — pyrage missing / vault absent
            return {"msg_id": msg_id, "ok": False,
                    "error": f"vault keystore unavailable: {exc}"}

        if msg_type == "secrets_status":
            try:
                services = ks.list_services()
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
                    return list(services)
                try:
                    return list(ks.resolve_spec(wish))
                except Exception:  # noqa: BLE001 — fail closed
                    return []

            assignments: dict[str, dict[str, Any]] = {}
            for agent_id, defn in self.runtime.registry._agents.items():
                wish = getattr(defn, "secrets_allowance", None)
                lg = live.get(agent_id)
                pending = self.runtime._pending_grants.get(agent_id)
                if not wish and lg is None and not pending:
                    continue  # nothing secret-related about this agent
                assignments[agent_id] = {
                    "name": getattr(defn, "name", agent_id),
                    "allowance_spec": wish,
                    "resolved": _resolve_wish(wish),
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
                for p in await self.runtime.provider_list():
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
            if service.lower() in ("none", "all"):
                return {"msg_id": msg_id, "ok": False,
                        "error": f"'{service}' is a reserved allowance-spec keyword"}
            if not isinstance(value, str) or not value:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'value' field"}
            try:
                existed = service in ks.list_services()
                ks.put_secret(service, value)
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
            }}

        if msg_type == "secrets_delete":
            service = str(msg.get("service", "")).strip()
            if not service:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'service' field"}
            try:
                removed = ks.delete_secret(service)
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

        return {"msg_id": msg_id, "ok": False,
                "error": f"Unknown secrets message: {msg_type}"}

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

    async def _handle_service_request(
        self, msg: dict[str, Any], msg_id: Any
    ) -> dict[str, Any]:
        """Provider-side service entry (docs/services_market.md §3).

        Wire frame: {spec_digest, request_id, args} → look up spec →
        validate payment (seam) → dispatch to the backing local tool →
        record the request in the provider's log → return the result.

        The backing implementation of a v1 service is a registered tool
        the OWNER chose to sell, so dispatch goes through
        ``runtime.tool_store.call`` with ``caller_id=None`` — the sale is
        the owner's sanction, not a lineage-scoped agent call.
        """
        spec_digest = msg.get("spec_digest", "")
        request_id = msg.get("request_id", "")
        args = msg.get("args") or {}
        client = msg.get("client", "") or msg.get("client_address", "")
        if not spec_digest or not request_id:
            return {"msg_id": msg_id, "ok": False,
                    "error": "Missing 'spec_digest' or 'request_id' field"}

        record = self.runtime.service_store.get(spec_digest)
        if record is None:
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Unknown service digest: {spec_digest[:16]}"}
        if record.retired:
            return {"msg_id": msg_id, "ok": False, "error": "Service retired"}

        ask = record.ask
        token = str(ask.get("token", ""))
        amount = str(ask.get("amount", ""))

        # Payment/voucher validation seam (contracts workstream owns it).
        if not self._validate_service_payment(msg, record):
            self.runtime.service_store.record_request(
                spec_digest, request_id, client, ok=False,
                amount=amount, token=token)
            return {"msg_id": msg_id, "ok": False,
                    "error": "Payment validation failed"}

        backing = record.backing_tool
        if not backing:
            self.runtime.service_store.record_request(
                spec_digest, request_id, client, ok=False,
                amount=amount, token=token)
            return {"msg_id": msg_id, "ok": False,
                    "error": "Service has no backing implementation"}

        tool_record = self.runtime.tool_store.get(backing)
        if tool_record is None:
            tool_record = self.runtime.tool_store.resolve(backing)
        if tool_record is None:
            self.runtime.service_store.record_request(
                spec_digest, request_id, client, ok=False,
                amount=amount, token=token)
            return {"msg_id": msg_id, "ok": False,
                    "error": f"Backing tool {backing[:16]} not found"}

        # Owner-sanctioned dispatch: selling a tool IS the owner's choice.
        result = await self.runtime.tool_store.call(
            tool_record, args, caller_id=None)
        ok = "error" not in result
        self.runtime.service_store.record_request(
            spec_digest, request_id, client, ok=ok,
            amount=amount, token=token)
        if not ok:
            return {"msg_id": msg_id, "ok": False,
                    "result": {"request_id": request_id, **result}}
        return {"msg_id": msg_id, "ok": True,
                "result": {"request_id": request_id, **result}}

    def _validate_service_payment(
        self, request: dict[str, Any], record: Any
    ) -> bool:
        """Seam for service payment/voucher validation.

        TODO(contracts): implement per docs/services_market.md §2 —
        channel-only settlement (postpaid escrow was deleted, not
        deferred): verify the request's signed cumulative voucher
        against the PaymentChannel's voucherHash/current state. Until
        then this returns True so the provider dispatch path is
        exercisable end-to-end.
        """
        return True

    async def _handle_register_on_chain(
        self,
        agent_id: str,
        is_root: bool = False,
        private_key: str = "",
        sponsor_address: str = "",
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

        # Substrate-native registration flow (Phase 12):
        #
        # The agent has its own keypair generated at agent creation time
        # (held by the daemon, not the user's wallet). Registration is
        # ALWAYS daemon-signed from the agent's own key — this includes
        # the root/orchestrator agent. The user's wallet only funds the
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
            balance = w3.eth.get_balance(w3.to_checksum_address(identity.address))
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

        result = await svc.register_agent(
            identity=identity,
            private_key=key,
            peer_id=peer_id,
        )
        if result.get("success"):
            identity.registered_on_chain = True
            identity.registration_tx = result.get("tx_hash")
            self.runtime.registry.persist_identity(agent_id)
        return result

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
        # daemon-held keypair (incl. the orchestrator). Check that
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
            record = await svc.get_agent_record(matched_address)
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
        for defn in agents:
            await rt.register_agent(defn)
            if defn.schedule or defn.heartbeat:
                await rt.activate_agent(defn.id)
        # Note: execution history is hydrated automatically in register_agent()
        if agents:
            log.info("Loaded %d agent(s)", len(agents))

    # Register the orchestrator meta-agent
    try:
        await rt.setup_orchestrator()
        log.info("Orchestrator registered")
    except Exception as exc:
        log.warning("Failed to register orchestrator: %s", exc)

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
