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
import atexit
import json
import logging
import os
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import Server as WSServer, ServerConnection

from .events import Event, EventBus, EventType
from .orchestrator.tools import execute_tool
from .runtime import Runtime
from .runtime.provider_manager import get_model_tier, get_tier_label

log = logging.getLogger(__name__)

# Suppress noisy websockets handshake errors (e.g. bare TCP probes from
# port-in-use checks, or clients connecting before the upgrade completes).
# These are non-fatal — the library handles them — but the full tracebacks
# confuse users.
logging.getLogger("websockets").setLevel(logging.CRITICAL)

# Default port
DEFAULT_PORT = 7700


class WebSocketBridge:
    """Bridges the ATN Runtime to WebSocket clients.

    Handles:
      - Routing incoming JSON messages to orchestrator tools
      - Broadcasting EventBus events to all connected clients
    """

    def __init__(self, runtime: Runtime, host: str = "localhost", port: int = DEFAULT_PORT) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self._server: WSServer | None = None
        self._clients: set[ServerConnection] = set()
        self._event_handler_registered = False

    async def start(self) -> None:
        """Start the WebSocket server."""
        # Subscribe to all EventBus events
        if not self._event_handler_registered:
            self.runtime.events.subscribe(None, self._on_event)
            self._event_handler_registered = True

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        log.info("WebSocket server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._event_handler_registered:
            self.runtime.events.unsubscribe(None, self._on_event)
            self._event_handler_registered = False
        log.info("WebSocket server stopped")

    # ------------------------------------------------------------------
    # Client handling
    # ------------------------------------------------------------------

    async def _handle_client(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket client connection."""
        self._clients.add(ws)
        remote = ws.remote_address
        log.info("Client connected: %s", remote)

        # Send initial snapshot so the UI has state immediately
        try:
            snapshot = self.runtime.snapshot()
            await ws.send(json.dumps({
                "type": "snapshot",
                "data": snapshot,
            }, default=str))
        except Exception:
            log.exception("Failed to send initial snapshot")

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({
                        "ok": False,
                        "error": "Invalid JSON",
                    }))
                    continue

                response = await self._handle_message(msg)
                await ws.send(json.dumps(response, default=str))

        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("Client handler error")
        finally:
            self._clients.discard(ws)
            log.info("Client disconnected: %s", remote)

    async def _handle_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Route an incoming message to the appropriate handler."""
        msg_id = msg.get("msg_id")
        msg_type = msg.get("type", "")

        if not msg_type:
            return {"msg_id": msg_id, "ok": False, "error": "Missing 'type' field"}

        # Special case: snapshot request
        if msg_type == "snapshot":
            return {
                "msg_id": msg_id,
                "ok": True,
                "result": self.runtime.snapshot(),
            }

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
            delivered = await self.runtime.send_delegate_message(agent_id, content)
            if not delivered:
                return {"msg_id": msg_id, "ok": False, "error": f"Delegate '{agent_id}' is not running"}
            return {"msg_id": msg_id, "ok": True, "result": {"status": "injected", "agent_id": agent_id}}

        # Send message to a cognitive agent (universal chat)
        if msg_type == "send_agent_message":
            agent_id = msg.get("agent_id", "")
            content = msg.get("content", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id' field"}
            if not content:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'content' field"}
            result = await self.runtime.send_agent_message(agent_id, content)
            if result.get("error"):
                return {"msg_id": msg_id, "ok": False, "error": result["error"]}
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

        # Confirm that a non-custodial registration tx landed on-chain
        if msg_type == "confirm_agent_registration":
            agent_id = msg.get("agent_id", "")
            tx_hash = msg.get("tx_hash", "")
            agent_address = msg.get("agent_address", "")
            if not agent_id:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'agent_id'"}
            agent_def = self.runtime.registry.get_agent(agent_id)
            if agent_def and agent_def.identity:
                agent_def.identity.registered_on_chain = True
                agent_def.identity.registration_tx = tx_hash
                if agent_address:
                    agent_def.identity.address = agent_address
                self.runtime.registry.persist_identity(agent_id)
                log.info("Agent %s confirmed on-chain: tx=%s addr=%s", agent_id, tx_hash, agent_address)
            return {"msg_id": msg_id, "ok": True, "result": {"confirmed": True}}

        # RPB on-chain state queries
        if msg_type == "rpb_state":
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_rpb_state()
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_agent_training":
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_agent_training_info(address)
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
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_agent_record(address)
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # RPB investment/funding pipeline
        if msg_type == "rpb_investor_info":
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_investor_info(address)
                return {"msg_id": msg_id, "ok": True, "result": result or {}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_accepted_tokens":
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_accepted_tokens()
                return {"msg_id": msg_id, "ok": True, "result": {"tokens": result}}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_purchase_shares":
            token = msg.get("token", "")
            amount = msg.get("amount")
            if not token:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'token'"}
            if amount is None:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'amount'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                call_data = svc.build_purchase_shares_call_data(token, int(amount))
                return {"msg_id": msg_id, "ok": True, "result": {
                    "call_data": call_data,
                    "to": self.runtime._config.rpb.rpb_contract_address,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_fund_training":
            token = msg.get("token", "")
            amount = msg.get("amount")
            if not token:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'token'"}
            if amount is None:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'amount'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                call_data = svc.build_fund_training_call_data(token, int(amount))
                return {"msg_id": msg_id, "ok": True, "result": {
                    "call_data": call_data,
                    "to": self.runtime._config.rpb.rpb_contract_address,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_claim_training_reward":
            token = msg.get("token", "")
            if not token:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'token'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                call_data = svc.build_claim_training_reward_call_data(token)
                return {"msg_id": msg_id, "ok": True, "result": {
                    "call_data": call_data,
                    "to": self.runtime._config.rpb.rpb_contract_address,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_claim_dividends":
            token = msg.get("token", "")
            if not token:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'token'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                call_data = svc.build_claim_dividends_call_data(token)
                return {"msg_id": msg_id, "ok": True, "result": {
                    "call_data": call_data,
                    "to": self.runtime._config.rpb.rpb_contract_address,
                }}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        # RPB sponsorship / inference pipeline
        if msg_type == "rpb_record_inference":
            requester = msg.get("requester", "")
            provider = msg.get("provider", "")
            units = msg.get("units")
            token = msg.get("token", "")
            cost = msg.get("cost")
            if not all([requester, provider, token]) or units is None or cost is None:
                return {"msg_id": msg_id, "ok": False,
                        "error": "Missing required fields: requester, provider, units, token, cost"}
            try:
                from .on_chain import OnChainService
                config = self.runtime._config.rpb
                svc = OnChainService(config)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                if not config.private_key:
                    return {"msg_id": msg_id, "ok": False,
                            "error": "Owner private_key not configured (required for onlyOwner call)"}
                result = await svc.record_inference(
                    requester=requester,
                    provider=provider,
                    units=int(units),
                    token_address=token,
                    cost=int(cost),
                    private_key=config.private_key,
                )
                return {"msg_id": msg_id, "ok": result.get("success", False), "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_sponsor_budget":
            address = msg.get("address", "")
            if not address:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'address'"}
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                result = await svc.get_sponsor_budget(address)
                return {"msg_id": msg_id, "ok": "error" not in result, "result": result}
            except Exception as e:
                return {"msg_id": msg_id, "ok": False, "error": str(e)}

        if msg_type == "rpb_registered_agents":
            try:
                from .on_chain import OnChainService
                svc = OnChainService(self.runtime._config.rpb)
                if not svc.available:
                    return {"msg_id": msg_id, "ok": False, "error": "RPB not configured"}
                agents_list = await svc.get_all_registered_agents()

                # Build lookup: on-chain address → local agent metadata
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
                # Also check the connected wallet (root agent's address may
                # be the user's wallet, not the generated identity address)
                connected_wallet = getattr(
                    getattr(self.runtime, "autonet", None),
                    "state", None
                )
                wallet_addr = getattr(connected_wallet, "wallet_address", "") if connected_wallet else ""
                if wallet_addr:
                    wallet_lower = wallet_addr.lower()
                    if wallet_lower not in local_meta:
                        # Root agent uses the wallet address on-chain
                        root = self.runtime.registry._agents.get("orchestrator")
                        if root:
                            local_meta[wallet_lower] = {
                                "display_name": root.name,
                                "display_description": root.description,
                                "agent_type": getattr(root, "agent_type", ""),
                                "model": root.model or "",
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
            if agent_id == ORCHESTRATOR_ID:
                return {"msg_id": msg_id, "ok": False, "error": "The root agent cannot be removed"}
            try:
                await self.runtime.unregister_agent(agent_id)
                return {"msg_id": msg_id, "ok": True, "result": {"status": "removed", "agent_id": agent_id}}
            except ValueError as exc:
                return {"msg_id": msg_id, "ok": False, "error": str(exc)}

        # Special case: inject user message into running orchestrator session.
        # If the bridge process isn't running (e.g. after a daemon restart),
        # fall through to the normal post_message path which will trigger
        # a new execution.
        if msg_type == "orchestrator_message":
            content = msg.get("content", "")
            if not content:
                return {"msg_id": msg_id, "ok": False, "error": "Missing 'content' field"}
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

        # Use caller_id from message if provided (e.g. UI acting on behalf of
        # a specific agent), otherwise default to the orchestrator.
        caller_id = msg.get("caller_id", "orchestrator")
        result = await execute_tool(msg_type, args, self.runtime, caller_id=caller_id)

        # Tools signal errors by returning {"error": "..."} with a truthy value.
        # Some tools include "error": None as a data field — that's not an error.
        if result.get("error"):
            return {"msg_id": msg_id, "ok": False, "error": result["error"]}
        return {"msg_id": msg_id, "ok": True, "result": result}

    # ------------------------------------------------------------------
    # On-chain registration helpers
    # ------------------------------------------------------------------

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
            return {"success": False, "error": "On-chain not configured (missing rpb_contract_address or rpc_url)"}

        # Look up the agent
        agent_def = self.runtime.registry.get_agent(agent_id)
        if not agent_def:
            return {"success": False, "error": f"Agent '{agent_id}' not found"}
        if not agent_def.identity:
            return {"success": False, "error": f"Agent '{agent_id}' has no identity"}

        identity = agent_def.identity
        system_prompt = agent_def.system_prompt or ""

        # Determine parent's on-chain address
        parent_address = ""
        if agent_def.parent_id:
            parent_def = self.runtime.registry.get_agent(agent_def.parent_id)
            if parent_def and parent_def.identity:
                parent_address = parent_def.identity.address

        if is_root and not private_key:
            # Non-custodial: return call data for frontend wallet
            try:
                call_data = svc.build_register_call_data(
                    identity=identity,
                    system_prompt=system_prompt,
                    parent_address=parent_address,
                    sponsor_address=sponsor_address,
                )
                return {
                    "success": True,
                    "mode": "sign_required",
                    "call_data": call_data,
                    "to": config.rpb_contract_address,
                    "agent_address": identity.address,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Custodial path: we have the private key
        if is_root:
            # Root agent — private_key was passed from frontend
            key = private_key
        else:
            # Child agent — daemon holds the key
            key = self.runtime.registry.get_agent_key(agent_id)
            if not key:
                return {"success": False, "error": f"No private key stored for agent '{agent_id}'"}

        result = await svc.register_agent(
            identity=identity,
            private_key=key,
            system_prompt=system_prompt,
            parent_address=parent_address,
            sponsor_address=sponsor_address,
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

        # Root agents register via the user's wallet, not the daemon-generated
        # keypair.  After a daemon restart the keypair is regenerated, so the
        # generated address will differ from the one that was actually
        # registered.  Check both addresses (generated + connected wallet).
        addrs_to_check = []
        if agent_def.identity.address:
            addrs_to_check.append(agent_def.identity.address)
        wallet = self.runtime.autonet.state.wallet_address
        if wallet and wallet not in addrs_to_check:
            addrs_to_check.append(wallet)

        registered = False
        matched_address = ""
        for addr in addrs_to_check:
            if await svc.is_registered(addr):
                registered = True
                matched_address = addr
                break

        result: dict[str, Any] = {
            "registered": registered,
            "available": True,
            "agent_address": matched_address or agent_def.identity.address,
        }

        # Sync daemon's in-memory flag with on-chain truth.
        # This handles both directions: newly registered or contract redeployed.
        if registered != agent_def.identity.registered_on_chain:
            agent_def.identity.registered_on_chain = registered
            if registered and matched_address:
                agent_def.identity.address = matched_address
            self.runtime.registry.persist_identity(agent_id)
            log.info("Agent %s on-chain status synced to %s (address: %s)",
                     agent_id, registered, matched_address or "n/a")

        if registered:
            record = await svc.get_agent_record(matched_address)
            if record:
                result["record"] = record

        return result

    # ------------------------------------------------------------------
    # Event broadcasting
    # ------------------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        """Broadcast an EventBus event to all connected clients."""
        if not self._clients:
            return

        payload = json.dumps({
            "type": "event",
            "event_type": event.type.value,
            "source": event.source,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
        }, default=str)

        # Broadcast to all connected clients, drop failures silently
        stale: list[ServerConnection] = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                stale.append(ws)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)


# ---------------------------------------------------------------------------
# Singleton process lock (pidfile)
# ---------------------------------------------------------------------------

_PIDFILE_NAME = "ws_server.pid"


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _port_in_use(host: str, port: int) -> bool:
    """Return True if *port* is already bound on *host*."""
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((host if family == socket.AF_INET else "::1", port))
                return True
        except OSError:
            continue
    return False


def _get_pidfile(data_dir: Path) -> Path:
    return data_dir / _PIDFILE_NAME


def _acquire_lock(data_dir: Path, host: str = "localhost", port: int = DEFAULT_PORT) -> Path:
    """Acquire the singleton lock.  Exits the process if another instance is running."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pidfile = _get_pidfile(data_dir)

    if pidfile.exists():
        try:
            old_pid = int(pidfile.read_text().strip())
        except (ValueError, OSError):
            old_pid = 0

        if old_pid and _pid_is_alive(old_pid):
            print(
                f"ATN daemon is already running (PID {old_pid}, port {port}).  "
                f"Kill it first or remove {pidfile}",
                file=sys.stderr,
            )
            sys.exit(1)

        log.info("Stale pidfile found (PID %d not running), removing", old_pid)
        pidfile.unlink(missing_ok=True)

    # Secondary check: even without a pidfile, something may hold the port
    # (e.g. pidfile was cleaned up but the process survived).
    if _port_in_use(host, port):
        print(
            f"ATN daemon port {port} is already in use (pidfile was missing).  "
            f"Another instance may be running.  Find it with:\n"
            f"  netstat -ano | findstr {port}",
            file=sys.stderr,
        )
        sys.exit(1)

    pidfile.write_text(str(os.getpid()))
    atexit.register(_release_lock, data_dir)
    return pidfile


def _release_lock(data_dir: Path) -> None:
    """Release the singleton lock by removing the pidfile."""
    pidfile = _get_pidfile(data_dir)
    try:
        if pidfile.exists():
            stored_pid = int(pidfile.read_text().strip())
            if stored_pid == os.getpid():
                pidfile.unlink(missing_ok=True)
    except (ValueError, OSError):
        pidfile.unlink(missing_ok=True)


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
    _acquire_lock(config.data_dir, host=host, port=port)

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
        _release_lock(config.data_dir)


def main() -> None:
    """Entry point for python -m atn.ws_server."""
    asyncio.run(run_standalone())


if __name__ == "__main__":
    main()
