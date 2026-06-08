"""LIVE two-listener handshake over a real WebSocket.

The other WS tests call _handle_message directly. This one starts the actual
WebSocketBridge with both listeners on real ports and drives a genuine
`websockets` client through connect -> auth_challenge -> sign -> auth_response
-> scoped snapshot. It proves the socket path (serve, unsolicited challenge,
signature recovery over the wire, scoped delivery) that the unit tests bypass.

This is the closest deterministic stand-in for the two-host test: a real remote
client authenticating against the auth-required listener with a throwaway key
acting as the configured owner. (A true browser+MetaMask cross-host test can't
be driven headlessly — wallet signing is interactive — so we drive the same
wire protocol with eth_account.)"""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets
from eth_account import Account
from eth_account.messages import encode_defunct

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentIdentity, AgentMode
from atn.runtime import Runtime
from atn.ws_server import WebSocketBridge

LOCAL_PORT = 27700
REMOTE_PORT = 27701


async def _make_runtime(tmp_path, owner_wallet):
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)

    def _agent(aid, pid, addr=None):
        ident = AgentIdentity(public_key=addr or "", address=addr or "") if addr else None
        return AgentDefinition(id=aid, name=aid, mode=AgentMode.COGNITIVE,
                               parent_id=pid, identity=ident, budgets={})

    await rt.registry.register_agent(_agent("orchestrator", None))
    await rt.registry.register_agent(_agent("a", "orchestrator"))
    await rt.registry.register_agent(_agent("a.1", "a"))
    await rt.registry.register_agent(_agent("b", "orchestrator"))
    return rt


async def _recv_json(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


@pytest.mark.asyncio
async def test_live_two_listener_handshake(tmp_path):
    acct = Account.create()
    rt = await _make_runtime(tmp_path, owner_wallet=acct.address)
    bridge = WebSocketBridge(
        rt, host="127.0.0.1", port=LOCAL_PORT,
        remote_host="127.0.0.1", remote_port=REMOTE_PORT,
        owner_wallet=acct.address,
    )
    await bridge.start()
    try:
        # --- LOCAL listener: immediate full snapshot, no handshake -----------
        async with websockets.connect(f"ws://127.0.0.1:{LOCAL_PORT}") as ws:
            first = await _recv_json(ws)
            assert first["type"] == "snapshot"
            assert set(first["data"]["agents"].keys()) == {"orchestrator", "a", "a.1", "b"}

        # --- REMOTE listener: challenge -> sign -> scoped snapshot -----------
        async with websockets.connect(f"ws://127.0.0.1:{REMOTE_PORT}") as ws:
            challenge_msg = await _recv_json(ws)
            assert challenge_msg["type"] == "auth_challenge"
            assert "challenge" in challenge_msg and "nonce" in challenge_msg

            # An unauthed request is refused.
            await ws.send(json.dumps({"type": "snapshot", "msg_id": "1"}))
            denied = await _recv_json(ws)
            assert denied["ok"] is False and denied["error"] == "unauthenticated"

            # Sign the literal challenge text the server sent, root at 'a'.
            sig = acct.sign_message(
                encode_defunct(text=challenge_msg["challenge"])).signature.hex()
            await ws.send(json.dumps({
                "type": "auth_response", "msg_id": "2", "signature": sig, "root": "a"}))
            ok = await _recv_json(ws)
            assert ok["type"] == "auth_ok" and ok["ok"] is True
            assert ok["root"] == "a"

            # Now a scoped snapshot request returns only the 'a' subtree, with
            # owner-global sections redacted.
            await ws.send(json.dumps({"type": "snapshot", "msg_id": "3"}))
            snap = await _recv_json(ws)
            assert snap["ok"] is True
            assert set(snap["result"]["agents"].keys()) == {"a", "a.1"}
            for secret in ("providers", "budget", "user", "connectors", "autonet"):
                assert secret not in snap["result"]
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_live_agent_self_auth_no_owner(tmp_path):
    """No owner_wallet configured: a holder of an AGENT's own key signs in as
    that agent (address-as-credential) and is scoped to its subtree, owner=False.

    This is the user's model: register agent -> export its key -> import to
    MetaMask -> connect as that agent. The remote listener serves it even with
    no owner wallet set."""
    agent_acct = Account.create()
    # Build a fleet where agent 'a' has agent_acct's address as its identity.
    data_dir = tmp_path / "d2"; agents_dir = tmp_path / "ag2"
    data_dir.mkdir(parents=True); agents_dir.mkdir(parents=True)
    from atn.config import ATNConfig
    cfg = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    cfg.autonet.enabled = False; cfg.voice.enabled = False
    cfg.autonet.owner_wallet = ""                    # NO owner
    rt = Runtime(EventBus(), data_dir=data_dir, config=cfg)

    def _ag(aid, pid, addr=None):
        ident = AgentIdentity(public_key=addr or "", address=addr or "") if addr else None
        return AgentDefinition(id=aid, name=aid, mode=AgentMode.COGNITIVE,
                               parent_id=pid, identity=ident, budgets={})
    await rt.registry.register_agent(_ag("orchestrator", None))
    await rt.registry.register_agent(_ag("a", "orchestrator", addr=agent_acct.address))
    await rt.registry.register_agent(_ag("a.1", "a"))
    await rt.registry.register_agent(_ag("b", "orchestrator"))

    bridge = WebSocketBridge(
        rt, host="127.0.0.1", port=LOCAL_PORT + 10,
        remote_host="127.0.0.1", remote_port=REMOTE_PORT + 10,
        owner_wallet="",
    )
    await bridge.start()
    try:
        assert bridge._remote_server is not None      # listener IS up, no owner
        async with websockets.connect(f"ws://127.0.0.1:{REMOTE_PORT + 10}") as ws:
            ch = await _recv_json(ws)
            assert ch["type"] == "auth_challenge"
            # Sign as agent 'a' (the user holds a's key).
            sig = agent_acct.sign_message(
                encode_defunct(text=ch["challenge"])).signature.hex()
            await ws.send(json.dumps({"type": "auth_response", "msg_id": "1", "signature": sig}))
            ok = await _recv_json(ws)
            assert ok["type"] == "auth_ok" and ok["ok"] is True
            assert ok["owner"] is False               # agent, not owner
            assert ok["root"] == "a"                  # rooted at itself
            await ws.send(json.dumps({"type": "snapshot", "msg_id": "2"}))
            snap = await _recv_json(ws)
            assert set(snap["result"]["agents"].keys()) == {"a", "a.1"}

        # A signer that is NEITHER owner NOR a known agent is denied.
        stranger = Account.create()
        async with websockets.connect(f"ws://127.0.0.1:{REMOTE_PORT + 10}") as ws:
            ch = await _recv_json(ws)
            sig = stranger.sign_message(
                encode_defunct(text=ch["challenge"])).signature.hex()
            await ws.send(json.dumps({"type": "auth_response", "msg_id": "1", "signature": sig}))
            denied = await _recv_json(ws)
            assert denied["type"] == "auth_denied" and denied["ok"] is False
    finally:
        await bridge.stop()
