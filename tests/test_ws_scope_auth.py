"""Integration tests for the WS server's two-listener auth + scoping + custody.

Drives WebSocketBridge._handle_message with a real Runtime + Session, no socket.
Covers the security-review attack surface: loopback bypass, remote fail-closed,
owner signature, subtree scoping, the caller_id-clamp-is-insufficient escape via
explicit target args, and localhost-only key export."""
from __future__ import annotations

import time

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from atn import ws_auth
from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentIdentity, AgentMode
from atn.runtime import Runtime
from atn.ws_auth import ClientSession
from atn.ws_server import WebSocketBridge


def _make_runtime(tmp_path, owner_wallet="") -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id, parent_id, *, address=None, key=None):
    identity = AgentIdentity(public_key=address or "", address=address or "") if address else None
    return AgentDefinition(id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
                           parent_id=parent_id, identity=identity, budgets={})


async def _fleet(tmp_path, owner_wallet=""):
    rt = _make_runtime(tmp_path, owner_wallet=owner_wallet)
    await rt.registry.register_agent(_agent("orchestrator", None))
    await rt.registry.register_agent(_agent("a", "orchestrator", address="0xAaA0000000000000000000000000000000000001"))
    await rt.registry.register_agent(_agent("a.1", "a"))
    await rt.registry.register_agent(_agent("b", "orchestrator"))
    # Give a.1 a stored private key so export tests have something to fetch.
    rt.registry._agent_keys["a.1"] = "deadbeef" * 8
    return rt


def _bridge(rt, owner_wallet=""):
    return WebSocketBridge(rt, owner_wallet=owner_wallet)


def _local_session():
    return ClientSession(local=True, authed=True, owner=True,
                         root_agent_id="orchestrator", scope_ids=None)


# ---------------------------------------------------------------------------
# Local listener: full control, no handshake (backward compat)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_session_snapshot_unrestricted(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    resp = await bridge._handle_message({"type": "snapshot", "msg_id": "1"}, _local_session())
    assert resp["ok"] is True
    assert set(resp["result"]["agents"].keys()) == {"orchestrator", "a", "a.1", "b"}


@pytest.mark.asyncio
async def test_local_session_can_export_key(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    resp = await bridge._handle_message(
        {"type": "export_agent_key", "agent_id": "a.1", "msg_id": "1"}, _local_session())
    assert resp["ok"] is True
    assert resp["result"]["private_key"] == "deadbeef" * 8


# ---------------------------------------------------------------------------
# Remote listener: fail-closed until authed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remote_unauthed_rejected(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    s = ClientSession(local=False, authed=False)
    # Any non-auth message is refused.
    for mtype in ("snapshot", "list_agents", "trigger_run", "kill_agent"):
        resp = await bridge._handle_message({"type": mtype, "msg_id": "1"}, s)
        assert resp["ok"] is False
        assert resp["error"] == "unauthenticated"


@pytest.mark.asyncio
async def test_remote_key_export_refused_even_if_somehow_authed(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    # A remote session that is fully authed as owner STILL cannot export keys —
    # the gate is the listener (local), not auth/owner.
    s = ClientSession(local=False, authed=True, owner=True, scope_ids=None)
    resp = await bridge._handle_message(
        {"type": "export_agent_key", "agent_id": "a.1", "msg_id": "1"}, s)
    assert resp["ok"] is False
    assert "localhost-only" in resp["error"]
    assert "private_key" not in str(resp)


@pytest.mark.asyncio
async def test_remote_inbound_private_key_handler_refused(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    s = ClientSession(local=False, authed=True, owner=True, scope_ids=None)
    resp = await bridge._handle_message(
        {"type": "autonet_publish_standards", "private_key": "0xabc", "msg_id": "1"}, s)
    assert resp["ok"] is False
    assert "localhost-only" in resp["error"]


# ---------------------------------------------------------------------------
# Owner signature handshake
# ---------------------------------------------------------------------------

def _sign_challenge(bridge, session, acct):
    challenge = ws_auth.build_challenge_text(
        session.nonce, daemon_id=bridge._daemon_id(),
        chain_id=int(bridge.runtime._config.autonet.chain_id or 0),
        owner_wallet=bridge.owner_wallet, conn_id=session.conn_id,
        issued_at=session.nonce_issued_at)
    return acct.sign_message(encode_defunct(text=challenge)).signature.hex()


@pytest.mark.asyncio
async def test_owner_signature_authorizes_full_fleet(tmp_path):
    acct = Account.create()
    rt = await _fleet(tmp_path, owner_wallet=acct.address)
    bridge = _bridge(rt, owner_wallet=acct.address)
    s = ClientSession(local=False, conn_id="c1", nonce=ws_auth.new_nonce(),
                      nonce_issued_at=time.time())
    sig = _sign_challenge(bridge, s, acct)
    resp = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "msg_id": "1"}, s)
    assert resp["ok"] is True and resp["owner"] is True
    assert resp["root"] == "orchestrator"
    assert s.authed and s.scope_ids is None


@pytest.mark.asyncio
async def test_non_owner_signature_denied(tmp_path):
    owner = Account.create()
    attacker = Account.create()
    rt = await _fleet(tmp_path, owner_wallet=owner.address)
    bridge = _bridge(rt, owner_wallet=owner.address)
    s = ClientSession(local=False, conn_id="c1", nonce=ws_auth.new_nonce(),
                      nonce_issued_at=time.time())
    sig = _sign_challenge(bridge, s, attacker)   # signed by the wrong wallet
    resp = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "msg_id": "1"}, s)
    assert resp["ok"] is False
    assert s.authed is False


@pytest.mark.asyncio
async def test_owner_can_root_at_subtree(tmp_path):
    acct = Account.create()
    rt = await _fleet(tmp_path, owner_wallet=acct.address)
    bridge = _bridge(rt, owner_wallet=acct.address)
    s = ClientSession(local=False, conn_id="c1", nonce=ws_auth.new_nonce(),
                      nonce_issued_at=time.time())
    sig = _sign_challenge(bridge, s, acct)
    resp = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "root": "a", "msg_id": "1"}, s)
    assert resp["ok"] is True and resp["root"] == "a"
    assert s.scope_ids == {"a", "a.1"}


@pytest.mark.asyncio
async def test_expired_and_replayed_nonce_rejected(tmp_path):
    acct = Account.create()
    rt = await _fleet(tmp_path, owner_wallet=acct.address)
    bridge = _bridge(rt, owner_wallet=acct.address)
    # Expired challenge.
    s = ClientSession(local=False, conn_id="c1", nonce=ws_auth.new_nonce(),
                      nonce_issued_at=time.time() - 999)
    sig = _sign_challenge(bridge, s, acct)
    resp = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "msg_id": "1"}, s)
    assert resp["ok"] is False and "expired" in resp["error"]
    # Nonce consumed: a second attempt (even fresh sig) has no nonce.
    resp2 = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "msg_id": "2"}, s)
    assert resp2["ok"] is False


# ---------------------------------------------------------------------------
# Scoped-session escalation: the caller_id-clamp-is-insufficient attack
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scoped_session_cannot_target_out_of_subtree(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    # Session scoped to subtree 'a' = {a, a.1}. caller_id is its own root (would
    # pass a naive caller_id clamp), but the TARGET is the orchestrator.
    s = ClientSession(local=False, authed=True, owner=True,
                      root_agent_id="a", scope_ids={"a", "a.1"})
    resp = await bridge._handle_message(
        {"type": "kill_agent", "agent_id": "orchestrator",
         "caller_id": "a", "msg_id": "1"}, s)
    assert resp["ok"] is False
    assert "outside the authorized subtree" in resp["error"]
    # Sibling 'b' is also out of subtree.
    resp_b = await bridge._handle_message(
        {"type": "trigger_run", "agent_id": "b", "caller_id": "a", "msg_id": "2"}, s)
    assert resp_b["ok"] is False


@pytest.mark.asyncio
async def test_scoped_session_owner_only_tool_refused(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    s = ClientSession(local=False, authed=True, owner=True,
                      root_agent_id="a", scope_ids={"a", "a.1"})
    resp = await bridge._handle_message(
        {"type": "get_user_profile", "msg_id": "1"}, s)
    assert resp["ok"] is False
    assert "owner-only" in resp["error"]


@pytest.mark.asyncio
async def test_scoped_snapshot_redacts_secret_sections(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    s = ClientSession(local=False, authed=True, owner=True,
                      root_agent_id="a", scope_ids={"a", "a.1"})
    resp = await bridge._handle_message({"type": "snapshot", "msg_id": "1"}, s)
    assert resp["ok"] is True
    for secret in ("providers", "budget", "user", "connectors", "autonet"):
        assert secret not in resp["result"]
    # Only the subtree's agents are present.
    assert set(resp["result"]["agents"].keys()) == {"a", "a.1"}
