"""Integration listener (third socket): bearer token clamped to ONE agent.

Covers docs/integration_listener.md: token store (hashed, revocable), the
owner-sentinel invariant against is_owner_caller, the caller clamp, the
owner-message deny-list + allowlist, revocation mid-connection, wrong-agent
requests, use_tool narrowing, the rate limit, and the live socket handshake
(401 before upgrade on a bad/revoked token)."""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from atn import ws_auth
from atn.agent_tools import OWNER_ID, _LEGACY_ROOT_ID, is_owner_caller
from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.ws_server import (
    INTEGRATION_ALLOWED_MESSAGES,
    INTEGRATION_DENIED_MESSAGES,
    KEY_LOCAL_ONLY_MESSAGES,
    WebSocketBridge,
    _SECRETS_MESSAGES,
)

INTEGRATION_PORT = 27710


async def _fleet(tmp_path):
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    for aid, pid in (("root", None), ("guest", "root"), ("other", "root")):
        await rt.registry.register_agent(AgentDefinition(
            id=aid, name=aid, mode=AgentMode.COGNITIVE, parent_id=pid, budgets={}))
    return rt


def _bridge(rt, **kw):
    b = WebSocketBridge(rt, host="127.0.0.1", port=0, **kw)
    b._integration_store = ws_auth.IntegrationTokenStore(rt._config.data_dir)
    return b


def _session(rec):
    return ws_auth.IntegrationSession(token_id=rec.token_id,
                                      token_hash=rec.token_hash,
                                      agent_id=rec.agent_id, label=rec.label)


_SCHEMA = {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Owner sentinels: the clamp can never produce an owner-trusted caller
# ---------------------------------------------------------------------------

def test_owner_sentinels_agree_with_is_owner_caller():
    # Every value is_owner_caller trusts as the owner...
    for s in (None, "", OWNER_ID, _LEGACY_ROOT_ID):
        assert is_owner_caller(s)
        # ...is never a bindable integration identity.
        assert not ws_auth.is_bindable_agent_id(s)
    # The literal set in ws_auth matches the agent_tools constants.
    assert ws_auth._OWNER_SENTINELS == {"", OWNER_ID, _LEGACY_ROOT_ID}
    # Whitespace-padded sentinels are refused too (no strip-then-trust).
    assert not ws_auth.is_bindable_agent_id(" user")
    assert ws_auth.is_bindable_agent_id("guest")


def test_store_refuses_sentinel_binding(tmp_path):
    store = ws_auth.IntegrationTokenStore(tmp_path)
    for s in ("", OWNER_ID, _LEGACY_ROOT_ID, None):
        with pytest.raises(ValueError):
            store.create(s, "x")
    assert store.list() == []


def test_store_hashes_and_revokes(tmp_path):
    store = ws_auth.IntegrationTokenStore(tmp_path)
    token, rec = store.create("guest", "odysseus")
    assert token.startswith(ws_auth.INTEGRATION_TOKEN_PREFIX)
    on_disk = (tmp_path / ws_auth.INTEGRATION_TOKENS_FILE).read_text()
    assert token not in on_disk                      # plaintext never stored
    assert ws_auth.hash_integration_token(token) in on_disk
    assert store.verify(token).agent_id == "guest"
    assert store.verify(token + "x") is None
    assert store.verify("") is None
    # A second store instance (the daemon) sees a revoke made by the CLI one.
    daemon_view = ws_auth.IntegrationTokenStore(tmp_path)
    assert daemon_view.verify(token) is not None
    assert [r.token_id for r in store.revoke("odysseus")] == [rec.token_id]
    assert daemon_view.verify(token) is None
    assert store.revoke("odysseus") == []            # already revoked


def test_store_refuses_hand_edited_sentinel(tmp_path):
    store = ws_auth.IntegrationTokenStore(tmp_path)
    token, _ = store.create("guest", "x")
    path = tmp_path / ws_auth.INTEGRATION_TOKENS_FILE
    raw = json.loads(path.read_text())
    raw["tokens"][0]["agent_id"] = OWNER_ID
    path.write_text(json.dumps(raw))
    assert ws_auth.IntegrationTokenStore(tmp_path).verify(token) is None


def test_parse_bearer():
    assert ws_auth.parse_bearer("Bearer atn_it_abc") == "atn_it_abc"
    assert ws_auth.parse_bearer("bearer  atn_it_abc ") == "atn_it_abc"
    assert ws_auth.parse_bearer("Basic atn_it_abc") == ""
    assert ws_auth.parse_bearer("Bearer something_else") == ""
    assert ws_auth.parse_bearer(None) == ""


# ---------------------------------------------------------------------------
# Deny-list / allow-list
# ---------------------------------------------------------------------------

def test_denylist_covers_owner_surfaces():
    assert KEY_LOCAL_ONLY_MESSAGES <= INTEGRATION_DENIED_MESSAGES
    assert _SECRETS_MESSAGES <= INTEGRATION_DENIED_MESSAGES
    for t in ("approve_adoption", "grant_tool", "set_owner_wallet",
              "export_agent_key", "set_budget", "set_credit_budget",
              "create_agent", "remove_agent", "pay_for_service",
              "register_agent_on_chain", "autonet_wallet_connect",
              "rotate_owner", "snapshot"):
        assert t in INTEGRATION_DENIED_MESSAGES, t
    assert not (INTEGRATION_ALLOWED_MESSAGES & INTEGRATION_DENIED_MESSAGES)


@pytest.mark.asyncio
async def test_owner_messages_denied(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt, integration_burst=10_000)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    for t in sorted(INTEGRATION_DENIED_MESSAGES):
        resp = await bridge._handle_integration_message(
            {"type": t, "msg_id": "1"}, sess)
        assert resp["ok"] is False and resp["code"] == "owner_only", t
    # Anything not on the allowlist is refused even if not explicitly denied.
    resp = await bridge._handle_integration_message(
        {"type": "economy_graph", "msg_id": "2"}, sess)
    assert resp["code"] == "not_allowed"


@pytest.mark.asyncio
async def test_use_tool_restricted_to_registered(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    # A core tool through use_tool would be an escalation: refused.
    resp = await bridge._handle_integration_message(
        {"type": "use_tool", "msg_id": "1", "name": "create_agent",
         "arguments": {"id": "x", "name": "x"}}, sess)
    assert resp["ok"] is False and resp["code"] == "not_allowed"
    assert rt.get_agent("x") is None


@pytest.mark.asyncio
async def test_register_tool_guest_origin(tmp_path):
    # Guest authoring is allowed ONLY as origin="integration" (always run on
    # the guest containment path, tests/atn/test_guest_tools.py). Connector-
    # backed manifests stay refused.
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    assert "register_tool" in INTEGRATION_ALLOWED_MESSAGES
    assert "register_tool" not in INTEGRATION_DENIED_MESSAGES
    resp = await bridge._handle_integration_message(
        {"type": "register_tool", "msg_id": "1", "name": "guest_x",
         "description": "x", "input_schema": _SCHEMA,
         "code": "print(1)", "connector_id": "gmail"}, sess)
    assert resp["ok"] is False and resp["code"] == "not_allowed", resp
    resp = await bridge._handle_integration_message(
        {"type": "register_tool", "msg_id": "2", "name": "guest_x",
         "description": "x", "input_schema": _SCHEMA,
         "code": "print(1)"}, sess)
    assert resp["ok"] is True, resp
    assert rt.tool_store.get(resp["result"]["digest"]).origin == "integration"


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_clamps_caller_to_bound_agent(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    # A forged private _caller_id (owner) is stripped: the call runs as the
    # bound agent, so publishing another agent's tool stays refused.
    tool = rt.tool_store.register(
        name="echo_other", description="echo", input_schema=_SCHEMA,
        author="other", code="print('hi')")
    resp = await bridge._handle_integration_message(
        {"type": "publish_tool", "msg_id": "1", "digest": tool["digest"],
         "_caller_id": OWNER_ID}, sess)
    assert resp["ok"] is False, resp
    assert rt.tool_store.resolve(tool["digest"]).published is False

    status = await bridge._handle_integration_message(
        {"type": "status", "msg_id": "3"}, sess)
    assert status["ok"] and status["result"]["agent_id"] == "guest"


@pytest.mark.asyncio
async def test_wrong_agent_refused(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    for key, val in (("caller_id", "other"), ("caller_id", OWNER_ID),
                     ("caller_id", _LEGACY_ROOT_ID), ("agent_id", "other"),
                     ("target", "root"), ("author", OWNER_ID)):
        resp = await bridge._handle_integration_message(
            {"type": "list_tools", "msg_id": "1", key: val}, sess)
        assert resp["ok"] is False and resp["code"] == "wrong_agent", (key, val)
    # Naming the bound agent itself is fine.
    resp = await bridge._handle_integration_message(
        {"type": "list_tools", "msg_id": "2", "caller_id": "guest"}, sess)
    assert resp["ok"] is True, resp

    # publish_tool on another agent's tool: refused by the author check.
    other_tool = rt.tool_store.register(
        name="other_tool", description="o", input_schema=_SCHEMA,
        author="other", code="print(1)")
    resp = await bridge._handle_integration_message(
        {"type": "publish_tool", "msg_id": "3",
         "digest": other_tool["digest"]}, sess)
    assert resp["ok"] is False


@pytest.mark.asyncio
async def test_token_for_missing_agent_unavailable(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    token, rec = bridge._integration_store.create("ghost", "g")
    assert bridge._integration_verify_header({"Authorization": f"Bearer {token}"}) is None
    resp = await bridge._handle_integration_message(
        {"type": "status", "msg_id": "1"}, _session(rec))
    assert resp["ok"] is False and resp["code"] == "agent_unavailable"


@pytest.mark.asyncio
async def test_revoked_token_mid_session(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    token, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    ok = await bridge._handle_integration_message({"type": "status"}, sess)
    assert ok["ok"] is True
    # Revoke from a separate store instance (the CLI process).
    ws_auth.IntegrationTokenStore(rt._config.data_dir).revoke(rec.token_id)
    resp = await bridge._handle_integration_message({"type": "status"}, sess)
    assert resp["ok"] is False and resp["code"] == "token_revoked"
    assert bridge._integration_verify_header({"Authorization": f"Bearer {token}"}) is None


@pytest.mark.asyncio
async def test_rate_limit_per_token(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt, integration_rate=0.0001, integration_burst=3)
    _, rec = bridge._integration_store.create("guest", "g")
    sess_a, sess_b = _session(rec), _session(rec)     # two connections, one token
    codes = []
    for s in (sess_a, sess_b, sess_a, sess_b):
        r = await bridge._handle_integration_message({"type": "status"}, s)
        codes.append(r.get("code") if not r["ok"] else "ok")
    assert codes == ["ok", "ok", "ok", "rate_limited"]
    # A different token has its own bucket.
    _, rec2 = bridge._integration_store.create("other", "o")
    r = await bridge._handle_integration_message({"type": "status"}, _session(rec2))
    assert r["ok"] is True


# ---------------------------------------------------------------------------
# Live socket
# ---------------------------------------------------------------------------

async def _recv(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


@pytest.mark.asyncio
async def test_live_integration_listener(tmp_path):
    rt = await _fleet(tmp_path)
    token, rec = ws_auth.IntegrationTokenStore(rt._config.data_dir).create("guest", "g")
    bridge = WebSocketBridge(rt, host="127.0.0.1", port=27720,
                             integration_host="127.0.0.1",
                             integration_port=INTEGRATION_PORT)
    await bridge.start()
    url = f"ws://127.0.0.1:{INTEGRATION_PORT}"
    try:
        # No token / wrong token: rejected before the upgrade.
        for headers in (None, {"Authorization": "Bearer atn_it_nope"}):
            with pytest.raises(websockets.InvalidStatus) as ei:
                async with websockets.connect(url, additional_headers=headers):
                    pass
            assert ei.value.response.status_code == 401

        async with websockets.connect(
                url, additional_headers={"Authorization": f"Bearer {token}"}) as ws:
            hello = await _recv(ws)
            assert hello["type"] == "integration_ready"
            assert hello["agent_id"] == "guest"
            # No snapshot, no event stream: the first frame is the hello.
            await ws.send(json.dumps({"type": "status", "msg_id": "1"}))
            r = await _recv(ws)
            assert r["ok"] and r["result"]["agent_id"] == "guest"
            await ws.send(json.dumps({"type": "export_agent_key", "msg_id": "2",
                                      "agent_id": "guest"}))
            r = await _recv(ws)
            assert r["ok"] is False and r["code"] == "owner_only"
            # Revoke while connected: next message is refused and the
            # server closes the socket.
            ws_auth.IntegrationTokenStore(rt._config.data_dir).revoke("g")
            await ws.send(json.dumps({"type": "status", "msg_id": "3"}))
            r = await _recv(ws)
            assert r["code"] == "token_revoked"
            with pytest.raises(websockets.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)

        with pytest.raises(websockets.InvalidStatus) as ei:
            async with websockets.connect(
                    url, additional_headers={"Authorization": f"Bearer {token}"}):
                pass
        assert ei.value.response.status_code == 401
    finally:
        await bridge.stop()


def test_cli_create_list_revoke(tmp_path, capsys):
    from atn.integration_token_cli import main
    dd = str(tmp_path)
    assert main(["--data-dir", dd, "create", "--agent", OWNER_ID]) == 2
    assert main(["--data-dir", dd, "create", "--agent", "guest",
                 "--label", "ody"]) == 0
    token = capsys.readouterr().out.strip()
    assert ws_auth.IntegrationTokenStore(tmp_path).verify(token).agent_id == "guest"
    assert main(["--data-dir", dd, "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["label"] == "ody" and "hash" not in rows[0]
    assert main(["--data-dir", dd, "revoke", "ody"]) == 0
    assert ws_auth.IntegrationTokenStore(tmp_path).verify(token) is None
    assert main(["--data-dir", dd, "revoke", "ody"]) == 1


# ---------------------------------------------------------------------------
# Public reads (find_services, tool_reviews, network_status)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_public_reads_need_no_grant_and_move_nothing(tmp_path, monkeypatch):
    from atn.ws_server import INTEGRATION_PUBLIC_READS
    assert INTEGRATION_PUBLIC_READS <= INTEGRATION_ALLOWED_MESSAGES
    # Money stays denied: browsing the market never opens paying.
    for t in ("pay_for_service", "request_service", "register_service"):
        assert t in INTEGRATION_DENIED_MESSAGES
    rt = await _fleet(tmp_path)
    # A guest whose bundle has no `services`: find_services still answers,
    # because it is not dispatched through execute_tool.
    rt.get_agent("guest").tools = ["unified_tools"]
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)

    seen = {}

    async def fake_find(runtime, args):
        seen.update(args)
        return {"services": [{"service_id": 1, "ask": 5}], "count": 1}

    import atn.agent_tools as agent_tools
    monkeypatch.setattr(agent_tools, "_find_services", fake_find)
    resp = await bridge._handle_integration_message(
        {"type": "find_services", "msg_id": "1", "query": "ocr",
         "_caller_id": OWNER_ID}, sess)
    assert resp["ok"] is True, resp
    assert resp["result"]["count"] == 1
    assert seen == {"query": "ocr", "limit": None}      # no _caller_id leaks in

    # Naming another identity is still refused before the read.
    resp = await bridge._handle_integration_message(
        {"type": "find_services", "msg_id": "2", "caller_id": "other"}, sess)
    assert resp["code"] == "wrong_agent"


@pytest.mark.asyncio
async def test_tool_reviews_scoped_to_visible_tools(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    mine = rt.tool_store.register(
        name="guest_tool", description="g", input_schema=_SCHEMA,
        author="guest", code="print(1)")
    theirs = rt.tool_store.register(
        name="other_private", description="o", input_schema=_SCHEMA,
        author="other", code="print(2)")
    resp = await bridge._handle_integration_message(
        {"type": "tool_reviews", "msg_id": "1", "digest": mine["digest"]}, sess)
    assert resp["ok"] is True, resp
    assert resp["result"]["digest"] == mine["digest"]
    assert resp["result"]["reviews"] == []
    # Another agent's private tool: its local review rows are not exposed.
    assert not rt.tool_store.allowed("guest", rt.tool_store.get(theirs["digest"]))
    resp = await bridge._handle_integration_message(
        {"type": "tool_reviews", "msg_id": "2", "digest": theirs["digest"]}, sess)
    assert resp["ok"] is False and resp["code"] == "unknown_tool"
    # A digest this daemon doesn't hold: public close state only.
    resp = await bridge._handle_integration_message(
        {"type": "tool_reviews", "msg_id": "3", "digest": "ab" * 32}, sess)
    assert resp["ok"] is True and resp["result"]["reviews"] == []
    resp = await bridge._handle_integration_message(
        {"type": "tool_reviews", "msg_id": "4"}, sess)
    assert resp["code"] == "bad_request"


@pytest.mark.asyncio
async def test_network_status_has_no_secrets(tmp_path):
    rt = await _fleet(tmp_path)
    rt._config.rpb.rpc_url = "https://rpc.example/v1/SECRETKEY"
    rt._config.rpb.chain_id = 127823
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    resp = await bridge._handle_integration_message(
        {"type": "network_status", "msg_id": "1"}, _session(rec))
    assert resp["ok"] is True, resp
    body = resp["result"]
    assert body["chain"]["chain_id"] == 127823
    assert body["chain"]["label"] == "Etherlink Shadownet (testnet)"
    assert body["chain"]["testnet"] is True
    assert "SECRETKEY" not in json.dumps(body)
    assert set(body) >= {"autonet", "chain", "p2p", "epoch", "epochs_closed"}


def test_remote_listener_env_overrides(monkeypatch):
    from types import SimpleNamespace

    from atn.cli import _remote_listener_settings
    an = SimpleNamespace(remote_ws_host="", remote_ws_port=7701, owner_wallet="")
    for k in ("ATN_REMOTE_WS_HOST", "ATN_REMOTE_WS_PORT", "ATN_OWNER_WALLET"):
        monkeypatch.delenv(k, raising=False)
    assert _remote_listener_settings(an) == ("", 7701, "")
    w = "0x" + "ab" * 20
    monkeypatch.setenv("ATN_REMOTE_WS_HOST", "0.0.0.0")
    monkeypatch.setenv("ATN_REMOTE_WS_PORT", "7801")
    monkeypatch.setenv("ATN_OWNER_WALLET", w)
    assert _remote_listener_settings(an) == ("0.0.0.0", 7801, w)
    # It also becomes the runtime's owner (earnings / claim identity).
    assert an.owner_wallet == w
    # The env never replaces an owner wallet the config already holds.
    cfg_w = "0x" + "cd" * 20
    an.owner_wallet = cfg_w
    assert _remote_listener_settings(an)[2] == cfg_w
    # A malformed env wallet is ignored, not trusted.
    an.owner_wallet = ""
    monkeypatch.setenv("ATN_OWNER_WALLET", "not-a-wallet")
    assert _remote_listener_settings(an)[2] == ""


@pytest.mark.asyncio
async def test_owner_tool_reviews_handler_unchanged(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    tool = rt.tool_store.register(
        name="owner_seen", description="o", input_schema=_SCHEMA,
        author="other", code="print(1)")
    owner = ws_auth.ClientSession(local=True, authed=True, owner=True,
                                  scope_ids=None)
    resp = await bridge._handle_message(
        {"type": "tool_reviews", "msg_id": "1", "digest": tool["digest"]}, owner)
    assert resp["ok"] is True, resp
    assert set(resp["result"]) == {"digest", "reviews", "position", "vetting", "usage"}


@pytest.mark.asyncio
async def test_list_tools_marks_the_bound_agents_own_tools(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("root", "r")
    mine = rt.tool_store.register(
        name="root_tool", description="r", input_schema=_SCHEMA,
        author="root", code="print(1)")
    child = rt.tool_store.register(
        name="guest_tool", description="g", input_schema=_SCHEMA,
        author="guest", code="print(2)")
    resp = await bridge._handle_integration_message(
        {"type": "list_tools", "msg_id": "1"}, _session(rec))
    assert resp["ok"] is True, resp
    by_digest = {t["digest"]: t for t in resp["result"]["tools"]}
    assert by_digest[mine["digest"]]["mine"] is True
    # Visible through lineage (root is guest's ancestor) but not authored.
    assert by_digest[child["digest"]]["mine"] is False
    assert by_digest[mine["digest"]]["published"] is False
