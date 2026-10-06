"""There is no privileged root agent: any number of top-level agents, each
with its own conversation, and every message names its agent.

Covers the daemon side of that model:
  - the per-daemon auth identity is persisted, not derived from the fleet;
  - an unscoped owner session is None, not an agent-like id;
  - per-agent session APIs require an agent id (no first-parentless guess);
  - a pipeline agent's cognitive step records into ITS OWN conversation;
  - the one-time migrations of pre-rename on-disk data.
"""
from __future__ import annotations

import time

import pytest
import yaml
from eth_account import Account
from eth_account.messages import encode_defunct

from atn import ws_auth
from atn.config import ATNConfig, _migrate_legacy_orchestrator_config, load_config
from atn.conversation import ConversationStore
from atn.events import EventBus
from atn.loader import load_agent_file
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.ws_auth import ClientSession
from atn.ws_server import WebSocketBridge


def _runtime(tmp_path, owner_wallet: str = "") -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id: str, parent_id: str | None = None) -> AgentDefinition:
    return AgentDefinition(id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
                           parent_id=parent_id, budgets={})


def _local_owner() -> ClientSession:
    return ClientSession(local=True, authed=True, owner=True, scope_ids=None)


# ---------------------------------------------------------------------------
# Per-daemon identity + unscoped sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daemon_id_is_persisted_and_independent_of_the_fleet(tmp_path):
    rt = _runtime(tmp_path)
    first = WebSocketBridge(rt)._daemon_id()
    assert first and first.startswith("atn-daemon-")
    # Registering agents (in any order) never changes it...
    await rt.registry.register_agent(_agent("zeta"))
    await rt.registry.register_agent(_agent("alpha"))
    # ...and a fresh bridge on the same data dir reads the same id back.
    assert WebSocketBridge(rt)._daemon_id() == first
    assert (tmp_path / "data" / ws_auth.DAEMON_ID_FILE).read_text().strip() == first


@pytest.mark.asyncio
async def test_owner_without_root_gets_an_unscoped_session(tmp_path):
    acct = Account.create()
    rt = _runtime(tmp_path, owner_wallet=acct.address)
    await rt.registry.register_agent(_agent("alpha"))
    await rt.registry.register_agent(_agent("beta"))
    bridge = WebSocketBridge(rt, owner_wallet=acct.address)
    s = ClientSession(local=False, conn_id="c1", nonce=ws_auth.new_nonce(),
                      nonce_issued_at=time.time())
    challenge = ws_auth.build_challenge_text(
        s.nonce, daemon_id=bridge._daemon_id(),
        chain_id=int(rt._config.autonet.chain_id or 0),
        owner_wallet=acct.address, conn_id=s.conn_id,
        issued_at=s.nonce_issued_at)
    sig = acct.sign_message(encode_defunct(text=challenge)).signature.hex()
    resp = await bridge._handle_message(
        {"type": "auth_response", "signature": sig, "msg_id": "1"}, s)
    assert resp["ok"] is True and resp["owner"] is True
    assert resp["root"] is None
    assert s.root_agent_id is None and s.scope_ids is None
    status = await bridge._handle_message({"type": "auth_status", "msg_id": "2"}, s)
    assert status["result"]["root"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("msg_type", ["session_stats", "session_context",
                                      "context_breakdown"])
async def test_per_agent_session_apis_require_an_agent_id(tmp_path, msg_type):
    rt = _runtime(tmp_path)
    await rt.registry.register_agent(_agent("alpha"))
    bridge = WebSocketBridge(rt)
    resp = await bridge._handle_message({"type": msg_type, "msg_id": "1"},
                                        _local_owner())
    assert resp["ok"] is False
    assert "agent_id" in resp["error"]


@pytest.mark.asyncio
async def test_two_top_level_agents_keep_separate_conversations(tmp_path):
    rt = _runtime(tmp_path)
    await rt.registry.register_agent(_agent("alpha"))
    await rt.registry.register_agent(_agent("beta"))
    rt.get_agent_conversation_store("alpha").add_user_turn("hello alpha")
    rt.get_agent_conversation_store("beta").add_user_turn("hello beta")
    bridge = WebSocketBridge(rt)
    a = await bridge._handle_message(
        {"type": "get_agent_conversation", "agent_id": "alpha", "msg_id": "1"},
        _local_owner())
    b = await bridge._handle_message(
        {"type": "get_agent_conversation", "agent_id": "beta", "msg_id": "2"},
        _local_owner())
    assert [t["content"] for t in a["result"]["turns"]] == ["hello alpha"]
    assert [t["content"] for t in b["result"]["turns"]] == ["hello beta"]
    # No shared conversation store exists on the runtime.
    assert not hasattr(rt, "conversation")


# ---------------------------------------------------------------------------
# One-time migrations of pre-rename on-disk data
# ---------------------------------------------------------------------------

def test_migrate_legacy_orchestrator_config_moves_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# keep me\n"
        "orchestrator:\n"
        "  model: claude-sonnet-4-6  # inline comment\n"
        "  provider: claude_max\n"
        "chat:\n"
        "  orchestrator_label: HQ\n",
        encoding="utf-8")
    assert _migrate_legacy_orchestrator_config(path) is True
    text = path.read_text(encoding="utf-8")
    assert "orchestrator" not in text
    assert "# keep me" in text and "# inline comment" in text
    data = yaml.safe_load(text)
    assert data["defaults"] == {"model": "claude-sonnet-4-6", "provider": "claude_max"}
    assert data["chat"]["root_label"] == "HQ"
    # Idempotent: a second load is a no-op, and load_config reads the result.
    assert _migrate_legacy_orchestrator_config(path) is False
    cfg = load_config(path)
    assert cfg.default_model == "claude-sonnet-4-6"
    assert cfg.default_provider == "claude_max"
    assert cfg.chat.root_label == "HQ"
    assert cfg.chat.bound_agent == ""


def test_migrate_legacy_orchestrator_config_defaults_section_wins(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("orchestrator:\n  model: old\ndefaults:\n  model: new\n",
                    encoding="utf-8")
    assert _migrate_legacy_orchestrator_config(path) is True
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {"defaults": {"model": "new"}}


def test_migrate_legacy_orchestrator_agent_yaml(tmp_path):
    agents = tmp_path / "agents"
    (agents / "orchestrator").mkdir(parents=True)
    (agents / "orchestrator" / "agent.yaml").write_text(
        yaml.safe_dump({"id": "orchestrator", "name": "Lead", "mode": "cognitive"}),
        encoding="utf-8")
    (agents / "kid").mkdir()
    kid = agents / "kid" / "agent.yaml"
    kid.write_text(yaml.safe_dump({
        "id": "kid", "name": "Kid", "parent_id": "orch",
        "steps": [{"type": "cognitive",
                   "config": {"prompt": "x", "tool_executors": "orchestrator"}}],
    }), encoding="utf-8")
    defn, errors = load_agent_file(kid)
    assert not errors and defn is not None
    # The short alias resolves to the sibling agent that really exists on disk.
    assert defn.parent_id == "orchestrator"
    assert defn.steps[0].config["tool_executors"] == "atn"
    # Persisted, so the migration runs once.
    on_disk = yaml.safe_load(kid.read_text(encoding="utf-8"))
    assert on_disk["parent_id"] == "orchestrator"
    assert on_disk["steps"][0]["config"]["tool_executors"] == "atn"


def test_migrate_legacy_orchestrator_agent_yaml_without_sibling(tmp_path):
    agents = tmp_path / "agents"
    (agents / "kid").mkdir(parents=True)
    kid = agents / "kid" / "agent.yaml"
    kid.write_text(yaml.safe_dump({"id": "kid", "name": "Kid", "mode": "cognitive",
                                  "parent_id": "orch"}),
                   encoding="utf-8")
    defn, errors = load_agent_file(kid)
    assert not errors and defn is not None
    assert defn.parent_id is None          # becomes a top-level agent
    assert "parent_id" not in yaml.safe_load(kid.read_text(encoding="utf-8"))


def test_migrate_legacy_orchestrator_label_dump_in_history(tmp_path):
    import json
    from datetime import datetime, timezone
    store_dir = tmp_path / "agent" / "conversations"
    store_dir.mkdir(parents=True)
    ts = datetime.now(timezone.utc).isoformat()
    # No "System:"/"User:" prefix and no "\nAssistant:": only the legacy
    # assistant label marks this turn as a history dump.
    dump = "Earlier context\nOrchestrator: reply\nUser: the real ask"
    (store_dir / "active.jsonl").write_text(
        json.dumps({"role": "user", "content": dump, "timestamp": ts}) + "\n",
        encoding="utf-8")
    store = ConversationStore(tmp_path / "agent")
    assert [t.content for t in store.get_turns()] == ["the real ask"]
