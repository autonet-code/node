"""Scoped sessions are default-deny on the inline WS surface (2026-10-26
survey: about 150 inline handlers ran for any authed session, so a
delegated remote session could restart the daemon, configure providers,
move budgets, spend the owner's wallet...).

A scoped session (scope_ids set) reaches only SCOPED_INLINE_ALLOWED among
the inline handlers, each clamped to its subtree; agent tools still fall
through to execute_tool's per-target check."""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.runtime import Runtime
from atn.ws_auth import ClientSession
from atn.ws_server import (
    SCOPED_INLINE_ALLOWED,
    SCOPED_REQUIRES_AGENT,
    WebSocketBridge,
)


def _bridge(tmp_path):
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir()
    agents_dir.mkdir()
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return WebSocketBridge(Runtime(EventBus(), data_dir=data_dir, config=config))


def _scoped():
    return ClientSession(local=False, authed=True, owner=True,
                         scope_ids={"a", "a.1"}, root_agent_id="a")


DANGEROUS = [
    "daemon_restart", "provider_configure", "custom_provider_add",
    "oauth_start", "set_budget", "update_sponsor_budget", "set_owner_wallet",
    "autonet_stop", "autonet_set_chain", "register_service", "invoke_service",
    "claude_session_import", "robot_export_packs", "sanctions",
    "voice_start", "skip_onboarding", "grant_tool", "set_local_model",
    "remove_agent", "clone_agent", "export_agent_key", "secrets_put",
]


def test_every_allowed_message_is_a_real_inline_handler():
    inline = WebSocketBridge._inline_message_types()
    assert SCOPED_INLINE_ALLOWED <= inline
    assert SCOPED_REQUIRES_AGENT <= SCOPED_INLINE_ALLOWED
    # The dangerous ones are inline and not allowed.
    for t in DANGEROUS:
        assert t in inline and t not in SCOPED_INLINE_ALLOWED, t


@pytest.mark.asyncio
@pytest.mark.parametrize("msg_type", DANGEROUS)
async def test_scoped_session_is_refused_owner_surfaces(tmp_path, msg_type):
    bridge = _bridge(tmp_path)
    out = await bridge._handle_message(
        {"msg_id": "1", "type": msg_type, "agent_id": "a"}, _scoped())
    assert out["ok"] is False
    assert "scoped session" in out["error"] or "localhost-only" in out["error"]


@pytest.mark.asyncio
async def test_allowed_messages_are_clamped_to_the_subtree(tmp_path):
    bridge = _bridge(tmp_path)
    s = _scoped()
    # Fleet-wide (no agent named) is refused for agent-bound messages.
    out = await bridge._handle_message(
        {"msg_id": "1", "type": "agent_processes_list"}, s)
    assert out["ok"] is False and "agent_id" in out["error"]
    # An agent outside the subtree is refused.
    bridge.runtime.registry._agents["b"] = object()
    out = await bridge._handle_message(
        {"msg_id": "2", "type": "agent_processes_list", "agent_id": "b"}, s)
    assert out["ok"] is False and "outside the authorized subtree" in out["error"]


@pytest.mark.asyncio
async def test_full_fleet_owner_is_unaffected(tmp_path):
    bridge = _bridge(tmp_path)
    owner = ClientSession(local=True, authed=True, owner=True, scope_ids=None)
    out = await bridge._handle_message(
        {"msg_id": "1", "type": "agent_processes_list"}, owner)
    assert out["ok"] is True
