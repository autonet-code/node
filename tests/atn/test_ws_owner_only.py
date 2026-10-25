"""Owner rotation is for the owner's full-fleet session only (2026-10-26
survey, seam report finding 2)."""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.runtime import Runtime
from atn.ws_auth import ClientSession
from atn.ws_server import WebSocketBridge


def _bridge(tmp_path):
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir()
    agents_dir.mkdir()
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return WebSocketBridge(Runtime(EventBus(), data_dir=data_dir, config=config))


@pytest.mark.asyncio
@pytest.mark.parametrize("msg_type", ["rotate_owner", "owner_binding_status"])
async def test_scoped_or_agent_sessions_cannot_rotate_owners(tmp_path, msg_type):
    bridge = _bridge(tmp_path)
    agent_session = ClientSession(local=False, authed=True, owner=False,
                                  scope_ids={"a", "a.1"})
    out = await bridge._handle_message({"msg_id": "1", "type": msg_type,
                                        "bindings": []}, agent_session)
    assert out["ok"] is False and "owner" in out["error"]
    scoped_owner = ClientSession(local=False, authed=True, owner=True,
                                 scope_ids={"a"})
    out = await bridge._handle_message({"msg_id": "2", "type": msg_type,
                                        "bindings": []}, scoped_owner)
    assert out["ok"] is False and "full-fleet" in out["error"]


@pytest.mark.asyncio
async def test_the_owners_full_fleet_session_reaches_the_handler(tmp_path):
    bridge = _bridge(tmp_path)
    owner = ClientSession(local=True, authed=True, owner=True, scope_ids=None)
    out = await bridge._handle_message({"msg_id": "1", "type": "rotate_owner",
                                        "bindings": []}, owner)
    assert "full-fleet session only" not in str(out.get("error", ""))
