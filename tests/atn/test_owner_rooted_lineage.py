"""Owner-rooted lineage seeding (docs/tool_substrate.md, Owner-rooted
registration): parentless agents hash their lineage from the owner
wallet when the daemon knows it — fleets hang from a human 0x, never an
installation. Child lineage chains from the parent as always; unknown
owner degrades to the empty root (pre-wallet behavior, byte-identical).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.agent_identity import compute_lineage_hash, generate_agent_identity
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode

OWNER = "0x00000000000000000000000000000000deadbeef"


def _make_runtime(tmp_path: Path, owner_wallet: str = ""):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _defn(agent_id, parent_id=None):
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5", parent_id=parent_id,
    )


def test_generate_identity_owner_root_unit():
    identity, _ = generate_agent_identity(
        "top", system_prompt="p", owner_root=OWNER)
    assert identity.parent_lineage_hash == OWNER
    assert identity.lineage_hash == compute_lineage_hash(
        "top", OWNER, "p", identity.public_key)

    # No owner known -> empty root, pre-wallet behavior.
    bare, _ = generate_agent_identity("top2", system_prompt="p")
    assert bare.parent_lineage_hash is None


@pytest.mark.asyncio
async def test_parentless_agents_root_in_owner_wallet(tmp_path):
    rt = _make_runtime(tmp_path, owner_wallet=OWNER)
    await rt.register_agent(_defn("top"))
    await rt.register_agent(_defn("kid", parent_id="top"))

    top = rt.get_agent("top").identity
    kid = rt.get_agent("kid").identity
    # Top-level roots in the human wallet...
    assert top.parent_lineage_hash == OWNER
    # ...children chain from their parent's lineage, not the wallet.
    assert kid.parent_lineage_hash == top.lineage_hash


@pytest.mark.asyncio
async def test_no_owner_wallet_keeps_empty_root(tmp_path):
    rt = _make_runtime(tmp_path)  # owner_wallet unset
    await rt.register_agent(_defn("top"))
    assert rt.get_agent("top").identity.parent_lineage_hash is None
