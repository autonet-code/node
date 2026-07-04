"""Agent cloning — human-only conversation branching (agent_clone.py)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import atn.agent_clone as agent_clone
from atn.agent_clone import clone_agent, merge_clone, _watch_merge
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode, HeartbeatConfig
from atn.runtime.agent_registry import AgentStatus


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def _register_original(rt, agent_id="orig", **over):
    defn = AgentDefinition(
        id=agent_id,
        name="Original",
        mode=AgentMode.COGNITIVE,
        system_prompt="You are the original.",
        cognitive_model="claude-sonnet-5",
        tools=["atn_core"],
        heartbeat=HeartbeatConfig(interval="5m"),
        **over,
    )
    await rt.register_agent(defn)
    store = rt.get_agent_conversation_store(agent_id)
    store.add_user_turn("hello")
    store.add_assistant_turn("hi, I remember things")
    return defn


class TestClone:
    @pytest.mark.asyncio
    async def test_clone_copies_definition_and_history(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)

        res = await clone_agent(rt, "orig")
        assert res["status"] == "cloned"
        clone_id = res["agent_id"]
        assert clone_id == "orig-clone"
        assert res["cloned_from"] == "orig"
        assert res["turns_forked"] == 2

        cdefn = rt.get_agent(clone_id)
        assert cdefn.system_prompt == "You are the original."
        assert cdefn.cognitive_model == "claude-sonnet-5"
        assert cdefn.tools == ["atn_core"]
        assert cdefn.cloned_from == "orig"
        assert cdefn.parent_id == "orig"          # budget rollup chain
        assert cdefn.notify_parent is False
        assert cdefn.heartbeat is None            # autonomy stripped
        assert cdefn.schedule is None
        assert rt.get_status(clone_id) == AgentStatus.ACTIVE

        turns = rt.get_agent_conversation_store(clone_id).get_turns()
        assert [t.content for t in turns] == ["hello", "hi, I remember things"]

    @pytest.mark.asyncio
    async def test_clone_gets_fresh_identity(self, tmp_path):
        rt = _make_runtime(tmp_path)
        odefn = await _register_original(rt)
        res = await clone_agent(rt, "orig")
        cdefn = rt.get_agent(res["agent_id"])
        # Fresh keypair minted at registration, nothing inherited.
        assert cdefn.identity is not None
        assert cdefn.identity.address != odefn.identity.address
        assert cdefn.identity.registered_on_chain is False

    @pytest.mark.asyncio
    async def test_clone_ids_are_unique(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        first = await clone_agent(rt, "orig")
        second = await clone_agent(rt, "orig")
        assert first["agent_id"] == "orig-clone"
        assert second["agent_id"] == "orig-clone2"

    @pytest.mark.asyncio
    async def test_pipeline_agent_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        defn = AgentDefinition(id="pipe", name="Pipe", mode=AgentMode.PIPELINE)
        await rt.register_agent(defn)
        res = await clone_agent(rt, "pipe")
        assert "error" in res

    @pytest.mark.asyncio
    async def test_missing_agent_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = await clone_agent(rt, "ghost")
        assert "error" in res

    @pytest.mark.asyncio
    async def test_clone_spend_rolls_up_to_original(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        res = await clone_agent(rt, "orig")
        clone_id = res["agent_id"]

        rt.registry.record_token_usage(clone_id, "anthropic", 1_000)
        # Ancestor rollup: the original's counter carries the clone's spend.
        assert rt.registry._budget_used["orig"]["anthropic"] == 1_000

    @pytest.mark.asyncio
    async def test_clone_definition_persisted(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        res = await clone_agent(rt, "orig")
        yaml_path = rt._config.agents_dir / res["agent_id"] / "agent.yaml"
        assert yaml_path.exists()
        assert "cloned_from: orig" in yaml_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_cloned_from_survives_loader_roundtrip(self, tmp_path):
        from atn.loader import load_agent_file, save_agent
        defn = AgentDefinition(
            id="c1", name="C", mode=AgentMode.COGNITIVE, cloned_from="orig",
        )
        save_agent(defn, tmp_path)
        loaded, errors = load_agent_file(tmp_path / "c1")
        assert not errors
        assert loaded.cloned_from == "orig"


class TestMerge:
    @pytest.mark.asyncio
    async def test_merge_rejects_non_clone(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        res = await merge_clone(rt, "orig")
        assert "error" in res and "not a clone" in res["error"]

    @pytest.mark.asyncio
    async def test_merge_requests_brief_from_clone(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        cres = await clone_agent(rt, "orig")
        clone_id = cres["agent_id"]

        rt.send_agent_message = AsyncMock(return_value={"status": "ok"})
        res = await merge_clone(rt, clone_id)
        assert res["status"] == "merging"
        assert res["original"] == "orig"
        target, instruction = rt.send_agent_message.call_args[0]
        assert target == clone_id
        assert "MERGE-BACK" in instruction and "orig" in instruction
        # Let the watcher task run its (fast-failing) course quietly.
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_watch_merge_delivers_brief_and_retires(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path)
        await _register_original(rt)
        cres = await clone_agent(rt, "orig")
        clone_id = cres["agent_id"]

        # The clone "wrote" its brief as its final assistant turn.
        rt.get_agent_conversation_store(clone_id).add_assistant_turn("THE BRIEF")

        monkeypatch.setattr(agent_clone, "_MERGE_START_TIMEOUT_S", 0.0)
        monkeypatch.setattr(agent_clone, "_MERGE_FINISH_TIMEOUT_S", 0.0)
        monkeypatch.setattr(agent_clone, "_MERGE_POLL_S", 0.01)
        rt.send_agent_message = AsyncMock(return_value={"status": "ok"})

        await _watch_merge(rt, clone_id, "orig")

        target, delivery = rt.send_agent_message.call_args[0]
        assert target == "orig"
        assert "SIDEQUEST BRIEF" in delivery and "THE BRIEF" in delivery
        assert rt.get_status(clone_id) == AgentStatus.STOPPED
