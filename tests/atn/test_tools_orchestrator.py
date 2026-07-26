"""Tests for orchestrator tools — delegate, message passing, agent CRUD.

Phase 3: delegates are cognitive agents executed through the unified Runtime.
Tests use a real Runtime with mocked BridgeProvider.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atn.agent_registry import DelegateRegistry, DelegateStatus
from atn.events import EventBus, EventType
from atn.models import (
    AgentDefinition,
    AgentMode,
    AgentStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    StepDefinition,
    StepType,
)
from atn.providers.base import ProviderResponse, ToolCall, Usage


def _make_runtime(bus: EventBus, tmp_path: Path):
    """Create a real Runtime for delegate tool testing."""
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False

    return Runtime(bus, data_dir=data_dir, config=config)


class TestDelegateSpawnAndCollect:
    """Tests for _delegate and _delegate_collect tool executors."""

    @pytest.mark.asyncio
    async def test_delegate_spawns_and_collects(self, tmp_path):
        bus = EventBus()
        events = []

        async def capture(e):
            events.append(e)

        bus.subscribe(None, capture)
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Found 3 files.",
            usage=Usage(input_tokens=500, output_tokens=200),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.orchestrator.tools import _create_agent, _delegate_collect

            result = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "Search for auth code",
                "agent_type": "explore",
                "name": "Auth search",
                "_caller_id": "orch",
            })
            assert result["status"] == "running"
            agent_id = result["agent_id"]

            collect = await _delegate_collect(rt, {"agent_id": agent_id})

        assert collect["status"] == "completed"
        assert "Found 3 files" in collect["result"]
        assert collect["usage"]["input_tokens"] == 500

        event_types = [e.type for e in events]
        assert EventType.DELEGATE_SPAWNED in event_types
        assert EventType.EXECUTION_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_delegate_failure_captured(self, tmp_path):
        bus = EventBus()
        rt = _make_runtime(bus, tmp_path)

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(side_effect=RuntimeError("crash"))
        mock_provider.close = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.orchestrator.tools import _create_agent, _delegate_collect

            spawn = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "Will fail", "agent_type": "implement",
                "_caller_id": "orch",
            })
            assert spawn["status"] == "running"

            collect = await _delegate_collect(rt, {"agent_id": spawn["agent_id"]})

        assert collect["status"] == "failed"
        assert "crash" in collect["error"]
        node = rt.delegate_registry.get_node(spawn["agent_id"])
        assert node.status == DelegateStatus.FAILED


class TestDelegateStatus:
    """Tests for the _delegate_status tool executor."""

    @pytest.mark.asyncio
    async def test_status_while_running(self, tmp_path):
        bus = EventBus()
        rt = _make_runtime(bus, tmp_path)

        proceed = asyncio.Event()
        mock_response = ProviderResponse(
            text="Done.", usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn", model="sonnet",
        )

        async def slow_orchestrate(**kwargs):
            await proceed.wait()
            return mock_response

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = slow_orchestrate
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.orchestrator.tools import _create_agent, _delegate_status, _delegate_collect

            spawn = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "Slow task", "agent_type": "implement",
                "_caller_id": "orch",
            })
            await asyncio.sleep(0.05)

            status = await _delegate_status(rt, {"agent_id": spawn["agent_id"]})
            assert status["status"] == "running"

            proceed.set()
            collect = await _delegate_collect(rt, {"agent_id": spawn["agent_id"]})
            assert collect["status"] == "completed"


class TestMessageTool:
    """Tests for post_message tool executor."""

    @pytest.mark.asyncio
    async def test_post_message_to_known_agent(self, tmp_path):
        bus = EventBus()
        rt = _make_runtime(bus, tmp_path)

        # Register a target agent
        defn = AgentDefinition(id="target-1", name="Target")
        await rt.register_agent(defn)

        from atn.orchestrator.tools import _post_message

        result = await _post_message(rt, {
            "target": "target-1",
            "message": "I found something important",
            "priority": "normal",
        })
        assert "message_id" in result
        assert result["target"] == "target-1"

    @pytest.mark.asyncio
    async def test_post_message_to_unknown_agent(self, tmp_path):
        bus = EventBus()
        rt = _make_runtime(bus, tmp_path)

        from atn.orchestrator.tools import _post_message

        result = await _post_message(rt, {
            "target": "nonexistent",
            "message": "hello",
        })
        assert "error" in result


class TestGetSnapshotTool:
    """Tests for the get_snapshot tool executor."""

    @pytest.mark.asyncio
    async def test_get_snapshot_returns_dict(self, tmp_path):
        bus = EventBus()
        rt = _make_runtime(bus, tmp_path)

        from atn.orchestrator.tools import _get_snapshot

        result = await _get_snapshot(rt, {})
        assert isinstance(result, dict)
        assert "agents" in result


# ---------------------------------------------------------------------------
# Marketplace inference binding — PARENT-ONLY authority
# (docs/services_market.md, ratified 2026-07-26: employer-chooses-the-tool)
# ---------------------------------------------------------------------------

_SVC_PROVIDER_ADDR = "0x1111111111111111111111111111111111111111"
_SVC_DIGEST = "ab" * 32
_SVC_BINDING = {"provider_address": _SVC_PROVIDER_ADDR,
                "spec_digest": _SVC_DIGEST}


class TestServiceProviderBindingAuthority:
    """An agent's substrate is its PARENT's choice. A parent may bind a child
    to a marketplace inference service it bought; an agent may never set or
    change its own binding, and never a stranger's.

    _update_agent has no blanket lineage gate, so this field carries its own
    (same shape as `budgets`). These tests are the guard on that gate.
    """

    async def _fleet(self, tmp_path):
        """parent-1 with child-1; sibling-1 under a different parent."""
        from atn.orchestrator.tools import execute_tool

        rt = _make_runtime(EventBus(), tmp_path)
        for aid, parent in (("parent-1", None), ("child-1", "parent-1"),
                            ("other-1", None), ("sibling-1", "other-1")):
            await rt.register_agent(AgentDefinition(
                id=aid, name=aid, mode=AgentMode.COGNITIVE,
                cognitive_model="sonnet", parent_id=parent,
                budgets={"claude_max": 100000},
            ))
        return rt, execute_tool

    @pytest.mark.asyncio
    async def test_parent_can_bind_its_child(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="parent-1")
        assert "error" not in res, res
        assert "service_provider" in res["changed"]
        assert rt.get_agent("child-1").service_provider == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_agent_cannot_bind_itself(self, tmp_path):
        """The core rule: no self-switching surface at all."""
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="child-1")
        assert "error" in res
        assert "its own" in res["error"]
        assert rt.get_agent("child-1").service_provider is None

    @pytest.mark.asyncio
    async def test_agent_cannot_unbind_itself(self, tmp_path):
        """Clearing is a change too — a bound child must not escape upward."""
        rt, execute_tool = await self._fleet(tmp_path)
        rt.get_agent("child-1").service_provider = dict(_SVC_BINDING)
        res = await execute_tool(
            "update_agent", {"agent_id": "child-1", "service_provider": None},
            rt, caller_id="child-1")
        assert "error" in res
        assert rt.get_agent("child-1").service_provider == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_non_parent_agent_cannot_bind(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="sibling-1")
        assert "error" in res
        assert "not the parent" in res["error"]
        assert rt.get_agent("child-1").service_provider is None

    @pytest.mark.asyncio
    async def test_child_cannot_bind_its_parent(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "parent-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="child-1")
        assert "error" in res
        assert rt.get_agent("parent-1").service_provider is None

    @pytest.mark.asyncio
    async def test_owner_may_bind_anyone(self, tmp_path):
        """caller_id None = the human owner surface, unconstrained."""
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "sibling-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id=None)
        assert "error" not in res, res
        assert rt.get_agent("sibling-1").service_provider == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_parent_can_unbind_its_child(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        rt.get_agent("child-1").service_provider = dict(_SVC_BINDING)
        res = await execute_tool(
            "update_agent", {"agent_id": "child-1", "service_provider": None},
            rt, caller_id="parent-1")
        assert "error" not in res, res
        assert rt.get_agent("child-1").service_provider is None

    @pytest.mark.asyncio
    async def test_malformed_binding_refused_with_a_clear_error(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "update_agent",
            {"agent_id": "child-1",
             "service_provider": {"provider_address": _SVC_PROVIDER_ADDR}},
            rt, caller_id="parent-1")
        assert "error" in res
        assert "spec_digest" in res["error"]
        assert rt.get_agent("child-1").service_provider is None

    @pytest.mark.asyncio
    async def test_binding_change_evicts_the_cached_provider(self, tmp_path):
        """The substrate AND the paying wallet change, so a cached instance
        would keep buying from the old seller on the old key."""
        rt, execute_tool = await self._fleet(tmp_path)
        stale = AsyncMock()
        stale.close = AsyncMock()
        rt.providers._active_providers["child-1"] = stale
        await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="parent-1")
        assert "child-1" not in rt.providers._active_providers
        stale.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_binding_persists_to_yaml(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="parent-1")
        import yaml
        raw = yaml.safe_load(
            (tmp_path / "agents" / "child-1" / "agent.yaml").read_text(
                encoding="utf-8"))
        assert raw["service_provider"] == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_create_agent_binds_the_new_child(self, tmp_path):
        """create_agent needs no authority check: parent_id is derived from the
        caller, so the created agent is the caller's child by construction."""
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "create_agent",
            {"id": "bought-1", "name": "bought-1",
             "budgets": {"claude_max": 10000},
             "service_provider": dict(_SVC_BINDING)},
            rt, caller_id="parent-1")
        assert "error" not in res, res
        defn = rt.get_agent(res["agent_id"])
        assert defn.service_provider == _SVC_BINDING
        assert defn.parent_id == "parent-1"
        assert res["service_provider"] == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_create_agent_refuses_a_malformed_binding(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        res = await execute_tool(
            "create_agent",
            {"id": "bad-1", "name": "bad-1",
             "budgets": {"claude_max": 10000},
             "service_provider": {"spec_digest": _SVC_DIGEST}},
            rt, caller_id="parent-1")
        assert "error" in res
        assert rt.get_agent("bad-1") is None

    @pytest.mark.asyncio
    async def test_get_agent_surfaces_the_binding(self, tmp_path):
        rt, execute_tool = await self._fleet(tmp_path)
        rt.get_agent("child-1").service_provider = dict(_SVC_BINDING)
        res = await execute_tool("get_agent", {"agent_id": "child-1"},
                                 rt, caller_id="parent-1")
        assert res["service_provider"] == _SVC_BINDING

    @pytest.mark.asyncio
    async def test_snapshot_surfaces_the_binding(self, tmp_path):
        rt, _ = await self._fleet(tmp_path)
        rt.get_agent("child-1").service_provider = dict(_SVC_BINDING)
        snap = rt.snapshot()
        assert snap["agents"]["child-1"]["service_provider"] == _SVC_BINDING
        assert "service_provider" not in snap["agents"]["parent-1"]
