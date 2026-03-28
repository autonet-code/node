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
