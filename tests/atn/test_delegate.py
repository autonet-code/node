"""Tests for the delegate sub-agent system.

Tests the DelegateRegistry, event types, and delegate tool executor
with a mocked bridge provider.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atn.agent_registry import DelegateNode, DelegateRegistry, DelegateStatus
from atn.delegate_prompts import build_delegate_prompt
from atn.events import EventBus, EventType


# ---------------------------------------------------------------------------
# DelegateRegistry unit tests
# ---------------------------------------------------------------------------

class TestDelegateRegistry:
    def test_generate_child_id(self):
        reg = DelegateRegistry()
        assert reg.generate_child_id("orch") == "orch.1"
        assert reg.generate_child_id("orch") == "orch.2"
        assert reg.generate_child_id("orch.1") == "orch.1.1"
        assert reg.generate_child_id("orch.1") == "orch.1.2"
        assert reg.generate_child_id("orch") == "orch.3"

    def test_register_and_get(self):
        reg = DelegateRegistry()
        node = reg.register("orch.1", "orch", "explore", "Find auth code", "Find auth")
        assert node.agent_id == "orch.1"
        assert node.parent_id == "orch"
        assert node.agent_type == "explore"
        assert node.status == DelegateStatus.PENDING

        retrieved = reg.get_node("orch.1")
        assert retrieved is node

        assert reg.get_node("nonexistent") is None

    def test_update_status(self):
        reg = DelegateRegistry()
        reg.register("orch.1", "orch", "implement", "Write code", "Code")

        node = reg.update_status(
            "orch.1", DelegateStatus.COMPLETED,
            result_preview="Done!",
            tokens_used=5000,
        )
        assert node is not None
        assert node.status == DelegateStatus.COMPLETED
        assert node.result_preview == "Done!"
        assert node.tokens_used == 5000
        assert node.completed_at is not None

    def test_update_status_unknown(self):
        reg = DelegateRegistry()
        assert reg.update_status("nope", DelegateStatus.RUNNING) is None

    def test_children_and_descendants(self):
        reg = DelegateRegistry()
        reg.register("orch.1", "orch", "implement", "Task 1", "T1")
        reg.register("orch.2", "orch", "explore", "Task 2", "T2")
        reg.register("orch.1.1", "orch.1", "debug", "Subtask", "S1")
        reg.register("orch.1.2", "orch.1", "review", "Subtask 2", "S2")

        children = reg.get_children("orch")
        assert len(children) == 2
        assert {c.agent_id for c in children} == {"orch.1", "orch.2"}

        children_of_1 = reg.get_children("orch.1")
        assert len(children_of_1) == 2

        descendants = reg.get_descendants("orch")
        assert len(descendants) == 4

        descendants_of_1 = reg.get_descendants("orch.1")
        assert len(descendants_of_1) == 2

    def test_get_active(self):
        reg = DelegateRegistry()
        reg.register("orch.1", "orch", "implement", "Task", "T")
        reg.register("orch.2", "orch", "explore", "Task 2", "T2")
        reg.update_status("orch.1", DelegateStatus.RUNNING)
        reg.update_status("orch.2", DelegateStatus.COMPLETED)

        active = reg.get_active()
        assert len(active) == 1
        assert active[0].agent_id == "orch.1"

    def test_get_tree(self):
        reg = DelegateRegistry()
        reg.register("orch.1", "orch", "implement", "Task", "T")
        tree = reg.get_tree()
        assert tree["total_count"] == 1
        assert tree["active_count"] == 1  # PENDING counts as active
        assert len(tree["nodes"]) == 1
        assert tree["nodes"][0]["agent_id"] == "orch.1"

    def test_cleanup_orphans(self):
        reg = DelegateRegistry()
        reg.register("orch.1", "orch", "implement", "Task", "T")
        reg.update_status("orch.1", DelegateStatus.RUNNING)
        reg.register("orch.2", "orch", "explore", "Task 2", "T2")
        reg.update_status("orch.2", DelegateStatus.COMPLETED)

        count = reg.cleanup_orphans()
        assert count == 1  # only orch.1 was running
        assert reg.get_node("orch.1").status == DelegateStatus.KILLED
        assert reg.get_node("orch.2").status == DelegateStatus.COMPLETED

    def test_persistence(self, tmp_path: Path):
        store = tmp_path / "delegates.json"

        # Save
        reg = DelegateRegistry(store_path=store)
        reg.register("orch.1", "orch", "implement", "Write tests", "Tests")
        reg.update_status("orch.1", DelegateStatus.COMPLETED, result_preview="All pass")
        reg.register("orch.2", "orch", "explore", "Find bugs", "Bugs")
        reg.save()

        assert store.exists()

        # Load into fresh registry
        reg2 = DelegateRegistry(store_path=store)
        reg2.load()

        node = reg2.get_node("orch.1")
        assert node is not None
        assert node.status == DelegateStatus.COMPLETED
        assert node.result_preview == "All pass"

        node2 = reg2.get_node("orch.2")
        assert node2 is not None

        # Child counters should be rebuilt correctly
        next_id = reg2.generate_child_id("orch")
        assert next_id == "orch.3"

    def test_to_dict_from_dict_roundtrip(self):
        node = DelegateNode(
            agent_id="orch.1.2",
            parent_id="orch.1",
            agent_type="debug",
            prompt="Fix the flaky test in test_bridge.py",
            title="Fix flaky test",
            status=DelegateStatus.COMPLETED,
            result_preview="Fixed by adding retry",
            tokens_used=3200,
            tool_calls=7,
        )
        d = node.to_dict()
        assert d["agent_id"] == "orch.1.2"
        assert d["status"] == "completed"

        restored = DelegateNode.from_dict(d)
        assert restored.agent_id == node.agent_id
        assert restored.status == node.status
        assert restored.tokens_used == node.tokens_used


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestDelegatePrompts:
    def test_all_agent_types(self):
        for agent_type in ("explore", "implement", "research", "debug", "review"):
            prompt = build_delegate_prompt(agent_type, "orch.1", "orch")
            assert "orch.1" in prompt
            assert agent_type in prompt.lower() or agent_type.title() in prompt

    def test_includes_common_tools(self):
        prompt = build_delegate_prompt("implement", "orch.1", "orch")
        assert "delegate" in prompt
        assert "post_message" in prompt
        assert "get_snapshot" in prompt

    def test_unknown_type_defaults_to_implement(self):
        prompt = build_delegate_prompt("unknown_type", "orch.1", "orch")
        assert "Implementation" in prompt


# ---------------------------------------------------------------------------
# Event types exist
# ---------------------------------------------------------------------------

def test_delegate_event_types():
    assert EventType.DELEGATE_SPAWNED.value == "delegate.spawned"
    assert EventType.DELEGATE_RUNNING.value == "delegate.running"
    assert EventType.DELEGATE_COMPLETED.value == "delegate.completed"
    assert EventType.DELEGATE_FAILED.value == "delegate.failed"
    assert EventType.DELEGATE_KILLED.value == "delegate.killed"


# ---------------------------------------------------------------------------
# Delegate tool integration test (mocked bridge)
# ---------------------------------------------------------------------------

def _make_mock_runtime(bus, tmp_path):
    """Create a mock runtime with delegate infrastructure."""
    import asyncio

    mock_runtime = MagicMock()
    mock_runtime.events = bus
    mock_runtime.delegate_registry = DelegateRegistry(
        store_path=tmp_path / "delegates.json",
    )
    mock_runtime._config = MagicMock()
    mock_runtime._config.orchestrator.model = "sonnet"
    mock_runtime.connectors = MagicMock()
    mock_runtime.connectors._sessions = {}
    mock_runtime.register_interrupt_hook = MagicMock()
    mock_runtime.unregister_interrupt_hook = MagicMock()
    # Real dicts/events for the async delegate lifecycle
    mock_runtime._delegate_providers = {}
    mock_runtime._delegate_tasks = {}
    mock_runtime._delegate_results = {}
    mock_runtime._delegate_done = {}
    mock_runtime.send_delegate_message = AsyncMock(return_value=True)
    return mock_runtime


@pytest.mark.asyncio
async def test_delegate_tool_executor(tmp_path: Path):
    """Test the delegate executor spawns a background task and returns immediately."""
    import asyncio
    from atn.events import EventBus
    from atn.providers.base import ProviderResponse, Usage

    bus = EventBus()
    events_captured: list = []

    async def _capture(e):
        events_captured.append(e)

    bus.subscribe(None, _capture)
    mock_runtime = _make_mock_runtime(bus, tmp_path)

    # Mock BridgeProvider
    mock_response = ProviderResponse(
        text="I explored the codebase and found 3 auth modules.",
        usage=Usage(input_tokens=1000, output_tokens=500),
        stop_reason="end_turn",
        model="sonnet",
    )

    mock_provider = AsyncMock()
    mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
    mock_provider.close = AsyncMock()
    mock_provider.interrupt = AsyncMock()

    with patch(
        "atn.providers.bridge.BridgeProvider",
        return_value=mock_provider,
    ):
        from atn.orchestrator.tools import _delegate, _delegate_collect

        # delegate() returns immediately with "spawned"
        spawn_result = await _delegate(mock_runtime, {
            "prompt": "Explore the auth system",
            "agent_type": "explore",
            "title": "Auth exploration",
        })

        assert spawn_result["status"] == "spawned"
        assert spawn_result["agent_id"] == "orch.1"
        assert spawn_result["agent_type"] == "explore"

        # Background task is running
        assert "orch.1" in mock_runtime._delegate_tasks

        # Collect the result (blocks until background task finishes)
        collect_result = await _delegate_collect(mock_runtime, {"agent_id": "orch.1"})

    assert collect_result["status"] == "completed"
    assert "explored the codebase" in collect_result["result"]
    assert collect_result["usage"]["input_tokens"] == 1000

    # Verify registry was updated
    node = mock_runtime.delegate_registry.get_node("orch.1")
    assert node is not None
    assert node.status == DelegateStatus.COMPLETED

    # Verify events were emitted
    event_types = [e.type for e in events_captured]
    assert EventType.DELEGATE_SPAWNED in event_types
    assert EventType.DELEGATE_RUNNING in event_types
    assert EventType.DELEGATE_COMPLETED in event_types

    # Verify interrupt hook was registered and unregistered
    mock_runtime.register_interrupt_hook.assert_called_once()
    mock_runtime.unregister_interrupt_hook.assert_called_once()

    # Verify bridge was closed
    mock_provider.close.assert_called_once()

    # Verify send_orchestrate was called with correct args
    call_args = mock_provider.send_orchestrate.call_args
    assert call_args.kwargs["message"] == "Explore the auth system"
    assert "explore" in call_args.kwargs["system"].lower() or "Exploration" in call_args.kwargs["system"]
    assert call_args.kwargs["max_turns"] == 50


@pytest.mark.asyncio
async def test_delegate_tool_failure(tmp_path: Path):
    """Test delegate handles bridge failures gracefully."""
    from atn.events import EventBus

    bus = EventBus()
    events_captured: list = []

    async def _capture(e):
        events_captured.append(e)

    bus.subscribe(None, _capture)
    mock_runtime = _make_mock_runtime(bus, tmp_path)

    # Mock BridgeProvider that raises
    mock_provider = AsyncMock()
    mock_provider.send_orchestrate = AsyncMock(
        side_effect=RuntimeError("Bridge subprocess crashed")
    )
    mock_provider.close = AsyncMock()

    with patch(
        "atn.providers.bridge.BridgeProvider",
        return_value=mock_provider,
    ):
        from atn.orchestrator.tools import _delegate, _delegate_collect

        spawn_result = await _delegate(mock_runtime, {
            "prompt": "This will fail",
            "agent_type": "implement",
        })

        assert spawn_result["status"] == "spawned"

        # Collect — should get the failure result
        collect_result = await _delegate_collect(mock_runtime, {"agent_id": "orch.1"})

    assert collect_result["status"] == "failed"
    assert "crashed" in collect_result["error"]

    # Verify registry reflects failure
    node = mock_runtime.delegate_registry.get_node("orch.1")
    assert node.status == DelegateStatus.FAILED

    # Verify failure event
    event_types = [e.type for e in events_captured]
    assert EventType.DELEGATE_FAILED in event_types


@pytest.mark.asyncio
async def test_delegate_status_tool(tmp_path: Path):
    """Test delegate_status returns correct state at different points."""
    from atn.events import EventBus
    from atn.providers.base import ProviderResponse, Usage

    bus = EventBus()
    mock_runtime = _make_mock_runtime(bus, tmp_path)

    # Create a delegate that we can control with an event
    proceed = asyncio.Event()
    mock_response = ProviderResponse(
        text="Done.", usage=Usage(input_tokens=100, output_tokens=50),
        stop_reason="end_turn", model="sonnet",
    )

    async def _slow_orchestrate(**kwargs):
        await proceed.wait()
        return mock_response

    mock_provider = AsyncMock()
    mock_provider.send_orchestrate = _slow_orchestrate
    mock_provider.close = AsyncMock()
    mock_provider.interrupt = AsyncMock()

    with patch(
        "atn.providers.bridge.BridgeProvider",
        return_value=mock_provider,
    ):
        from atn.orchestrator.tools import _delegate, _delegate_status, _delegate_collect

        spawn = await _delegate(mock_runtime, {
            "prompt": "Do something", "agent_type": "implement",
        })
        agent_id = spawn["agent_id"]

        # Give the background task a moment to start
        await asyncio.sleep(0.05)

        # Status should show running
        status = await _delegate_status(mock_runtime, {"agent_id": agent_id})
        assert status["status"] == "running"

        # Let it finish
        proceed.set()
        result = await _delegate_collect(mock_runtime, {"agent_id": agent_id})
        assert result["status"] == "completed"
