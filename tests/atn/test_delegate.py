"""Tests for the delegate sub-agent system.

Tests the DelegateRegistry, event types, and delegate tool executor
with a mocked bridge provider.  Phase 3: delegates are cognitive agents
executed through the unified Runtime._execute_cognitive_agent() path.
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
    def test_generate_child_id(self, tmp_path):
        """generate_child_id is now on AgentRegistry (via Runtime)."""
        from atn.config import ATNConfig
        from atn.runtime import Runtime
        from atn.events import EventBus

        data_dir = tmp_path / "data"
        agents_dir = tmp_path / "agents"
        data_dir.mkdir(parents=True, exist_ok=True)
        agents_dir.mkdir(parents=True, exist_ok=True)
        config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
        config.autonet.enabled = False
        config.voice.enabled = False
        rt = Runtime(EventBus(), data_dir=data_dir, config=config)

        assert rt.generate_child_id("orch") == "orch.1"
        assert rt.generate_child_id("orch") == "orch.2"
        assert rt.generate_child_id("orch.1") == "orch.1.1"
        assert rt.generate_child_id("orch.1") == "orch.1.2"
        assert rt.generate_child_id("orch") == "orch.3"

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

        # Verify all nodes loaded correctly
        assert len(reg2.get_children("orch")) == 2

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
        # The type specialization layer is still in the system prompt; agent
        # IDENTITY (agent_id/parent) is deliberately NOT — it moved to the
        # first user message (build_identity_header) so the system-prompt
        # prefix stays byte-identical across agents and hits the prompt cache.
        from atn.delegate_prompts import build_identity_header
        for agent_type in ("explore", "implement", "research", "debug", "review"):
            prompt = build_delegate_prompt(agent_type, "orch.1", "orch")
            assert agent_type in prompt.lower() or agent_type.title() in prompt
            assert "orch.1" not in prompt  # identity is no longer in the prefix
            header = build_identity_header(
                agent_id="orch.1", agent_type=agent_type, parent_id="orch")
            assert "orch.1" in header and agent_type in header

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
# Delegate tool integration tests (mocked bridge, real Runtime)
# ---------------------------------------------------------------------------

def _make_runtime(bus, tmp_path):
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


@pytest.mark.asyncio
async def test_delegate_tool_executor(tmp_path: Path):
    """Test the delegate executor spawns a cognitive agent and returns immediately."""
    from atn.events import EventBus
    from atn.providers.base import ProviderResponse, Usage

    bus = EventBus()
    events_captured: list = []

    async def _capture(e):
        events_captured.append(e)

    bus.subscribe(None, _capture)
    rt = _make_runtime(bus, tmp_path)

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

    with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
        from atn.orchestrator.tools import _create_agent, _delegate_collect

        # create_agent with mode=cognitive and prompt auto-activates and triggers
        spawn_result = await _create_agent(rt, {
            "mode": "cognitive",
            "prompt": "Explore the auth system",
            "agent_type": "explore",
            "name": "Auth exploration",
            "_caller_id": "orch",
        })

        assert spawn_result["status"] == "running"
        agent_id = spawn_result["agent_id"]
        assert agent_id == "orch.1"

        # Agent is registered in the unified registry as cognitive
        defn = rt.get_agent(agent_id)
        assert defn is not None
        assert defn.mode.value == "cognitive"
        assert defn.parent_id == "orch"

        # Collect the result (blocks until execution finishes)
        collect_result = await _delegate_collect(rt, {"agent_id": agent_id})

    assert collect_result["status"] == "completed"
    assert "explored the codebase" in collect_result["result"]
    assert collect_result["usage"]["input_tokens"] == 1000

    # Verify delegate registry was synced for UI
    node = rt.delegate_registry.get_node(agent_id)
    assert node is not None
    assert node.status == DelegateStatus.COMPLETED

    # Verify events were emitted
    event_types = [e.type for e in events_captured]
    assert EventType.DELEGATE_SPAWNED in event_types

    # Verify bridge was closed
    mock_provider.close.assert_called_once()

    # Verify send_orchestrate was called with correct args
    call_args = mock_provider.send_orchestrate.call_args
    assert "Explore the auth system" in call_args.kwargs["message"]
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
    rt = _make_runtime(bus, tmp_path)

    # Mock BridgeProvider that raises
    mock_provider = AsyncMock()
    mock_provider.send_orchestrate = AsyncMock(
        side_effect=RuntimeError("Bridge subprocess crashed")
    )
    mock_provider.close = AsyncMock()

    with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
        from atn.orchestrator.tools import _create_agent, _delegate_collect

        spawn_result = await _create_agent(rt, {
            "mode": "cognitive",
            "prompt": "This will fail",
            "agent_type": "implement",
            "_caller_id": "orch",
        })

        assert spawn_result["status"] == "running"
        agent_id = spawn_result["agent_id"]

        # Collect — should get the failure result
        collect_result = await _delegate_collect(rt, {"agent_id": agent_id})

    assert collect_result["status"] == "failed"
    assert "crashed" in collect_result["error"]

    # Verify delegate registry reflects failure
    node = rt.delegate_registry.get_node(agent_id)
    assert node.status == DelegateStatus.FAILED

    # Verify failure event
    event_types = [e.type for e in events_captured]
    assert EventType.EXECUTION_FAILED in event_types


@pytest.mark.asyncio
async def test_delegate_status_tool(tmp_path: Path):
    """Test delegate_status returns correct state at different points."""
    from atn.events import EventBus
    from atn.providers.base import ProviderResponse, Usage

    bus = EventBus()
    rt = _make_runtime(bus, tmp_path)

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

    with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
        from atn.orchestrator.tools import _create_agent, _delegate_status, _delegate_collect

        spawn = await _create_agent(rt, {
            "mode": "cognitive",
            "prompt": "Do something", "agent_type": "implement",
            "_caller_id": "orch",
        })
        agent_id = spawn["agent_id"]

        # Give the background task a moment to start
        await asyncio.sleep(0.05)

        # Status should show running
        status = await _delegate_status(rt, {"agent_id": agent_id})
        assert status["status"] == "running"

        # Let it finish
        proceed.set()
        result = await _delegate_collect(rt, {"agent_id": agent_id})
        assert result["status"] == "completed"
