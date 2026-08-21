"""Tests for Phase 2: Cognitive mode execution + innate wake-up callbacks.

Tests cover:
- Cognitive agent execution via Runtime
- Innate wake-up (child completion -> parent inbox message)
- generate_child_id at Runtime level
- Delegate tools wrapping cognitive agents
- _active_providers tracking
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atn.agent_registry import DelegateRegistry, DelegateStatus
from atn.events import Event, EventBus, EventType
from atn.models import (
    AgentDefinition,
    AgentMode,
    AgentStatus,
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    StepDefinition,
    StepType,
)
from atn.providers.base import ProviderResponse, Usage


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def captured_events(bus: EventBus) -> list[Event]:
    events: list[Event] = []

    async def _capture(e: Event):
        events.append(e)

    bus.subscribe(None, _capture)
    return events


def _make_runtime(bus: EventBus, tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False

    rt = Runtime(bus, data_dir=data_dir, config=config)
    return rt


async def _wait_runs(rt, timeout: float = 5.0):
    """Wait for every in-flight execution task to finish.

    Mocked providers return instantly, but the completion path
    (persist, registry sync, parent notify) spans many awaits — a
    fixed one-tick sleep races it. Must be called INSIDE the
    provider patch so a still-running agent can never construct a
    real provider after the mock goes away.
    """
    tasks = list(rt.engine._tasks.values())
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)


class TestGenerateChildId:
    """Tests for Runtime.generate_child_id."""

    def test_generates_hierarchical_ids(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        assert rt.generate_child_id("orch") == "orch.1"
        assert rt.generate_child_id("orch") == "orch.2"
        assert rt.generate_child_id("orch.1") == "orch.1.1"
        assert rt.generate_child_id("orch") == "orch.3"
        assert rt.generate_child_id("orch.1") == "orch.1.2"

    @pytest.mark.asyncio
    async def test_counters_rebuild_from_registered_agents(self, bus, tmp_path):
        """Restart safety: re-registering persisted hierarchical ids
        must push the counters past them, or the next spawned child
        collides with (and shadows) an existing agent."""
        rt = _make_runtime(bus, tmp_path)
        for aid in ("orch.1", "orch.2", "orch.1.3"):
            await rt.register_agent(AgentDefinition(
                id=aid, name=aid, mode=AgentMode.COGNITIVE,
                provider="sonnet", description="restored",
            ))
        assert rt.generate_child_id("orch") == "orch.3"
        assert rt.generate_child_id("orch.1") == "orch.1.4"

    def test_independent_counters(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        assert rt.generate_child_id("a") == "a.1"
        assert rt.generate_child_id("b") == "b.1"
        assert rt.generate_child_id("a") == "a.2"
        assert rt.generate_child_id("b") == "b.2"


class TestCognitiveAgentExecution:
    """Tests for _execute_cognitive_agent."""

    @pytest.mark.asyncio
    async def test_cognitive_agent_completes(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Task completed successfully.",
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        defn = AgentDefinition(
            id="cog-1",
            name="Test Cognitive Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            cognitive_model="sonnet",
            system_prompt="You are a test agent.",
            agent_type="implement",
            tools=["sdk_builtin"],
            description="Do something useful.",
        )

        await rt.register_agent(defn)
        await rt.activate_agent("cog-1")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            eid = await rt.trigger_run("cog-1", source="test")
            assert eid is not None
            # Wait for execution
            await _wait_runs(rt)

        # Check agent reached COMPLETED status
        assert rt.get_status("cog-1") == AgentStatus.COMPLETED

        # Check output store has result
        output = rt.output_store.read("cog-1")
        assert output is not None
        assert output.data["result"] == "Task completed successfully."

        # Check events
        event_types = [e.type for e in captured_events]
        assert EventType.EXECUTION_STARTED in event_types
        assert EventType.EXECUTION_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_cognitive_agent_failure(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(side_effect=RuntimeError("bridge exploded"))
        mock_provider.close = AsyncMock()

        defn = AgentDefinition(
            id="cog-fail",
            name="Failing Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            description="Will fail.",
        )

        await rt.register_agent(defn)
        await rt.activate_agent("cog-fail")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            eid = await rt.trigger_run("cog-fail", source="test")
            await _wait_runs(rt)

        assert rt.get_status("cog-fail") == AgentStatus.ERROR

        event_types = [e.type for e in captured_events]
        assert EventType.EXECUTION_FAILED in event_types

    @pytest.mark.asyncio
    async def test_pipeline_agents_still_work(self, bus, tmp_path, captured_events):
        """Ensure pipeline mode is not broken by cognitive mode additions."""
        rt = _make_runtime(bus, tmp_path)

        defn = AgentDefinition(
            id="pipe-agent",
            name="Pipeline Agent",
            steps=[
                StepDefinition(
                    type=StepType.SCRIPT,
                    config={"command": "echo pipeline-ok", "timeout": 5},
                    name="echo",
                ),
            ],
        )

        await rt.register_agent(defn)
        await rt.activate_agent("pipe-agent")
        eid = await rt.trigger_run("pipe-agent")
        assert eid is not None
        await asyncio.sleep(0.5)  # real echo subprocess

        event_types = [e.type for e in captured_events]
        assert EventType.EXECUTION_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_active_providers_tracked(self, bus, tmp_path):
        """Verify _active_providers is set during execution and cleaned up after."""
        rt = _make_runtime(bus, tmp_path)
        provider_captured = {}

        proceed = asyncio.Event()
        mock_response = ProviderResponse(
            text="Done.", usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn", model="sonnet",
        )

        async def slow_orchestrate(**kwargs):
            # Record that the provider is tracked
            provider_captured["during"] = "cog-track" in rt._active_providers
            proceed.set()
            return mock_response

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = slow_orchestrate
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        defn = AgentDefinition(
            id="cog-track",
            name="Tracked Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            description="Track me.",
        )

        await rt.register_agent(defn)
        await rt.activate_agent("cog-track")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            eid = await rt.trigger_run("cog-track", source="test")
            await proceed.wait()
            await _wait_runs(rt)

        assert provider_captured["during"] is True
        # After completion the provider STAYS registered: sessions are
        # sticky so delegate_message / follow-up injection can reach a
        # completed-but-resumable agent. Release happens on session
        # reset (session_manager / model_switch), not per-run.
        assert rt._active_providers.get("cog-track") is mock_provider


class TestInnateWakeUp:
    """Tests for child completion -> parent inbox notification."""

    @pytest.mark.asyncio
    async def test_parent_gets_inbox_message(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)

        # Register parent agent
        parent = AgentDefinition(
            id="parent-1",
            name="Parent Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
        )
        await rt.register_agent(parent)
        await rt.activate_agent("parent-1")

        mock_response = ProviderResponse(
            text="Child result here.",
            usage=Usage(input_tokens=50, output_tokens=20),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        # Register child agent with parent_id
        child = AgentDefinition(
            id="child-1",
            name="Child Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            parent_id="parent-1",
            description="Do child work.",
        )
        await rt.register_agent(child)
        await rt.activate_agent("child-1")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            eid = await rt.trigger_run("child-1", source="test")
            await _wait_runs(rt)

        # Parent should have a NORMAL-priority WORK message in its inbox.
        # NORMAL is the batching default (Task B): a completion queues into
        # the parent's next natural run rather than waking an idle parent
        # (that is the opt-in wake_parent_on_child path). The notification is
        # LEAN by design: no result blob — the parent pulls details via
        # get_children_status / get_output.
        msgs = rt.inbox.peek("parent-1")
        assert len(msgs) >= 1
        child_msg = [m for m in msgs if m.data.get("type") == "child_completed"]
        assert len(child_msg) == 1
        assert child_msg[0].priority == MessagePriority.NORMAL
        assert child_msg[0].data["child_id"] == "child-1"
        assert child_msg[0].data["status"] == "completed"
        assert "get_children_status" in child_msg[0].data["instruction"]

    @pytest.mark.asyncio
    async def test_no_notification_without_parent(self, bus, tmp_path):
        """Agents without parent_id should not trigger wake-up."""
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Done.", usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn", model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        defn = AgentDefinition(
            id="orphan",
            name="No Parent Agent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            description="No parent.",
        )
        await rt.register_agent(defn)
        await rt.activate_agent("orphan")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            await rt.trigger_run("orphan", source="test")
            await _wait_runs(rt)

        # No messages should be posted (no parent to notify)
        # Check all registered agents' inboxes
        for aid in rt._agents:
            assert rt.inbox.count(aid) == 0

    @pytest.mark.asyncio
    async def test_no_live_inject_when_parent_idle(self, bus, tmp_path):
        """A parent that merely HAS a cached provider but is NOT running gets
        the completion via the inbox, not a live injection. Live injection is
        gated on the parent actually running (_running_count > 0); an idle
        parent with a warm provider must still batch via the inbox."""
        rt = _make_runtime(bus, tmp_path)

        parent = AgentDefinition(
            id="parent-2",
            name="Active Parent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
        )
        await rt.register_agent(parent)
        await rt.activate_agent("parent-2")

        # Cached provider present but the parent is idle (running_count 0).
        parent_bridge = AsyncMock()
        parent_bridge.send_user_message = AsyncMock()
        rt._active_providers["parent-2"] = parent_bridge

        mock_response = ProviderResponse(
            text="Child done.",
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        child = AgentDefinition(
            id="child-2",
            name="Child",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            parent_id="parent-2",
            description="Child task.",
        )
        await rt.register_agent(child)
        await rt.activate_agent("child-2")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            await rt.trigger_run("child-2", source="test")
            await _wait_runs(rt)

        # Parent idle → no live injection; the completion lands in the inbox.
        parent_bridge.send_user_message.assert_not_called()
        msgs = rt.inbox.peek("parent-2")
        child_msg = [m for m in msgs if m.data.get("type") == "child_completed"]
        assert len(child_msg) == 1
        assert child_msg[0].data["child_id"] == "child-2"


class TestDelegateToolsUnified:
    """Tests for delegate tools wrapping cognitive agents."""

    def _make_mock_runtime(self, bus: EventBus, tmp_path: Path):
        """Create a real Runtime for delegate tool testing."""
        return _make_runtime(bus, tmp_path)

    @pytest.mark.asyncio
    async def test_delegate_creates_cognitive_agent(self, bus, tmp_path, captured_events):
        rt = self._make_mock_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Found files.",
            usage=Usage(input_tokens=500, output_tokens=200),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import _create_agent, _delegate_collect

            result = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "Search for auth code",
                "agent_type": "explore",
                "name": "Auth search",
                "_caller_id": "orch",
            })
            assert result["status"] == "running"
            agent_id = result["agent_id"]

            # Agent should be registered as cognitive
            defn = rt.get_agent(agent_id)
            assert defn is not None
            assert defn.mode == AgentMode.COGNITIVE
            assert defn.parent_id == "orch"
            assert defn.agent_type == "explore"

            # Wait for execution
            collect = await _delegate_collect(rt, {"agent_id": agent_id})

        assert collect["status"] == "completed"
        assert "Found files" in str(collect.get("result", "")) or "Found files" in str(collect.get("output", ""))

    @pytest.mark.asyncio
    async def test_delegate_status_checks_cognitive(self, bus, tmp_path):
        rt = self._make_mock_runtime(bus, tmp_path)

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
            from atn.agent_tools import _create_agent, _delegate_status, _delegate_collect

            spawn = await _create_agent(rt, {"mode": "cognitive", "prompt": "Slow task", "_caller_id": "orch"})
            await asyncio.sleep(0.01)  # yield to event loop

            status = await _delegate_status(rt, {"agent_id": spawn["agent_id"]})
            assert status["status"] == "running"

            proceed.set()
            collect = await _delegate_collect(rt, {"agent_id": spawn["agent_id"]})
            assert collect["status"] == "completed"

    @pytest.mark.asyncio
    async def test_delegate_message_reaches_cognitive(self, bus, tmp_path):
        rt = self._make_mock_runtime(bus, tmp_path)

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
        mock_provider.send_user_message = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import _create_agent, _delegate_message, _delegate_collect

            spawn = await _create_agent(rt, {"mode": "cognitive", "prompt": "Working", "_caller_id": "orch"})
            agent_id = spawn["agent_id"]
            await asyncio.sleep(0.01)  # yield to event loop

            msg_result = await _delegate_message(rt, {
                "agent_id": agent_id,
                "content": "Update: new info",
            })
            assert msg_result["status"] == "delivered"
            mock_provider.send_user_message.assert_called_once_with("Update: new info")

            proceed.set()
            await _delegate_collect(rt, {"agent_id": agent_id})


class TestCompletionCallbacks:
    """Tests for _completion_callbacks tracking."""

    @pytest.mark.asyncio
    async def test_callback_registered_via_delegate(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Done.", usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn", model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import _create_agent

            result = await _create_agent(rt, {"mode": "cognitive", "prompt": "task", "_caller_id": "orch"})
            agent_id = result["agent_id"]
            # Completion callbacks may be tracked via delegate registry or done events
            assert agent_id in rt._delegate_done or agent_id in getattr(rt, '_completion_callbacks', {})

            await _wait_runs(rt)


class TestDelegateRegistryBackwardCompat:
    """Ensure delegate registry stays in sync for UI/snapshot."""

    @pytest.mark.asyncio
    async def test_delegate_registry_synced(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Result text.",
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import _create_agent, _delegate_collect

            result = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "do work",
                "agent_type": "implement",
                "_caller_id": "orch",
            })
            agent_id = result["agent_id"]
            collect = await _delegate_collect(rt, {"agent_id": agent_id})

        assert collect["status"] == "completed"

        # Delegate registry should be updated
        node = rt.delegate_registry.get_node(agent_id)
        assert node is not None
        assert node.status == DelegateStatus.COMPLETED
        assert "Result text" in node.result_preview


class TestFractality:
    """Tests for fractal agent spawning — sub-agents spawning sub-sub-agents."""

    @pytest.mark.asyncio
    async def test_delegate_tools_include_create_agent(self):
        """Cognitive agents must get create_agent and trigger_run tools."""
        from atn.agent_tools import _get_delegate_tools

        tools = _get_delegate_tools()
        tool_names = {t["name"] for t in tools}
        assert "create_agent" in tool_names
        assert "trigger_run" in tool_names
        assert "get_output" in tool_names
        assert "delegate_status" in tool_names
        assert "delegate_collect" in tool_names
        assert "delegate_message" in tool_names
        assert "post_message" in tool_names
        assert "get_snapshot" in tool_names

    @pytest.mark.asyncio
    async def test_sub_agent_spawns_child_with_correct_parent(self, bus, tmp_path):
        """When a sub-agent (orch.1) calls delegate, the child's parent should be orch.1."""
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Child done.",
            usage=Usage(input_tokens=50, output_tokens=20),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import _create_agent

            # Simulate orch.1 (a sub-agent) calling create_agent
            result = await _create_agent(rt, {
                "mode": "cognitive",
                "prompt": "Sub-sub task",
                "agent_type": "explore",
                "_caller_id": "orch.1",
            })

            assert result["status"] == "running"
            agent_id = result["agent_id"]

            # The child should be orch.1.1, with parent_id = orch.1
            assert agent_id == "orch.1.1"
            defn = rt.get_agent(agent_id)
            assert defn is not None
            assert defn.parent_id == "orch.1"

    @pytest.mark.asyncio
    async def test_create_agent_cognitive_with_caller_context(self, bus, tmp_path):
        """create_agent with mode=cognitive should set parent_id from caller context."""
        rt = _make_runtime(bus, tmp_path)

        mock_response = ProviderResponse(
            text="Done.", usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn", model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            from atn.agent_tools import execute_tool

            result = await execute_tool("create_agent", {
                "id": "child-cog",
                "name": "Child Cognitive",
                "mode": "cognitive",
                "description": "Autonomous child task",
                "prompt": "Do something",
                "agent_type": "implement",
            }, rt, caller_id="orch.2")

            assert result["status"] == "running"
            defn = rt.get_agent("child-cog")
            assert defn is not None
            assert defn.mode == AgentMode.COGNITIVE
            assert defn.parent_id == "orch.2"
            assert defn.created_by == "orch.2"

    @pytest.mark.asyncio
    async def test_create_agent_pipeline_with_caller_context(self, bus, tmp_path):
        """create_agent with pipeline mode should also set parent_id from caller context."""
        rt = _make_runtime(bus, tmp_path)

        from atn.agent_tools import execute_tool

        result = await execute_tool("create_agent", {
            "id": "child-pipe",
            "name": "Child Pipeline",
            "steps": [
                {"name": "echo", "type": "script", "config": {"command": "echo ok"}},
            ],
        }, rt, caller_id="orch.3")

        assert result["status"] == "registered"
        defn = rt.get_agent("child-pipe")
        assert defn is not None
        assert defn.parent_id == "orch.3"
        assert defn.created_by == "orch.3"

    @pytest.mark.asyncio
    async def test_tool_executor_passes_caller_id(self, bus, tmp_path):
        """The cognitive agent's tool executor should pass caller_id to execute_tool."""
        rt = _make_runtime(bus, tmp_path)

        # Register parent agent so it exists
        parent = AgentDefinition(
            id="parent-frac",
            name="Parent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
        )
        await rt.register_agent(parent)
        await rt.activate_agent("parent-frac")

        captured_calls = []

        # Create a mock response that calls delegate tool
        async def mock_orchestrate(**kwargs):
            tool_executor = kwargs.get("tool_executor")
            if tool_executor:
                # The agent calls get_snapshot — we capture to verify caller_id is passed
                result = await tool_executor("get_snapshot", {})
                captured_calls.append(result)
            return ProviderResponse(
                text="Parent done.",
                usage=Usage(input_tokens=50, output_tokens=20),
                stop_reason="end_turn",
                model="sonnet",
            )

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = mock_orchestrate
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            eid = await rt.trigger_run("parent-frac", source="test")
            await asyncio.sleep(0.5)

        # get_snapshot should have been called and returned valid data
        assert len(captured_calls) == 1
        assert "agents" in captured_calls[0]


class TestInnateWakeUpInstruction:
    """Tests for the instruction field in child_completed messages."""

    @pytest.mark.asyncio
    async def test_instruction_field_in_completion_message(self, bus, tmp_path):
        """The child_completed message should include an instruction with get_output hint."""
        rt = _make_runtime(bus, tmp_path)

        parent = AgentDefinition(
            id="parent-inst",
            name="Parent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
        )
        await rt.register_agent(parent)
        await rt.activate_agent("parent-inst")

        mock_response = ProviderResponse(
            text="Child finished work.",
            usage=Usage(input_tokens=50, output_tokens=20),
            stop_reason="end_turn",
            model="sonnet",
        )
        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(return_value=mock_response)
        mock_provider.close = AsyncMock()
        mock_provider.interrupt = AsyncMock()

        child = AgentDefinition(
            id="child-inst",
            name="Worker Child",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            parent_id="parent-inst",
            description="Do child work.",
        )
        await rt.register_agent(child)
        await rt.activate_agent("child-inst")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            await rt.trigger_run("child-inst", source="test")
            await asyncio.sleep(0.5)

        msgs = rt.inbox.peek("parent-inst")
        child_msgs = [m for m in msgs if m.data.get("type") == "child_completed"]
        assert len(child_msgs) == 1

        data = child_msgs[0].data
        assert "instruction" in data
        # Lean-notification contract: the instruction points the parent
        # at the pull tools; no result/output preview blobs ride along.
        assert "get_children_status" in data["instruction"]
        assert "Worker Child" in data["instruction"]
        assert data["child_id"] == "child-inst"
        assert "result_preview" not in data

    @pytest.mark.asyncio
    async def test_failed_child_instruction(self, bus, tmp_path):
        """Failed child should have instruction mentioning failure."""
        rt = _make_runtime(bus, tmp_path)

        parent = AgentDefinition(
            id="parent-fail",
            name="Parent",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
        )
        await rt.register_agent(parent)
        await rt.activate_agent("parent-fail")

        mock_provider = AsyncMock()
        mock_provider.send_orchestrate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_provider.close = AsyncMock()

        child = AgentDefinition(
            id="child-fail",
            name="Failing Child",
            mode=AgentMode.COGNITIVE,
            provider="sonnet",
            parent_id="parent-fail",
        )
        await rt.register_agent(child)
        await rt.activate_agent("child-fail")

        with patch("atn.runtime.provider_manager.BridgeProvider", return_value=mock_provider):
            await rt.trigger_run("child-fail", source="test")
            await asyncio.sleep(0.5)

        # A FAILED child notifies via the always-on failure path, which posts
        # a child_error ALERT (not a child_completed). Failures always wake the
        # parent (HIGH priority) regardless of wake_parent_on_child.
        msgs = rt.inbox.peek("parent-fail")
        child_msgs = [m for m in msgs if m.data.get("type") == "child_error"]
        assert len(child_msgs) == 1

        data = child_msgs[0].data
        assert "failed" in data["instruction"]
        assert data["status"] == "failed"
        assert child_msgs[0].priority == MessagePriority.HIGH
