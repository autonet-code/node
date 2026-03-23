"""Tests for Runtime agent lifecycle — register, activate, execute, deactivate.

Uses a mock-friendly approach: we instantiate Runtime with a test config
and mock external dependencies (providers, filesystem, autonet bridge).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atn.events import Event, EventBus, EventType
from atn.models import (
    AgentDefinition,
    AgentStatus,
    ExecutionStatus,
    StepDefinition,
    StepType,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def captured_events(bus: EventBus) -> list[Event]:
    """Subscribe a wildcard handler and return the list it populates."""
    events: list[Event] = []

    async def _capture(e: Event):
        events.append(e)

    bus.subscribe(None, _capture)
    return events


def _make_runtime(bus: EventBus, tmp_path: Path):
    """Create a Runtime with a temp data dir and minimal config."""
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    # Disable autonet bridge to avoid network calls
    config.autonet.enabled = False
    config.voice.enabled = False

    rt = Runtime(bus, data_dir=data_dir, config=config)
    return rt


def _simple_agent(agent_id: str = "test-agent", schedule: str | None = None) -> AgentDefinition:
    """Create a minimal agent with a single script step."""
    return AgentDefinition(
        id=agent_id,
        name="Test Agent",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo hello", "timeout": 5},
                name="greet",
            ),
        ],
        schedule=schedule,
    )


class TestRuntimeRegister:
    """Tests for agent registration."""

    @pytest.mark.asyncio
    async def test_register_agent(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)
        agent = _simple_agent()
        aid = await rt.register_agent(agent)

        assert aid == "test-agent"
        assert rt.get_agent("test-agent") is agent
        assert rt.get_status("test-agent") == AgentStatus.REGISTERED

        reg_events = [e for e in captured_events if e.type == EventType.AGENT_REGISTERED]
        assert len(reg_events) == 1
        assert reg_events[0].data["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    async def test_register_multiple_agents(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent("a1"))
        await rt.register_agent(_simple_agent("a2"))
        agents = rt.list_agents()
        assert len(agents) == 2
        ids = {a[0].id for a in agents}
        assert ids == {"a1", "a2"}

    @pytest.mark.asyncio
    async def test_register_agent_with_schedule(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        agent = _simple_agent(schedule="30s")
        await rt.register_agent(agent)
        assert "test-agent" in rt._schedule_table
        assert rt._schedule_table["test-agent"] == 30.0

    @pytest.mark.asyncio
    async def test_get_agent_nonexistent(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        assert rt.get_agent("nope") is None

    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        assert rt.get_status("nope") is None


class TestRuntimeActivateDeactivate:
    """Tests for agent activation and deactivation."""

    @pytest.mark.asyncio
    async def test_activate_agent(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        await rt.activate_agent("test-agent")

        assert rt.get_status("test-agent") == AgentStatus.ACTIVE
        act_events = [e for e in captured_events if e.type == EventType.AGENT_ACTIVATED]
        assert len(act_events) == 1

    @pytest.mark.asyncio
    async def test_deactivate_agent(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        await rt.activate_agent("test-agent")
        await rt.deactivate_agent("test-agent")

        assert rt.get_status("test-agent") == AgentStatus.STOPPED
        deact_events = [e for e in captured_events if e.type == EventType.AGENT_DEACTIVATED]
        assert len(deact_events) == 1

    @pytest.mark.asyncio
    async def test_activate_unknown_agent_raises(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        with pytest.raises(ValueError, match="Unknown agent"):
            await rt.activate_agent("nonexistent")

    @pytest.mark.asyncio
    async def test_deactivate_unknown_agent_raises(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        with pytest.raises(ValueError, match="Unknown agent"):
            await rt.deactivate_agent("nonexistent")


class TestRuntimeUnregister:
    """Tests for unregistering agents."""

    @pytest.mark.asyncio
    async def test_unregister_agent(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        await rt.unregister_agent("test-agent")

        assert rt.get_agent("test-agent") is None
        assert rt.get_status("test-agent") is None
        unreg_events = [e for e in captured_events if e.type == EventType.AGENT_UNREGISTERED]
        assert len(unreg_events) == 1

    @pytest.mark.asyncio
    async def test_unregister_clears_inbox(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        from atn.models import InboxMessage, MessageType, MessagePriority
        rt.inbox.post(InboxMessage(
            id="m1", source="user", target="test-agent",
            type=MessageType.WORK, priority=MessagePriority.NORMAL,
        ))
        assert rt.inbox.count("test-agent") == 1
        await rt.unregister_agent("test-agent")
        assert rt.inbox.count("test-agent") == 0


class TestRuntimeTriggerRun:
    """Tests for triggering agent execution."""

    @pytest.mark.asyncio
    async def test_trigger_run_returns_execution_id(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        await rt.activate_agent("test-agent")
        eid = await rt.trigger_run("test-agent")
        assert eid is not None
        # Wait for the script step to complete
        await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_trigger_run_unknown_agent_raises(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        with pytest.raises(ValueError, match="Unknown agent"):
            await rt.trigger_run("nonexistent")

    @pytest.mark.asyncio
    async def test_trigger_run_concurrency_limit(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        # Agent with concurrency=1 and a slow step
        agent = AgentDefinition(
            id="slow",
            name="Slow Agent",
            steps=[
                StepDefinition(
                    type=StepType.SCRIPT,
                    config={"command": "sleep 5", "timeout": 10},
                    name="wait",
                ),
            ],
            concurrency=1,
        )
        await rt.register_agent(agent)
        await rt.activate_agent("slow")

        eid1 = await rt.trigger_run("slow")
        assert eid1 is not None
        # Second run should be blocked by concurrency
        eid2 = await rt.trigger_run("slow")
        assert eid2 is None

        # Kill the running execution
        await rt.kill_agent("slow")

    @pytest.mark.asyncio
    async def test_execution_completes_successfully(self, bus, tmp_path, captured_events):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent())
        await rt.activate_agent("test-agent")

        eid = await rt.trigger_run("test-agent")
        # Wait for the echo command to finish
        await asyncio.sleep(2.0)

        # Check events for execution lifecycle
        event_types = [e.type for e in captured_events]
        assert EventType.EXECUTION_STARTED in event_types
        assert EventType.EXECUTION_COMPLETED in event_types


class TestRuntimeListAgents:
    """Tests for listing agents with their statuses."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        assert rt.list_agents() == []

    @pytest.mark.asyncio
    async def test_list_agents_returns_tuples(self, bus, tmp_path):
        rt = _make_runtime(bus, tmp_path)
        await rt.register_agent(_simple_agent("a"))
        await rt.register_agent(_simple_agent("b"))
        await rt.activate_agent("b")

        agents = rt.list_agents()
        assert len(agents) == 2
        statuses = {a[0].id: a[1] for a in agents}
        assert statuses["a"] == AgentStatus.REGISTERED
        assert statuses["b"] == AgentStatus.ACTIVE
