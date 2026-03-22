"""Tests for the scheduler: agents with schedule intervals fire automatically."""
import asyncio
import shutil
from pathlib import Path

from atn.config import ATNConfig
from atn.events import EventBus, EventType
from atn.models import (
    AgentDefinition, StepDefinition, StepType, ExecutionStatus, AgentStatus,
)
from atn.runtime import Runtime

TEST_DIR = Path("./test_data_scheduler")


async def test():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    config = ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=config)

    events = []
    async def collect(e):
        events.append(e)
    bus.subscribe(None, collect)

    await rt.start()

    # ===== Test 1: _parse_interval unit tests =====
    assert Runtime._parse_interval("5s") == 5.0
    assert Runtime._parse_interval("2m") == 120.0
    assert Runtime._parse_interval("1h") == 3600.0
    assert Runtime._parse_interval("  30 S  ") == 30.0
    try:
        Runtime._parse_interval("abc")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        Runtime._parse_interval("10d")
        assert False, "Should have raised ValueError for 'd'"
    except ValueError:
        pass
    print("Test 1 PASS: _parse_interval correctly parses s/m/h and rejects invalid")

    # ===== Test 2: Scheduled agent fires automatically =====
    counter_agent = AgentDefinition(
        id="tick", name="Tick Agent",
        schedule="2s",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo tick", "timeout": 5},
            name="tick",
        )],
    )
    await rt.register_agent(counter_agent)
    # Verify schedule is registered
    assert "tick" in rt._schedule_table
    assert rt._schedule_table["tick"] == 2.0
    print("Test 2a PASS: Schedule table populated on register")

    # Agent must be ACTIVE for scheduler to fire
    await rt.activate_agent("tick")
    assert rt._status["tick"] == AgentStatus.ACTIVE

    # Wait for the scheduler to fire at least twice.
    # Scheduler loop sleeps 1s per iteration. With a 2s interval:
    #   - First fire at ~0s (immediately on first check, since last=None)
    #   - Second fire at ~2s after first
    # Plus inbox watcher 0.5s delay + execution time.
    # Wait 6s to be safe for at least 2 firings.
    await asyncio.sleep(6)

    # Count scheduler events
    sched_events = [
        e for e in events
        if e.type == EventType.SCHEDULE_TRIGGERED and e.data["agent_id"] == "tick"
    ]
    print(f"  Scheduler fired {len(sched_events)} times in 6s (interval=2s)")
    assert len(sched_events) >= 2, f"Expected >=2 scheduler firings, got {len(sched_events)}"

    # Verify executions actually ran
    completed = [
        e for e in events
        if e.type == EventType.EXECUTION_COMPLETED and e.data.get("agent_id") == "tick"
    ]
    assert len(completed) >= 2, f"Expected >=2 completed executions, got {len(completed)}"
    print(f"Test 2b PASS: Scheduled agent executed {len(completed)} times automatically")

    # ===== Test 3: Deactivated agent stops getting scheduled =====
    events.clear()
    # Deactivate the agent
    rt._status["tick"] = AgentStatus.REGISTERED
    await asyncio.sleep(4)  # Wait for 2 potential schedule cycles

    sched_after = [
        e for e in events
        if e.type == EventType.SCHEDULE_TRIGGERED and e.data["agent_id"] == "tick"
    ]
    assert len(sched_after) == 0, f"Deactivated agent should not be scheduled, got {len(sched_after)}"
    print("Test 3 PASS: Deactivated agent is not scheduled")

    # ===== Test 4: Re-activate resumes scheduling =====
    events.clear()
    await rt.activate_agent("tick")
    await asyncio.sleep(5)

    sched_reactivated = [
        e for e in events
        if e.type == EventType.SCHEDULE_TRIGGERED and e.data["agent_id"] == "tick"
    ]
    assert len(sched_reactivated) >= 1, f"Re-activated agent should schedule, got {len(sched_reactivated)}"
    print(f"Test 4 PASS: Re-activated agent resumed scheduling ({len(sched_reactivated)} fires)")

    # ===== Test 5: Unregister clears the schedule table =====
    await rt.unregister_agent("tick")
    assert "tick" not in rt._schedule_table
    assert "tick" not in rt._last_scheduled
    print("Test 5 PASS: Unregister clears schedule table entries")

    # ===== Test 6: Agent without schedule has no schedule entry =====
    no_sched = AgentDefinition(
        id="nosched", name="No Schedule Agent",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo nope", "timeout": 5},
            name="nope",
        )],
    )
    await rt.register_agent(no_sched)
    assert "nosched" not in rt._schedule_table
    print("Test 6 PASS: Agent without schedule has no schedule table entry")

    await rt.stop()

    print("\nAll scheduler tests passed!")

    # Cleanup
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    asyncio.run(test())
