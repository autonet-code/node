"""End-to-end test for JSONL execution log persistence."""
import asyncio
import shutil
import json
from pathlib import Path

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, StepDefinition, StepType, ExecutionStatus
from atn.runtime import Runtime


TEST_DIR = Path("./test_data_persistence")
AGENTS_DIR = TEST_DIR / "agents"


async def test():
    # Clean slate
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    bus = EventBus()
    config = ATNConfig(data_dir=TEST_DIR, agents_dir=AGENTS_DIR)
    rt = Runtime(bus, data_dir=TEST_DIR, config=config)
    await rt.start()

    # ===== Register and run an agent =====
    agent = AgentDefinition(
        id="persist01", name="Persist Test Agent",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo hello-persist", "timeout": 5},
                name="greet",
            ),
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo step-two", "timeout": 5},
                name="second",
            ),
        ],
    )
    await rt.register_agent(agent)

    # Run it twice
    eid1 = await rt.trigger_run("persist01")
    await asyncio.sleep(2)
    eid2 = await rt.trigger_run("persist01")
    await asyncio.sleep(2)

    # ===== Test 1: JSONL file exists =====
    jsonl_path = AGENTS_DIR / "persist01" / "execution.jsonl"
    assert jsonl_path.exists(), f"JSONL file not created at {jsonl_path}"
    print("Test 1 PASS: JSONL file exists")

    # ===== Test 2: File has 2 records =====
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2, f"Expected 2 records, got {len(lines)}"
    print("Test 2 PASS: 2 records written")

    # ===== Test 3: Records are valid JSON with correct fields =====
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["agent_id"] == "persist01"
        assert rec["status"] == "completed"
        assert len(rec["step_results"]) == 2
        assert rec["step_results"][0]["step_name"] == "greet"
        assert rec["step_results"][1]["step_name"] == "second"
        assert rec["started_at"] is not None
        assert rec["completed_at"] is not None
    print("Test 3 PASS: Records are valid JSON with correct structure")

    # ===== Test 4: load_from_disk round-trip =====
    loaded = rt.execution_log.load_from_disk("persist01")
    assert len(loaded) == 2
    assert loaded[0].agent_id == "persist01"
    assert loaded[0].status == ExecutionStatus.COMPLETED
    assert len(loaded[0].step_results) == 2
    assert loaded[1].execution_id != loaded[0].execution_id
    print("Test 4 PASS: load_from_disk round-trip works")

    # ===== Test 5: disk_summary =====
    summary = rt.execution_log.disk_summary()
    assert summary == {"persist01": 2}, f"Unexpected summary: {summary}"
    print("Test 5 PASS: disk_summary correct")

    # ===== Test 6: Run a failing agent and verify persistence =====
    fail_agent = AgentDefinition(
        id="failpersist", name="Failing Agent",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "exit 1", "timeout": 5},
            name="fail_step",
        )],
    )
    await rt.register_agent(fail_agent)
    await rt.trigger_run("failpersist")
    await asyncio.sleep(2)

    fail_path = AGENTS_DIR / "failpersist" / "execution.jsonl"
    assert fail_path.exists()
    fail_lines = [l for l in fail_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(fail_lines) == 1
    fail_rec = json.loads(fail_lines[0])
    assert fail_rec["status"] == "failed"
    assert fail_rec["step_results"][0]["status"] == "failed"
    print("Test 6 PASS: Failed execution persisted correctly")

    # ===== Test 7: clear_agent =====
    cleared = rt.execution_log.clear_agent("persist01")
    assert cleared is True, "clear_agent should return True when file exists"
    assert not jsonl_path.exists(), "JSONL file should be deleted"
    assert rt.execution_log.get_history("persist01") == []
    print("Test 7 PASS: clear_agent removes memory + disk")

    # ===== Test 8: clear_all =====
    n = rt.execution_log.clear_all()
    assert n == 1, f"Expected 1 file deleted, got {n}"  # failpersist.jsonl
    assert not fail_path.exists()
    print("Test 8 PASS: clear_all removes all files")

    # ===== Test 9: Run again after clear, verify fresh file =====
    await rt.trigger_run("persist01")
    await asyncio.sleep(2)
    assert jsonl_path.exists()
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    print("Test 9 PASS: Fresh persistence after clear")

    # ===== Test 10: No .running.json files after clean completion =====
    running_files = list(AGENTS_DIR.glob("*/*.running.json"))
    assert len(running_files) == 0, f"Expected no running files, found {running_files}"
    print("Test 10 PASS: No .running.json files after clean completion")

    # ===== Test 11: persist_running + recover_running round-trip =====
    from atn.models import ExecutionRecord
    fake_record = ExecutionRecord(
        execution_id="fake-crash-001",
        agent_id="persist01",
        status=ExecutionStatus.RUNNING,
        trigger_source="test",
    )
    # Simulate a step result being recorded
    from atn.models import StepResult, StepType as ST
    from datetime import datetime, timezone
    fake_record.step_results.append(StepResult(
        step_index=0, step_name="greet", step_type=ST.SCRIPT,
        status=ExecutionStatus.COMPLETED,
        output={"stdout": "hello-persist"},
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    ))
    rt.execution_log.persist_running(fake_record)

    running_path = AGENTS_DIR / "persist01" / "fake-crash-001.running.json"
    assert running_path.exists(), "Running file should exist after persist_running"
    print("Test 11a PASS: persist_running creates .running.json file")

    # Now simulate restart — recover_running should pick it up
    recovered = rt.execution_log.recover_running()
    assert len(recovered) == 1
    assert recovered[0].execution_id == "fake-crash-001"
    assert recovered[0].status == ExecutionStatus.FAILED
    assert "crashed" in (recovered[0].error or "").lower()
    assert len(recovered[0].step_results) == 1
    assert not running_path.exists(), "Running file should be cleaned up after recovery"
    print("Test 11b PASS: recover_running loads crashed execution and cleans up")

    await rt.stop()

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("\nAll persistence tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
