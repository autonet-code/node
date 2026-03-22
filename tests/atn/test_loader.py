"""End-to-end tests for agent definition loading and config system."""
import asyncio
import shutil
from pathlib import Path

import yaml

from atn.config import load_config, ATNConfig
from atn.events import EventBus
from atn.loader import load_agent_file, load_agents_dir, LoadError
from atn.models import StepType, ExecutionStatus
from atn.runtime import Runtime


TEST_DIR = Path("./test_data_loader")


def setup():
    """Create a fresh test directory structure."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    (TEST_DIR / "agents").mkdir(parents=True)
    (TEST_DIR / "execution").mkdir(parents=True)


def write_yaml(name: str, data: dict) -> Path:
    path = TEST_DIR / "agents" / name
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


async def test():
    setup()

    # ===== Test 1: Load a valid agent file =====
    write_yaml("good.yaml", {
        "id": "good01",
        "name": "Good Agent",
        "description": "A valid agent",
        "steps": [
            {"name": "hello", "type": "script", "config": {"command": "echo hi", "timeout": 5}},
        ],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "good.yaml")
    assert defn is not None, f"Should load: {errors}"
    assert defn.id == "good01"
    assert defn.name == "Good Agent"
    assert len(defn.steps) == 1
    assert defn.steps[0].type == StepType.SCRIPT
    assert defn.steps[0].name == "hello"
    assert defn.concurrency == 1
    assert defn.schedule is None
    print("Test 1 PASS: Single valid agent loaded")

    # ===== Test 2: Agent with schedule and concurrency =====
    write_yaml("scheduled.yaml", {
        "id": "sched01",
        "name": "Scheduled Agent",
        "schedule": "30s",
        "concurrency": 3,
        "steps": [
            {"name": "tick", "type": "script", "config": {"command": "echo tick", "timeout": 5}},
        ],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "scheduled.yaml")
    assert defn is not None
    assert defn.schedule == "30s"
    assert defn.concurrency == 3
    print("Test 2 PASS: Schedule and concurrency parsed")

    # ===== Test 3: Multi-step agent =====
    write_yaml("multi.yaml", {
        "id": "multi01",
        "name": "Multi Agent",
        "steps": [
            {"name": "fetch", "type": "script", "config": {"command": "echo 1", "timeout": 5}},
            {"name": "pull", "type": "pull", "config": {"source": "other"}},
            {"name": "send", "type": "message", "config": {"target": "other", "mode": "fire_and_forget"}},
        ],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "multi.yaml")
    assert defn is not None
    assert len(defn.steps) == 3
    assert defn.steps[0].type == StepType.SCRIPT
    assert defn.steps[1].type == StepType.PULL
    assert defn.steps[2].type == StepType.MESSAGE
    print("Test 3 PASS: Multi-step with mixed types")

    # ===== Test 4: Missing required fields =====
    write_yaml("bad_no_id.yaml", {
        "name": "No ID",
        "steps": [{"type": "script", "config": {"command": "echo x"}}],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "bad_no_id.yaml")
    assert defn is None
    assert any("id" in str(e) for e in errors)
    print("Test 4 PASS: Missing 'id' rejected")

    # ===== Test 5: Invalid step type =====
    write_yaml("bad_type.yaml", {
        "id": "badtype", "name": "Bad Type",
        "steps": [{"name": "x", "type": "quantum", "config": {}}],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "bad_type.yaml")
    assert defn is None
    assert any("quantum" in str(e) for e in errors)
    print("Test 5 PASS: Invalid step type rejected")

    # ===== Test 6: Missing step config =====
    write_yaml("bad_config.yaml", {
        "id": "badcfg", "name": "Bad Config",
        "steps": [{"name": "x", "type": "script"}],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "bad_config.yaml")
    assert defn is None
    assert any("config" in str(e) for e in errors)
    print("Test 6 PASS: Missing step config rejected")

    # ===== Test 7: Empty steps list =====
    write_yaml("bad_empty.yaml", {
        "id": "empty", "name": "Empty",
        "steps": [],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "bad_empty.yaml")
    assert defn is None
    assert any("non-empty" in str(e) for e in errors)
    print("Test 7 PASS: Empty steps rejected")

    # ===== Test 8: load_agents_dir — loads good, skips bad =====
    agents, errors = load_agents_dir(TEST_DIR / "agents")
    good_ids = {a.id for a in agents}
    assert "good01" in good_ids
    assert "sched01" in good_ids
    assert "multi01" in good_ids
    assert "badtype" not in good_ids
    assert "badcfg" not in good_ids
    assert len(errors) > 0  # bad files produce errors
    print(f"Test 8 PASS: Directory load — {len(agents)} valid, {len(errors)} errors")

    # ===== Test 9: Duplicate ID detection =====
    # Create a subdirectory agent with the same ID as an existing flat file
    dup_dir = TEST_DIR / "agents" / "dup_agent"
    dup_dir.mkdir(parents=True, exist_ok=True)
    (dup_dir / "agent.yaml").write_text(yaml.dump({
        "id": "good01",  # same as good.yaml
        "name": "Duplicate",
        "steps": [{"name": "x", "type": "script", "config": {"command": "echo dup"}}],
    }, default_flow_style=False), encoding="utf-8")
    agents, errors = load_agents_dir(TEST_DIR / "agents")
    # good01 from dup_agent/agent.yaml (subdir) loads first, then good.yaml (flat) is silently skipped.
    # But if two subdirectories have the same ID, that's an error.
    # Create a second subdir with the same ID to test real duplicate detection.
    dup_dir2 = TEST_DIR / "agents" / "dup_agent2"
    dup_dir2.mkdir(parents=True, exist_ok=True)
    (dup_dir2 / "agent.yaml").write_text(yaml.dump({
        "id": "good01",  # same ID again
        "name": "Duplicate 2",
        "steps": [{"name": "x", "type": "script", "config": {"command": "echo dup2"}}],
    }, default_flow_style=False), encoding="utf-8")
    agents, errors = load_agents_dir(TEST_DIR / "agents")
    dup_errors = [e for e in errors if "duplicate" in str(e)]
    assert len(dup_errors) == 1
    print("Test 9 PASS: Duplicate ID detected")
    # Clean up dups
    shutil.rmtree(dup_dir)
    shutil.rmtree(dup_dir2)

    # ===== Test 10: Config loading with defaults =====
    config = load_config(TEST_DIR / "nonexistent.yaml")
    assert config.data_dir == Path.home() / ".atn"
    assert config.agents_dir == Path("agents")
    assert config.providers == {}
    print("Test 10 PASS: Default config")

    # ===== Test 11: Config from YAML =====
    config_path = TEST_DIR / "config.yaml"
    config_path.write_text(yaml.dump({
        "data_dir": str(TEST_DIR.resolve()),
        "agents_dir": str((TEST_DIR / "agents").resolve()),
        "providers": {
            "anthropic": {
                "api_key": "sk-test-1234",
                "default_model": "claude-sonnet-4-20250514",
            },
            "openai": {
                "api_key": "sk-openai-5678",
                "default_model": "gpt-4o",
                "base_url": "https://custom.endpoint/v1",
            },
        },
    }), encoding="utf-8")
    config = load_config(config_path)
    assert config.data_dir == TEST_DIR.resolve()
    assert config.agents_dir == (TEST_DIR / "agents").resolve()
    assert "anthropic" in config.providers
    assert config.providers["anthropic"].api_key == "sk-test-1234"
    assert config.providers["anthropic"].default_model == "claude-sonnet-4-20250514"
    assert config.providers["openai"].base_url == "https://custom.endpoint/v1"
    print("Test 11 PASS: Config from YAML with providers")

    # ===== Test 12: Full integration — load from files, register, run =====
    # Clean out bad files for a clean integration test
    for bad in ("bad_no_id.yaml", "bad_type.yaml", "bad_config.yaml", "bad_empty.yaml"):
        p = TEST_DIR / "agents" / bad
        if p.exists():
            p.unlink()

    bus = EventBus()
    test_config = ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents")
    rt = Runtime(bus, data_dir=TEST_DIR, config=test_config)
    await rt.start()

    agents, errors = load_agents_dir(TEST_DIR / "agents")
    assert len(errors) == 0, f"Unexpected errors: {errors}"
    for defn in agents:
        await rt.register_agent(defn)
        if defn.schedule:
            await rt.activate_agent(defn.id)

    assert len(rt.list_agents()) == 3

    # Run the echo agent
    eid = await rt.trigger_run("good01")
    await asyncio.sleep(2)
    rec = rt.execution_log.get_latest("good01")
    assert rec.status == ExecutionStatus.COMPLETED
    print("Test 12 PASS: Full integration — YAML -> register -> run -> completed")

    # Run the multi-step agent (pull and message will fail since targets don't exist,
    # but the script step should work — testing that the loader wired types correctly)
    eid = await rt.trigger_run("multi01")
    await asyncio.sleep(2)
    rec = rt.execution_log.get_latest("multi01")
    # First step (script) should have succeeded
    assert rec.step_results[0].status == ExecutionStatus.COMPLETED
    assert rec.step_results[0].step_type == StepType.SCRIPT
    print("Test 13 PASS: Multi-step agent — step types correctly wired from YAML")

    await rt.stop()

    # ===== Test 14: Budgets field parsed correctly =====
    write_yaml("budgeted.yaml", {
        "id": "budgeted01",
        "name": "Budgeted Agent",
        "budgets": {"gemini": 50000, "claude_max": 100000},
        "steps": [
            {"name": "think", "type": "script", "config": {"command": "echo ok"}},
        ],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "budgeted.yaml")
    assert defn is not None, f"Should load: {errors}"
    assert defn.budgets == {"gemini": 50000, "claude_max": 100000}
    print("Test 14 PASS: Budgets parsed correctly")

    # ===== Test 15: Invalid budgets rejected =====
    write_yaml("bad_budget.yaml", {
        "id": "badbud", "name": "Bad Budget",
        "budgets": {"gemini": "lots"},
        "steps": [
            {"name": "x", "type": "script", "config": {"command": "echo x"}},
        ],
    })
    defn, errors = load_agent_file(TEST_DIR / "agents" / "bad_budget.yaml")
    assert defn is None
    assert any("budgets" in str(e) for e in errors)
    print("Test 15 PASS: Invalid budgets rejected")

    # ===== Test 16: No budgets field defaults to empty =====
    defn, errors = load_agent_file(TEST_DIR / "agents" / "good.yaml")
    assert defn is not None
    assert defn.budgets == {}
    print("Test 16 PASS: Missing budgets defaults to empty dict")

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("\nAll loader + config tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
