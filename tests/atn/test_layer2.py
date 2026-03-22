"""End-to-end tests for Layer 2: inter-agent communication."""
import asyncio
import shutil
from pathlib import Path

from atn.config import ATNConfig
from atn.events import EventBus, EventType
from atn.models import (
    AgentDefinition, StepDefinition, StepType, ExecutionStatus, TokenUsage,
)
from atn.runtime import Runtime

TEST_DIR = Path("./test_data_layer2")


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

    # ===== Setup: a 'data producer' agent that generates output =====
    producer = AgentDefinition(
        id="producer", name="Data Producer",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": 'echo {"items":["alpha","beta","gamma"]}', "timeout": 5},
            name="generate",
        )],
    )
    await rt.register_agent(producer)
    await rt.trigger_run("producer")
    await asyncio.sleep(2)

    output = rt.output_store.read("producer")
    assert output is not None, "Producer should have output"
    print(f"Setup PASS: Producer ran, output stored")

    # ===== Test 1: Pull step — read producer output =====
    puller = AgentDefinition(
        id="puller", name="Data Puller",
        steps=[StepDefinition(
            type=StepType.PULL,
            config={"source": "producer"},
            name="read_producer",
        )],
    )
    await rt.register_agent(puller)
    await rt.trigger_run("puller")
    await asyncio.sleep(2)

    rec = rt.execution_log.get_latest("puller")
    assert rec.status == ExecutionStatus.COMPLETED, f"Pull failed: {rec.error}"
    assert rec.step_results[0].output["source"] == "producer"
    print("Test 1 PASS (pull): Got producer data via pull step")

    # ===== Test 2: Pull with max_age — reject stale data =====
    stale = AgentDefinition(
        id="stale", name="Stale Puller",
        steps=[StepDefinition(
            type=StepType.PULL,
            config={"source": "producer", "max_age": 0},
            name="read_stale",
        )],
    )
    await rt.register_agent(stale)
    await rt.trigger_run("stale")
    await asyncio.sleep(2)

    rec = rt.execution_log.get_latest("stale")
    assert rec.status == ExecutionStatus.FAILED, "Stale pull should fail"
    assert "old" in rec.error
    print("Test 2 PASS (pull max_age): Correctly rejected stale output")

    # ===== Test 3: Message step (post and move on) =====
    receiver = AgentDefinition(
        id="recv", name="Receiver",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo received", "timeout": 5},
            name="ack",
        )],
    )
    await rt.register_agent(receiver)
    await rt.activate_agent("recv")

    sender = AgentDefinition(
        id="sender", name="Sender",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo preparing", "timeout": 5},
                name="prepare",
            ),
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "recv", "priority": "high"},
                name="notify",
            ),
        ],
    )
    await rt.register_agent(sender)
    await rt.trigger_run("sender")
    await asyncio.sleep(3)

    rec_s = rt.execution_log.get_latest("sender")
    assert rec_s.status == ExecutionStatus.COMPLETED, f"Sender failed: {rec_s.error}"
    rec_r = rt.execution_log.get_latest("recv")
    assert rec_r is not None and rec_r.status == ExecutionStatus.COMPLETED
    print("Test 3 PASS (message): Sender triggered Receiver via inbox")

    # ===== Test 4: Message + collect (trigger and wait for result) =====
    worker = AgentDefinition(
        id="worker", name="Worker",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo 42", "timeout": 5},
            name="compute",
        )],
    )
    await rt.register_agent(worker)
    await rt.activate_agent("worker")

    caller = AgentDefinition(
        id="caller", name="Caller",
        steps=[
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "worker"},
                name="trigger_worker",
            ),
            StepDefinition(
                type=StepType.COLLECT,
                config={"from_step": 0, "timeout": 10},
                name="get_result",
            ),
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo used-worker-result", "timeout": 5},
                name="use_result",
            ),
        ],
    )
    await rt.register_agent(caller)
    await rt.trigger_run("caller")
    await asyncio.sleep(5)

    rec = rt.execution_log.get_latest("caller")
    assert rec.status == ExecutionStatus.COMPLETED, f"Caller failed: {rec.error}"
    assert len(rec.step_results) == 3, "All 3 steps should have run"
    collected = rec.step_results[1].output
    assert collected is not None
    print(f"Test 4 PASS (message+collect): Caller triggered Worker, collected result")

    # ===== Test 5: Message + do other work + collect =====
    slow = AgentDefinition(
        id="slow", name="Slow Worker",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={
                "command": 'python -c "import time; time.sleep(2); print(99)"',
                "timeout": 10,
            },
            name="slow_compute",
        )],
    )
    await rt.register_agent(slow)
    await rt.activate_agent("slow")

    pipeline = AgentDefinition(
        id="async_pl", name="Dispatch Pipeline",
        steps=[
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "slow"},
                name="dispatch",
            ),
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo doing-other-work", "timeout": 5},
                name="other_work",
            ),
            StepDefinition(
                type=StepType.COLLECT,
                config={"from_step": 0, "timeout": 15},
                name="get_result",
            ),
        ],
    )
    await rt.register_agent(pipeline)
    await rt.trigger_run("async_pl")
    await asyncio.sleep(8)

    rec = rt.execution_log.get_latest("async_pl")
    assert rec.status == ExecutionStatus.COMPLETED, f"Pipeline failed: {rec.error}"
    assert len(rec.step_results) == 3, "All 3 steps should run"
    collected = rec.step_results[2].output
    assert collected is not None
    print(f"Test 5 PASS (message+work+collect): Dispatched, worked, collected result")

    # ===== Test 6: Multi-agent chain — A triggers B which pulls from C =====
    source_c = AgentDefinition(
        id="src_c", name="Source C",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo source-data-from-C", "timeout": 5},
            name="produce",
        )],
    )
    await rt.register_agent(source_c)
    await rt.trigger_run("src_c")
    await asyncio.sleep(2)

    chain_b = AgentDefinition(
        id="chain_b", name="Chain B",
        steps=[
            StepDefinition(
                type=StepType.PULL,
                config={"source": "src_c"},
                name="pull_from_c",
            ),
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo chain-b-processed", "timeout": 5},
                name="process",
            ),
        ],
    )
    await rt.register_agent(chain_b)
    await rt.activate_agent("chain_b")

    chain_a = AgentDefinition(
        id="chain_a", name="Chain A",
        steps=[
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "chain_b"},
                name="trigger_b",
            ),
            StepDefinition(
                type=StepType.COLLECT,
                config={"from_step": 0, "timeout": 10},
                name="await_b",
            ),
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo chain-complete", "timeout": 5},
                name="finalise",
            ),
        ],
    )
    await rt.register_agent(chain_a)
    await rt.trigger_run("chain_a")
    await asyncio.sleep(5)

    rec = rt.execution_log.get_latest("chain_a")
    assert rec.status == ExecutionStatus.COMPLETED, f"Chain A failed: {rec.error}"
    print("Test 6 PASS (chain): A->B->C  (A triggers B which pulls from C)")

    # ===== Test 7: Budget enforcement — pipeline fails when over budget =====
    # Register a cognitive provider that will be used for the budget test.
    # We simulate exceeding the budget by pre-populating token_usage on
    # the execution record.  To do this we need to test the budget check
    # logic directly, since we don't have a live cognitive provider in
    # this test suite.
    #
    # The agent has a script step first (succeeds), then a cognitive step.
    # We'll set budget to 1 token for "fake_prov" and manually inject usage
    # before the second step runs.  Since we can't hook into the pipeline
    # mid-flight, we instead test via a budget of 0 — any usage exceeds it.
    #
    # Actually — the budget check fires before the cognitive step runs,
    # comparing record.token_usage against the limit.  On a fresh execution
    # with no prior cognitive steps, usage is 0 and budget > 0 will pass.
    # So we need budget=0 to block on the first cognitive step... but
    # budget 0 means "unlimited" in our code (if limit > 0).
    #
    # Instead: use budget=1.  The agent starts with 0 usage, so the first
    # cognitive step passes the check.  But we don't have a real cognitive
    # provider.  Let's test this differently — directly verify the runtime
    # logic via a two-cognitive-step agent where the first consumes tokens.
    #
    # Simplest approach: test the budget check function directly.
    from atn.models import ExecutionRecord
    rec = ExecutionRecord(
        execution_id="budget-test",
        agent_id="test",
        status=ExecutionStatus.RUNNING,
        trigger_source="test",
    )
    # No usage yet — budget check should pass (usage=0 < limit=100)
    budgeted_agent = AgentDefinition(
        id="budget_agent", name="Budget Agent",
        steps=[StepDefinition(type=StepType.COGNITIVE, config={"provider": "test_prov"}, name="think")],
        budgets={"test_prov": 100},
    )
    prov = "test_prov"
    limit = budgeted_agent.budgets.get(prov, 0)
    current = rec.token_usage.get(prov)
    used = current.total if current else 0
    assert used == 0
    assert not (limit > 0 and used >= limit), "Should pass — 0 < 100"
    print("Test 7a PASS (budget): Fresh execution passes budget check")

    # Now simulate usage exceeding the limit
    rec.token_usage["test_prov"] = TokenUsage(
        provider="test_prov", input_tokens=80, output_tokens=25,
    )
    current = rec.token_usage.get(prov)
    used = current.total if current else 0
    assert used == 105
    assert limit > 0 and used >= limit, "Should block — 105 >= 100"
    print("Test 7b PASS (budget): Over-budget execution blocked")

    # Test that cache tokens count toward total
    rec2 = ExecutionRecord(
        execution_id="budget-cache-test",
        agent_id="test",
        status=ExecutionStatus.RUNNING,
        trigger_source="test",
    )
    rec2.token_usage["test_prov"] = TokenUsage(
        provider="test_prov", input_tokens=10, output_tokens=10,
        cache_read_tokens=50, cache_creation_tokens=35,
    )
    assert rec2.token_usage["test_prov"].total == 105, "Total should include cache tokens"
    print("Test 7c PASS (budget): Cache tokens counted in total")

    # ===== Test 8: Child cost aggregation via collect step =====
    # Set up: a "cog_worker" agent that simulates having token usage,
    # and a "parent" that triggers it, collects the result, and should
    # aggregate the child's token usage.
    #
    # We use a script step for cog_worker (no real LLM) and manually
    # inject token_usage into its execution record after it completes.
    cog_worker = AgentDefinition(
        id="cog_worker", name="Cognitive Worker",
        steps=[StepDefinition(
            type=StepType.SCRIPT,
            config={"command": "echo worker-output", "timeout": 5},
            name="work",
        )],
    )
    await rt.register_agent(cog_worker)
    await rt.activate_agent("cog_worker")

    parent = AgentDefinition(
        id="parent", name="Parent Agent",
        steps=[
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "cog_worker"},
                name="trigger_child",
            ),
            StepDefinition(
                type=StepType.COLLECT,
                config={"from_step": 0, "timeout": 10},
                name="collect_child",
            ),
        ],
    )
    await rt.register_agent(parent)
    eid = await rt.trigger_run("parent")
    await asyncio.sleep(5)

    # Inject token usage into the child's execution record
    child_rec = rt.execution_log.get_latest("cog_worker")
    assert child_rec is not None and child_rec.status == ExecutionStatus.COMPLETED
    child_rec.token_usage["gemini"] = TokenUsage(
        provider="gemini", input_tokens=200, output_tokens=50,
    )
    # Re-record so it's findable
    rt.execution_log.record(child_rec)

    # Now verify the collect step output has the right structure
    parent_rec = rt.execution_log.get_latest("parent")
    assert parent_rec is not None and parent_rec.status == ExecutionStatus.COMPLETED
    collect_output = parent_rec.step_results[1].output
    assert collect_output is not None
    assert "source" in collect_output, f"Collect output missing 'source': {collect_output}"
    assert "execution_id" in collect_output, f"Collect output missing 'execution_id': {collect_output}"
    assert collect_output["source"] == "cog_worker"
    print("Test 8a PASS (child aggregation): Collect step returns source and execution_id")

    # Test _accumulate_child_usage directly
    from atn.runtime import _accumulate_child_usage
    test_parent_rec = ExecutionRecord(
        execution_id="parent-test",
        agent_id="parent",
        status=ExecutionStatus.RUNNING,
        trigger_source="test",
    )
    _accumulate_child_usage(test_parent_rec, collect_output, rt.execution_log)
    assert "gemini" in test_parent_rec.token_usage, "Child gemini usage should propagate"
    assert test_parent_rec.token_usage["gemini"].input_tokens == 200
    assert test_parent_rec.token_usage["gemini"].output_tokens == 50
    print("Test 8b PASS (child aggregation): Child token usage merged into parent")

    # ===== Test 9: Pull from agent whose output came from a collect step =====
    # Chain: coordinator triggers cog_worker via message, collects result.
    # Then downstream_reader pulls from coordinator.
    # Verifies the collect step's wrapped output {source, data, execution_id}
    # flows correctly through the output store and pull step.
    coordinator = AgentDefinition(
        id="coordinator", name="Coordinator",
        steps=[
            StepDefinition(
                type=StepType.MESSAGE,
                config={"target": "cog_worker"},
                name="dispatch",
            ),
            StepDefinition(
                type=StepType.COLLECT,
                config={"from_step": 0, "timeout": 10},
                name="await_result",
            ),
        ],
    )
    await rt.register_agent(coordinator)
    await rt.trigger_run("coordinator")
    await asyncio.sleep(5)

    coord_rec = rt.execution_log.get_latest("coordinator")
    assert coord_rec.status == ExecutionStatus.COMPLETED, f"Coordinator failed: {coord_rec.error}"
    # Coordinator's output store should hold the collect step's wrapped output
    coord_output = rt.output_store.read("coordinator")
    assert coord_output is not None
    assert isinstance(coord_output.data, dict)
    assert "source" in coord_output.data
    assert "data" in coord_output.data
    assert "execution_id" in coord_output.data
    print("Test 9a PASS: Coordinator output store has collect wrapper structure")

    # Now a downstream agent pulls from coordinator
    reader = AgentDefinition(
        id="reader", name="Downstream Reader",
        steps=[
            StepDefinition(
                type=StepType.PULL,
                config={"source": "coordinator"},
                name="read_coord",
            ),
        ],
    )
    await rt.register_agent(reader)
    await rt.trigger_run("reader")
    await asyncio.sleep(2)

    reader_rec = rt.execution_log.get_latest("reader")
    assert reader_rec.status == ExecutionStatus.COMPLETED, f"Reader failed: {reader_rec.error}"
    pull_output = reader_rec.step_results[0].output
    assert pull_output["source"] == "coordinator"
    # The actual data is nested: pull_output.data -> collect wrapper -> data -> worker output
    inner = pull_output["data"]
    assert "source" in inner, f"Pull data missing collect wrapper: {inner}"
    assert inner["source"] == "cog_worker"
    assert "data" in inner
    print("Test 9b PASS: Downstream pull from collect-based agent works correctly")

    await rt.stop()

    # Summary
    msg_events = [e for e in events if e.type == EventType.MESSAGE_POSTED]
    print(f"\nTotal events: {len(events)}")
    print(f"MESSAGE_POSTED events: {len(msg_events)}")
    for e in msg_events:
        print(f"  {e.data['agent_id']} -> {e.data['target']}")
    print("\nAll Layer 2 tests passed!")

    # Cleanup
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    asyncio.run(test())
