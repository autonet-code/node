"""Tests for the cognitive loop — multi-turn tool use + tool execution.

Uses mock providers to verify:
  - Multi-turn loop mechanics (tool calls -> tool results -> next turn)
  - Tool execution against a real Runtime
  - Tool definitions are complete
"""
import asyncio
import shutil
from pathlib import Path
from typing import Any

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import (
    AgentDefinition,
    ExecutionStatus,
    StepDefinition,
    StepType,
)
from atn.agent_tools import (
    execute_tool,
    get_tool_definitions,
    get_tool_executor,
)
from atn.providers.base import Provider, ProviderResponse, ToolCall, ToolDefinition, Usage
from atn.runtime import Runtime
from atn.steps.cognitive import CognitiveStepExecutor


TEST_DIR = Path("./test_data_cognitive_loop")


# ---------------------------------------------------------------------------
# Mock provider that follows a scripted conversation
# ---------------------------------------------------------------------------

class ScriptedProvider(Provider):
    """Mock provider that returns pre-scripted responses in sequence."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_log: list[dict] = []

    @property
    def name(self) -> str:
        return "mock"

    async def send(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.call_log.append({
            "messages": messages,
            "system": system,
            "tools": [t.name for t in tools] if tools else [],
        })
        if self._call_index >= len(self._responses):
            return ProviderResponse(text="(no more scripted responses)", stop_reason="end_turn")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


def setup():
    """Create a fresh test directory."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)


async def test():
    setup()

    # ==================================================================
    # Test 1: Single-turn cognitive step (backward compat)
    # ==================================================================
    print("Test 1: Single-turn cognitive step...")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents"))
    await rt.start()

    mock = ScriptedProvider([
        ProviderResponse(
            text="Hello world",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="mock-v1",
        ),
    ])
    cognitive = rt._executors[StepType.COGNITIVE]
    assert isinstance(cognitive, CognitiveStepExecutor)
    cognitive.register_provider(mock)

    agent = AgentDefinition(
        id="single_turn",
        name="Single Turn Test",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={"provider": "mock", "prompt": "Say hello"},
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.activate_agent(agent.id)
    eid = await rt.trigger_run(agent.id)
    await asyncio.sleep(0.05)  # mocked provider completes instantly

    rec = rt.execution_log.get_latest(agent.id)
    assert rec.status == ExecutionStatus.COMPLETED, f"Expected COMPLETED, got {rec.status}"
    assert rec.step_results[0].output["text"] == "Hello world"
    assert rec.step_results[0].output["stop_reason"] == "end_turn"
    assert "turns" not in rec.step_results[0].output  # single turn, no history
    print("  PASS: Single-turn returns text, no turn history")

    await rt.stop()

    # ==================================================================
    # Test 2: Multi-turn with tool calls (no executor — returns tool_calls)
    # ==================================================================
    print("Test 2: Multi-turn with tool calls but no executor...")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents"))
    await rt.start()

    mock = ScriptedProvider([
        ProviderResponse(
            text="Let me check",
            tool_calls=[ToolCall(id="tc1", name="get_weather", input={"city": "London"})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=20, output_tokens=10),
            model="mock-v1",
        ),
    ])
    cognitive = rt._executors[StepType.COGNITIVE]
    assert isinstance(cognitive, CognitiveStepExecutor)
    cognitive.register_provider(mock)

    agent = AgentDefinition(
        id="no_exec",
        name="No Executor Test",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={"provider": "mock", "prompt": "What's the weather?", "max_turns": 5},
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.activate_agent(agent.id)
    await rt.trigger_run(agent.id)
    await asyncio.sleep(0.05)  # mocked provider completes instantly

    rec = rt.execution_log.get_latest(agent.id)
    assert rec.status == ExecutionStatus.COMPLETED
    assert rec.step_results[0].output["tool_calls"][0]["name"] == "get_weather"
    assert "turns" not in rec.step_results[0].output  # only 1 turn, no history
    # Provider should only have been called once (no executor -> no loop)
    assert len(mock.call_log) == 1
    print("  PASS: Tool calls returned as-is when no executor configured")

    await rt.stop()

    # ==================================================================
    # Test 3: Multi-turn with ATN tools (legacy tool_executors alias)
    # ==================================================================
    print("Test 3: Multi-turn tool loop (legacy executor-set name)...")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents"))
    await rt.start()

    # Script: turn 1 = list_agents tool call, turn 2 = end_turn with summary
    mock = ScriptedProvider([
        # Turn 1: LLM calls list_agents
        ProviderResponse(
            text="Let me check what agents exist.",
            tool_calls=[ToolCall(id="tc1", name="list_agents", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=100, output_tokens=30),
            model="mock-v1",
        ),
        # Turn 2: LLM sees the tool result and responds
        ProviderResponse(
            text="There are no agents registered yet.",
            stop_reason="end_turn",
            usage=Usage(input_tokens=150, output_tokens=20),
            model="mock-v1",
        ),
    ])
    cognitive = rt._executors[StepType.COGNITIVE]
    assert isinstance(cognitive, CognitiveStepExecutor)
    cognitive.register_provider(mock)

    agent = AgentDefinition(
        id="loop_test",
        name="Loop Test",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "mock",
                "prompt": "What agents exist?",
                "max_turns": 10,
                # LEGACY-DATA: persisted agent defs may still name the old
                # executor set; it must keep resolving to the ATN tools.
                "tool_executors": "orchestrator",
            },
            name="orchestrate",
        )],
    )
    await rt.register_agent(agent)
    await rt.activate_agent(agent.id)
    await rt.trigger_run(agent.id)
    await asyncio.sleep(0.05)  # mocked provider completes instantly

    rec = rt.execution_log.get_latest(agent.id)
    assert rec.status == ExecutionStatus.COMPLETED
    output = rec.step_results[0].output
    assert output["text"] == "There are no agents registered yet."
    assert output["mode"] == "orchestrate"
    # Usage is cumulative
    assert output["usage"]["input_tokens"] == 250  # 100 + 150
    assert output["usage"]["output_tokens"] == 50   # 30 + 20
    # Provider was called twice (via send_orchestrate -> send_stream loop)
    assert len(mock.call_log) == 2
    # Second call should have assistant + tool_result messages
    msgs = mock.call_log[1]["messages"]
    assert len(msgs) == 3  # user, assistant (tool_use), user (tool_result)
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    print("  PASS: 2-turn loop with list_agents tool")

    await rt.stop()

    # ==================================================================
    # Test 5: Tool definitions are complete
    # ==================================================================
    print("Test 5: Tool definitions...")
    tools = get_tool_definitions()
    tool_names = {t.name for t in tools}
    expected = {
        "find_services", "register_service", "request_service", "pay_for_service",
        "list_agents", "get_agent", "create_agent", "update_agent", "remove_agent",
        "activate_agent", "deactivate_agent", "trigger_run",
        "get_execution", "get_output", "kill_execution", "kill_agent",
        "post_message", "get_snapshot", "get_history", "list_connectors",
        "add_connector", "remove_connector", "get_connector_tools",
        "use_connector",
        # Unified tools
        "list_tools", "use_tool",
        # Tool substrate (docs/tool_substrate.md)
        "register_tool", "attest_tools", "publish_tool", "vet_tool",
        "adopt_tool", "probe_tools",
        # Validator rail (vetting bundle)
        "check_evidence", "run_trial",
        # Planning & goal tools
        "get_goals", "add_goal", "update_goal",
        "get_projects", "add_project", "update_project",
        "get_credit_budget", "set_credit_budget", "get_usage",
        "propose_task", "list_tasks", "get_user_profile", "update_user_profile",
        # Delegation
        "delegate_status", "delegate_message", "delegate_collect",
        "get_latest_thought", "get_children_status",
        # Budgets & chain
        "get_my_budget_status", "register_on_chain", "metering_report",
        # Manual compaction (spec §15)
        "compact_agent",
    }
    assert tool_names == expected, f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"
    # Each tool has a name, description, and input_schema
    for t in tools:
        assert t.name, "Tool must have a name"
        assert t.description, f"Tool {t.name} must have a description"
        assert isinstance(t.input_schema, dict), f"Tool {t.name} must have input_schema"
    print(f"  PASS: {len(tools)} tools defined with schemas")

    # ==================================================================
    # Test 6: Individual tool execution
    # ==================================================================
    print("Test 6: Individual tool execution...")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents"))
    await rt.start()

    # list_agents on empty runtime
    result = await execute_tool("list_agents", {}, rt)
    assert result["agents"] == []
    print("  PASS: list_agents returns empty list")

    # create_agent
    result = await execute_tool("create_agent", {
        "id": "test01",
        "name": "Test Agent",
        "steps": [{"name": "run", "type": "script", "config": {"command": "echo ok"}}],
    }, rt)
    assert result["agent_id"] == "test01"
    assert result["status"] == "registered"
    print("  PASS: create_agent registers agent")

    # list_agents now has one
    result = await execute_tool("list_agents", {}, rt)
    assert len(result["agents"]) == 1
    assert result["agents"][0]["id"] == "test01"
    print("  PASS: list_agents shows created agent")

    # get_agent
    result = await execute_tool("get_agent", {"agent_id": "test01"}, rt)
    assert result["id"] == "test01"
    assert result["name"] == "Test Agent"
    assert len(result["steps"]) == 1
    print("  PASS: get_agent returns details")

    # activate_agent
    result = await execute_tool("activate_agent", {"agent_id": "test01"}, rt)
    assert result["status"] == "active"
    print("  PASS: activate_agent works")

    # trigger_run
    result = await execute_tool("trigger_run", {"agent_id": "test01"}, rt)
    assert "execution_id" in result
    assert result["status"] == "started"
    await asyncio.sleep(0.5)  # real echo subprocess
    print("  PASS: trigger_run starts execution")

    # get_execution
    result = await execute_tool("get_execution", {"agent_id": "test01"}, rt)
    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "completed"
    print("  PASS: get_execution returns result")

    # get_output
    result = await execute_tool("get_output", {"agent_id": "test01"}, rt)
    assert result["agent_id"] == "test01"
    print("  PASS: get_output returns data")

    # get_snapshot
    result = await execute_tool("get_snapshot", {}, rt)
    assert "agents" in result
    assert "test01" in result["agents"]
    print("  PASS: get_snapshot returns system state")

    # post_message
    result = await execute_tool("post_message", {"target": "test01", "type": "info", "data": {"msg": "hi"}}, rt)
    assert result["target"] == "test01"
    print("  PASS: post_message delivers")

    # deactivate_agent
    result = await execute_tool("deactivate_agent", {"agent_id": "test01"}, rt)
    assert result["status"] == "stopped"
    print("  PASS: deactivate_agent works")

    # remove_agent
    result = await execute_tool("remove_agent", {"agent_id": "test01"}, rt)
    assert result["status"] == "removed"
    result = await execute_tool("get_agent", {"agent_id": "test01"}, rt)
    assert "error" in result
    print("  PASS: remove_agent removes agent")

    # Unknown tool
    result = await execute_tool("nonexistent", {}, rt)
    assert "error" in result
    print("  PASS: unknown tool returns error")

    # Error cases
    result = await execute_tool("get_agent", {"agent_id": "nope"}, rt)
    assert "error" in result
    print("  PASS: get_agent on missing agent returns error")

    result = await execute_tool("create_agent", {
        "id": "bad", "name": "Bad", "steps": [{"name": "x", "type": "quantum", "config": {}}],
    }, rt)
    assert "error" in result
    print("  PASS: create_agent with invalid step type returns error")

    await rt.stop()

    # ==================================================================
    # Test 8: Max turns limit is respected
    # ==================================================================
    print("Test 8: Max turns limit...")
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents"))
    await rt.start()

    # Provider that always returns tool calls (infinite loop attempt)
    infinite_responses = [
        ProviderResponse(
            text=f"Turn {i+1}",
            tool_calls=[ToolCall(id=f"tc{i}", name="list_agents", input={})],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        for i in range(20)
    ]
    mock = ScriptedProvider(infinite_responses)
    cognitive = rt._executors[StepType.COGNITIVE]
    assert isinstance(cognitive, CognitiveStepExecutor)
    cognitive.register_provider(mock)

    agent = AgentDefinition(
        id="max_turns_test",
        name="Max Turns Test",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "mock",
                "prompt": "Do something",
                "max_turns": 3,
                "tool_executors": "atn",
            },
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.activate_agent(agent.id)
    await rt.trigger_run(agent.id)
    await asyncio.sleep(0.05)  # mocked provider completes instantly

    rec = rt.execution_log.get_latest(agent.id)
    assert rec.status == ExecutionStatus.COMPLETED
    # Should have stopped at 3 turns even though provider keeps returning tool_use
    assert len(mock.call_log) == 3
    print("  PASS: Stopped at max_turns=3")

    await rt.stop()

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("\nAll cognitive loop tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
