"""Tests for the cognitive step and provider abstraction.

Two modes:
  - Unit tests with a mock provider (always runs)
  - Live integration test with Anthropic (only if ANTHROPIC_API_KEY is set)
"""
import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from atn.config import ATNConfig, ProviderConfig
from atn.events import EventBus, EventType
from atn.models import (
    AgentDefinition, StepDefinition, StepType, ExecutionStatus,
)
from atn.providers.base import (
    Provider, ProviderResponse, ToolCall, ToolDefinition, Usage,
)
from atn.runtime import Runtime
from atn.steps.cognitive import CognitiveStepExecutor


TEST_DIR = Path("./test_data_cognitive")


# ---------------------------------------------------------------------------
# Mock provider for unit tests
# ---------------------------------------------------------------------------

class MockProvider(Provider):
    """A fake LLM provider that returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.response = ProviderResponse(
            text="Mock LLM response",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            model="mock-model",
        )

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
        self.calls.append({
            "messages": messages,
            "system": system,
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "temperature": temperature,
        })
        return self.response


async def test_unit():
    """Unit tests with mock provider."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    bus = EventBus()
    events: list = []
    async def collect(e):
        events.append(e)
    bus.subscribe(None, collect)

    # Create runtime with no real providers
    test_config = ATNConfig(data_dir=TEST_DIR, agents_dir=TEST_DIR / "agents")
    rt = Runtime(bus, data_dir=TEST_DIR, config=test_config)

    # Manually register mock provider on the cognitive executor
    cognitive = rt._executors[StepType.COGNITIVE]
    mock = MockProvider()
    cognitive.register_provider(mock)

    await rt.start()

    # ===== Test 1: Basic cognitive step =====
    agent = AgentDefinition(
        id="thinker01", name="Thinker",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "mock",
                "system": "You are a helpful assistant.",
                "prompt": "What is 2+2?",
                "max_tokens": 100,
            },
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.trigger_run("thinker01")
    await asyncio.sleep(1)

    rec = rt.execution_log.get_latest("thinker01")
    assert rec.status == ExecutionStatus.COMPLETED, f"Failed: {rec.error}"
    assert rec.step_results[0].output["text"] == "Mock LLM response"
    assert rec.step_results[0].output["model"] == "mock-model"
    assert rec.step_results[0].output["usage"]["input_tokens"] == 10
    assert len(mock.calls) == 1
    assert mock.calls[0]["system"] == "You are a helpful assistant."
    assert "What is 2+2?" in mock.calls[0]["messages"][0]["content"]
    print("Test 1 PASS: Basic cognitive step works")

    # ===== Test 2: Prompt template with {prev} =====
    pipeline = AgentDefinition(
        id="pipeline01", name="Script+Cognitive Pipeline",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo raw-data-here", "timeout": 5},
                name="fetch",
            ),
            StepDefinition(
                type=StepType.COGNITIVE,
                config={
                    "provider": "mock",
                    "system": "Analyse data",
                    "prompt": "Please analyse this data: {prev}",
                },
                name="analyse",
            ),
        ],
    )
    await rt.register_agent(pipeline)
    mock.calls.clear()
    await rt.trigger_run("pipeline01")
    await asyncio.sleep(2)

    rec = rt.execution_log.get_latest("pipeline01")
    assert rec.status == ExecutionStatus.COMPLETED, f"Pipeline failed: {rec.error}"
    assert "raw-data-here" in mock.calls[0]["messages"][0]["content"]
    print("Test 2 PASS: Prompt template {prev} substitution works")

    # ===== Test 3: Missing provider fails clearly =====
    bad_agent = AgentDefinition(
        id="bad_provider", name="Bad Provider",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={"provider": "nonexistent", "prompt": "hello"},
            name="fail",
        )],
    )
    await rt.register_agent(bad_agent)
    await rt.trigger_run("bad_provider")
    await asyncio.sleep(1)

    rec = rt.execution_log.get_latest("bad_provider")
    assert rec.status == ExecutionStatus.FAILED
    assert "nonexistent" in rec.error
    assert "no configured provider" in rec.error.lower()
    print("Test 3 PASS: Missing provider fails with clear error")

    # ===== Test 4: Missing provider key in config =====
    no_provider = AgentDefinition(
        id="no_provider", name="No Provider",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={"prompt": "hello"},
            name="fail",
        )],
    )
    await rt.register_agent(no_provider)
    await rt.trigger_run("no_provider")
    await asyncio.sleep(1)

    rec = rt.execution_log.get_latest("no_provider")
    assert rec.status == ExecutionStatus.FAILED
    assert "provider" in rec.error.lower()
    print("Test 4 PASS: Missing provider config key fails clearly")

    # ===== Test 5: Tool calls in response =====
    mock.response = ProviderResponse(
        text="I'll check the weather.",
        tool_calls=[
            ToolCall(id="tc_001", name="get_weather", input={"city": "London"}),
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=15, output_tokens=20),
        model="mock-model",
    )
    tool_agent = AgentDefinition(
        id="tool_agent", name="Tool Agent",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "mock",
                "prompt": "What's the weather in London?",
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                ],
            },
            name="ask",
        )],
    )
    await rt.register_agent(tool_agent)
    mock.calls.clear()
    await rt.trigger_run("tool_agent")
    await asyncio.sleep(1)

    rec = rt.execution_log.get_latest("tool_agent")
    assert rec.status == ExecutionStatus.COMPLETED
    output = rec.step_results[0].output
    assert output["stop_reason"] == "tool_use"
    assert len(output["tool_calls"]) == 1
    assert output["tool_calls"][0]["name"] == "get_weather"
    assert output["tool_calls"][0]["input"] == {"city": "London"}
    # Verify tools were sent to provider
    assert mock.calls[0]["tools"] is not None
    assert mock.calls[0]["tools"][0].name == "get_weather"
    print("Test 5 PASS: Tool definitions sent and tool calls returned")

    # ===== Test 6: Events emitted for cognitive step =====
    step_events = [e for e in events if e.type in (EventType.STEP_STARTED, EventType.STEP_COMPLETED)
                   and e.data.get("step_type") == "cognitive"]
    assert len(step_events) >= 2  # at least one started + completed pair
    started = [e for e in step_events if e.type == EventType.STEP_STARTED]
    completed = [e for e in step_events if e.type == EventType.STEP_COMPLETED]
    assert len(started) > 0
    assert len(completed) > 0
    print("Test 6 PASS: Step events emitted for cognitive steps")

    # ===== Test 7: Previous output without explicit prompt =====
    mock.response = ProviderResponse(
        text="Analysed the data",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="mock-model",
    )
    implicit = AgentDefinition(
        id="implicit01", name="Implicit Prompt",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": "echo some-input-data", "timeout": 5},
                name="produce",
            ),
            StepDefinition(
                type=StepType.COGNITIVE,
                config={"provider": "mock", "system": "Process data"},
                name="consume",
            ),
        ],
    )
    await rt.register_agent(implicit)
    mock.calls.clear()
    await rt.trigger_run("implicit01")
    await asyncio.sleep(2)

    rec = rt.execution_log.get_latest("implicit01")
    assert rec.status == ExecutionStatus.COMPLETED
    # Without explicit prompt, previous step output should be the user message
    assert "some-input-data" in mock.calls[0]["messages"][0]["content"]
    print("Test 7 PASS: Implicit prompt from previous step output")

    await rt.stop()
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    print("\nAll unit tests passed!")


async def test_live():
    """Live integration test with real Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\nSKIPPED: Live test (set ANTHROPIC_API_KEY to run)")
        return

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    config = ATNConfig(
        data_dir=TEST_DIR,
        providers={"anthropic": ProviderConfig(
            name="anthropic",
            api_key=api_key,
            default_model="claude-sonnet-4-20250514",
        )},
    )
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=config)

    events: list = []
    async def collect(e):
        events.append(e)
    bus.subscribe(None, collect)

    await rt.start()

    # Live test: simple question
    agent = AgentDefinition(
        id="live01", name="Live Thinker",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "anthropic",
                "system": "Reply with only the number, nothing else.",
                "prompt": "What is 2+2?",
                "max_tokens": 10,
            },
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.trigger_run("live01")
    await asyncio.sleep(10)

    rec = rt.execution_log.get_latest("live01")
    assert rec.status == ExecutionStatus.COMPLETED, f"Live test failed: {rec.error}"
    output = rec.step_results[0].output
    assert "4" in output["text"], f"Expected '4' in response, got: {output['text']}"
    assert output["usage"]["input_tokens"] > 0
    assert output["usage"]["output_tokens"] > 0
    print(f"\nLive test PASS: Got '{output['text'].strip()}' (model={output['model']}, tokens={output['usage']})")

    # Live test 2: pipeline (script -> cognitive)
    pipeline = AgentDefinition(
        id="live_pipe", name="Live Pipeline",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": 'echo {"temp": 22, "city": "Sydney"}', "timeout": 5},
                name="data",
            ),
            StepDefinition(
                type=StepType.COGNITIVE,
                config={
                    "provider": "anthropic",
                    "system": "You receive JSON weather data. Reply with a single short sentence.",
                    "prompt": "Summarise this weather reading: {prev}",
                    "max_tokens": 50,
                },
                name="summarise",
            ),
        ],
    )
    await rt.register_agent(pipeline)
    await rt.trigger_run("live_pipe")
    await asyncio.sleep(10)

    rec = rt.execution_log.get_latest("live_pipe")
    assert rec.status == ExecutionStatus.COMPLETED, f"Live pipeline failed: {rec.error}"
    assert len(rec.step_results) == 2
    summary = rec.step_results[1].output["text"]
    print(f"Live pipeline PASS: Summary = '{summary.strip()}'")

    await rt.stop()
    shutil.rmtree(TEST_DIR)
    print("\nAll live tests passed!")


async def main():
    await test_unit()
    await test_live()


if __name__ == "__main__":
    asyncio.run(main())
