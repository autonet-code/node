"""Live integration test with Ollama."""
import asyncio
import shutil
from pathlib import Path

import httpx
import pytest

from atn.config import ATNConfig, ProviderConfig
from atn.events import EventBus, EventType
from atn.models import AgentDefinition, StepDefinition, StepType, ExecutionStatus
from atn.runtime import Runtime

TEST_DIR = Path("./test_data_ollama")

OLLAMA_URL = "http://localhost:11434"


def _ollama_reachable() -> bool:
    try:
        httpx.get(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
async def test():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    config = ATNConfig(
        data_dir=TEST_DIR,
        providers={"ollama": ProviderConfig(
            name="ollama",
            default_model="qwen3.5:4b",
        )},
    )
    bus = EventBus()
    rt = Runtime(bus, data_dir=TEST_DIR, config=config)

    events = []
    async def collect(e):
        events.append(e)
    bus.subscribe(None, collect)

    await rt.start()

    # ===== Test 1: Simple question =====
    agent = AgentDefinition(
        id="ollama01", name="Ollama Thinker",
        steps=[StepDefinition(
            type=StepType.COGNITIVE,
            config={
                "provider": "ollama",
                "system": "Reply with only the number, nothing else. Do not explain.",
                "prompt": "What is 2+2?",
                "max_tokens": 20,
                "temperature": 0.0,
            },
            name="think",
        )],
    )
    await rt.register_agent(agent)
    await rt.trigger_run("ollama01")
    await asyncio.sleep(15)  # local models can be slow on first load

    rec = rt.execution_log.get_latest("ollama01")
    assert rec is not None, "No execution record"
    assert rec.status == ExecutionStatus.COMPLETED, f"Failed: {rec.error}"
    output = rec.step_results[0].output
    print(f"Test 1 PASS: Got '{output['text'].strip()}' (model={output['model']})")
    print(f"  tokens: in={output['usage']['input_tokens']} out={output['usage']['output_tokens']}")

    # ===== Test 2: Pipeline — script feeds data to LLM =====
    pipeline = AgentDefinition(
        id="ollama_pipe", name="Ollama Pipeline",
        steps=[
            StepDefinition(
                type=StepType.SCRIPT,
                config={"command": 'echo {"city": "Sydney", "temp_c": 22}', "timeout": 5},
                name="data",
            ),
            StepDefinition(
                type=StepType.COGNITIVE,
                config={
                    "provider": "ollama",
                    "system": "You receive JSON weather data. Reply with one short sentence.",
                    "prompt": "Summarise: {prev}",
                    "max_tokens": 60,
                },
                name="summarise",
            ),
        ],
    )
    await rt.register_agent(pipeline)
    await rt.trigger_run("ollama_pipe")
    await asyncio.sleep(15)

    rec = rt.execution_log.get_latest("ollama_pipe")
    assert rec.status == ExecutionStatus.COMPLETED, f"Pipeline failed: {rec.error}"
    assert len(rec.step_results) == 2
    summary = rec.step_results[1].output["text"]
    print(f"Test 2 PASS: Pipeline summary = '{summary.strip()}'")

    # ===== Test 3: Events emitted =====
    cog_started = [e for e in events if e.type == EventType.STEP_STARTED
                   and e.data.get("step_type") == "cognitive"]
    cog_completed = [e for e in events if e.type == EventType.STEP_COMPLETED
                     and e.data.get("step_type") == "cognitive"]
    assert len(cog_started) >= 2
    assert len(cog_completed) >= 2
    print(f"Test 3 PASS: {len(cog_started)} cognitive steps started, {len(cog_completed)} completed")

    await rt.stop()
    shutil.rmtree(TEST_DIR)
    print("\nAll Ollama live tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
