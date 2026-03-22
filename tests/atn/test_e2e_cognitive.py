"""End-to-end cognitive agent tests across all three providers.

Runs each agent (Gemini, Ollama, Bridge) through a [script → cognitive] pipeline
and verifies the cognitive step produces meaningful output.

Usage:
    python test_e2e_cognitive.py              # run all
    python test_e2e_cognitive.py gemini       # run one
    python test_e2e_cognitive.py ollama
    python test_e2e_cognitive.py bridge
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atn.config import load_config
from atn.events import Event, EventBus
from atn.loader import load_agents_dir
from atn.runtime import Runtime

# Collect events for inspection
collected: list[Event] = []


async def _collect(event: Event) -> None:
    collected.append(event)
    # Print a compact line
    parts = []
    for k in ("agent_id", "step_name", "step_type", "status", "error"):
        v = event.data.get(k)
        if v:
            parts.append(f"{k}={v}")
    detail = "  ".join(parts)
    ts = event.timestamp.strftime("%H:%M:%S")
    print(f"  {ts}  {event.type.value:<26}  {detail}")


async def run_agent(runtime: Runtime, agent_id: str) -> dict | None:
    """Trigger an agent and wait for completion. Returns the final output."""
    print(f"\n{'='*60}")
    print(f"  Running: {agent_id}")
    print(f"{'='*60}")

    collected.clear()

    eid = await runtime.trigger_run(agent_id, source="test")
    if not eid:
        print(f"  FAILED: Could not trigger {agent_id}")
        return None

    # Wait for completion
    for _ in range(120):  # 120 seconds max
        await asyncio.sleep(1)
        # Check if execution is done
        if eid not in runtime._executions:
            break
    else:
        print(f"  TIMEOUT waiting for {agent_id}")
        await runtime.kill_execution(eid)
        return None

    # Get the result
    record = runtime.execution_log.get_history(agent_id)
    if not record:
        print(f"  No execution history for {agent_id}")
        return None

    last = record[-1]
    print(f"\n  Status: {last.status.value}")
    if last.error:
        print(f"  Error:  {last.error}")

    # Print step results
    for sr in last.step_results:
        print(f"\n  Step [{sr.step_index}] {sr.step_name} ({sr.step_type.value}): {sr.status.value}")
        if sr.error:
            print(f"    Error: {sr.error}")
        if sr.output:
            output_str = json.dumps(sr.output, indent=2, default=str) if isinstance(sr.output, dict) else str(sr.output)
            # Truncate long output
            if len(output_str) > 500:
                output_str = output_str[:500] + "..."
            print(f"    Output: {output_str}")

    return last.output


async def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["gemini", "ollama", "bridge"]

    config = load_config()
    event_bus = EventBus()
    runtime = Runtime(event_bus, data_dir=config.data_dir, config=config)

    event_bus.subscribe(None, _collect)

    await runtime.start()

    # Load agents
    agents, errors = load_agents_dir(config.agents_dir)
    for err in errors:
        print(f"  Load error: {err}")
    for defn in agents:
        await runtime.register_agent(defn)
    print(f"Loaded {len(agents)} agents: {[a.id for a in agents]}")

    results = {}

    try:
        if "gemini" in targets:
            results["gemini"] = await run_agent(runtime, "gemini-analyst")

        if "ollama" in targets:
            results["ollama"] = await run_agent(runtime, "ollama-analyst")

        if "bridge" in targets:
            results["bridge"] = await run_agent(runtime, "bridge-analyst")

    finally:
        await runtime.stop()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, output in results.items():
        status = "PASS" if output is not None else "FAIL"
        print(f"  {name:<10} {status}")
        if output and isinstance(output, dict):
            # Show key fields
            if "text" in output:
                text = str(output["text"])[:120]
                print(f"             text: {text}")
            if "tool_calls" in output:
                for tc in output["tool_calls"]:
                    print(f"             tool: {tc['name']}({json.dumps(tc['input'], default=str)[:100]})")
            if "usage" in output:
                u = output["usage"]
                print(f"             tokens: in={u.get('input_tokens',0)} out={u.get('output_tokens',0)}"
                      f" cache_read={u.get('cache_read_tokens',0)}")


if __name__ == "__main__":
    asyncio.run(main())
