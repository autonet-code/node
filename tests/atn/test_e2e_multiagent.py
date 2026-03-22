"""End-to-end multi-agent workflow test.

Chain: collector → analyst → reporter
  1. collector gathers system metrics (script), posts to analyst's inbox
  2. analyst receives metrics, runs cognitive step (Gemini), posts verdict to reporter
  3. reporter receives verdict and produces final report

We trigger the collector and wait for all three agents to complete.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atn.config import load_config
from atn.events import Event, EventBus, EventType
from atn.loader import load_agents_dir
from atn.runtime import Runtime

completed_agents: list[str] = []


async def _print_event(event: Event) -> None:
    parts = []
    for k in ("agent_id", "step_name", "step_type", "status", "error", "output_preview"):
        v = event.data.get(k)
        if v:
            if k == "output_preview":
                v = str(v)[:80]
            parts.append(f"{k}={v}")
    detail = "  ".join(parts)
    ts = event.timestamp.strftime("%H:%M:%S")
    print(f"  {ts}  {event.type.value:<26}  {detail}")

    # Track agent completions
    if event.type in (EventType.EXECUTION_COMPLETED, EventType.EXECUTION_FAILED):
        agent_id = event.data.get("agent_id", "")
        if agent_id:
            completed_agents.append(agent_id)


async def main():
    config = load_config()
    event_bus = EventBus()
    runtime = Runtime(event_bus, data_dir=config.data_dir, config=config)

    event_bus.subscribe(None, _print_event)

    await runtime.start()

    # Load all agents
    agents, errors = load_agents_dir(config.agents_dir)
    for err in errors:
        print(f"  Load error: {err}")
    for defn in agents:
        await runtime.register_agent(defn)
        # Activate all agents so inbox watcher can trigger them
        await runtime.activate_agent(defn.id)

    agent_ids = [a.id for a in agents]
    print(f"Loaded and activated {len(agents)} agents: {agent_ids}")

    # Trigger the collector — this starts the chain
    print(f"\n{'='*60}")
    print("  Triggering collector -> analyst -> reporter chain")
    print(f"{'='*60}\n")

    eid = await runtime.trigger_run("collector", source="test")
    if not eid:
        print("FAILED: Could not trigger collector")
        await runtime.stop()
        return

    # Wait for all three agents to complete (collector, analyst, reporter)
    expected = {"collector", "analyst", "reporter"}
    for _ in range(60):  # 60 seconds max
        await asyncio.sleep(1)
        if expected.issubset(set(completed_agents)):
            break
    else:
        print(f"\nTIMEOUT — completed: {completed_agents}, expected: {expected}")

    # Give a moment for final events
    await asyncio.sleep(1)

    # Print results for each agent
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")

    for agent_id in ["collector", "analyst", "reporter"]:
        records = runtime.execution_log.get_history(agent_id)
        if not records:
            print(f"\n  {agent_id}: NO HISTORY")
            continue

        last = records[-1]
        print(f"\n  {agent_id}: {last.status.value}")
        if last.error:
            print(f"    Error: {last.error}")
        for sr in last.step_results:
            print(f"    Step [{sr.step_index}] {sr.step_name} ({sr.step_type.value}): {sr.status.value}")
            if sr.output:
                out_str = json.dumps(sr.output, indent=2, default=str) if isinstance(sr.output, dict) else str(sr.output)
                if len(out_str) > 300:
                    out_str = out_str[:300] + "..."
                print(f"      Output: {out_str}")

    await runtime.stop()

    # Final verdict
    print(f"\n{'='*60}")
    all_done = expected.issubset(set(completed_agents))
    print(f"  MULTI-AGENT WORKFLOW: {'PASS' if all_done else 'FAIL'}")
    print(f"  Agents completed: {list(set(completed_agents) & expected)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
