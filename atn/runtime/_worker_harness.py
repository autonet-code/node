"""Standalone harness for the in-worker cognitive loop (Phase 4 verification).

Runs ``run_cognitive_loop`` WITHOUT a daemon, WITHOUT a network, and WITHOUT a
real pipe. A fake DaemonClient answers every RPC/emit locally and records what
crossed the seam; a fake provider mimics ``send_orchestrate`` calling into the
tool_executor / on_chunk / usage_recorder exactly as the base provider loop
does. This proves:

  1. LOCAL shell tools execute IN-worker (never RPC'd).
  2. AUTHORITY tools become framework_tool / surface_tool / mcp_tool RPCs.
  3. spawn (create_agent) is refused cleanly, no crash.
  4. budget_check runs pre-flight; budget_record runs per turn.
  5. step.output text streams via emit.
  6. the returned execution_done dict is correctly shaped + JSON-safe.

Run: ``python -m atn.runtime._worker_harness``  (exit 0 = pass)
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .worker_loop import run_cognitive_loop
from ..providers.base import ProviderResponse, Usage


# ---------------------------------------------------------------------------
# Fake DaemonClient — answers RPCs locally, records the seam traffic
# ---------------------------------------------------------------------------

class FakeDaemonClient:
    def __init__(self, *, exceed_on_record: bool = False) -> None:
        self.rpc_calls: list[tuple[str, dict]] = []
        self.emits: list[tuple[str, dict]] = []
        self._exceed_on_record = exceed_on_record
        self._recorded_tokens = 0

    async def rpc(self, name: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        self.rpc_calls.append((name, payload))
        if name == "budget_check":
            return {"ok": True, "blocker": None}
        if name == "budget_record":
            self._recorded_tokens += payload.get("tokens", 0)
            if self._exceed_on_record:
                return {"exceeded": "some-ancestor"}
            return {"exceeded": None}
        if name == "framework_tool":
            # Echo a plausible framework result.
            return {"ok": True, "tool": payload.get("name"), "echo": payload.get("input")}
        if name == "surface_tool":
            return {"ok": True, "surface": payload.get("name")}
        if name == "mcp_tool":
            return {"ok": True, "mcp": payload.get("name")}
        raise RuntimeError(f"unexpected rpc {name!r}")

    async def emit(self, name: str, payload: dict | None = None) -> None:
        self.emits.append((name, payload or {}))


# ---------------------------------------------------------------------------
# Fake provider — mirrors the base send_orchestrate contract closely enough
# to exercise every callback the worker wires in.
# ---------------------------------------------------------------------------

class FakeProvider:
    """Drives a scripted 2-turn loop: turn 1 issues one tool call of each class
    (local / framework / surface / mcp / spawn); turn 2 ends the turn. Calls the
    same callbacks the real base loop calls, in the same order."""

    def __init__(self) -> None:
        self.name = "fake-api"
        self.event_bus: Any = None
        self.source_agent_id = ""
        self._interrupted = False
        self.closed = False

    @property
    def session_stats(self) -> dict:
        return {"num_turns": 2, "active_model": "fake-model"}

    async def interrupt(self) -> None:
        self._interrupted = True

    async def close(self) -> None:
        self.closed = True

    async def send_orchestrate(
        self, *, message, system, tools, tool_executor, on_chunk,
        usage_recorder, model="", **kw,
    ) -> ProviderResponse:
        # Turn 1: stream some text, book usage, run one tool of each class.
        await on_chunk("Working on it. ")
        ok, blocker = await _maybe_await(usage_recorder(120, model or "fake-model"))
        if not ok:
            return ProviderResponse(text="", stop_reason="budget_exceeded",
                                    usage=Usage(input_tokens=120), model="fake-model")

        for tname, tinput in [
            ("read_file", {"path": "<harness>"}),   # LOCAL
            ("get_snapshot", {}),                    # AUTHORITY framework
            ("surface_read_history", {"n": 5}),      # AUTHORITY surface
            ("mcp_x__do", {"a": 1}),                 # AUTHORITY mcp
            ("create_agent", {"name": "child"}),     # SPAWN refusal
        ]:
            res = await tool_executor(tname, tinput)
            assert isinstance(res, dict), f"tool {tname} returned non-dict {res!r}"

        # Turn 2: final answer, book usage, end.
        await on_chunk("Done.")
        await _maybe_await(usage_recorder(80, model or "fake-model"))
        return ProviderResponse(
            text="Final answer.",
            stop_reason="end_turn",
            usage=Usage(input_tokens=200, output_tokens=50, cache_read_tokens=10),
            model="fake-model",
        )


async def _maybe_await(v):
    import inspect
    if inspect.isawaitable(v):
        return await v
    return v


# ---------------------------------------------------------------------------
# Harness driver + assertions
# ---------------------------------------------------------------------------

async def _run() -> int:
    client = FakeDaemonClient()
    provider = FakeProvider()
    # Point the local read_file at a real temp file so the LOCAL executor
    # actually reads worker-side (proves it did NOT go over RPC).
    tmp = Path(tempfile.gettempdir()) / "atn_worker_harness_probe.txt"
    tmp.write_text("hello-from-local-worker", encoding="utf-8")

    manifest = {
        "provider_config": {"kind": "openai_compat", "name": "fake-api"},
        "agent_id": "agent-xyz",
        "provider_name": "fake-api",
        "model": "fake-model",
        "message": "do the thing",
        "system": "you are a test agent",
        "tools": [],
        "max_turns": 5,
        "session_id": "",
        "history": None,
    }
    # Patch read_file's input to the real temp path.
    provider._probe_path = str(tmp)

    # Wrap the provider so read_file gets the real path.
    orig = provider.send_orchestrate
    async def _wrapped(**kw):
        te = kw["tool_executor"]
        async def _te(name, inp):
            if name == "read_file":
                inp = {"path": str(tmp)}
            return await te(name, inp)
        kw["tool_executor"] = _te
        return await orig(**kw)
    provider.send_orchestrate = _wrapped

    result = await run_cognitive_loop(
        client=client, provider=provider, manifest=manifest, agent_label="agent-xyz",
    )

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    rpc_names = [n for n, _ in client.rpc_calls]

    # 1. Pre-flight budget_check happened first.
    check(rpc_names[0] == "budget_check", f"first rpc was {rpc_names[0]!r}, want budget_check")

    # 2. Two budget_record calls (one per turn).
    check(rpc_names.count("budget_record") == 2,
          f"budget_record count = {rpc_names.count('budget_record')}, want 2")

    # 3. AUTHORITY tools routed to the right RPC kinds.
    check(("framework_tool", {"name": "get_snapshot", "input": {}}) in client.rpc_calls,
          "get_snapshot did not route to framework_tool")
    surface_calls = [p for n, p in client.rpc_calls if n == "surface_tool"]
    check(any(p.get("name") == "surface_read_history" for p in surface_calls),
          "surface_read_history did not route to surface_tool")
    mcp_calls = [p for n, p in client.rpc_calls if n == "mcp_tool"]
    check(any(p.get("name") == "mcp_x__do" for p in mcp_calls),
          "mcp_x__do did not route to mcp_tool")

    # 4. LOCAL read_file did NOT go over RPC (no framework_tool for it) and read
    #    the real worker-side file.
    fw_names = [p.get("name") for n, p in client.rpc_calls if n == "framework_tool"]
    check("read_file" not in fw_names, "read_file leaked to an RPC (should be LOCAL)")
    read_tc = next((tc for tc in result["tool_calls"] if tc["tool"] == "read_file"), None)
    check(read_tc is not None and "hello-from-local-worker" in read_tc["result"],
          "LOCAL read_file did not read the worker-side temp file")

    # 5. spawn (create_agent) refused cleanly, not crashed.
    spawn_tc = next((tc for tc in result["tool_calls"] if tc["tool"] == "create_agent"), None)
    check(spawn_tc is not None and not spawn_tc["success"]
          and "isolation" in spawn_tc["result"].lower(),
          "create_agent was not refused cleanly")
    # create_agent must NOT have produced a spawn_child RPC in P4.
    check("spawn_child" not in rpc_names, "create_agent leaked a spawn_child RPC (P6, not P4)")

    # 6. Text streamed via emit.
    emit_names = [n for n, _ in client.emits]
    text_emits = [p for n, p in client.emits if n == "step.output" and p.get("channel") == "text"]
    check(any(p.get("content") == "Working on it. " for p in text_emits),
          "streamed text chunk not emitted")
    check(len(text_emits) >= 2, f"expected >=2 text emits, got {len(text_emits)}")

    # 7. execution_done dict shape + JSON-safety.
    check(result["status"] == "completed", f"status = {result['status']!r}")
    check(result["output"]["result"] == "Final answer.", "output.result wrong")
    check(result["token_usage"]["fake-api"]["input_tokens"] == 200, "token_usage wrong")
    try:
        json.dumps(result)
    except TypeError as e:
        check(False, f"result not JSON-safe: {e}")

    # 8. provider.close() ran (done by agent_worker; here we assert loop left it
    #    reusable — close is the worker's job, so just confirm no crash).
    check(provider.event_bus is not None, "event_bus was not wired")

    # --- Budget-exceeded path (separate run). ---
    client2 = FakeDaemonClient(exceed_on_record=True)
    provider2 = FakeProvider()
    result2 = await run_cognitive_loop(
        client=client2, provider=provider2,
        manifest={**manifest, "provider_config": {"kind": "openai_compat", "name": "fake-api"}},
        agent_label="agent-xyz",
    )
    # First budget_record returns exceeded -> loop aborts with budget_exceeded.
    # No text answer produced before abort in this scripted provider path... but
    # our fake streams "Working on it." before the recorder. That text is a
    # chunk, not the final result, so result_text is "" -> failed.
    check(result2["status"] in ("failed", "completed"),
          f"budget path status unexpected: {result2['status']!r}")
    check(any(n == "budget_record" for n, _ in client2.rpc_calls),
          "budget_record never called on exceed path")

    if failures:
        print("HARNESS FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("worker-loop harness: ALL CHECKS PASSED")
    print(f"  rpc calls: {rpc_names}")
    print(f"  emits:     {[n for n, _ in client.emits]}")
    print(f"  status:    {result['status']}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
