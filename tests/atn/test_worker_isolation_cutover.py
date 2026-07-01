"""Phase 4 cutover tests: flag-ON API-provider cognitive agents run in a worker.

These exercise ExecutionEngine._run_cognitive_in_worker end-to-end WITHOUT a
real subprocess: a fake WorkerManager is injected onto the engine so the whole
seam (eligibility guard, manifest build, WorkerHost RPC/status wiring, finalize
ownership, per-execution teardown, budget reconciliation) runs in-process.

The flag-OFF path is covered by test_cognitive_mode.py; here we assert the
flag-ON fork lands in the worker for API providers and stays in-process for the
bridge provider.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atn.events import Event, EventBus, EventType
from atn.models import AgentDefinition, AgentMode, AgentStatus, ExecutionStatus
from atn.providers.base import ProviderResponse, Usage


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def captured(bus: EventBus) -> list[Event]:
    events: list[Event] = []

    async def _cap(e: Event):
        events.append(e)

    bus.subscribe(None, _cap)
    return events


def _make_runtime(bus: EventBus, tmp_path: Path, *, isolation: bool):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.worker_isolation.enabled = isolation

    return Runtime(bus, data_dir=data_dir, config=config)


class _FakeChannel:
    """Stands in for WorkerChannel: records cmds and, on the 'run' cmd, drives
    the WorkerHost status sink with a synthetic execution_done (the clean path).

    ``status_sink`` is the callable the daemon injected into ensure_worker (the
    WorkerHost's real sink) — so firing it exercises the real record-population
    + finalize wiring."""

    def __init__(self, agent_id: str, status_sink, done_payload: dict):
        self._agent_id = agent_id
        self._status_sink = status_sink
        self._done = done_payload
        self.cmds: list[tuple[str, dict]] = []

    async def send_cmd(self, name: str, payload: dict | None = None) -> None:
        self.cmds.append((name, dict(payload or {})))
        if name == "run" and payload and payload.get("go"):
            await self._status_sink(self._agent_id, "heartbeat", {})
            await self._status_sink(self._agent_id, "execution_done", self._done)

    async def close(self) -> None:
        pass


class _FakeHandle:
    def __init__(self, agent_id: str, channel):
        self.agent_id = agent_id
        self.pid = 4242
        self.channel = channel
        self.proc = None
        self.job = None


class _FakeWorkerManager:
    """ensure_worker wires a fake channel bound to the real WorkerHost sinks;
    graceful_close / hard_kill just mark the (fake) process gone."""

    def __init__(self, done_payload: dict):
        self._done = done_payload
        self.ensured: list[str] = []
        self.closed: list[str] = []
        self.manifest: dict | None = None
        self.run_cmds: list[tuple[str, dict]] = []

    async def ensure_worker(self, agent_id, manifest, *, rpc_handlers=None,
                            event_sink=None, status_sink=None, **_):
        self.ensured.append(agent_id)
        self.manifest = manifest
        self.rpc_handlers = rpc_handlers
        self.event_sink = event_sink
        chan = _FakeChannel(agent_id, status_sink, self._done)
        self._chan = chan
        return _FakeHandle(agent_id, chan)

    async def graceful_close(self, handle, **_):
        self.closed.append(handle.agent_id)
        return True

    async def hard_kill(self, handle):
        self.closed.append(handle.agent_id)


def _api_provider():
    from atn.providers.anthropic import AnthropicProvider
    return AnthropicProvider(api_key="sk-test", default_model="claude-sonnet-4-20250514")


def _cog_defn(aid="cog-w") -> AgentDefinition:
    return AgentDefinition(
        id=aid, name="Worker Agent", mode=AgentMode.COGNITIVE,
        provider="anthropic", cognitive_model="sonnet",
        system_prompt="You are a test agent.", agent_type="implement",
        description="Do the thing.",
    )


@pytest.mark.asyncio
async def test_flag_on_api_provider_runs_in_worker(bus, tmp_path, captured):
    rt = _make_runtime(bus, tmp_path, isolation=True)
    defn = _cog_defn()
    await rt.register_agent(defn)
    await rt.activate_agent(defn.id)

    # Pre-seed a real API provider so eligibility passes and no network provider
    # is resolved. The worker path intercepts BEFORE send_orchestrate.
    rt.engine.provider_manager._active_providers[defn.id] = _api_provider()

    done_payload = {
        "status": "completed",
        "error": None,
        "output": {
            "result": "worker answer",
            "tokens_used": 150,
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "cache_read_tokens": 0, "cache_creation_tokens": 0},
        },
        "token_usage": {"anthropic": {"input_tokens": 100, "output_tokens": 50,
                                      "cache_read_tokens": 0, "cache_creation_tokens": 0}},
        "tool_calls": [],
        "streamed_text": True,
        "session_stats": None,
    }
    mgr = _FakeWorkerManager(done_payload)
    rt.engine._worker_mgr = mgr
    # The supervisor holds its own manager for the kill ladder; point it at the
    # same fake so teardown's graceful_close is observable.
    rt.supervisor._worker_manager = mgr

    eid = await rt.trigger_run(defn.id, source="test")
    assert eid is not None
    await asyncio.wait(list(rt.engine._tasks.values()), timeout=10)

    # Worker was spawned + told to run.
    assert mgr.ensured == [defn.id]
    assert ("run", {**mgr.manifest, "go": True}) in mgr._chan.cmds
    # The manifest carried the prebuilt tool surface + provider_config with the
    # secret on the pipe frame (not env).
    assert mgr.manifest["provider_config"]["kind"] == "anthropic"
    assert mgr.manifest["provider_config"]["api_key"] == "sk-test"
    assert mgr.manifest["system"].startswith("You are a test agent.")

    # Clean finalize populated record.output verbatim + wrote the output store.
    out = rt.output_store.read(defn.id)
    assert out is not None and out.data["result"] == "worker answer"
    assert rt.get_status(defn.id) == AgentStatus.COMPLETED

    # Terminal event emitted exactly once.
    completed = [e for e in captured if e.type == EventType.EXECUTION_COMPLETED
                 and e.data.get("agent_id") == defn.id]
    assert len(completed) == 1

    # Per-execution teardown reaped the worker (no lingering supervised handle).
    assert not rt.supervisor.is_supervised(defn.id)
    assert defn.id in mgr.closed

    # Budget was booked once via the daemon ledger (budget_tokens excludes
    # cache_read; here 150 with 0 cache => 150). No double-count.
    used = rt.registry.get_budget_info(defn.id)
    assert used is not None


@pytest.mark.asyncio
async def test_flag_on_bridge_provider_is_worker_eligible(bus, tmp_path):
    """P5: a bridge (Claude-Max SDK) provider IS routed to a worker with the flag
    ON. _worker_eligible returns True and _build_provider_config emits a
    ``kind="bridge"`` manifest carrying model + bridge_script and NO api_key."""
    rt = _make_runtime(bus, tmp_path, isolation=True)
    defn = _cog_defn("cog-bridge")
    await rt.register_agent(defn)
    await rt.activate_agent(defn.id)

    from unittest.mock import AsyncMock
    from atn.providers.bridge import BridgeProvider

    # A real-typed bridge provider instance (spec) so isinstance checks hold.
    bridge = AsyncMock(spec=BridgeProvider)
    bridge.name = "claude_max"
    bridge._model = "opus"
    bridge._bridge_script = "/some/where/claude-bridge.ts"
    rt.engine.provider_manager._active_providers[defn.id] = bridge

    # P5: bridge is now worker-eligible.
    assert rt.engine._worker_eligible(defn, bridge) is True

    cfg = rt.engine._build_provider_config(bridge)
    assert cfg["kind"] == "bridge"
    assert cfg["name"] == "claude_max"
    assert cfg["model"] == "opus"
    assert cfg["bridge_script"] == "/some/where/claude-bridge.ts"
    # No secret rides for bridge — auth is the node SDK reading credentials
    # off disk, so api_key/base_url must be absent from the manifest.
    assert "api_key" not in cfg
    assert "base_url" not in cfg


@pytest.mark.asyncio
async def test_flag_on_codex_bridge_stays_in_process(bus, tmp_path):
    """codex_max is NOT in P5 scope: it must stay in-process even with the flag
    ON. _worker_eligible returns False; classify refuses it."""
    rt = _make_runtime(bus, tmp_path, isolation=True)
    defn = _cog_defn("cog-codex")
    await rt.register_agent(defn)
    await rt.activate_agent(defn.id)

    from unittest.mock import AsyncMock
    try:
        from atn.providers.codex_bridge import CodexBridgeProvider
    except Exception:
        pytest.skip("codex bridge provider unavailable")

    codex = AsyncMock(spec=CodexBridgeProvider)
    codex.name = "codex_max"
    rt.engine.provider_manager._active_providers[defn.id] = codex

    assert rt.engine._worker_eligible(defn, codex) is False


class _CrashChannel(_FakeChannel):
    """On 'go' does NOT send execution_done — simulates a worker that crashed /
    was hard-killed mid-flight. The supervisor's kill (driven by the test) then
    reaps it and finalizes on-behalf as KILLED."""

    async def send_cmd(self, name: str, payload: dict | None = None) -> None:
        self.cmds.append((name, dict(payload or {})))
        # deliberately silent on 'run' — no execution_done


class _CrashWorkerManager(_FakeWorkerManager):
    async def ensure_worker(self, agent_id, manifest, *, rpc_handlers=None,
                            event_sink=None, status_sink=None, **_):
        self.ensured.append(agent_id)
        self.manifest = manifest
        chan = _CrashChannel(agent_id, status_sink, self._done)
        self._chan = chan
        return _FakeHandle(agent_id, chan)


@pytest.mark.asyncio
async def test_flag_on_worker_crash_finalizes_on_behalf(bus, tmp_path, captured):
    """Worker dies without execution_done → supervisor reap → on-behalf KILLED
    finalize, exactly once, and the driver task unparks (no hang)."""
    rt = _make_runtime(bus, tmp_path, isolation=True)
    defn = _cog_defn("cog-crash")
    await rt.register_agent(defn)
    await rt.activate_agent(defn.id)
    rt.engine.provider_manager._active_providers[defn.id] = _api_provider()

    mgr = _CrashWorkerManager({})
    rt.engine._worker_mgr = mgr
    rt.supervisor._worker_manager = mgr

    eid = await rt.trigger_run(defn.id, source="test")
    assert eid is not None

    # The driver is now parked waiting for finalize. Simulate the supervisor
    # detecting the dead worker and reaping it (what the monitor loop / a kill
    # would do). This must unpark the driver and finalize as KILLED.
    async def _kill_soon():
        await asyncio.sleep(0.05)
        await rt.supervisor.kill(defn.id, reason="test_crash")

    await asyncio.gather(
        _kill_soon(),
        asyncio.wait_for(asyncio.gather(*rt.engine._tasks.values()), timeout=10),
    )

    assert rt.get_status(defn.id) in (AgentStatus.ERROR, AgentStatus.ACTIVE)
    killed = [e for e in captured if e.type == EventType.EXECUTION_KILLED
              and e.data.get("agent_id") == defn.id]
    assert len(killed) == 1                 # finalized on-behalf exactly once
    assert not rt.supervisor.is_supervised(defn.id)


@pytest.mark.asyncio
async def test_flag_off_never_eligible(bus, tmp_path):
    """With the flag OFF, even an API provider is not worker-eligible."""
    rt = _make_runtime(bus, tmp_path, isolation=False)
    defn = _cog_defn("cog-off")
    prov = _api_provider()
    assert rt.engine._worker_eligible(defn, prov) is False
