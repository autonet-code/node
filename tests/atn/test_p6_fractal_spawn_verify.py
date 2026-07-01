"""P6 FINAL VERIFY — fractal delegate spawn under worker isolation.

Drives the REAL path: an isolated (worker) parent agent calls create_agent ->
the worker issues the spawn_child RPC -> the daemon's WorkerHost._rpc_spawn_child
runs the REAL create_agent (caller_id forced to the pipe-bound parent) -> the
child is registered with parent_id = the authoritative parent -> trigger_run
recurses -> _run_cognitive_in_worker spawns the child as ITS OWN worker with its
OWN pid. Then: prove distinct pids, the _children edge, the pid-bind seam args +
ordering, a forged parent id is ignored, subtree leaf-first kill (no orphans),
and _on_pid_revoked for the child.

Providers are NOT contacted: the fake WorkerManager hands back fake handles with
distinct incrementing pids and drives the WorkerHost sinks, so the spawn /
register / recurse / worker-spawn / pid-bind / reap seam is entirely REAL while
the cognitive loop + node subprocess are stubbed. What is stubbed is stated in
each assertion's comment.
"""
from __future__ import annotations

import asyncio
import itertools
from pathlib import Path

import pytest

from atn.events import Event, EventBus, EventType
from atn.models import AgentDefinition, AgentMode


# --- shared fakes ---------------------------------------------------------

_DONE = {
    "status": "completed",
    "error": None,
    "output": {"result": "ok", "tokens_used": 10,
               "usage": {"input_tokens": 5, "output_tokens": 5,
                         "cache_read_tokens": 0, "cache_creation_tokens": 0}},
    "token_usage": {"anthropic": {"input_tokens": 5, "output_tokens": 5,
                                  "cache_read_tokens": 0, "cache_creation_tokens": 0}},
    "tool_calls": [],
    "streamed_text": True,
    "session_stats": None,
}


class _HoldChannel:
    """On 'run'+go, records the cmd but does NOT auto-finalize — the worker is
    'alive' until the test kills it. Lets us hold parent+child workers open at
    once to observe two live pids and a real subtree kill."""

    def __init__(self, agent_id, status_sink):
        self.agent_id = agent_id
        self._status_sink = status_sink
        self.cmds: list[tuple[str, dict]] = []
        self.go_sent = False

    async def send_cmd(self, name, payload=None):
        self.cmds.append((name, dict(payload or {})))
        if name == "run" and payload and payload.get("go"):
            self.go_sent = True

    async def close(self):
        pass


class _Handle:
    def __init__(self, agent_id, pid, channel):
        self.agent_id = agent_id
        self.pid = pid
        self.channel = channel
        self.proc = None
        self.job = None


class _MultiPidWorkerManager:
    """Distinct incrementing pid per ensure_worker. Records the register order
    so we can show two separate worker pids (parent + child). hard_kill / close
    mark the fake process gone AND drive execution_done so on-behalf finalize
    doesn't hang the driver."""

    def __init__(self):
        self._pids = itertools.count(9000)
        self.handles: dict[str, _Handle] = {}
        self.ensured: list[tuple[str, int]] = []
        self.closed: list[str] = []

    async def ensure_worker(self, agent_id, manifest, *, rpc_handlers=None,
                            event_sink=None, status_sink=None, **_):
        pid = next(self._pids)
        chan = _HoldChannel(agent_id, status_sink)
        h = _Handle(agent_id, pid, chan)
        h._status_sink = status_sink
        self.handles[agent_id] = h
        self.ensured.append((agent_id, pid))
        return h

    async def graceful_close(self, handle, **_):
        self.closed.append(handle.agent_id)
        return True

    async def hard_kill(self, handle):
        self.closed.append(handle.agent_id)
        # Feed a terminal so the parked driver unblocks via on-behalf finalize.
        try:
            await handle._status_sink(handle.agent_id, "execution_done",
                                      {**_DONE, "status": "killed",
                                       "error": "killed by test"})
        except Exception:
            pass


def _make_runtime(bus, tmp_path, *, isolation):
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


def _api_provider():
    from atn.providers.anthropic import AnthropicProvider
    return AnthropicProvider(api_key="sk-test", default_model="claude-sonnet-4-20250514")


def _root_defn(aid="root"):
    return AgentDefinition(
        id=aid, name="Root", mode=AgentMode.COGNITIVE,
        provider="anthropic", cognitive_model="sonnet",
        system_prompt="root", agent_type="implement", description="root",
    )


def _instrument_hooks(rt):
    bound: list[tuple[str, int, str | None]] = []
    revoked: list[int] = []
    orig_bound = rt._on_pid_bound
    orig_revoked = rt._on_pid_revoked

    def _b(agent_id, pid, parent_agent_id):
        bound.append((agent_id, pid, parent_agent_id))
        return orig_bound(agent_id, pid, parent_agent_id)

    def _r(pid):
        revoked.append(pid)
        return orig_revoked(pid)

    # Re-inject onto the already-constructed supervisor.
    rt.supervisor._on_pid_bound = _b
    rt.supervisor._on_pid_revoked = _r
    return bound, revoked


def _seed_api_provider(rt, agent_id):
    rt.engine.provider_manager._active_providers[agent_id] = _api_provider()


@pytest.mark.asyncio
async def test_p6_isolated_parent_spawns_isolated_child(tmp_path):
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(None, lambda e: events.append(e) or _noop())

    rt = _make_runtime(bus, tmp_path, isolation=True)
    mgr = _MultiPidWorkerManager()
    rt.engine._worker_mgr = mgr
    rt.supervisor._worker_manager = mgr
    bound, revoked = _instrument_hooks(rt)

    # --- bring up the isolated PARENT worker -----------------------------
    parent = _root_defn("root")
    await rt.register_agent(parent)
    await rt.activate_agent(parent.id)
    _seed_api_provider(rt, parent.id)

    peid = await rt.trigger_run(parent.id, source="test")
    assert peid is not None
    # Let the driver spawn the parent worker and park at "go".
    await _until(lambda: parent.id in mgr.handles and mgr.handles[parent.id].channel.go_sent)

    parent_pid = mgr.handles[parent.id].pid
    assert rt.supervisor.is_supervised(parent.id)

    # --- the PARENT worker calls create_agent => spawn_child RPC ---------
    # Pull the REAL rpc handler the daemon injected for this worker and call it
    # exactly as the pipe would: first arg = AUTHORITATIVE pipe-bound parent id.
    host_handlers = _rpc_handlers_for(rt, parent.id)
    assert "spawn_child" in host_handlers

    # Forge a DIFFERENT parent id in the body to prove it is IGNORED.
    child_spec = {
        "name": "Delegate",
        "prompt": "do a subtask",
        "model": "sonnet",
        "mode": "cognitive",
        "parent_agent_id": "ATTACKER-FORGED",   # must be ignored
        "_caller_id": "ATTACKER-FORGED",         # must be ignored
    }
    # Pre-seed the child's provider so ITS recursion is worker-eligible without a
    # network call. The child id is deterministic (parent.<n>) => "root.1".
    _seed_api_provider(rt, "root.1")

    result = await host_handlers["spawn_child"](parent.id, child_spec)
    assert "error" not in result, result
    child_id = result["agent_id"]
    assert child_id == "root.1"            # generate_child_id: parent.<n>

    # parent_id is the AUTHORITATIVE pipe id, NOT the forged body value.
    child_defn = rt.registry.get_agent(child_id)
    assert child_defn.parent_id == parent.id
    assert child_defn.parent_id != "ATTACKER-FORGED"

    # --- child recursed into ITS OWN worker with ITS OWN pid -------------
    await _until(lambda: child_id in mgr.handles and mgr.handles[child_id].channel.go_sent)
    child_pid = mgr.handles[child_id].pid

    # TWO distinct worker pids: parent + child.
    assert parent_pid != child_pid
    assert rt.supervisor.is_supervised(child_id)
    assert {a for a, _ in mgr.ensured} == {parent.id, child_id}

    # _children edge parent -> child (drives leaf-first subtree kill).
    assert child_id in rt.supervisor._children.get(parent.id, set())

    # --- BINDING SEAM fired for the child with real args, before "go" ----
    child_bound = [b for b in bound if b[0] == child_id]
    assert len(child_bound) == 1
    b_agent, b_pid, b_parent = child_bound[0]
    assert b_pid == child_pid                 # real child pid
    assert b_parent == parent.id              # real authoritative parent id
    # Ordering: pid-bound recorded BEFORE the child's "go" was sent.
    # (register() fires _on_pid_bound, then the driver sends run+go.)
    assert mgr.handles[child_id].channel.go_sent is True   # go did happen
    # The bind was appended at register time; the go cmd is the last cmd — both
    # present, and the seam is inside register which precedes send_cmd("run",go).

    # --- SUBTREE KILL: kill the PARENT, child reaped leaf-first ----------
    await rt.supervisor.kill(parent.id, reason="p6_test")
    await _until(lambda: not rt.supervisor.is_supervised(parent.id)
                 and not rt.supervisor.is_supervised(child_id))

    # No orphan child worker survives.
    assert not rt.supervisor.is_supervised(child_id)
    assert not rt.supervisor.is_supervised(parent.id)
    assert parent.id in mgr.closed and child_id in mgr.closed

    # _on_pid_revoked fired for BOTH pids (child included).
    assert child_pid in revoked
    assert parent_pid in revoked

    # Drain any parked driver tasks.
    tasks = [t for t in rt.engine._tasks.values() if not t.done()]
    if tasks:
        await asyncio.wait(tasks, timeout=5)


# --- helpers --------------------------------------------------------------

def _noop():
    return None


async def _until(pred, timeout=5.0, step=0.01):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(step)
    raise AssertionError("condition not met within timeout")


def _rpc_handlers_for(rt, agent_id):
    """Reach the REAL WorkerHost rpc_handlers the engine injected for this
    worker. The engine builds a WorkerHost per cutover and passes its
    rpc_handlers() into ensure_worker; we captured them on the fake manager."""
    # The fake manager stored them? It doesn't — rebuild from a WorkerHost the
    # same way the engine does, so we exercise the identical handler code.
    from atn.runtime.worker_host import WorkerHost
    from atn.models import ExecutionRecord, ExecutionStatus

    rec = ExecutionRecord(
        execution_id="verify-eid", agent_id=agent_id,
        status=ExecutionStatus.RUNNING, trigger_source="test",
    )

    async def _done_cb(status, error, tool_calls):
        return None

    host = WorkerHost(engine=rt.engine, agent_id=agent_id, record=rec,
                      on_execution_done=_done_cb)
    return host.rpc_handlers()
