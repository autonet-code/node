"""P5 — secret_* tool surface (worker requests from its own PID; monitor tripwire).

Three proofs, per the phase brief:

  1. GRANT -> STAGE -> LEAK-DETECTED: a granted worker calls
     ``secret_request_secret`` -> gets {var_name, path}, NEVER the value; the
     broker's value-push feeds the daemon SecurityMonitor with value+path; when
     the worker then EMITS the value (or the path) in STEP_OUTPUT, a
     SECURITY_ALARM fires naming the secret + the leaking agent.

  2. DENY: requesting a NON-granted service returns {ok: False}, no staging, no
     value.

  3. FLAG-OFF / UNGRANTED: the secret_* schemas are NOT injected into the
     manifest (the model cannot even name the tools); and the worker handler is
     fail-closed refused outside a worker context.

The broker pipe itself is exercised in P2's E2E; here we stub WorkerBrokerClient
so the proof is deterministic and pipe-free.
"""
import asyncio
import os

import pytest

from atn.events import Event, EventBus, EventType
from atn.runtime import worker_loop
from atn.runtime.worker_loop import (
    SECRET_TOOL_SCHEMAS, _SECRET_TOOLS, _handle_secret_tool, _in_worker_context,
)
from atn.runtime.security_monitor import SecurityMonitor


# ── stub broker (deterministic, pipe-free) ────────────────────────────────────
class _FakeWorkerBroker:
    """Mimics WorkerBrokerClient over a fixed grant set. Staging a value here is
    what the real broker does server-side; the test then hand-delivers the
    value-push to the monitor (the pipe transport is P2's concern)."""

    def __init__(self, granted, staged, pushed):
        self._granted = granted        # {service: (var_name, value)}
        self._staged = staged          # service -> path (records staging)
        self._pushed = pushed          # list of value-push exposures

    def list_services(self):
        return {"ok": True, "services": sorted(self._granted)}

    def request(self, service):
        if service not in self._granted:
            return {"ok": False, "error": f"service '{service}' not granted"}
        var_name, value = self._granted[service]
        path = rf"C:\Users\x\AppData\Local\Temp\{os.urandom(8).hex()}"
        self._staged[service] = path
        # The real broker pushes {value, staged_path, ...} to the daemon here.
        self._pushed.append({
            "value": value, "staged_path": path, "secret_name": service,
        })
        return {"ok": True, "service": service, "var_name": var_name, "path": path}

    def release(self, service):
        return {"ok": bool(self._staged.pop(service, None))}


def _install_fake_broker(monkeypatch, granted):
    staged, pushed = {}, []
    fake = _FakeWorkerBroker(granted, staged, pushed)
    monkeypatch.setattr(
        "atn.runtime.broker_client.WorkerBrokerClient", lambda *a, **k: fake)
    return fake, staged, pushed


# ── 1. GRANT -> STAGE -> LEAK-DETECTED ────────────────────────────────────────
def test_grant_stage_leak_detected(tmp_path, monkeypatch):
    async def go():
        monkeypatch.setenv("ATN_IS_WORKER", "1")
        granted = {"openai_key": ("OPENAI_API_KEY", "sk-LEAKED-VALUE-9x8y")}
        _fake, staged, pushed = _install_fake_broker(monkeypatch, granted)

        # Worker requests the granted secret.
        res = await _handle_secret_tool("secret_request_secret", {"service": "openai_key"})
        assert res["ok"] is True
        assert res["var_name"] == "OPENAI_API_KEY"
        assert res["path"]
        # SAFETY-CRITICAL: the VALUE is NOT in the tool result.
        assert "sk-LEAKED-VALUE-9x8y" not in str(res)
        assert set(res) == {"ok", "var_name", "path"}  # exactly the safe fields
        # A value WAS staged (a file path exists) and the push carries the value.
        assert staged["openai_key"] == res["path"]
        assert pushed and pushed[0]["value"] == "sk-LEAKED-VALUE-9x8y"

        # Wire the daemon monitor and feed it the broker's push (as P2 does).
        bus = EventBus()
        alarms = []

        async def _catch(ev):
            alarms.append(ev)

        bus.subscribe(EventType.SECURITY_ALARM, _catch)
        mon = SecurityMonitor(bus, data_dir=tmp_path)
        bus.subscribe(EventType.STEP_OUTPUT, mon._on_step_output)
        exp = pushed[0]
        mon.add_exposure(value=exp["value"], staged_path=exp["staged_path"],
                         secret_name=exp["secret_name"], agent_id="agent-A", pid=1234)

        # The worker leaks the VALUE into its transcript.
        await bus.emit(Event(type=EventType.STEP_OUTPUT, source="agent-A",
                             data={"content": f"the key is {exp['value']}"}))
        assert len(alarms) == 1
        assert alarms[0].source == "agent-A"
        assert alarms[0].data == {"names": ["openai_key"]}
        assert "agent_id" not in alarms[0].data
        assert exp["value"] not in str(alarms[0].data)  # value never in the event

        # It ALSO catches a PATH leak (the staged path is registered too). Use a
        # fresh agent so the monitor's per-(agent,secret) dedup window doesn't
        # suppress this second alarm.
        alarms.clear()
        mon.add_exposure(value=exp["value"], staged_path=exp["staged_path"],
                         secret_name=exp["secret_name"], agent_id="agent-B", pid=5678)
        await bus.emit(Event(type=EventType.STEP_OUTPUT, source="agent-B",
                             data={"content": f"reading {exp['staged_path']}"}))
        assert len(alarms) == 1 and alarms[0].data == {"names": ["openai_key"]}
        assert alarms[0].source == "agent-B"

    asyncio.run(go())


# ── 2. DENY (non-granted service) ─────────────────────────────────────────────
def test_request_non_granted_denied_no_staging(monkeypatch):
    async def go():
        monkeypatch.setenv("ATN_IS_WORKER", "1")
        granted = {"openai_key": ("OPENAI_API_KEY", "sk-abc")}
        _fake, staged, pushed = _install_fake_broker(monkeypatch, granted)

        res = await _handle_secret_tool("secret_request_secret", {"service": "aws_key"})
        assert res["ok"] is False
        assert "not granted" in res["error"]
        assert "path" not in res and "var_name" not in res
        assert staged == {} and pushed == []  # nothing staged, nothing pushed

    asyncio.run(go())


def test_list_services_names_only(monkeypatch):
    async def go():
        monkeypatch.setenv("ATN_IS_WORKER", "1")
        granted = {"openai_key": ("K", "sk-VAL1"), "stripe_key": ("K2", "sk-VAL2")}
        _install_fake_broker(monkeypatch, granted)
        res = await _handle_secret_tool("secret_list_services", {})
        assert res == {"ok": True, "services": ["openai_key", "stripe_key"]}
        assert "sk-VAL1" not in str(res) and "sk-VAL2" not in str(res)  # never a value
    asyncio.run(go())


# ── 3. worker-context fail-closed ─────────────────────────────────────────────
def test_secret_tool_refused_outside_worker(monkeypatch):
    async def go():
        monkeypatch.delenv("ATN_IS_WORKER", raising=False)
        assert _in_worker_context() is False
        # Even with a broker that WOULD grant, no worker context => refuse, and
        # the broker is never touched (so nothing stages).
        granted = {"openai_key": ("K", "sk-should-not-stage")}
        _fake, staged, pushed = _install_fake_broker(monkeypatch, granted)
        res = await _handle_secret_tool("secret_request_secret", {"service": "openai_key"})
        assert res == {"ok": False, "error": "secret tools are worker-only"}
        assert staged == {} and pushed == []
    asyncio.run(go())


# ── double-gate: schema injection (daemon-side manifest builder) ──────────────
class _FakeRuntime:
    def __init__(self, pending):
        self._pending_grants = pending


class _GateHarness:
    """Minimal stand-in exercising ONLY the double-gate expression the manifest
    builder uses: (flag ON) AND (non-empty grant) -> schemas appended."""

    def __init__(self, iso_on, pending):
        self._iso_on = iso_on
        self._runtime_ref = _FakeRuntime(pending)

    def _worker_isolation_enabled(self):
        return self._iso_on

    def build(self, agent_id):
        agent_tools = []
        rt = getattr(self, "_runtime_ref", None)
        if (rt is not None and self._worker_isolation_enabled()
                and rt._pending_grants.get(agent_id)):
            agent_tools.extend(SECRET_TOOL_SCHEMAS)
        return agent_tools


def _names(tools):
    return {t["name"] for t in tools}


def test_schemas_injected_only_when_granted_and_flag_on():
    # granted + flag on => schemas present
    h = _GateHarness(iso_on=True, pending={"a": ["openai_key"]})
    assert _names(h.build("a")) == set(_SECRET_TOOLS)


def test_no_schemas_when_flag_off():
    h = _GateHarness(iso_on=False, pending={"a": ["openai_key"]})
    assert h.build("a") == []


def test_no_schemas_when_empty_grant():
    h = _GateHarness(iso_on=True, pending={})
    assert h.build("a") == []
    h2 = _GateHarness(iso_on=True, pending={"a": []})  # explicit empty list
    assert h2.build("a") == []


def test_inprocess_route_refuses_secret_tools():
    """Defense-in-depth: if a granted execution falls back to the in-process
    path (worker spawn failure), route_tool_call refuses the secret_* tools —
    the daemon holds no broker session and must never stage a secret for itself.
    """
    import asyncio as _asyncio
    from atn.runtime.execution_engine import ExecutionEngine

    async def go():
        # Reach the guard directly on an unbound method: it short-circuits before
        # any self.* dependency, so a bare object is enough.
        class _E:
            route_tool_call = ExecutionEngine.route_tool_call
        e = _E()
        for name in ("secret_request_secret", "secret_list_services", "secret_release"):
            res = await e.route_tool_call(name, {"service": "openai_key"}, "agent-A")
            assert res == {"ok": False, "error": "secret tools require an isolated worker"}

    _asyncio.run(go())


def test_schemas_are_wellformed():
    for s in SECRET_TOOL_SCHEMAS:
        assert set(s) == {"name", "description", "input_schema"}
        assert s["name"] in _SECRET_TOOLS
        assert s["input_schema"]["type"] == "object"
    assert _names(SECRET_TOOL_SCHEMAS) == set(_SECRET_TOOLS)
