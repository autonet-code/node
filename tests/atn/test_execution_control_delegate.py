"""§5 delivery-ack + §11 kill-ladder wiring in ExecutionControl.

Unit-level: a fake engine (inbox + registry), a fake provider_manager
(_active_providers), and fake providers that model the bridge's consumption
ack / subprocess-termination contract. No real subprocess is spawned.
"""
import asyncio

import pytest

from atn.runtime.execution_control import ExecutionControl
from atn.models import MessagePriority, MessageType


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeInbox:
    def __init__(self):
        self.posted = []
    def post(self, msg):
        self.posted.append(msg)


class FakeRegistry:
    def __init__(self, agent_ids):
        self._agents = {a: object() for a in agent_ids}


class FakeEngine:
    def __init__(self, agent_ids=("delegate-1",)):
        self.inbox = FakeInbox()
        self.registry = FakeRegistry(agent_ids)
        self._executions = {}
        self._cancels = {}
        self._tasks = {}
        self._interrupt_hooks = {}
        self._config = None  # worker_isolation off


class FakeProviderManager:
    def __init__(self):
        self._active_providers = {}


class ConsumingProvider:
    """Bridge-like provider: send_user_message returns an id; the injection is
    consumed after ``consume_after`` polls."""
    def __init__(self, consume_after=1):
        self._seq = 0
        self._pending = {}
        self._polls = {}
        self._consume_after = consume_after
    async def send_user_message(self, content):
        self._seq += 1
        inj = f"inj-{self._seq}"
        self._pending[inj] = content
        self._polls[inj] = 0
        return inj
    def injection_consumed(self, inj_id):
        self._polls[inj_id] = self._polls.get(inj_id, 0) + 1
        if self._polls[inj_id] >= self._consume_after:
            self._pending.pop(inj_id, None)
        return inj_id not in self._pending
    def take_unconsumed_injection(self, inj_id):
        return self._pending.pop(inj_id, None)


class NeverConsumingProvider:
    """Injection is written but never acked → forces the inbox fallback."""
    def __init__(self):
        self._seq = 0
        self._pending = {}
    async def send_user_message(self, content):
        self._seq += 1
        inj = f"inj-{self._seq}"
        self._pending[inj] = content
        return inj
    def injection_consumed(self, inj_id):
        return inj_id not in self._pending
    def take_unconsumed_injection(self, inj_id):
        return self._pending.pop(inj_id, None)


class LegacyProvider:
    """Old provider whose send_user_message returns None (no ack contract)."""
    def __init__(self):
        self.sent = []
    async def send_user_message(self, content):
        self.sent.append(content)
        return None


class KillableProvider:
    def __init__(self, confirmed=True):
        self.terminated = False
        self._confirmed = confirmed
    async def interrupt(self):
        pass
    async def terminate_subprocess(self, reason=""):
        self.terminated = True
        return self._confirmed


def make_control(agent_ids=("delegate-1",)):
    engine = FakeEngine(agent_ids)
    pm = FakeProviderManager()
    return ExecutionControl(engine, pm), engine, pm


# ---------------------------------------------------------------------------
# Worker-isolation fakes (supervised path)
# ---------------------------------------------------------------------------

class _FakeWIConfig:
    class _WI:
        enabled = True
    worker_isolation = _WI()


class _FakeChannel:
    """Records send_cmd_await calls and returns a scripted ack (or raises)."""
    def __init__(self, ack=None, raises=False):
        self.awaited = []
        self._ack = ack
        self._raises = raises

    async def send_cmd_await(self, name, payload=None, *, timeout=5.0):
        self.awaited.append((name, dict(payload or {})))
        if self._raises:
            raise RuntimeError("pipe gone")
        return self._ack


class _FakeWorker:
    def __init__(self, channel):
        self.handle = type("H", (), {"channel": channel})()


class _FakeSupervisor:
    def __init__(self, workers):
        self._workers = workers   # {agent_id: _FakeWorker}

    def is_supervised(self, agent_id):
        return agent_id in self._workers

    def get(self, agent_id):
        return self._workers.get(agent_id)


def make_supervised_control(agent_id="delegate-1", ack=None, raises=False):
    engine = FakeEngine((agent_id,))
    engine._config = _FakeWIConfig()      # flag ON
    pm = FakeProviderManager()
    channel = _FakeChannel(ack=ack, raises=raises)
    sup = _FakeSupervisor({agent_id: _FakeWorker(channel)})
    control = ExecutionControl(engine, pm, supervisor=sup)
    return control, engine, channel


# ---------------------------------------------------------------------------
# send_delegate_message — worker-path consumption ack (IPC parity, gap 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_injection_consumed_reports_injected():
    control, engine, channel = make_supervised_control(ack={"status": "injected"})
    status = await control.send_delegate_message("delegate-1", "steer")
    assert status == "injected"
    assert channel.awaited == [("send_user_message", {"content": "steer"})]
    assert engine.inbox.posted == []       # no fallback


@pytest.mark.asyncio
async def test_worker_injection_unconsumed_falls_back_to_inbox():
    control, engine, channel = make_supervised_control(
        ack={"status": "inbox_fallback", "content": "reclaimed text"})
    status = await control.send_delegate_message("delegate-1", "reclaimed text")
    assert status == "inbox_fallback"
    # Daemon re-posted the worker-reclaimed content as HIGH WORK.
    assert len(engine.inbox.posted) == 1
    msg = engine.inbox.posted[0]
    assert msg.target == "delegate-1"
    assert msg.type == MessageType.WORK
    assert msg.priority == MessagePriority.HIGH
    assert msg.data["instruction"] == "reclaimed text"


@pytest.mark.asyncio
async def test_worker_injection_cmd_error_falls_back_to_inbox():
    control, engine, channel = make_supervised_control(raises=True)
    status = await control.send_delegate_message("delegate-1", "unreachable")
    assert status == "inbox_fallback"
    assert len(engine.inbox.posted) == 1
    assert engine.inbox.posted[0].data["instruction"] == "unreachable"


# ---------------------------------------------------------------------------
# send_delegate_message — §5 ack + fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_message_consumed_reports_injected():
    control, engine, pm = make_control()
    pm._active_providers["delegate-1"] = ConsumingProvider(consume_after=1)

    status = await control.send_delegate_message("delegate-1", "steer")
    assert status == "injected"
    assert engine.inbox.posted == []  # no fallback needed


@pytest.mark.asyncio
async def test_delegate_message_unconsumed_falls_back_to_inbox():
    control, engine, pm = make_control()
    prov = NeverConsumingProvider()
    pm._active_providers["delegate-1"] = prov

    status = await control.send_delegate_message("delegate-1", "lost message")
    assert status == "inbox_fallback"
    # The content was re-posted to the inbox as HIGH WORK.
    assert len(engine.inbox.posted) == 1
    msg = engine.inbox.posted[0]
    assert msg.target == "delegate-1"
    assert msg.type == MessageType.WORK
    assert msg.priority == MessagePriority.HIGH
    assert msg.data["instruction"] == "lost message"
    # The pending injection was reclaimed (not left dangling).
    assert prov._pending == {}


@pytest.mark.asyncio
async def test_delegate_message_legacy_provider_reports_injected():
    control, engine, pm = make_control()
    prov = LegacyProvider()
    pm._active_providers["delegate-1"] = prov

    status = await control.send_delegate_message("delegate-1", "hi")
    assert status == "injected"
    assert prov.sent == ["hi"]
    assert engine.inbox.posted == []


@pytest.mark.asyncio
async def test_delegate_message_no_provider_uses_inbox_when_agent_exists():
    control, engine, pm = make_control(agent_ids=("delegate-1",))
    # No provider registered → not running live, but the agent exists.
    status = await control.send_delegate_message("delegate-1", "queued")
    assert status == "inbox_fallback"
    assert len(engine.inbox.posted) == 1


@pytest.mark.asyncio
async def test_delegate_message_unknown_agent_returns_false():
    control, engine, pm = make_control(agent_ids=())  # no agents
    status = await control.send_delegate_message("ghost", "x")
    assert status is False
    assert engine.inbox.posted == []


@pytest.mark.asyncio
async def test_await_injection_consumed_timeout(monkeypatch):
    control, engine, pm = make_control()
    prov = NeverConsumingProvider()
    # Fast timeout so the test is quick.
    inj_id = await prov.send_user_message("x")
    consumed = await control._await_injection_consumed(prov, inj_id, timeout=0.1, interval=0.02)
    assert consumed is False


# ---------------------------------------------------------------------------
# kill ladder wiring — §11
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_agent_verifies_subprocess_killed():
    control, engine, pm = make_control()
    prov = KillableProvider(confirmed=True)
    pm._active_providers["delegate-1"] = prov

    # No in-process executions for this agent → count 0, but the subprocess
    # kill still runs.
    n = await control.kill_agent("delegate-1")
    assert n == 0
    assert prov.terminated is True


@pytest.mark.asyncio
async def test_kill_agent_no_terminate_method_is_noop():
    control, engine, pm = make_control()
    # A provider without terminate_subprocess (in-process) must not error.
    pm._active_providers["delegate-1"] = LegacyProvider()
    n = await control.kill_agent("delegate-1")
    assert n == 0  # no crash


@pytest.mark.asyncio
async def test_verify_provider_subprocess_unconfirmed_logs(caplog):
    control, engine, pm = make_control()
    prov = KillableProvider(confirmed=False)
    pm._active_providers["delegate-1"] = prov
    await control._verify_provider_subprocess_killed("delegate-1")
    assert prov.terminated is True
