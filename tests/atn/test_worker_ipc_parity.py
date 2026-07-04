"""Worker IPC parity for the three in-process hardening features (docs/agentic_loop.md).

Under ATN_WORKER_ISOLATION the agent's provider lives in the worker process, so
three daemon-side features have to travel over the IPC command channel:

  1. injection consumption acks   — send_delegate_message injected/inbox_fallback
  2. undelivered-steering re-post  — worker reports leftover steering; host inboxes it
  3. manual compaction request     — request_compaction forwarded to the live worker

These tests exercise the REAL wire path (mp.Pipe + PipeTransport + DaemonClient +
WorkerChannel + AgentWorker.on_cmd) — no daemon, no subprocess. asyncio_mode=auto,
so bare ``async def`` tests are collected automatically (matches test_worker_rpc.py).
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest

from atn.agent_worker import AgentWorker
from atn.runtime.worker_rpc import (
    DaemonClient,
    Envelope,
    PipeTransport,
    WorkerChannel,
)


# ---------------------------------------------------------------------------
# Wiring helper: a real daemon<->worker pipe with a live AgentWorker on the
# worker end and a bare WorkerChannel on the daemon end.
# ---------------------------------------------------------------------------

async def _wire_worker(provider=None):
    daemon_end, worker_end = mp.Pipe(duplex=True)
    daemon_t = PipeTransport(daemon_end)
    worker_t = PipeTransport(worker_end)

    client = DaemonClient(worker_t)
    worker = AgentWorker(client)
    worker._agent_label = "test-agent"
    worker._provider = provider           # simulate a mid-run live provider
    client.set_cmd_handler(worker.on_cmd)

    channel = WorkerChannel("agent-AUTH", daemon_t)
    client.start()
    channel.start()

    async def _close():
        await client.close()
        await channel.close()

    return channel, worker, _close


# ---------------------------------------------------------------------------
# Provider doubles (mirror tests/atn/test_execution_control_delegate.py)
# ---------------------------------------------------------------------------

class ConsumingProvider:
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


class GenericQueueProvider:
    """Base-style provider whose send_user_message returns True (no id contract)."""
    def __init__(self):
        self.sent = []
        self._compact_requested = False
        self._compact_requested_by = ""
        self._active = True   # steering queue open

    async def send_user_message(self, content):
        self.sent.append(content)
        return True

    def request_compaction(self, requested_by=""):
        if not self._active:
            return False
        self._compact_requested = True
        self._compact_requested_by = requested_by
        return True


# ---------------------------------------------------------------------------
# Gap 1: injection consumption ack over cmd_res
# ---------------------------------------------------------------------------

async def test_injection_consumed_reports_injected():
    channel, worker, close = await _wire_worker(ConsumingProvider(consume_after=1))
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("send_user_message", {"content": "steer"}),
            timeout=5,
        )
        assert ack == {"status": "injected"}
    finally:
        await close()


async def test_injection_unconsumed_reports_inbox_fallback_with_content():
    prov = NeverConsumingProvider()
    channel, worker, close = await _wire_worker(prov)
    try:
        # Short timeout so the worker's consumption poll gives up fast.
        worker._await_injection_consumed = _fast_poll(worker)
        ack = await asyncio.wait_for(
            channel.send_cmd_await("send_user_message", {"content": "lost msg"}),
            timeout=5,
        )
        assert ack["status"] == "inbox_fallback"
        assert ack["content"] == "lost msg"       # exact reclaimed content
        assert prov._pending == {}                # reclaimed, not dangling
    finally:
        await close()


async def test_injection_no_live_provider_reports_inbox_fallback():
    channel, worker, close = await _wire_worker(provider=None)
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("send_user_message", {"content": "queued"}),
            timeout=5,
        )
        assert ack == {"status": "inbox_fallback", "content": "queued"}
    finally:
        await close()


async def test_injection_generic_true_contract_reports_injected():
    prov = GenericQueueProvider()
    channel, worker, close = await _wire_worker(prov)
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("send_user_message", {"content": "hi"}),
            timeout=5,
        )
        # No id contract (returns True) -> injected (prior behavior).
        assert ack == {"status": "injected"}
        assert prov.sent == ["hi"]
    finally:
        await close()


def _fast_poll(worker):
    """Wrap the worker's poll with a tiny timeout so the never-consumed test
    doesn't wait the full 2 s window."""
    async def _p(provider, inj_id, timeout=0.1, interval=0.02):
        return await AgentWorker._await_injection_consumed(
            worker, provider, inj_id, timeout=0.1, interval=0.02)
    return _p


# ---------------------------------------------------------------------------
# Gap 3: manual compaction request forwarded over cmd_res
# ---------------------------------------------------------------------------

async def test_request_compaction_accepted():
    prov = GenericQueueProvider()
    channel, worker, close = await _wire_worker(prov)
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("request_compaction", {"requested_by": "owner"}),
            timeout=5,
        )
        assert ack == {"accepted": True}
        # Flag actually set on the live provider, with the requester.
        assert prov._compact_requested is True
        assert prov._compact_requested_by == "owner"
    finally:
        await close()


async def test_request_compaction_not_active_reports_false():
    prov = GenericQueueProvider()
    prov._active = False   # orchestration already ended -> request_compaction False
    channel, worker, close = await _wire_worker(prov)
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("request_compaction", {"requested_by": "owner"}),
            timeout=5,
        )
        assert ack == {"accepted": False}
    finally:
        await close()


async def test_request_compaction_no_provider_reports_false():
    channel, worker, close = await _wire_worker(provider=None)
    try:
        ack = await asyncio.wait_for(
            channel.send_cmd_await("request_compaction", {"requested_by": "owner"}),
            timeout=5,
        )
        assert ack == {"accepted": False}
    finally:
        await close()


# ---------------------------------------------------------------------------
# cmd_res protocol invariants
# ---------------------------------------------------------------------------

async def test_fire_and_forget_cmd_gets_no_reply():
    """A cmd with no id (send_cmd, not send_cmd_await) must NOT produce a
    cmd_res — the reader only replies when env.id is set. We assert the
    interrupt cmd is handled and no pending-cmd future is orphaned."""
    prov = GenericQueueProvider()
    channel, worker, close = await _wire_worker(prov)
    try:
        await channel.send_cmd("interrupt", {})
        # Give the worker a beat to handle it; nothing should come back.
        await asyncio.sleep(0.1)
        assert channel._pending_cmds == {}
    finally:
        await close()


async def test_send_cmd_await_times_out_when_worker_silent():
    """If no worker is wired to reply, send_cmd_await raises TimeoutError and
    cleans up its pending future."""
    daemon_end, worker_end = mp.Pipe(duplex=True)
    daemon_t = PipeTransport(daemon_end)
    # Deliberately leave the worker end unread (no DaemonClient).
    channel = WorkerChannel("agent-1", daemon_t)
    channel.start()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await channel.send_cmd_await("request_compaction", {}, timeout=0.2)
        assert channel._pending_cmds == {}
    finally:
        await channel.close()
        try:
            worker_end.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Gap 2: undelivered-steering re-post at execution_done (WorkerHost)
# ---------------------------------------------------------------------------

class _FakeInbox:
    def __init__(self):
        self.posted = []

    def post(self, msg):
        self.posted.append(msg)


class _FakeEvents:
    async def emit(self, event):
        pass


class _FakeEngine:
    def __init__(self):
        self.inbox = _FakeInbox()
        self.events = _FakeEvents()
        self._booked_by_execution = {}
        self.supervisor = None
        self.provider_manager = None


async def test_worker_undelivered_steering_reposted_to_inbox():
    """WorkerHost._handle_execution_done must re-post any undelivered_steering the
    worker reports, mirroring the in-process finalize re-post (WORK/HIGH,
    source=steering-fallback, data={instruction})."""
    from atn.runtime.worker_host import WorkerHost
    from atn.models import (
        ExecutionRecord, ExecutionStatus, MessagePriority, MessageType,
    )

    engine = _FakeEngine()
    record = ExecutionRecord(
        execution_id="exec-1", agent_id="agent-AUTH", status=ExecutionStatus.RUNNING, trigger_source="test",
    )

    done = {}

    async def _on_done(status, error, tool_calls):
        done["status"] = status

    host = WorkerHost(engine=engine, agent_id="agent-AUTH", record=record,
                      on_execution_done=_on_done)

    payload = {
        "status": "completed",
        "error": None,
        "output": {"result": "ok", "tokens_used": 0, "usage": {}},
        "token_usage": {},
        "tool_calls": [],
        "undelivered_steering": ["steer A", "steer B"],
    }
    await host._status_sink("agent-AUTH", "execution_done", payload)

    assert done["status"] == ExecutionStatus.COMPLETED
    assert len(engine.inbox.posted) == 2
    for msg, expected in zip(engine.inbox.posted, ["steer A", "steer B"]):
        assert msg.source == "steering-fallback"
        assert msg.target == "agent-AUTH"
        assert msg.type == MessageType.WORK
        assert msg.priority == MessagePriority.HIGH
        assert msg.data["instruction"] == expected


async def test_worker_no_undelivered_steering_posts_nothing():
    from atn.runtime.worker_host import WorkerHost
    from atn.models import ExecutionRecord, ExecutionStatus

    engine = _FakeEngine()
    record = ExecutionRecord(
        execution_id="exec-2", agent_id="agent-AUTH", status=ExecutionStatus.RUNNING, trigger_source="test",
    )

    async def _on_done(status, error, tool_calls):
        pass

    host = WorkerHost(engine=engine, agent_id="agent-AUTH", record=record,
                      on_execution_done=_on_done)
    await host._status_sink("agent-AUTH", "execution_done", {
        "status": "completed", "output": {"result": ""}, "token_usage": {},
        "tool_calls": [],
    })
    assert engine.inbox.posted == []


async def test_channel_close_fails_pending_cmd():
    """Closing the channel with an outstanding cmd future fails it rather than
    hanging the awaiter."""
    daemon_end, worker_end = mp.Pipe(duplex=True)
    daemon_t = PipeTransport(daemon_end)
    channel = WorkerChannel("agent-1", daemon_t)
    channel.start()

    async def _await():
        return await channel.send_cmd_await("request_compaction", {}, timeout=5)

    task = asyncio.create_task(_await())
    await asyncio.sleep(0.05)         # let the send happen + future register
    await channel.close()
    with pytest.raises((ConnectionError, asyncio.CancelledError)):
        await task
    try:
        worker_end.close()
    except Exception:
        pass
