"""Unit tests for the daemon<->worker IPC contract (atn/runtime/worker_rpc.py).

Isolated: uses an in-process multiprocessing.Pipe pair (which exposes
send_bytes/recv_bytes exactly like the inherited worker pipe will) to round-trip
envelopes through PipeTransport, DaemonClient, and WorkerChannel — no real
subprocess, no torch, no daemon graph.

The security-critical invariants under test:
  - framing is length-prefixed JSON (never pickle)
  - identity is pipe-bound: WorkerChannel ignores any agent_id in the body and
    stamps events/rpc with its authoritative agent_id
  - rpc round-trips by id; events are droppable; rpc requests are not
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import struct

import pytest

from atn.runtime import worker_rpc as wr
from atn.runtime.worker_rpc import (
    DaemonClient,
    Envelope,
    PipeTransport,
    WorkerChannel,
    _decode,
    _encode,
)


# ---------------------------------------------------------------------------
# Framing: length-prefixed JSON, no pickle
# ---------------------------------------------------------------------------

def test_encode_is_length_prefixed_json():
    env = Envelope.rpc_req("budget_check", {"amount": 5})
    frame = _encode(env.to_dict())
    # First 4 bytes are a big-endian length of the JSON body.
    (n,) = struct.unpack(">I", frame[:4])
    body = frame[4:]
    assert len(body) == n
    # Body is valid JSON — decodable without any pickle machinery.
    d = json.loads(body.decode("utf-8"))
    assert d["kind"] == "rpc_req"
    assert d["name"] == "budget_check"
    assert d["payload"] == {"amount": 5}
    assert d["v"] == wr.PROTO_VERSION


def test_encode_is_not_pickle():
    # A pickle stream starts with the PROTO opcode b"\x80". Our frame starts
    # with a 4-byte length whose first byte is 0 for any reasonable payload.
    frame = _encode(Envelope.event("step.output", {"text": "hi"}).to_dict())
    assert frame[0:1] == b"\x00"
    assert b"__reduce__" not in frame  # no pickle reduce protocol anywhere


def test_decode_roundtrip():
    env = Envelope.status("ready", {"pid": 4242})
    d = _decode(_encode(env.to_dict()))
    back = Envelope.from_dict(d)
    assert back.kind == "status"
    assert back.name == "ready"
    assert back.payload == {"pid": 4242}


def test_decode_rejects_truncated():
    frame = _encode(Envelope.event("x", {}).to_dict())
    with pytest.raises(ValueError):
        _decode(frame[:-2])  # chop the body


def test_from_dict_rejects_bad_kind():
    with pytest.raises(ValueError):
        Envelope.from_dict({"kind": "pickle", "name": "", "id": None, "payload": {}})


def test_no_pickle_guard_rejects_non_connection():
    class Fake:  # has neither send_bytes nor recv_bytes
        pass
    with pytest.raises(TypeError):
        PipeTransport(Fake())


def test_module_never_calls_pickle_send_recv():
    # Belt-and-suspenders: the module source must not use the pickle path on the
    # raw pipe Connection (self._conn.send / self._conn.recv). Only the *_bytes
    # methods are permitted on the Connection.
    import inspect
    src = inspect.getsource(wr)
    assert "self._conn.send_bytes" in src and "self._conn.recv_bytes" in src
    assert "self._conn.send(" not in src
    assert "self._conn.recv(" not in src


# ---------------------------------------------------------------------------
# PipeTransport round-trip over a real Pipe pair
# ---------------------------------------------------------------------------

async def test_pipe_transport_roundtrip():
    a, b = mp.Pipe(duplex=True)
    ta, tb = PipeTransport(a), PipeTransport(b)
    try:
        await ta.send(Envelope.cmd("run", {"go": True}))
        got = await tb.recv()
        assert got.kind == "cmd"
        assert got.name == "run"
        assert got.payload == {"go": True}
    finally:
        ta.close()
        tb.close()


# ---------------------------------------------------------------------------
# Full demux: DaemonClient (worker side) <-> WorkerChannel (daemon side)
# ---------------------------------------------------------------------------

async def _wire():
    daemon_end, worker_end = mp.Pipe(duplex=True)
    return PipeTransport(daemon_end), PipeTransport(worker_end)


async def test_rpc_roundtrip_and_pipe_bound_identity():
    daemon_t, worker_t = await _wire()
    seen = {}

    async def budget_check(agent_id, payload):
        # agent_id is the AUTHORITATIVE pipe-bound id, NOT what the worker claims.
        seen["agent_id"] = agent_id
        seen["payload"] = payload
        return {"ok": True, "remaining": 100}

    channel = WorkerChannel(
        "agent-AUTHORITATIVE",
        daemon_t,
        rpc_handlers={"budget_check": budget_check},
    )
    client = DaemonClient(worker_t)
    channel.start()
    client.start()
    try:
        # Worker sends a spoofed agent_id in the body — must be ignored.
        result = await asyncio.wait_for(
            client.rpc("budget_check", {"amount": 7, "agent_id": "agent-SPOOF"}),
            timeout=5,
        )
        assert result == {"ok": True, "remaining": 100}
        assert seen["agent_id"] == "agent-AUTHORITATIVE"
        # The body still contains the spoof field, but the handler got the real id.
        assert seen["payload"]["amount"] == 7
    finally:
        await client.close()
        await channel.close()


async def test_unwired_rpc_returns_error():
    daemon_t, worker_t = await _wire()
    channel = WorkerChannel("agent-1", daemon_t)  # no handlers wired (P4 does that)
    client = DaemonClient(worker_t)
    channel.start()
    client.start()
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(client.rpc("spawn_child", {}), timeout=5)
    finally:
        await client.close()
        await channel.close()


async def test_unknown_rpc_name_rejected_worker_side():
    daemon_t, worker_t = await _wire()
    client = DaemonClient(worker_t)
    try:
        with pytest.raises(ValueError):
            await client.rpc("definitely_not_a_real_rpc", {})
    finally:
        await client.close()
        daemon_t.close()


async def test_event_source_overwritten_by_channel():
    daemon_t, worker_t = await _wire()
    events = []

    async def event_sink(agent_id, name, payload):
        events.append((agent_id, name, payload))

    channel = WorkerChannel("agent-REAL", daemon_t, event_sink=event_sink)
    client = DaemonClient(worker_t)
    channel.start()
    client.start()
    try:
        await client.emit("step.output", {"text": "hello", "source": "agent-FAKE"})
        # Give the daemon side a moment to drain the event queue.
        for _ in range(50):
            if events:
                break
            await asyncio.sleep(0.01)
        assert events, "event never reached the sink"
        agent_id, name, payload = events[0]
        assert agent_id == "agent-REAL"  # channel-stamped, not the body's source
        assert name == "step.output"
        assert payload["text"] == "hello"
    finally:
        await client.close()
        await channel.close()


async def test_status_forwarded():
    daemon_t, worker_t = await _wire()
    statuses = []

    async def status_sink(agent_id, name, payload):
        statuses.append((agent_id, name, payload))

    channel = WorkerChannel("agent-S", daemon_t, status_sink=status_sink)
    client = DaemonClient(worker_t)
    channel.start()
    client.start()
    try:
        await client.send_status("ready", {"pid": 999})
        for _ in range(50):
            if statuses:
                break
            await asyncio.sleep(0.01)
        assert statuses == [("agent-S", "ready", {"pid": 999})]
    finally:
        await client.close()
        await channel.close()


async def test_cmd_delivered_to_worker_handler():
    daemon_t, worker_t = await _wire()
    got = []

    async def on_cmd(env: Envelope):
        got.append((env.name, env.payload))

    channel = WorkerChannel("agent-C", daemon_t)
    client = DaemonClient(worker_t, on_cmd=on_cmd)
    channel.start()
    client.start()
    try:
        await channel.send_cmd("shutdown", {})
        for _ in range(50):
            if got:
                break
            await asyncio.sleep(0.01)
        assert got == [("shutdown", {})]
    finally:
        await client.close()
        await channel.close()


async def test_event_queue_drops_oldest_on_overflow():
    daemon_t, worker_t = await _wire()
    # maxsize=2, no sink drain running until we start it, so we can force overflow.
    channel = WorkerChannel("agent-Q", daemon_t, event_queue_maxsize=2)
    # Directly exercise the offer path (bounded, drop-oldest+warn).
    channel._offer_event(Envelope.event("e1", {}))
    channel._offer_event(Envelope.event("e2", {}))
    channel._offer_event(Envelope.event("e3", {}))  # overflow -> drop e1
    names = []
    while not channel._events.empty():
        names.append(channel._events.get_nowait().name)
    assert names == ["e2", "e3"]  # oldest (e1) dropped
    daemon_t.close()
    worker_t.close()
