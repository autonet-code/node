"""P3 tripwire — SecurityMonitor value/path/chunk-split leak detection.

Exercises the monitor directly against a real EventBus (no broker/pipes): feed a
synthetic exposure, emit STEP_OUTPUT events, assert SECURITY_ALARM fires with the
secret NAME + source agent, carries NO agent_id key, never the value; and that
drop_exposure stops future alarms and flag-off/empty-live-set is silent.
"""
import asyncio

import pytest

from atn.events import Event, EventBus, EventType
from atn.runtime.security_monitor import SecurityMonitor


def _mk(tmp_path):
    bus = EventBus()
    alarms = []

    async def _catch(ev):
        alarms.append(ev)

    bus.subscribe(EventType.SECURITY_ALARM, _catch)
    mon = SecurityMonitor(bus, data_dir=tmp_path)
    # Subscribe to STEP_OUTPUT but do NOT start the sweeper thread (deterministic).
    bus.subscribe(EventType.STEP_OUTPUT, mon._on_step_output)
    return bus, mon, alarms


async def _emit_output(bus, src, content):
    await bus.emit(Event(type=EventType.STEP_OUTPUT, source=src,
                         data={"agent_id": src, "channel": "text",
                               "content": content, "cognitive": True}))


def test_value_leak_fires_alarm(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        mon.add_exposure(value="SEKRET123", staged_path=r"C:\x\ab",
                         secret_name="openai_key", agent_id="agent-A", pid=4242)
        await _emit_output(bus, "agent-A", "here is the token SEKRET123 oops")
        assert len(alarms) == 1
        ev = alarms[0]
        assert ev.type == EventType.SECURITY_ALARM
        assert ev.source == "agent-A"           # leaking agent = source
        assert ev.data == {"names": ["openai_key"]}
        assert "agent_id" not in ev.data        # subtree-scoping trap avoided
        # The value NEVER appears anywhere in the event.
        assert "SEKRET123" not in str(ev.data)
    asyncio.run(go())


def test_path_leak_fires_alarm(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        path = r"C:\Users\x\AppData\Local\Temp\vault-staged-abcdef0123456789"
        mon.add_exposure(value="SEKRET123", staged_path=path,
                         secret_name="aws_key", agent_id="agent-B", pid=99)
        await _emit_output(bus, "agent-B", f"reading {path} now")
        assert len(alarms) == 1
        assert alarms[0].data == {"names": ["aws_key"]}
    asyncio.run(go())


def test_chunk_split_value_caught_via_tail(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        mon.add_exposure(value="SUPERSECRETVALUE", staged_path=None,
                         secret_name="stripe_key", agent_id="agent-C", pid=7)
        # Split the value across two events; neither half contains it whole.
        await _emit_output(bus, "agent-C", "prefix text SUPERSEC")
        assert alarms == []                     # not yet visible
        await _emit_output(bus, "agent-C", "RETVALUE suffix")
        assert len(alarms) == 1                 # caught across the tail boundary
        assert alarms[0].data == {"names": ["stripe_key"]}
    asyncio.run(go())


def test_drop_exposure_stops_alarms(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        mon.add_exposure(value="DROPME1234", staged_path=None,
                         secret_name="k", agent_id="agent-D", pid=555)
        mon.drop_exposure(pid=555, agent_id="agent-D")
        await _emit_output(bus, "agent-D", "leaking DROPME1234 now")
        assert alarms == []                     # dropped => no scan target
        assert not mon.is_live(555, "k")
    asyncio.run(go())


def test_empty_live_set_is_silent(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        await _emit_output(bus, "agent-E", "totally normal output, no secrets")
        assert alarms == []
    asyncio.run(go())


def test_short_value_not_scanned_but_path_is(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        # Value below _MIN_VALUE_LEN (6) must NOT be a scan target.
        mon.add_exposure(value="abc", staged_path=r"C:\p\longenoughpath",
                         secret_name="short", agent_id="agent-F", pid=1)
        await _emit_output(bus, "agent-F", "abc appears everywhere abc abc")
        assert alarms == []                     # short value ignored
        await _emit_output(bus, "agent-F", r"path C:\p\longenoughpath here")
        assert len(alarms) == 1                 # path still catches it
    asyncio.run(go())


def test_dedup_within_window(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        mon.add_exposure(value="REPEATED999", staged_path=None,
                         secret_name="k", agent_id="agent-G", pid=2)
        await _emit_output(bus, "agent-G", "REPEATED999")
        await _emit_output(bus, "agent-G", "REPEATED999 again")
        assert len(alarms) == 1                 # second hit dedup'd in-window
    asyncio.run(go())


def test_other_agent_value_does_not_implicate_src(tmp_path):
    async def go():
        bus, mon, alarms = _mk(tmp_path)
        # Secret staged for agent-A; agent-B's transcript happens to contain it.
        mon.add_exposure(value="ONLYFORA123", staged_path=None,
                         secret_name="k", agent_id="agent-A", pid=10)
        await _emit_output(bus, "agent-B", "ONLYFORA123 in B's output")
        assert alarms == []                     # only A's own leak implicates A
        await _emit_output(bus, "agent-A", "ONLYFORA123 in A's output")
        assert len(alarms) == 1
        assert alarms[0].source == "agent-A"
    asyncio.run(go())
