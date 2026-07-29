"""Idle MCP connectors are reclaimed on a running daemon.

An MCP connector is a long-lived server process, started lazily and — before
the reaper — never stopped while the daemon ran: ``stop_all`` was reachable
only from ``Runtime.stop`` and manual removal. Using a heavy connector once
held its memory until restart, and ``set_tool_enabled`` does not help (it
refuses FUTURE calls; it does not reclaim a running server).

Pinned tools need no equivalent and must not grow one: they are per-call
subprocesses that exit on return, so an unused pinned tool already costs
nothing.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from atn.config import load_config
from atn.connectors_manager import ConnectorManager, ConnectorSession


def _manager(idle_timeout_s=1.0, live=()):
    m = ConnectorManager({}, idle_timeout_s=idle_timeout_s)
    for cid in live:
        m._sessions[cid] = ConnectorSession()
        m._touch(cid)
    return m


def test_idle_connector_is_reaped():
    """PROPERTY: past the window, the process is stopped."""
    async def go():
        m = _manager(0.05, live=["stale"])
        await asyncio.sleep(0.12)
        return await m.reap_idle(), m
    stopped, m = asyncio.run(go())
    assert stopped == ["stale"]
    assert "stale" not in m._sessions


def test_recently_used_connector_survives():
    """PROPERTY: use resets the clock. A connector called every few minutes
    must not thrash — restarting costs a spawn + handshake + tool discovery."""
    async def go():
        m = _manager(0.05, live=["busy"])
        await asyncio.sleep(0.12)
        m._touch("busy")          # used just now
        return await m.reap_idle(), m
    stopped, m = asyncio.run(go())
    assert stopped == []
    assert "busy" in m._sessions


def test_zero_timeout_disables_reaping():
    """PROPERTY: <= 0 is an explicit opt-out (keep-alive-forever), not an
    accident that reaps everything immediately."""
    async def go():
        m = _manager(0, live=["forever"])
        await asyncio.sleep(0.05)
        return await m.reap_idle(), m
    stopped, m = asyncio.run(go())
    assert stopped == []
    assert "forever" in m._sessions


def test_stop_clears_the_usage_stamp():
    """PROPERTY: no stale timestamp survives a stop — a restarted connector
    must not inherit the old one and look instantly idle."""
    async def go():
        m = _manager(60, live=["c"])
        await m.stop("c")
        return m
    m = asyncio.run(go())
    assert "c" not in m._last_used
    assert m.idle_seconds("c") is None


def test_idle_seconds_is_none_when_not_running():
    m = _manager(60)
    assert m.idle_seconds("never-started") is None


def test_reaper_survives_a_failing_stop():
    """PROPERTY: one bad connector must not block the sweep — the whole point
    is bounding total footprint."""
    async def go():
        m = _manager(0.05, live=["bad", "good"])

        original = m._cleanup_session

        async def boom(cs):
            # fail for the first, succeed after
            if not getattr(boom, "fired", False):
                boom.fired = True
                raise RuntimeError("cleanup exploded")
            return await original(cs)

        m._cleanup_session = boom
        await asyncio.sleep(0.12)
        return await m.reap_idle(), m

    stopped, m = asyncio.run(go())
    # One failed, but the sweep continued and reaped the other.
    assert len(stopped) >= 1
    assert len(m._sessions) <= 1


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@pytest.mark.parametrize("body,expected", [
    ("{}", 900.0),
    ("connector_idle_timeout_s: 120\n", 120.0),
    ("connector_idle_timeout_s: 0\n", 0.0),
    ('connector_idle_timeout_s: "abc"\n', 900.0),   # malformed -> default
])
def test_config_idle_timeout(body, expected):
    """PROPERTY: settable, disable-able, and a malformed value never breaks
    boot."""
    d = Path(tempfile.mkdtemp())
    p = d / "config.yaml"
    p.write_text(body, encoding="utf-8")
    assert load_config(p).connector_idle_timeout_s == expected
