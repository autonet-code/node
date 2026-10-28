"""The autonet wiring waiters keep retrying past their first window.

`_wire_event_gossip_when_ready`, `_wire_chain_submission_when_ready` and
`_wire_peer_attribution_when_ready` used to give up silently after 30 s,
so a host or WorldService that came up late left the daemon unwired for
the whole session. They now log one warning and keep polling with
backoff until wired or the service stops.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from atn.autonet_service import AutonetBridge, AutonetStatus
from atn.config import AutonetConfig


class _FakeHost:
    def __init__(self) -> None:
        self._ready_event = threading.Event()
        self.peer_id = "12D3KooFake"


class _FakeService:
    def __init__(self) -> None:
        self._world_service = object()
        self._federated_close_driver = None
        self.gossip_hosts: list = []
        self.chain_attached = 0

    def attach_event_gossip(self, host) -> bool:
        self.gossip_hosts.append(host)
        return True

    def attach_chain_submission(self, **_kw) -> bool:
        self.chain_attached += 1
        return True


def _bridge() -> tuple[AutonetBridge, _FakeService, _FakeHost]:
    bridge = AutonetBridge(AutonetConfig())
    service, host = _FakeService(), _FakeHost()
    bridge._service = service
    bridge._p2p_host = host
    bridge.state.status = AutonetStatus.RUNNING
    return bridge, service, host


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "still waiting" in r.getMessage()]


@pytest.mark.asyncio
async def test_event_gossip_wires_when_host_is_ready_late(caplog):
    bridge, service, host = _bridge()
    caplog.set_level(logging.DEBUG, logger="atn.autonet_service")
    task = asyncio.create_task(bridge._wire_event_gossip_when_ready(timeout=0.1))
    await asyncio.sleep(0.6)  # well past the first window
    assert not task.done() and service.gossip_hosts == []
    host._ready_event.set()
    await asyncio.wait_for(task, timeout=5)
    assert service.gossip_hosts == [host]
    assert len(_warnings(caplog)) == 1  # warned once, then quiet


@pytest.mark.asyncio
async def test_chain_submission_wires_when_driver_is_late(caplog):
    bridge, service, _host = _bridge()
    caplog.set_level(logging.DEBUG, logger="atn.autonet_service")
    task = asyncio.create_task(
        bridge._wire_chain_submission_when_ready(timeout=0.1))
    await asyncio.sleep(0.6)
    assert not task.done() and service.chain_attached == 0
    service._federated_close_driver = object()
    await asyncio.wait_for(task, timeout=5)
    assert service.chain_attached == 1
    assert len(_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_waiter_ends_when_the_service_stops():
    bridge, service, _host = _bridge()
    task = asyncio.create_task(bridge._wire_event_gossip_when_ready(timeout=0.1))
    await asyncio.sleep(0.3)
    bridge.state.status = AutonetStatus.STOPPED
    await asyncio.wait_for(task, timeout=15)
    assert service.gossip_hosts == []


@pytest.mark.asyncio
async def test_peer_attribution_waits_for_a_late_host(monkeypatch):
    bridge, _service, host = _bridge()
    bridge.config.substrate_address = "0x" + "11" * 20
    bridge.config.rpc_url = "http://127.0.0.1:1"
    built = []

    async def _fake_executor(_self, _ex, fn):
        built.append(fn)
        return object(), 0

    monkeypatch.setattr(asyncio.BaseEventLoop, "run_in_executor",
                        _fake_executor)
    task = asyncio.create_task(
        bridge._wire_peer_attribution_when_ready(timeout=0.1))
    await asyncio.sleep(0.6)
    assert not task.done() and built == []
    host._ready_event.set()
    await asyncio.wait_for(task, timeout=5)
    assert len(built) == 1
    assert getattr(host, "_peer_attribution", None) is not None
