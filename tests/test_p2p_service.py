"""Paid service calls over libp2p (/autonet/service/1.0.0, issue #36):
direct calls, one hop through a relay between peers that are not connected
to each other, the hop budget, unreachable and silent providers, and the
provider side running the same payment gate as the WS path. Real
localhost libp2p hosts, no network mocking.

Run: python -m pytest tests/test_p2p_service.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import trio

from nodes.common.p2p import (
    SERVICE_PROTOCOL,
    AutonetHost,
    NodeCapability,
    _read_frame_no_close,
    _write_frame_no_close,
)


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, relay=False):
    cap = NodeCapability(peer_id="", node_id=name, roles=["relay"]) if relay else None
    return AutonetHost(node_id=name, listen_host="127.0.0.1", capability=cap)


def _echo(calls):
    async def serve(msg):
        calls.append(msg)
        return {"msg_id": msg.get("msg_id"), "ok": True,
                "result": {"echo": msg.get("args")}}
    return serve


async def _raw(host, peer, envelope):
    """Send one hand-built envelope and read the reply."""
    from libp2p.peer.id import ID
    stream = await host._host.new_stream(ID.from_base58(peer), [SERVICE_PROTOCOL])
    await _write_frame_no_close(stream, json.dumps(envelope).encode())
    reply = json.loads((await _read_frame_no_close(stream)).decode())
    await stream.close()
    return reply


@trio_test
async def test_direct_call():
    consumer, provider = _host("buyer"), _host("seller")
    calls = []
    async with consumer.run(), provider.run():
        provider.set_service_handler(_echo(calls))
        assert await consumer.connect_to_peer(provider.addrs[0])
        with trio.fail_after(10):
            reply = await consumer.request_service(
                provider.peer_id, {"msg_id": "m1", "args": {"x": 1}})
        assert reply == {"msg_id": "m1", "ok": True, "result": {"echo": {"x": 1}}}
        assert len(calls) == 1


@trio_test
async def test_call_through_a_relay():
    """A and C both dialled out to B only; A reaches C's service via B."""
    a, b, c = _host("nat-a"), _host("hub-b", relay=True), _host("nat-c")
    calls = []
    async with a.run(), b.run(), c.run():
        c.set_service_handler(_echo(calls))
        assert await a.connect_to_peer(b.addrs[0])
        assert await c.connect_to_peer(b.addrs[0])
        assert c.peer_id not in a.get_connected_peers()
        with trio.fail_after(15):
            reply = await a.request_service(c.peer_id, {"msg_id": "m2", "args": {}})
        assert reply["ok"] is True and reply["msg_id"] == "m2"
        assert len(calls) == 1  # served once


@trio_test
async def test_hop_budget_is_never_exceeded():
    """A - B - D - C: B is not connected to C, so the request dies at B
    rather than travelling a second hop through D."""
    a, b, d, c = _host("a"), _host("b", relay=True), _host("d"), _host("c")
    calls = []
    async with a.run(), b.run(), d.run(), c.run():
        c.set_service_handler(_echo(calls))
        assert await a.connect_to_peer(b.addrs[0])
        assert await d.connect_to_peer(b.addrs[0])
        assert await c.connect_to_peer(d.addrs[0])
        with trio.fail_after(15), pytest.raises(ConnectionError):
            await a.request_service(c.peer_id, {"msg_id": "m3"})
        # A hop budget above the cap is clamped; zero hops means no forward.
        with trio.fail_after(10):
            over = await _raw(a, b.peer_id, {"target": c.peer_id, "hops": 9, "msg": {}})
            none = await _raw(a, b.peer_id, {"target": d.peer_id, "hops": 0, "msg": {}})
            bare = await _raw(a, b.peer_id, {"target": d.peer_id, "msg": {}})
        assert "not connected" in over["relay_error"]
        assert "no hops left" in none["relay_error"]
        assert "no hops left" in bare["relay_error"]
        assert calls == []


@trio_test
async def test_unreachable_provider_fails_cleanly():
    a, b, lonely = _host("a"), _host("b", relay=True), _host("lonely")
    async with a.run(), b.run(), lonely.run():
        lonely.set_service_handler(_echo([]))
        assert await a.connect_to_peer(b.addrs[0])
        with trio.fail_after(10), pytest.raises(ConnectionError, match="no path"):
            await a.request_service(lonely.peer_id, {"msg_id": "m4"})


@trio_test
async def test_silent_provider_times_out():
    a, b, c = _host("a"), _host("b", relay=True), _host("c")

    async def hang(_msg):
        await trio.sleep_forever()
    async with a.run(), b.run(), c.run():
        c.set_service_handler(hang)
        assert await a.connect_to_peer(b.addrs[0])
        assert await c.connect_to_peer(b.addrs[0])
        # Through the relay: the relay gives up at the buyer's deadline and
        # says so, the buyer is not left hanging.
        with trio.fail_after(10):
            reply = await a.request_service(c.peer_id, {"msg_id": "m5"}, timeout=1.0)
        assert reply["ok"] is False and "did not reply" in reply["error"]
        # Direct: a delivered request that is never answered is a timeout.
        assert await a.connect_to_peer(c.addrs[0])
        with trio.fail_after(10), pytest.raises(TimeoutError):
            await a.request_service(c.peer_id, {"msg_id": "m6"}, timeout=1.0)


# ---------------------------------------------------------------------------
# End to end through service_client and the daemon's shared handler
# ---------------------------------------------------------------------------

class _Hosts:
    """Consumer, relay and provider hosts in a background trio thread, the
    way the daemon runs its host next to its asyncio loop."""

    def __init__(self):
        self.consumer, self.relay, self.provider = (
            _host("buyer"), _host("relay", relay=True), _host("seller"))
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=lambda: trio.run(self._main), daemon=True)

    async def _main(self):
        async with self.consumer.run(), self.relay.run(), self.provider.run():
            await self.consumer.connect_to_peer(self.relay.addrs[0])
            await self.provider.connect_to_peer(self.relay.addrs[0])
            self.ready.set()
            while not self.stop.is_set():
                await trio.sleep(0.05)

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(20)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(10)


@pytest.mark.asyncio
async def test_relayed_request_runs_the_providers_payment_gate(tmp_path, monkeypatch):
    """The libp2p path lands on WebSocketBridge._handle_service_request
    (local=False): a request signed by a key that did not pay is refused,
    the payer's own request passes the gate."""
    import atn.on_chain as oc
    from atn import service_client
    from atn.ws_server import WebSocketBridge
    from tests.test_service_payment_gate import (
        PAYER_KEY, _make_runtime, _payer, _register_service)

    rt = _make_runtime(tmp_path, chain=True)
    record = _register_service(rt)
    provider_addr = record.spec["author_pubkey"]

    async def _verify(self, *, request_id, recipient, min_amount,
                      tx_hash=None, payer=None):
        if payer != _payer():
            return {"verified": False, "reason": "payer mismatch"}
        return {"verified": True, "reason": "ok"}
    monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)

    server = WebSocketBridge(rt)
    with _Hosts() as hosts:
        hosts.provider.set_service_handler(service_client.host_service_handler(
            server._serve_p2p_service_request, asyncio.get_running_loop()))

        async def buy(rid, key):
            return await service_client.request_service(
                rt._config.rpb, provider_addr, record.digest, {},
                tx_hash="0x" + "ab" * 32, request_id=rid, payer_key=key,
                p2p_host=hosts.consumer, peer_id=hosts.provider.peer_id,
                timeout=20)

        thief = await buy("0x" + "0c" * 32, "0x" + "22" * 32)
        assert thief["transport"] == "libp2p"
        assert thief["ok"] is False and "Payment validation failed" in thief["error"]

        paid = await buy("0x" + "0d" * 32, PAYER_KEY)
        assert paid["transport"] == "libp2p"
        # Past the gate: the service's backing tool is a placeholder.
        assert "Payment validation" not in (paid["error"] or "")
        assert "Backing tool" in (paid["error"] or "")


@pytest.mark.asyncio
async def test_undelivered_libp2p_request_falls_back_to_wss(tmp_path):
    """A provider with no libp2p path is tried at its wss endpoint; both
    failures are reported."""
    from atn import service_client

    with _Hosts() as hosts:
        stranger = "12D3KooWJWoaqZhDaoEFshF7Rh1bpY9ohihFhzcW6d69Lr2NASuq"
        out = await service_client.request_service(
            None, "0x" + "33" * 20, "ab" * 32, {},
            p2p_host=hosts.consumer, peer_id=stranger,
            endpoint="ws://127.0.0.1:9", timeout=5)
    assert out["transport"] == "wss"
    assert "libp2p" in out["error"] and "no path" in out["error"]
