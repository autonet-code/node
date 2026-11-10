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


def _host(name, relay=False, seeds=None):
    cap = NodeCapability(peer_id="", node_id=name, roles=["relay"]) if relay else None
    return AutonetHost(node_id=name, listen_host="127.0.0.1", capability=cap,
                       bootstrap_peers=seeds)


async def _learn_relay(relay, *peers):
    """``relay`` advertises its role; wait until every peer has heard it
    (a relay does so on an interval in a real network)."""
    await relay.advertise_capability()
    with trio.fail_after(10):
        for p in peers:
            while relay.peer_id not in p.known_capabilities:
                await trio.sleep(0.05)


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
        await _learn_relay(b, a)
        with trio.fail_after(15):
            reply = await a.request_service(c.peer_id, {"msg_id": "m2", "args": {}})
        assert reply["ok"] is True and reply["msg_id"] == "m2"
        assert len(calls) == 1  # served once


@trio_test
async def test_only_relays_are_asked_to_forward():
    """A peer that is neither a seed nor an advertised relay never sees the
    request: its args are plaintext to whoever forwards it."""
    a, d, c = _host("nat-a"), _host("daemon-d"), _host("nat-c")
    calls, seen = [], []
    real = d._handle_service_stream

    async def spy(stream):
        seen.append(1)
        await real(stream)
    async with a.run(), d.run(), c.run():
        d._host.set_stream_handler(SERVICE_PROTOCOL, spy)
        c.set_service_handler(_echo(calls))
        assert await a.connect_to_peer(d.addrs[0])
        assert await c.connect_to_peer(d.addrs[0])
        with trio.fail_after(10), pytest.raises(ConnectionError,
                                                match="no connected relay"):
            await a.request_service(c.peer_id, {"msg_id": "m7"})
        assert seen == [] and calls == []


@trio_test
async def test_a_seed_forwards_before_it_advertises():
    """A seed (from the DAO Registry) counts as a relay from the moment it
    is connected, before its first capability advertisement."""
    b = _host("hub-b", relay=True)
    async with b.run():
        a = _host("nat-a", seeds=[b.addrs[0]])
        c = _host("nat-c")
        calls = []
        async with a.run(), c.run():
            c.set_service_handler(_echo(calls))
            with trio.fail_after(10):
                while b.peer_id not in a.get_connected_peers():
                    await trio.sleep(0.05)
            assert await c.connect_to_peer(b.addrs[0])
            assert b.peer_id not in a.known_capabilities
            with trio.fail_after(15):
                reply = await a.request_service(c.peer_id, {"msg_id": "m8"})
            assert reply["ok"] is True and len(calls) == 1


class _ForgingRelay(AutonetHost):
    """A relay that answers every request itself, ok, with made-up output."""

    async def _handle_service_stream(self, stream):
        envelope = json.loads((await _read_frame_no_close(stream)).decode())
        self.forged = getattr(self, "forged", 0) + 1
        msg = envelope.get("msg") or {}
        await _write_frame_no_close(stream, json.dumps(
            {"msg_id": msg.get("msg_id"), "ok": True,
             "result": "forged"}).encode())
        await stream.close()


@trio_test
async def test_a_reply_the_buyer_refuses_moves_on_to_the_next_relay():
    a, c = _host("nat-a"), _host("nat-c")
    honest = _host("relay-ok", relay=True)
    forger = _ForgingRelay(node_id="relay-bad", listen_host="127.0.0.1",
                           capability=NodeCapability(peer_id="", node_id="bad",
                                                     roles=["relay"]))
    calls = []

    def accept(reply):
        return reply.get("result") != "forged"
    async with a.run(), c.run(), honest.run(), forger.run():
        c.set_service_handler(_echo(calls))
        for r in (honest, forger):
            assert await a.connect_to_peer(r.addrs[0])
            assert await c.connect_to_peer(r.addrs[0])
        await _learn_relay(honest, a)
        await _learn_relay(forger, a)
        with trio.fail_after(15):
            reply = await a.request_service(c.peer_id, {"msg_id": "m9"},
                                            accept=accept)
        assert reply["ok"] is True and reply["result"] != "forged"
        assert len(calls) == 1
        # With the forger as the only path, its reply comes back for the
        # caller to refuse: the request may have reached the target.
        with trio.fail_after(15):
            only = await a.request_service(
                honest.peer_id + "x", {"msg_id": "m10"}, accept=accept)
        assert only["result"] == "forged"


@trio_test
async def test_a_peer_describes_only_itself():
    """A capability is kept under the peer that sent it, whatever peer id
    its payload names, so nobody can claim or strip another's relay role."""
    liar, victim, watcher = _host("liar"), _host("victim", relay=True), _host("w")
    async with liar.run(), victim.run(), watcher.run():
        assert await liar.connect_to_peer(watcher.addrs[0])
        fake = NodeCapability(peer_id=victim.peer_id, node_id="victim",
                              roles=[])
        await liar.advertise_capability(fake)
        with trio.fail_after(10):
            while liar.peer_id not in watcher.known_capabilities:
                await trio.sleep(0.05)
        assert victim.peer_id not in watcher.known_capabilities
        assert watcher.known_capabilities[liar.peer_id].peer_id == liar.peer_id


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
        await _learn_relay(b, a)
        with trio.fail_after(15), pytest.raises(ConnectionError,
                                                match="not connected"):
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
        await _learn_relay(b, a)
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

    def __init__(self, relay=None):
        self.consumer, self.provider = _host("buyer"), _host("seller")
        self.relay = relay or _host("relay", relay=True)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=lambda: trio.run(self._main), daemon=True)

    async def _main(self):
        async with self.consumer.run(), self.relay.run(), self.provider.run():
            await self.consumer.connect_to_peer(self.relay.addrs[0])
            await self.provider.connect_to_peer(self.relay.addrs[0])
            await _learn_relay(self.relay, self.consumer)
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
        PAYER_KEY, _make_runtime, _patch_listing, _payer, _register_service)

    _patch_listing(monkeypatch)
    rt = _make_runtime(tmp_path, chain=True)
    server = WebSocketBridge(rt)
    # The entry is registered under the key that signs the owner's
    # services, so its replies are signed by the address the buyer pays.
    provider_addr = server._service_provider_address("user")
    record = _register_service(rt, provider=provider_addr)
    served = []

    class _Tool:
        manifest: dict = {}

    async def _call(tool, args, caller_id=None):
        served.append(args)
        return {"result": "served"}
    monkeypatch.setattr(rt.tool_store, "get",
                        lambda d: _Tool() if d == record.backing_tool else None)
    monkeypatch.setattr(rt.tool_store, "call", _call)

    async def _verify(self, *, request_id, recipient, min_amount,
                      tx_hash=None, payer=None):
        if payer != _payer():
            return {"verified": False, "reason": "payer mismatch"}
        return {"verified": True, "reason": "ok"}
    monkeypatch.setattr(oc.OnChainService, "verify_service_payment", _verify)

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
        assert thief["authenticated"] is True   # the provider's own "no"

        paid = await buy("0x" + "0d" * 32, PAYER_KEY)
        assert paid["transport"] == "libp2p"
        assert paid["ok"] is True and paid["authenticated"] is True
        assert paid["result"]["result"] == "served"
        assert served == [{}]


class _Unsigned:
    """A provider-side handler that answers ok without a receipt_sig, as a
    peer impersonating the provider would."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, msg):
        self.calls += 1
        return {"msg_id": msg.get("msg_id"), "ok": True,
                "result": {"content": "made up"}}


@pytest.mark.asyncio
async def test_a_reply_the_provider_did_not_sign_is_not_served():
    """Whatever answers, a reply without the paid address's signature over
    this request is not served: ``ok`` is False, the request counts as
    delivered (it may have reached the provider), the made-up result is
    dropped. Through a forging relay, and from a peer that does not sign."""
    from atn import service_client

    forger = _ForgingRelay(node_id="relay-bad", listen_host="127.0.0.1",
                           capability=NodeCapability(peer_id="", node_id="bad",
                                                     roles=["relay"]))
    with _Hosts(relay=forger) as hosts:
        out = await service_client.request_service(
            None, "0x" + "33" * 20, "ab" * 32, {"messages": []},
            p2p_host=hosts.consumer, peer_id=hosts.provider.peer_id,
            timeout=10)
    assert out["transport"] == "libp2p" and out["delivered"] is True
    assert out["ok"] is False and out["authenticated"] is False
    assert out["result"] is None
    assert "not signed by the provider" in out["error"]

    unsigned = _Unsigned()
    with _Hosts() as hosts:
        hosts.provider.set_service_handler(unsigned)
        out = await service_client.request_service(
            None, "0x" + "33" * 20, "ab" * 32, {},
            p2p_host=hosts.consumer, peer_id=hosts.provider.peer_id,
            timeout=10)
    assert unsigned.calls == 1
    assert out["ok"] is False and out["authenticated"] is False


def test_service_requests_have_their_own_bounded_pool(monkeypatch):
    """A libp2p request holds a thread of service_client's own pool, never
    the default executor, and one past the bound is refused as undelivered
    instead of queued."""
    from atn import service_client

    class _Host:
        _trio_token = object()

    held = []
    for _ in range(service_client.P2P_REQUEST_WORKERS):
        assert service_client._p2p_slots.acquire(blocking=False)
        held.append(1)
    try:
        with pytest.raises(ConnectionError, match="in flight"):
            asyncio.run(service_client.p2p_request(_Host(), "peer", {}, 1.0))
    finally:
        for _ in held:
            service_client._p2p_slots.release()

    seen = []

    def _fake_run(fn, trio_token=None):
        import threading
        seen.append(threading.current_thread().name)
        return {"ok": True}
    monkeypatch.setattr(trio.from_thread, "run", _fake_run)

    async def go():
        return await service_client.p2p_request(_Host(), "peer", {}, 1.0)
    assert asyncio.run(go()) == {"ok": True}
    assert seen and seen[0].startswith("service-p2p")


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


@pytest.mark.asyncio
async def test_service_spec_gossip_reaches_a_peer_through_a_relay():
    """A provider's spec announcement crosses a relay (subscribed to the
    services topic, as ``atn relay`` is) to a buyer that reaches the
    provider only through it, and the buyer's catalog keeps it because the
    chain names its digest (Decision 2026-11-10)."""
    from atn.service_catalog import announcement, spec_digest, topic_for
    from tests.atn.test_service_catalog import ADDR_A, _catalog, _entry, _spec

    cat = _catalog()
    spec = _spec("transcribe_audio", "speech to text per audio minute")
    cat._entries[1] = _entry(1, spec, ADDR_A)
    topic = topic_for("1337:0x" + "da" * 20)
    got = threading.Event()

    def on_spec(_t, data):
        if cat.accept_announcement(data):
            got.set()

    with _Hosts() as hosts:
        hosts.relay.subscribe_topic_threadsafe(topic, lambda _t, _d: None)
        hosts.provider.subscribe_topic_threadsafe(topic, lambda _t, _d: None)
        hosts.consumer.subscribe_topic_threadsafe(topic, on_spec)
        # The mesh forms on the gossipsub heartbeat: announce until heard.
        for _ in range(40):
            hosts.provider.publish_topic_threadsafe(topic, announcement(spec, 1))
            if got.wait(0.5):
                break
    assert got.is_set(), "the announcement never crossed the relay"
    assert cat.spec(spec_digest(spec)) == spec
    assert [r["service_id"] for r in cat.search("speech to text")] == [1]
