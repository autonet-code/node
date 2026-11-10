"""Relayed connections (circuit relay v2) and paid service streams between
real localhost libp2p hosts (nodes/common/circuit_relay.py,
nodes/common/service_stream.py; docs/services_market.md, Decision
2026-11-10 "a stream between buyer and seller").

Run: python -m pytest tests/test_service_stream.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import trio

from nodes.common.circuit_relay import RelayCaps, dial_via_relay, is_relayed
from nodes.common.p2p import AutonetHost, NodeCapability
from nodes.common.service_stream import (
    Accept,
    StreamClosed,
    StreamHandler,
    StreamRefused,
    VoucherCheck,
)

ECHO = "/test/echo/1.0.0"


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, relay=False, caps=None, announce=None, transport="tcp"):
    cap = NodeCapability(peer_id="", node_id=name, roles=["relay"]) if relay else None
    h = AutonetHost(node_id=name, listen_host="127.0.0.1", capability=cap,
                    announce_addrs=announce, transport=transport)
    if relay:
        h.circuit_relay_caps = caps or RelayCaps()
    return h


async def _meet(relay, *peers):
    """Each peer dials the relay and both sides learn each other's
    capability (a relay and a daemon advertise on an interval)."""
    for p in peers:
        assert await p.connect_to_peer(relay.addrs[0])
    await relay.advertise_capability()
    for p in peers:
        await p.advertise_capability()
    with trio.fail_after(10):
        for p in peers:
            while relay.peer_id not in p.known_capabilities or \
                    p.peer_id not in relay.known_capabilities:
                await trio.sleep(0.05)


def _echo_handler(host):
    async def echo(stream):
        try:
            while True:
                data = await stream.read(65535)
                if not data:
                    break
                for off in range(0, len(data), 60000):
                    await stream.write(data[off:off + 60000])
        except Exception:
            pass
        try:
            await stream.close()
        except Exception:
            pass
    host._host.set_stream_handler(ECHO, echo)


async def _echo_through(host, peer_id, nbytes):
    """Send ``nbytes`` on an echo stream and count what comes back."""
    from libp2p.peer.id import ID
    stream = await host._host.new_stream(ID.from_base58(peer_id), [ECHO])
    payload = bytes(range(256)) * (nbytes // 256 + 1)
    payload = payload[:nbytes]
    back = bytearray()

    async def reader():
        while len(back) < nbytes:
            chunk = await stream.read(65535)
            if not chunk:
                break
            back.extend(chunk)
    async with trio.open_nursery() as n:
        n.start_soon(reader)
        for off in range(0, nbytes, 60000):
            await stream.write(payload[off:off + 60000])
    await stream.close()
    return bytes(back) == payload


# ---------------------------------------------------------------------------
# Circuit relay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_a_relayed_connection_carries_bulk_both_ways(transport):
    """A and C dialled out to R only. A gets a connection to C through R
    that carries megabytes (py-libp2p 0.6's own relay breaks past 64 KiB),
    on both transports the daemon runs."""
    async def main():
        a, r, c = (_host("nat-a", transport=transport),
                   _host("relay", relay=True, transport=transport),
                   _host("nat-c", transport=transport))
        async with a.run(), r.run(), c.run():
            _echo_handler(c)
            await _meet(r, a, c)
            assert c.peer_id not in a.get_connected_peers()
            with trio.fail_after(20):
                conn = await dial_via_relay(a._host, r.peer_id, c.peer_id)
            assert is_relayed(conn)
            assert a.connection_path(c.peer_id) == "relayed"
            assert c.connection_path(a.peer_id) == "relayed"
            assert a.connection_path(r.peer_id) == "direct"
            with trio.fail_after(60):
                assert await _echo_through(a, c.peer_id, 3 * 1024 * 1024)
            assert r.circuit_relay.bytes_relayed >= 6 * 1024 * 1024
    trio.run(main)


@trio_test
async def test_a_relay_only_reaches_peers_connected_to_it():
    a, r, c = _host("a"), _host("relay", relay=True), _host("lonely")
    async with a.run(), r.run(), c.run():
        await _meet(r, a)
        with trio.fail_after(20), pytest.raises(ConnectionError, match="not connected"):
            await dial_via_relay(a._host, r.peer_id, c.peer_id)
        assert r.circuit_relay.active == 0


@trio_test
async def test_a_host_without_relay_caps_relays_nothing():
    a, d, c = _host("a"), _host("daemon-d"), _host("c")
    async with a.run(), d.run(), c.run():
        assert await a.connect_to_peer(d.addrs[0])
        assert await c.connect_to_peer(d.addrs[0])
        with trio.fail_after(20), pytest.raises(ConnectionError, match="no circuit"):
            await dial_via_relay(a._host, d.peer_id, c.peer_id)


@trio_test
async def test_the_operators_caps_end_circuits():
    """Unlimited by default; a capped relay ends a circuit at its byte cap
    or its duration cap, and refuses circuits past its concurrent cap."""
    caps = RelayCaps(max_bytes=200_000, max_circuits=1)
    a, r, c = _host("a"), _host("relay", relay=True, caps=caps), _host("c")
    async with a.run(), r.run(), c.run():
        _echo_handler(c)
        await _meet(r, a, c)
        with trio.fail_after(20):
            await dial_via_relay(a._host, r.peer_id, c.peer_id)
        ok = False
        with trio.move_on_after(20):
            try:
                ok = await _echo_through(a, c.peer_id, 1024 * 1024)
            except Exception:
                ok = False
        assert not ok
        with trio.fail_after(10):
            while r.circuit_relay.active:
                await trio.sleep(0.05)
        # One circuit at once: hold one open, the next is refused.
        b = _host("b")
        async with b.run():
            await _meet(r, b)
            with trio.fail_after(20):
                await dial_via_relay(a._host, r.peer_id, c.peer_id)
            with trio.fail_after(20), pytest.raises(ConnectionError, match="at most 1"):
                await dial_via_relay(b._host, r.peer_id, c.peer_id)

    timed = _host("relay2", relay=True, caps=RelayCaps(max_duration=1.0))
    a, c = _host("a2"), _host("c2")
    async with a.run(), timed.run(), c.run():
        await _meet(timed, a, c)
        with trio.fail_after(20):
            await dial_via_relay(a._host, timed.peer_id, c.peer_id)
        assert timed.circuit_relay.active == 1
        with trio.fail_after(10):
            while timed.circuit_relay.active:
                await trio.sleep(0.05)


# ---------------------------------------------------------------------------
# Service streams
# ---------------------------------------------------------------------------

class _Gate(StreamHandler):
    """A provider whose opening needs a channel voucher covering ``units``
    and whose later vouchers each cover the units they name; ``app`` is the
    application run on an accepted stream."""

    def __init__(self, app=None, units=2):
        self.app, self.units = app, units
        self.accepted, self.kept, self.released, self.checked = [], [], [], []

    async def accept(self, msg):
        if msg.get("channel_id") is None:
            return Accept(reply={"ok": False, "charged": False,
                                 "error": "Payment validation failed: no payment proof"})
        self.accepted.append(msg["request_id"])
        return Accept(reply={"ok": True, "charged": None,
                             "result": {"request_id": msg["request_id"],
                                        "units": self.units}},
                      units=self.units, token=msg["request_id"])

    async def check_voucher(self, open_msg, frame):
        self.checked.append(frame)
        if frame.get("args", {}).get("stream") != open_msg["request_id"]:
            return VoucherCheck(ok=False, error="voucher names another stream")
        return VoucherCheck(ok=True, units=int(frame["units"]),
                            token=frame["request_id"])

    async def keep(self, open_msg, token):
        self.kept.append(token)

    async def release(self, open_msg, token):
        self.released.append(token)

    async def serve(self, stream):
        if self.app is not None:
            await self.app(stream)


def _opening(rid="open-1", paid=True):
    msg = {"type": "service_request", "msg_id": "m", "spec_digest": "ab" * 32,
           "request_id": rid, "args": {}}
    if paid:
        msg.update({"channel_id": 7, "cumulative_amount": 20, "signature": "0xsig",
                    "payer_signature": "0xpayer"})
    return msg


class _Buyer:
    """The buyer's voucher interface: hands out ``limit`` vouchers and
    records each one's outcome."""

    def __init__(self, open_rid="open-1", limit=100):
        self.open_rid, self.limit, self.issued, self.outcomes = open_rid, limit, [], {}

    async def voucher_source(self, units):
        if len(self.issued) >= self.limit:
            return None
        rid = f"v{len(self.issued) + 1}"
        self.issued.append(rid)
        return {"request_id": rid, "args": {"stream": self.open_rid, "units": units},
                "channel_id": 7, "cumulative_amount": 20 + 10 * len(self.issued),
                "signature": "0xsig", "payer_signature": "0xpayer"}

    def on_receipt(self, rid, charged):
        self.outcomes[rid] = charged


async def _metered_echo(stream):
    """Echo each message back, one unit per message served."""
    while True:
        data = await stream.receive()
        if not data:
            return
        await stream.consume(1)
        await stream.send(data)


async def _talk(stream, n):
    got = []
    for i in range(n):
        await stream.send(f"msg {i}".encode())
        reply = await stream.receive()
        if not reply:
            break
        got.append(reply.decode())
    return got


@trio_test
async def test_a_direct_stream_to_a_connected_provider():
    buyer, seller = _host("buyer"), _host("seller")
    gate = _Gate(app=_metered_echo, units=100)
    async with buyer.run(), seller.run():
        seller.set_service_stream_handler(gate)
        assert await buyer.connect_to_peer(seller.addrs[0])
        b = _Buyer()
        with trio.fail_after(20):
            async with buyer.service_stream(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source,
                    on_receipt=b.on_receipt) as s:
                assert s.path == "direct"
                assert s.accept_reply["result"]["units"] == 100
                assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]
        assert gate.kept == ["open-1"]          # usage entered the opening's units
        assert b.outcomes == {"open-1": True}   # and the buyer heard it


@trio_test
async def test_a_reachable_provider_is_dialled_directly_not_relayed():
    """The buyer knows the provider only through a relay, but the provider
    advertises a reachable address: the stream goes direct."""
    r = _host("relay", relay=True)
    async with r.run():
        seller = _host("seller", announce=["/ip4/127.0.0.1/tcp/0"])
        # Announce the real port once it is known (a reachable daemon
        # announces its public address).
        buyer = _host("buyer")
        async with seller.run(), buyer.run():
            port = seller._listen_port_actual()
            seller._announce_addrs = [f"/ip4/127.0.0.1/tcp/{port}"]
            seller._announce_resolved = seller._resolve_announce_addrs()
            seller._capability.listen_addrs = seller.addrs
            seller._capability.reachable = True
            seller.set_service_stream_handler(_Gate(app=_metered_echo, units=10))
            await _meet(r, buyer, seller)
            assert seller.peer_id not in buyer.get_connected_peers()
            with trio.fail_after(20):
                assert (await buyer.relay_peer_infos(seller.peer_id))[r.peer_id]["reachable"]
            with trio.fail_after(30):
                async with buyer.service_stream(seller.peer_id, _opening()) as s:
                    assert s.path == "direct" and s.relay == ""
                    assert await _talk(s, 2) == ["msg 0", "msg 1"]
            assert buyer.connection_path(seller.peer_id) == "direct"
            assert r.circuit_relay.total == 0


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_a_stream_through_a_relay(transport):
    """Neither side can dial the other: the stream runs over a circuit
    through the relay, and the circuit closes with the stream."""
    async def main():
        buyer, r, seller = (_host("nat-buyer", transport=transport),
                            _host("relay", relay=True, transport=transport),
                            _host("nat-seller", transport=transport))
        gate = _Gate(app=_metered_echo, units=50)
        async with buyer.run(), r.run(), seller.run():
            seller.set_service_stream_handler(gate)
            await _meet(r, buyer, seller)
            with trio.fail_after(30):
                async with buyer.service_stream(seller.peer_id, _opening()) as s:
                    assert s.path == "relayed" and s.relay == r.peer_id
                    assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]
                    big = bytes(300_000)
                    await s.send(big)
                    back = bytearray()
                    while len(back) < len(big):
                        chunk = await s.receive()
                        assert chunk
                        back.extend(chunk)
                    assert bytes(back) == big
            assert gate.kept == ["open-1"]
            with trio.fail_after(10):
                while buyer.connection_path(seller.peer_id) is not None:
                    await trio.sleep(0.05)
            assert r.circuit_relay.total == 1
    trio.run(main)


@trio_test
async def test_an_unpaid_stream_is_refused_before_anything_is_served():
    buyer, seller = _host("buyer"), _host("seller")
    served = []

    async def app(stream):
        served.append(1)
    gate = _Gate(app=app)
    async with buyer.run(), seller.run():
        seller.set_service_stream_handler(gate)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(20), pytest.raises(StreamRefused, match="no payment proof"):
            async with buyer.service_stream(seller.peer_id, _opening(paid=False)):
                pass
        assert served == [] and gate.accepted == []
        # A reply the buyer cannot verify is refused by the buyer, paid or
        # not: it hangs up, nothing is used, so the opening voucher is
        # released, not kept.
        with trio.fail_after(20), pytest.raises(StreamRefused, match="not signed"):
            async with buyer.service_stream(seller.peer_id, _opening(),
                                            verify=lambda reply, rid: False):
                pass
        with trio.fail_after(10):
            while not gate.released:
                await trio.sleep(0.05)
    assert gate.kept == [] and gate.released == ["open-1"]


@trio_test
async def test_vouchers_travel_on_the_stream_as_usage_accrues():
    """The opening covers 2 units; the provider asks for more as it serves
    and keeps each voucher once usage enters it."""
    buyer, seller = _host("buyer"), _host("seller")
    gate = _Gate(app=_metered_echo, units=2)
    b = _Buyer()
    async with buyer.run(), seller.run():
        seller.set_service_stream_handler(gate)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_stream(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source,
                    on_receipt=b.on_receipt) as s:
                assert await _talk(s, 7) == [f"msg {i}" for i in range(7)]
    # 7 units served: the opening (2) and three vouchers of 2 (8 covered).
    assert gate.kept[:4] == ["open-1", "v1", "v2", "v3"]
    assert all(f["args"]["stream"] == "open-1" for f in gate.checked)
    assert {rid: b.outcomes[rid] for rid in gate.kept} == {rid: True for rid in gate.kept}
    # A voucher fetched ahead but never used is released, and said so.
    for rid in gate.released:
        assert b.outcomes[rid] is False


@trio_test
async def test_the_provider_stops_serving_when_vouchers_stop():
    buyer, seller = _host("buyer"), _host("seller")
    stopped = []

    async def app(stream):
        try:
            await _metered_echo(stream)
        except StreamClosed as e:
            stopped.append(str(e))
            raise
    gate = _Gate(app=app, units=2)
    b = _Buyer(limit=0)  # pays the opening only
    async with buyer.run(), seller.run():
        seller.set_service_stream_handler(gate)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_stream(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source,
                    on_receipt=b.on_receipt) as s:
                got = await _talk(s, 5)
                assert await s.receive() == b""
                assert s.closed
    assert got == ["msg 0", "msg 1"]   # what the opening covered, no more
    assert gate.kept == ["open-1"] and b.outcomes["open-1"] is True
    assert stopped


@trio_test
async def test_a_free_stream_is_unmetered():
    buyer, seller = _host("buyer"), _host("seller")

    class _Free(_Gate):
        async def accept(self, msg):
            return Accept(reply={"ok": True, "charged": False,
                                 "result": {"units": None}}, units=None)
    gate = _Free(app=_metered_echo)
    async with buyer.run(), seller.run():
        seller.set_service_stream_handler(gate)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(20):
            async with buyer.service_stream(seller.peer_id,
                                            _opening(paid=False)) as s:
                assert await _talk(s, 5) == [f"msg {i}" for i in range(5)]
    assert gate.checked == []


@trio_test
async def test_no_path_raises_before_anything_is_sent():
    buyer, r, lonely = _host("buyer"), _host("relay", relay=True), _host("lonely")
    async with buyer.run(), r.run(), lonely.run():
        lonely.set_service_stream_handler(_Gate())
        await _meet(r, buyer)
        with trio.fail_after(30), pytest.raises(ConnectionError, match="no path"):
            async with buyer.service_stream(lonely.peer_id, _opening()):
                pass


@trio_test
async def test_reachability_is_learned_through_a_relay():
    """A buyer not connected to a provider learns from a relay whether the
    provider says it is reachable (what catalog rows show)."""
    r = _host("relay", relay=True)
    buyer, nat = _host("buyer"), _host("nat")
    reachable = _host("public", announce=["/ip4/127.0.0.1/tcp/4999"])
    async with r.run(), buyer.run(), nat.run(), reachable.run():
        assert reachable._capability.reachable is True
        assert nat._capability.reachable is False
        await _meet(r, buyer, nat, reachable)
        assert buyer.peer_reachable(nat.peer_id) is None   # asks in the background
        with trio.fail_after(10):
            while buyer.peer_reachable(nat.peer_id) is None or \
                    buyer.peer_reachable(reachable.peer_id) is None:
                await trio.sleep(0.05)
        assert buyer.peer_reachable(nat.peer_id) is False
        assert buyer.peer_reachable(reachable.peer_id) is True
        assert buyer.peer_reachable(r.peer_id) is False    # its own capability


# ---------------------------------------------------------------------------
# atn relay
# ---------------------------------------------------------------------------

def test_relay_flags_and_env_set_the_caps(monkeypatch):
    from nodes import relay
    args = relay.parse_args(["--data-dir", "x"])
    assert (args.circuit_max_duration, args.circuit_max_bytes,
            args.max_circuits, args.no_circuits) == (0.0, 0, 0, False)
    monkeypatch.setenv("AUTONET_RELAY_CIRCUIT_MAX_DURATION", "120")
    monkeypatch.setenv("AUTONET_RELAY_CIRCUIT_MAX_BYTES", "131072")
    monkeypatch.setenv("AUTONET_RELAY_MAX_CIRCUITS", "8")
    args = relay.parse_args(["--data-dir", "x"])
    assert (args.circuit_max_duration, args.circuit_max_bytes,
            args.max_circuits) == (120.0, 131072, 8)
    args = relay.parse_args(["--data-dir", "x", "--max-circuits", "2", "--no-circuits"])
    assert args.max_circuits == 2 and args.no_circuits is True


@trio_test
async def test_a_stream_crosses_atn_relay(tmp_path):
    """The relay node relays circuits (unlimited by default): two NAT'd
    daemons connected only to it get a paid stream."""
    from nodes import relay
    args = relay.parse_args([
        "--port", "0", "--host", "127.0.0.1", "--no-upnp", "--transport", "tcp",
        "--data-dir", str(tmp_path / "relay"), "--advertise-interval", "0.5"])
    async with trio.open_nursery() as nursery:
        relay_addr = []

        async def _relay():
            import builtins
            real_print = builtins.print

            def _capture(*a, **kw):
                if a and str(a[0]).startswith("relay address: "):
                    relay_addr.append(str(a[0])[len("relay address: "):])
                real_print(*a, **kw)
            builtins.print = _capture
            try:
                await relay.run(args)
            finally:
                builtins.print = real_print

        nursery.start_soon(_relay)
        with trio.fail_after(10):
            while not relay_addr:
                await trio.sleep(0.05)
        buyer = AutonetHost(node_id="nat-buyer", listen_host="127.0.0.1",
                            bootstrap_peers=[relay_addr[0]])
        seller = AutonetHost(node_id="nat-seller", listen_host="127.0.0.1")
        gate = _Gate(app=_metered_echo, units=10)
        async with buyer.run(), seller.run():
            seller.set_service_stream_handler(gate)
            assert await seller.connect_to_peer(relay_addr[0])
            with trio.fail_after(10):
                while relay_addr[0].split("/p2p/")[1] not in buyer.get_connected_peers():
                    await trio.sleep(0.05)
            with trio.fail_after(30):
                async with buyer.service_stream(seller.peer_id, _opening()) as s:
                    assert s.path == "relayed"
                    assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]
        assert gate.kept == ["open-1"]
        nursery.cancel_scope.cancel()
