"""Service sessions between real localhost libp2p hosts over the mesh
(nodes/common/service_session.py; docs/services_market.md, Decision
2026-11-12; docs/p2p_networking.md): opening, data both ways, payment
beside the data (charges, vouchers asked for and checked, the session ended
when none comes), direct connections, relay circuits and the relay's limits
(unlimited unless the operator caps them). A host built with
``advertise_listen_addrs=False`` advertises only its circuit addresses,
which is how a NAT'd daemon looks to the others; hole punching itself needs
real NATs (scripts/mesh_nat_harness.py).

Run: python -m pytest tests/test_service_session.py -v
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import trio

from nodes.common import p2p_mesh
from nodes.common.p2p import (
    RELAY_CIRCUIT_BYTES,
    RELAY_CIRCUIT_DURATION,
    RELAY_UNLIMITED_BYTES,
    AutonetHost,
    NodeCapability,
)
from nodes.common.service_session import (
    KIND_CONTROL,
    Accept,
    Ledger,
    SessionHandler,
    SessionRefused,
    SessionUnpaid,
    VoucherCheck,
    serve_incoming,
)

ECHO = "/test/echo/1.0.0"


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, *, relay=False, nat=False, limits=None, announce=None,
          transport="tcp+ws", mesh_interval=0.5):
    """A relay (``relay_hop``, the relay role), a NAT'd daemon (circuit
    addresses only) or a plain daemon, on the mesh."""
    cap = NodeCapability(peer_id="", node_id=name, roles=["relay"]) if relay else None
    h = AutonetHost(node_id=name, listen_host="127.0.0.1", capability=cap,
                    announce_addrs=announce, transport=transport,
                    relay_hop=relay, advertise_listen_addrs=not nat,
                    mesh_interval=mesh_interval)
    if limits:
        h.relay_limits = dict(limits)
    return h


async def _until(cond, timeout=20.0, step=0.1):
    with trio.move_on_after(timeout):
        while not cond():
            await trio.sleep(step)
    return cond()


def _direct_addr(host, ws=False):
    return next(a for a in host.addrs
                if p2p_mesh.is_ws(a) == ws and not p2p_mesh.is_circuit(a))


async def _natd_pair(relay, a, c, ws=False):
    """a and c dial only the relay; c reserves on it and advertises its
    circuit address, which a learns (as capability gossip or the DHT would
    tell it)."""
    assert await a.connect_to_peer(_direct_addr(relay, ws))
    assert await c.connect_to_peer(_direct_addr(relay, ws))
    assert await _until(lambda: relay.peer_id in c.reservations, 20)
    assert any(p2p_mesh.is_circuit(x) for x in c.addrs), c.addrs
    a._known_capabilities[c.peer_id] = c._capability


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
    payload = (bytes(range(256)) * (nbytes // 256 + 1))[:nbytes]
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
# Relayed connections: circuit relay v2 on the mesh
# ---------------------------------------------------------------------------

def test_relayed_connections_are_unlimited_by_default():
    """No duration cap and no data cap unless the relay's operator sets
    one. (py-libp2p 0.8.1 reads a data limit of 0 as zero bytes in its
    per-reservation fallback, so "no cap" is a count no circuit reaches.)"""
    assert RELAY_CIRCUIT_DURATION == 0
    assert RELAY_CIRCUIT_BYTES == RELAY_UNLIMITED_BYTES >= 1 << 60


@pytest.mark.parametrize("ws", [False, True])
def test_a_relayed_connection_carries_bulk_both_ways(ws):
    async def main():
        relay = _host("relay", relay=True)
        a, c = _host("nat-a", nat=True), _host("nat-c", nat=True)
        async with relay.run(), a.run(), c.run():
            _echo_handler(c)
            await _natd_pair(relay, a, c, ws=ws)
            assert await a.reach_peer(c.peer_id, timeout=20)
            assert a.connection_path(c.peer_id) == "relayed"
            assert a.connection_path(relay.peer_id) == "direct"
            with trio.fail_after(90):
                assert await _echo_through(a, c.peer_id, 3 * 1024 * 1024)
    trio.run(main)


@trio_test
async def test_the_operators_caps_end_circuits():
    """A relay whose operator caps bytes ends a circuit past the cap; one
    that caps duration ends it when the time runs out. (WebSocket only, so
    DCUtR, which is TCP only and punches loopback at once, cannot move the
    traffic off the circuit under test.)"""
    relay = _host("relay", relay=True, limits={"bytes": 200_000}, transport="ws")
    a = _host("nat-a", nat=True, mesh_interval=0, transport="ws")
    c = _host("nat-c", nat=True, mesh_interval=0, transport="ws")
    async with relay.run(), a.run(), c.run():
        assert relay._circuit_protocol.limits.data == 200_000
        _echo_handler(c)
        await _natd_pair(relay, a, c, ws=True)
        assert await a.reach_peer(c.peer_id, timeout=20)
        assert a.connection_path(c.peer_id) == "relayed"
        ok = False
        with trio.move_on_after(30):
            try:
                ok = await _echo_through(a, c.peer_id, 1024 * 1024)
            except Exception:
                ok = False
        assert not ok

    timed = _host("relay2", relay=True, limits={"duration": 2}, transport="ws")
    a2 = _host("nat-a2", nat=True, mesh_interval=0, transport="ws")
    c2 = _host("nat-c2", nat=True, mesh_interval=0, transport="ws")
    async with timed.run(), a2.run(), c2.run():
        assert timed._circuit_protocol.limits.duration == 2
        _echo_handler(c2)
        await _natd_pair(timed, a2, c2, ws=True)
        assert await a2.reach_peer(c2.peer_id, timeout=20)
        assert a2.connection_path(c2.peer_id) == "relayed"
        await trio.sleep(4)
        ok = False
        with trio.move_on_after(15):
            try:
                ok = await _echo_through(a2, c2.peer_id, 64 * 1024)
            except Exception:
                ok = False
        assert not ok


@trio_test
async def test_a_relay_without_circuits_holds_no_reservation():
    """atn relay --no-circuits: gossip, blobs and service requests only."""
    relay = AutonetHost(node_id="relay", listen_host="127.0.0.1",
                        capability=NodeCapability(peer_id="", node_id="relay",
                                                  roles=["relay"]),
                        relay_hop=False, mesh_interval=0.5)
    c = _host("nat-c", nat=True)
    async with relay.run(), c.run():
        assert await c.connect_to_peer(_direct_addr(relay))
        await trio.sleep(3)
        assert relay.peer_id not in c.reservations


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class _Credits(Ledger):
    """The buyer's credits as the provider counts them."""

    def __init__(self, cumulative=0):
        self.consumed, self.cumulative = 0, cumulative

    def charge(self, amount):
        self.consumed += amount

    def paid_up(self):
        return self.consumed <= self.cumulative

    def balance(self):
        return self.consumed, self.cumulative


class _Provider(SessionHandler):
    """Accepts openings that name a channel (refuses the rest), checks a
    voucher by its signature, and runs ``app`` on an accepted session."""

    def __init__(self, app=None, ledger=None, voucher_wait=None, check_delay=0.0):
        self.app, self.ledger, self.voucher_wait = app, ledger, voucher_wait
        self.check_delay = check_delay
        self.accepted, self.checked, self.ends = [], [], []
        self.unpaid = []

    async def accept(self, msg):
        if msg.get("channel_id") is None:
            return Accept(reply={"ok": False, "error": "voucher needed: no credits"})
        self.accepted.append(msg["request_id"])
        consumed, cumulative = self.ledger.balance() if self.ledger else (0, 0)
        return Accept(reply={"ok": True, "result": {
            "request_id": msg["request_id"], "voucher": 20,
            "consumed": consumed, "cumulative": cumulative}}, ledger=self.ledger)

    async def check_voucher(self, open_msg, frame):
        await trio.sleep(self.check_delay)
        self.checked.append(dict(frame))
        if frame.get("signature") != "0xok":
            return VoucherCheck(ok=False, error="not the channel's client")
        self.ledger.cumulative = max(self.ledger.cumulative,
                                     int(frame["cumulative_amount"]))
        return VoucherCheck(ok=True)

    async def serve(self, session):
        if self.voucher_wait is not None:
            session.voucher_wait = self.voucher_wait
        try:
            if self.app is not None:
                await self.app(session)
        except SessionUnpaid as e:
            self.unpaid.append(str(e))
            raise

    async def ended(self, session):
        self.ends.append((session.request_id, session.close_reason))


class _Buyer:
    """The buyer's side of payment: each voucher raises the running total
    by ``size``; ``limit`` vouchers at most (then it stops paying)."""

    def __init__(self, size=20, limit=100, delay=0.0, signature="0xok"):
        self.size, self.limit, self.delay, self.signature = size, limit, delay, signature
        self.asked, self.balances = [], []

    async def voucher_source(self, consumed, cumulative):
        self.asked.append((consumed, cumulative, time.monotonic()))
        await trio.sleep(self.delay)
        if len(self.asked) > self.limit:
            return None
        return {"channel_id": 7, "cumulative_amount": max(consumed, cumulative) + self.size,
                "signature": self.signature}

    def on_balance(self, consumed, cumulative):
        self.balances.append((consumed, cumulative))


def _opening(rid="open-1", paid=True):
    msg = {"spec_digest": "ab" * 32, "request_id": rid, "args": {}}
    if paid:
        msg["channel_id"] = 7
    return msg


async def _echo(session):
    """Free: echo every message back."""
    while True:
        data = await session.receive()
        if not data:
            return
        await session.send(data)


def _metered(price=10, ahead=10):
    """Charge ``price`` per message, serve only when paid up, ask ahead
    when less than ``ahead`` is left (the daemon's pipe does the same)."""
    async def app(session):
        while True:
            data = await session.receive()
            if not data:
                return
            session.charge(price)
            await session.paid_up()
            await session.low(ahead)
            await session.send(data)
    return app


async def _talk(session, n):
    got = []
    for i in range(n):
        await session.send(f"msg {i}".encode())
        reply = await session.receive()
        if not reply:
            break
        got.append(reply.decode())
    return got


@trio_test
async def test_an_accepted_session_carries_data_both_ways():
    buyer, seller = _host("buyer"), _host("seller")
    provider = _Provider(app=_echo)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_session(seller.peer_id, _opening()) as s:
                assert s.path == "direct" and s.role == "buyer"
                assert s.accept_reply["ok"] and s.accept_reply["result"]["voucher"] == 20
                assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]
                big = bytes(range(256)) * 4096        # 1 MiB: more than one frame
                await s.send(big)
                back = bytearray()
                while len(back) < len(big):
                    chunk = await s.receive()
                    assert chunk
                    back.extend(chunk)
                assert bytes(back) == big
            assert s.closed
        with trio.fail_after(10):
            while not provider.ends:
                await trio.sleep(0.05)
    assert provider.accepted == ["open-1"]
    assert provider.ends == [("open-1", "closed by the buyer")]


@trio_test
async def test_a_refused_opening_serves_nothing():
    buyer, seller = _host("buyer"), _host("seller")
    served = []

    async def app(session):
        served.append(1)
    provider = _Provider(app=app)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(20), pytest.raises(SessionRefused, match="voucher needed") as e:
            async with buyer.service_session(seller.peer_id, _opening(paid=False)):
                pass
        assert e.value.reply["ok"] is False
        # A node with no handler refuses every session.
        seller.set_service_session_handler(None)
        with trio.fail_after(20), pytest.raises(SessionRefused, match="no service sessions"):
            async with buyer.service_session(seller.peer_id, _opening()):
                pass
    assert served == [] and provider.accepted == [] and provider.ends == []


@trio_test
async def test_the_provider_charges_and_asks_for_vouchers_as_credits_run_low():
    """20 paid ahead at the opening, 10 per message, a voucher asked for
    when less than 10 is left: each voucher raises the total by 20, the
    provider checks it and reports the new balance."""
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=20)
    provider = _Provider(app=_metered(price=10, ahead=10), ledger=credits)
    b = _Buyer(size=20)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source,
                    on_balance=b.on_balance) as s:
                assert await _talk(s, 7) == [f"msg {i}" for i in range(7)]
    assert (credits.consumed, credits.cumulative) == (70, 80)
    totals = [int(f["cumulative_amount"]) for f in provider.checked]
    assert totals == sorted(totals) and totals[0] == 40
    assert all(f["channel_id"] == 7 for f in provider.checked)
    # One ask at a time: never two vouchers for the same shortfall.
    assert len(b.asked) == len(provider.checked) == len(set(totals))
    # The opening's balance first, then the provider's count with each ask
    # and after each voucher.
    assert b.balances[0] == (0, 20) and b.balances[-1] == (60, 80)


@trio_test
async def test_a_voucher_asked_ahead_does_not_hold_up_replies():
    """The buyer makes its voucher (here slowly) and the provider checks it
    (also slowly) beside the session's bytes: what is paid for keeps
    flowing meanwhile."""
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=40)
    provider = _Provider(app=_metered(price=10, ahead=20), ledger=credits,
                         check_delay=1.0)
    b = _Buyer(size=40, delay=1.0)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source,
                    on_balance=b.on_balance) as s:
                times = []
                for i in range(4):
                    await s.send(f"msg {i}".encode())
                    assert await s.receive() == f"msg {i}".encode()
                    times.append(time.monotonic())
                # msg 2 left 10 of 40: the ask went out with it; msgs 2 and
                # 3 are still paid for and come back without waiting.
                assert len(b.asked) == 1
                assert times[3] - b.asked[0][2] < 0.8
                assert await _talk(s, 1) == ["msg 0"]   # the 5th waits for the voucher
    assert credits.consumed == 50 and credits.cumulative == 80
    assert len(provider.checked) == 1


@trio_test
async def test_one_voucher_is_asked_for_at_a_time():
    """Serving goes on while a voucher is being checked; the provider does
    not ask again meanwhile, so the buyer never pays two vouchers ahead."""
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=40)
    provider = _Provider(app=_metered(price=10, ahead=20), ledger=credits,
                         check_delay=1.5)
    b = _Buyer(size=40)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source) as s:
                assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]   # asks at msg 2
                await trio.sleep(0.5)            # the voucher is being checked
                assert len(provider.checked) == 0
                assert await _talk(s, 1) == ["msg 0"]                      # still paid for
                await trio.sleep(1.5)
    assert len(b.asked) == 1 and len(provider.checked) == 1
    assert credits.cumulative == 80


@trio_test
async def test_no_voucher_in_time_ends_the_session():
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=20)
    provider = _Provider(app=_metered(price=10, ahead=10), ledger=credits,
                         voucher_wait=1.0)
    b = _Buyer(size=20, delay=30.0)          # never in time
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(30):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source) as s:
                got = await _talk(s, 5)
                assert await s.receive() == b""
                assert s.closed and "no voucher came" in s.close_reason
        with trio.fail_after(10):
            while not provider.ends:
                await trio.sleep(0.05)
    assert got == ["msg 0", "msg 1"]          # what the 20 paid ahead covered
    assert provider.unpaid and "no voucher came" in provider.unpaid[0]
    assert "no voucher came" in provider.ends[0][1]
    assert provider.checked == []


@trio_test
async def test_a_buyer_that_stops_paying_says_so():
    """``no_voucher``: the provider serves what is paid, then ends the
    session at once instead of waiting out ``voucher_wait``."""
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=20)
    provider = _Provider(app=_metered(price=10, ahead=10), ledger=credits)
    b = _Buyer(size=20, limit=0)
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        t0 = time.monotonic()
        with trio.fail_after(20):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source) as s:
                got = await _talk(s, 5)
                assert s.closed and "stopped paying" in s.close_reason
        assert time.monotonic() - t0 < 10     # not the 30 s voucher_wait
    assert got == ["msg 0", "msg 1"]
    assert provider.unpaid and "stopped paying" in provider.unpaid[0]


@trio_test
async def test_a_refused_voucher_ends_the_session():
    buyer, seller = _host("buyer"), _host("seller")
    credits = _Credits(cumulative=20)
    provider = _Provider(app=_metered(price=10, ahead=10), ledger=credits)
    b = _Buyer(size=20, signature="0xstolen")
    async with buyer.run(), seller.run():
        seller.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(seller.addrs[0])
        with trio.fail_after(20):
            async with buyer.service_session(
                    seller.peer_id, _opening(), voucher_source=b.voucher_source) as s:
                got = await _talk(s, 5)
                assert s.closed and "voucher refused" in s.close_reason
    assert got[:2] == ["msg 0", "msg 1"] and len(got) <= 2
    assert credits.cumulative == 20


@trio_test
async def test_an_accepted_session_whose_answer_cannot_be_sent_is_ended():
    """The handler took what an accepted session holds (a tool lease):
    ``ended`` gives it back even when the buyer never heard the answer,
    paid session or free."""
    class _Gone:
        def __init__(self, data):
            self.data = data

        async def read(self, n):
            chunk, self.data = self.data[:n], self.data[n:]
            return chunk

        async def write(self, data):
            raise ConnectionResetError("the buyer is gone")

        async def close(self):
            pass
    opening = json.dumps({"type": "service_session", "request_id": "r1",
                          "spec_digest": "ab" * 32, "channel_id": 7}).encode()
    provider = _Provider(app=_echo)                 # free: no ledger
    await serve_incoming(_Gone(bytes([KIND_CONTROL]) + struct.pack(">I", len(opening))
                               + opening), provider)
    assert provider.accepted == ["r1"]
    assert provider.ends == [("r1", "the acceptance could not be sent")]


@trio_test
async def test_a_reachable_provider_is_dialled_directly_not_relayed():
    """The buyer knows the provider only from its capability (its listen
    addresses, as gossip carries them): the session goes direct."""
    relay = _host("relay", relay=True)
    buyer, seller = _host("buyer", nat=True), _host("seller")
    async with relay.run(), buyer.run(), seller.run():
        seller.set_service_session_handler(_Provider(app=_echo))
        assert await buyer.connect_to_peer(_direct_addr(relay))
        assert await seller.connect_to_peer(_direct_addr(relay))
        assert seller.peer_id not in buyer.get_connected_peers()
        buyer._known_capabilities[seller.peer_id] = seller._capability
        with trio.fail_after(30):
            async with buyer.service_session(seller.peer_id, _opening()) as s:
                assert s.path == "direct"
                assert await _talk(s, 2) == ["msg 0", "msg 1"]
        assert buyer.connection_path(seller.peer_id) == "direct"


@pytest.mark.parametrize("ws", [False, True])
def test_a_session_through_a_relay(ws):
    """Neither side can dial the other: the session runs over the seller's
    circuit through the relay, carries bulk, and is paid on the way."""
    async def main():
        relay = _host("relay", relay=True)
        buyer, seller = _host("nat-buyer", nat=True), _host("nat-seller", nat=True)
        credits = _Credits(cumulative=20)
        provider = _Provider(app=_metered(price=10, ahead=10), ledger=credits)
        b = _Buyer(size=20)
        async with relay.run(), buyer.run(), seller.run():
            seller.set_service_session_handler(provider)
            await _natd_pair(relay, buyer, seller, ws=ws)
            with trio.fail_after(60):
                async with buyer.service_session(
                        seller.peer_id, _opening(),
                        voucher_source=b.voucher_source) as s:
                    assert s.path == "relayed"
                    assert await _talk(s, 4) == [f"msg {i}" for i in range(4)]
                    big = bytes(300_000)
                    await s.send(big)
                    back = bytearray()
                    while len(back) < len(big):
                        chunk = await s.receive()
                        assert chunk
                        back.extend(chunk)
                    assert bytes(back) == big
        assert credits.consumed == 50 and provider.checked
    trio.run(main)


@trio_test
async def test_no_path_raises_before_anything_is_sent():
    """A NAT'd provider with no relay reservation and no known address."""
    buyer, r = _host("buyer"), _host("relay", relay=True)
    lonely = _host("lonely", nat=True)
    provider = _Provider(app=_echo)
    async with buyer.run(), r.run(), lonely.run():
        lonely.set_service_session_handler(provider)
        assert await buyer.connect_to_peer(_direct_addr(r))
        with trio.fail_after(60), pytest.raises(ConnectionError, match="no path"):
            async with buyer.service_session(lonely.peer_id, _opening()):
                pass
    assert provider.accepted == []


@trio_test
async def test_reachability_is_learned_through_a_relay():
    """A buyer not connected to a provider learns from a relay whether the
    provider says it is reachable (what catalog rows show)."""
    r = _host("relay", relay=True)
    buyer, nat = _host("buyer"), _host("nat", nat=True)
    reachable = _host("public", announce=["/ip4/127.0.0.1/tcp/4999"])
    async with r.run(), buyer.run(), nat.run(), reachable.run():
        assert reachable._capability.reachable is True
        assert nat._capability.reachable is False
        for p in (buyer, nat, reachable):
            assert await p.connect_to_peer(_direct_addr(r))
        await r.advertise_capability()
        for p in (buyer, nat, reachable):
            await p.advertise_capability()
        with trio.fail_after(10):
            for p in (buyer, nat, reachable):
                while p.peer_id not in r.known_capabilities:
                    await trio.sleep(0.05)
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
    """Unlimited unless the operator caps it (0 = no limit); the caps reach
    the host as ``relay_limits`` (nodes/relay.py run)."""
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
async def test_a_session_crosses_atn_relay(tmp_path):
    """The relay node relays circuits (unlimited by default): two NAT'd
    daemons connected only to it get a session."""
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
                text = str(a[0]) if a else ""
                if text.startswith("relay address: ") and "/ws" not in text:
                    relay_addr.append(text[len("relay address: "):])
                real_print(*a, **kw)
            builtins.print = _capture
            try:
                await relay.run(args)
            finally:
                builtins.print = real_print

        nursery.start_soon(_relay)
        with trio.fail_after(15):
            while not relay_addr:
                await trio.sleep(0.05)
        relay_id = relay_addr[0].split("/p2p/")[1]
        buyer = _host("nat-buyer", nat=True)
        seller = _host("nat-seller", nat=True)
        provider = _Provider(app=_echo)
        async with buyer.run(), seller.run():
            seller.set_service_session_handler(provider)
            assert await seller.connect_to_peer(relay_addr[0])
            assert await buyer.connect_to_peer(relay_addr[0])
            assert await _until(lambda: relay_id in seller.reservations, 30)
            buyer._known_capabilities[seller.peer_id] = seller._capability
            with trio.fail_after(60):
                async with buyer.service_session(seller.peer_id, _opening()) as s:
                    assert s.path == "relayed"
                    assert await _talk(s, 3) == ["msg 0", "msg 1", "msg 2"]
        assert provider.accepted == ["open-1"]
        nursery.cancel_scope.cancel()
