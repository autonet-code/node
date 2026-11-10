"""The daemon's side of service streams (atn/service_streams.py): the spec's
``protocol`` field, the backend address and the metering, reachability for
the app and the catalog, and the provider's real payment gate
(WebSocketBridge._validate_service_payment, chain mocked) on streams between
real libp2p hosts that reach each other only through a relay.

Run: python -m pytest tests/atn/test_service_streams.py -v
"""

from __future__ import annotations

import asyncio
import socket
import threading
import types

import pytest
import trio

from atn import service_streams
from atn.service_spec import build_service_spec, stream_protocol, validate_service_spec
from atn.service_streams import meter_for, stream_backend
from tests.test_service_payment_gate import (
    ASK_AMOUNT,
    PAYER_KEY,
    SCHEMA,
    _make_runtime,
    _patch_chain,
    _patch_listing,
    _payer,
)


# ---------------------------------------------------------------------------
# Spec, backend, metering
# ---------------------------------------------------------------------------

def _spec(**kw):
    return build_service_spec(name="chat", description="a chat stream",
                              input_schema=SCHEMA, author="user",
                              unit="one minute of chat", **kw)


def test_the_spec_names_its_application_protocol():
    spec = _spec(protocol={"id": " openai-chat/1 ", "description": " OpenAI chat "})
    assert spec["protocol"] == {"id": "openai-chat/1", "description": "OpenAI chat"}
    assert stream_protocol(spec)["id"] == "openai-chat/1"
    assert _spec(protocol={"id": "webrtc-sdp/1"})["protocol"]["description"] == ""
    assert stream_protocol(_spec()) == {} and "protocol" not in _spec()
    for bad, why in (({"id": "openai-chat"}, "version"),
                     ({"id": "/1"}, "version"),
                     ({"id": "chat/v1"}, "version"),
                     ({"id": "chat/1", "port": 80}, "unknown keys"),
                     ({"id": "chat/1", "description": 5}, "text"),
                     ("chat/1", "dict")):
        with pytest.raises(ValueError, match=why):
            _spec(protocol=bad)
    assert validate_service_spec({**_spec(), "protocol": None})


def test_a_stream_backend_is_a_local_address():
    rec = types.SimpleNamespace
    assert stream_backend(rec(backing_tool="tcp://127.0.0.1:8080")) == ("127.0.0.1", 8080)
    assert stream_backend(rec(backing_tool="tcp://localhost:9/")) == ("localhost", 9)
    assert stream_backend(rec(backing_tool="tcp://[::1]:7")) == ("::1", 7)
    for bad in ("deadbeef", "tcp://10.0.0.5:80", "tcp://example.org:80",
                "tcp://127.0.0.1:0", "tcp://127.0.0.1:x", ""):
        assert stream_backend(rec(backing_tool=bad)) is None


def test_the_pipe_meters_by_the_specs_unit():
    assert meter_for({"unit": "minute"}) == ("time", 60)
    assert meter_for({"unit": "per_second"}) == ("time", 1)
    assert meter_for({"unit": "MiB"}) == ("size", 1024 ** 2)
    assert meter_for({"unit": "kb"}) == ("size", 1000)
    assert meter_for({"unit": "per_item"}) == ("stream", 0)
    assert meter_for({}) == ("stream", 0)
    assert meter_for({"unit": {"odd": 1}}) == ("stream", 0)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

class _FakeHost:
    def __init__(self, reachable, peers=None):
        self._trio_token = object()
        self._capability = types.SimpleNamespace(reachable=reachable)
        self.addrs = ["/ip4/127.0.0.1/tcp/1/p2p/me"]
        self.peer_id = "me"
        self._peer_attribution = types.SimpleNamespace(
            peer_for_agent=lambda a: (types.SimpleNamespace(peer_id=(peers or {})[a.lower()])
                                      if a.lower() in (peers or {}) else None))
        self.asked = []

    def peer_reachable(self, pid):
        self.asked.append(pid)
        return {"p-open": True, "p-nat": False}.get(pid)


def test_the_app_is_told_when_this_daemon_is_not_reachable():
    out = service_streams.own_reachability(_FakeHost(False))
    assert out["reachable"] is False and "relay" in out["advice"]
    out = service_streams.own_reachability(_FakeHost(True))
    assert out["reachable"] is True and "advice" not in out
    assert service_streams.own_reachability(types.SimpleNamespace(_trio_token=None)) \
        ["reachable"] is None


def test_listings_show_whether_the_provider_is_reachable(monkeypatch):
    from atn import service_client
    from atn.service_catalog import ServiceCatalog

    host = _FakeHost(True, {"0xaa": "p-open", "0xbb": "p-nat", "0xcc": "me"})
    monkeypatch.setattr(service_client, "_p2p_host", host)
    assert service_streams.provider_reachable("0xAA") is True
    assert service_streams.provider_reachable("0xbb") is False
    assert service_streams.provider_reachable("0xcc") is True    # this daemon
    assert service_streams.provider_reachable("0xdd") is None    # not on chain
    row = ServiceCatalog._row({"service_id": 1, "provider": "0xbb",
                               "spec_digest": "ab" * 32, "active": True}, {}, set())
    assert row["reachable"] is False


# ---------------------------------------------------------------------------
# The provider's gate on real streams
# ---------------------------------------------------------------------------

def _host(name, relay=False):
    from nodes.common.circuit_relay import RelayCaps
    from nodes.common.p2p import AutonetHost, NodeCapability
    cap = NodeCapability(peer_id="", node_id=name, roles=["relay"]) if relay else None
    h = AutonetHost(node_id=name, listen_host="127.0.0.1", capability=cap)
    if relay:
        h.circuit_relay_caps = RelayCaps()
    return h


class _Hosts:
    """Buyer, relay and provider hosts in a background trio thread, the way
    the daemon runs its host next to its asyncio loop. Buyer and provider
    are connected to the relay only."""

    def __init__(self):
        self.buyer, self.relay, self.provider = _host("buyer"), _host("relay", True), _host("seller")
        self.ready, self.stop = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=lambda: trio.run(self._main), daemon=True)

    async def _main(self):
        async with self.buyer.run(), self.relay.run(), self.provider.run():
            await self.buyer.connect_to_peer(self.relay.addrs[0])
            await self.provider.connect_to_peer(self.relay.addrs[0])
            await self.relay.advertise_capability()
            with trio.fail_after(10):
                while self.relay.peer_id not in self.buyer.known_capabilities:
                    await trio.sleep(0.05)
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

    async def trio(self, fn):
        """Run trio coroutine function ``fn`` in the hosts' thread."""
        token = self.buyer._trio_token
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: trio.from_thread.run(fn, trio_token=token))


def _echo_backend():
    """The provider owner's application: a local TCP echo server."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return

            def one(c):
                with c:
                    while True:
                        data = c.recv(65536)
                        if not data:
                            return
                        c.sendall(data)
            threading.Thread(target=one, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return srv


def _provider(tmp_path, monkeypatch, *, protocol=True, backing=None):
    from atn.ws_server import WebSocketBridge
    _patch_listing(monkeypatch)
    rt = _make_runtime(tmp_path, chain=True)
    server = WebSocketBridge(rt)
    provider = server._service_provider_address("user")
    srv = _echo_backend()
    res = rt.service_store.register(
        name="echo_stream", description="Bytes back.", input_schema=SCHEMA,
        author="user", unit="one KiB echoed",
        backing_tool=backing or f"tcp://127.0.0.1:{srv.getsockname()[1]}",
        protocol={"id": "echo/1", "description": "every byte comes back"}
        if protocol else None)
    record = rt.service_store.get(res["digest"])
    record.service_id = 7
    record.test_listing = {"service_id": 7, "provider": provider,
                           "ask_amount": ASK_AMOUNT, "active": True}
    _patch_chain(monkeypatch, voucher={"valid": True, "reason": "ok",
                                       "provider": provider, "client": _payer(),
                                       "status": 1})
    return rt, server, record, provider, srv


def _opening(record, provider, *, paid=True, cumulative=ASK_AMOUNT, rid=None):
    from atn.service_client import new_request_id, sign_request
    rid = rid or new_request_id()
    msg = {"type": "service_request", "msg_id": "m1", "spec_digest": record.digest,
           "request_id": rid, "args": {}}
    if paid:
        msg.update({"channel_id": 7, "cumulative_amount": cumulative, "signature": "0xsig",
                    "payer_signature": sign_request(PAYER_KEY, rid, record.digest, {},
                                                    provider)})
    return msg


async def _try_open(hosts, msg, record, provider):
    """Open a stream with ``msg`` and return the provider's reply."""
    from atn.service_client import reply_from_provider
    from nodes.common.service_stream import StreamRefused

    def verify(reply, rid):
        return reply_from_provider(reply, rid, record.digest, provider)

    async def go():
        try:
            async with hosts.buyer.service_stream(hosts.provider.peer_id, msg,
                                                  verify=verify) as s:
                return {"accepted": True, **s.accept_reply}
        except StreamRefused as e:
            return {"accepted": False, "verified": verify(e.reply, msg["request_id"]),
                    **e.reply}
    return await hosts.trio(go)


@pytest.mark.asyncio
async def test_the_gate_refuses_an_unpaid_stream(tmp_path, monkeypatch):
    from atn.service_streams import stream_handler_for
    rt, server, record, provider, srv = _provider(tmp_path, monkeypatch)
    try:
        with _Hosts() as hosts:
            hosts.provider.set_service_stream_handler(stream_handler_for(
                server._serve_p2p_service_request, asyncio.get_running_loop()))
            out = await _try_open(hosts, _opening(record, provider, paid=False),
                                  record, provider)
            assert out["accepted"] is False and out["charged"] is False
            assert "Payment validation failed" in out["error"]
            assert out["verified"] is True     # the provider's own signed "no"
            # A voucher signed by a key that does not hold the channel.
            thief = _opening(record, provider)
            from atn.service_client import sign_request
            thief["payer_signature"] = sign_request("0x" + "22" * 32,
                                                    thief["request_id"],
                                                    record.digest, {}, provider)
            out = await _try_open(hosts, thief, record, provider)
            assert out["accepted"] is False and "signed by the channel" in out["error"]
    finally:
        srv.close()
    assert rt.service_store.get_channel_voucher(7) is None
    assert rt.service_store.last_channel_cumulative(7) == 0


@pytest.mark.asyncio
async def test_a_request_reply_service_is_not_opened_as_a_stream(tmp_path, monkeypatch):
    from atn.service_streams import stream_handler_for
    rt, server, record, provider, srv = _provider(tmp_path, monkeypatch, protocol=False)
    try:
        with _Hosts() as hosts:
            hosts.provider.set_service_stream_handler(stream_handler_for(
                server._serve_p2p_service_request, asyncio.get_running_loop()))
            out = await _try_open(hosts, _opening(record, provider), record, provider)
    finally:
        srv.close()
    assert out["accepted"] is False and "not a stream" in out["error"]
    # Refused before the gate: the voucher is unspent.
    assert rt.service_store.last_channel_cumulative(7) == 0


@pytest.mark.asyncio
async def test_a_paid_stream_from_an_agent_port_to_the_providers_backend(
        tmp_path, monkeypatch):
    """open_local_stream: the buyer's voucher opens the stream through the
    relay, a local program's bytes reach the provider's backend and come
    back, and the provider keeps the voucher once the stream is used."""
    import atn.service_channels as sc
    from atn.service_streams import open_local_stream, stream_handler_for

    rt, server, record, provider, srv = _provider(tmp_path, monkeypatch)
    payer = _payer()
    issued = []

    async def voucher_for(book, smc, payer_key, payer_addr, provider_addr, ask,
                          deposit=0, *, units=1):
        if book.get(payer_addr, provider_addr) is None:
            book.set_open(payer_addr, provider_addr, 7, 100 * ASK_AMOUNT)
        cum = (int(book.get(payer_addr, provider_addr)["cumulative"])
               + int(ask) * int(units))
        issued.append(cum)
        return {"channel_id": 7, "cumulative_amount": cum, "signature": "0xsig"}
    monkeypatch.setattr(sc, "voucher_for", voucher_for)
    try:
        with _Hosts() as hosts:
            hosts.provider.set_service_stream_handler(stream_handler_for(
                server._serve_p2p_service_request, asyncio.get_running_loop()))
            out = await open_local_stream(
                rt, payer_key=PAYER_KEY, payer=payer, provider_address=provider,
                spec_digest=record.digest, units=3, ask=ASK_AMOUNT,
                p2p_host=hosts.buyer, peer_id=hosts.provider.peer_id)
            assert out.get("ok"), out
            assert out["path"] == "relayed" and out["protocol"]["id"] == "echo/1"
            assert out["units"] == 3
            reader, writer = await asyncio.open_connection("127.0.0.1", out["port"])
            writer.write(b"hello over a relayed stream")
            await writer.drain()
            got = await asyncio.wait_for(reader.readexactly(27), 20)
            assert got == b"hello over a relayed stream"
            writer.close()
            for _ in range(200):
                if rt.service_store.get_channel_voucher(7):
                    break
                await asyncio.sleep(0.05)
            ch = None
            for _ in range(200):
                ch = rt.channel_book.get(payer, provider)
                if ch and ch.get("pending") is None and int(ch["cumulative"]):
                    break
                await asyncio.sleep(0.05)
    finally:
        srv.close()
    kept = rt.service_store.get_channel_voucher(7)
    assert kept and kept["cumulative"] == 3 * ASK_AMOUNT     # one voucher of 3 units
    assert issued == [3 * ASK_AMOUNT]
    assert ch["cumulative"] == 3 * ASK_AMOUNT                # the buyer counted it
    # The provider's log names the payer the gate verified.
    logged = rt.service_store.summary()[record.digest]
    assert logged["ok_count"] == 1 and logged["unique_clients"] == 1


@pytest.mark.asyncio
async def test_vouchers_follow_metered_usage_through_the_gate(tmp_path, monkeypatch):
    """A size-metered stream (1 unit = 1 kB carried either way), 2 units a
    voucher: 8 kB of traffic takes the opening and three more vouchers, each
    checked by the gate and kept as usage enters it. A voucher fetched ahead
    and never used is released on both sides."""
    import atn.service_channels as sc
    from atn.service_streams import open_local_stream, stream_handler_for

    rt, server, record, provider, srv = _provider(tmp_path, monkeypatch)
    record.spec["unit"] = "kb"
    payer = _payer()
    issued = []

    async def voucher_for(book, smc, payer_key, payer_addr, provider_addr, ask,
                          deposit=0, *, units=1):
        if book.get(payer_addr, provider_addr) is None:
            book.set_open(payer_addr, provider_addr, 7, 100 * ASK_AMOUNT)
        cum = (int(book.get(payer_addr, provider_addr)["cumulative"])
               + int(ask) * int(units))
        issued.append(cum)
        return {"channel_id": 7, "cumulative_amount": cum, "signature": "0xsig"}
    monkeypatch.setattr(sc, "voucher_for", voucher_for)
    ch = None
    try:
        with _Hosts() as hosts:
            hosts.provider.set_service_stream_handler(stream_handler_for(
                server._serve_p2p_service_request, asyncio.get_running_loop()))
            out = await open_local_stream(
                rt, payer_key=PAYER_KEY, payer=payer, provider_address=provider,
                spec_digest=record.digest, units=2, ask=ASK_AMOUNT,
                p2p_host=hosts.buyer, peer_id=hosts.provider.peer_id)
            assert out.get("ok"), out
            reader, writer = await asyncio.open_connection("127.0.0.1", out["port"])
            writer.write(b"x" * 4000)
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(4000), 30) == b"x" * 4000
            writer.close()
            for _ in range(300):
                ch = rt.channel_book.get(payer, provider)
                kept = rt.service_store.get_channel_voucher(7)
                if (ch and ch.get("pending") is None and kept
                        and kept["cumulative"] == 8 * ASK_AMOUNT
                        and rt.service_store.last_channel_cumulative(7)
                        == 8 * ASK_AMOUNT):
                    break
                await asyncio.sleep(0.05)
    finally:
        srv.close()
    assert rt.service_store.get_channel_voucher(7)["cumulative"] == 8 * ASK_AMOUNT
    assert issued[:4] == [2 * ASK_AMOUNT, 4 * ASK_AMOUNT, 6 * ASK_AMOUNT,
                          8 * ASK_AMOUNT]
    # The buyer counts exactly what the provider kept, nothing pending.
    assert ch["cumulative"] == 8 * ASK_AMOUNT and ch.get("pending") is None
    assert rt.service_store.last_channel_cumulative(7) == 8 * ASK_AMOUNT
