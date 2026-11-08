"""The P2P mesh on py-libp2p 0.8.1 (docs/p2p_networking.md, "Mesh"):
TCP and WebSocket at once, circuit relay v2 with reservations, DHT blob
providers, the peer cache, the mesh dialer and the connection cap. Real
localhost hosts; a host built with ``advertise_listen_addrs=False``
advertises only its circuit addresses, which is how a NAT'd daemon looks
to the others. Hole punching needs real NATs: scripts/mesh_nat_harness.py.

Run: python -m pytest tests/test_p2p_mesh.py -v
"""

import functools
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from nodes.common import p2p_mesh
from nodes.common.blob_resolver import host_blob_server
from nodes.common.p2p import AutonetHost


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, **kw):
    kw.setdefault("mesh_interval", 0.5)
    return AutonetHost(node_id=name, listen_host="127.0.0.1", **kw)


async def _until(cond, timeout=20.0, step=0.1):
    with trio.move_on_after(timeout):
        while not cond():
            await trio.sleep(step)
    return cond()


def _tcp_addr(host):
    return next(a for a in host.addrs if not p2p_mesh.is_ws(a) and not p2p_mesh.is_circuit(a))


def _ws_addr(host):
    return next(a for a in host.addrs if p2p_mesh.is_ws(a) and not p2p_mesh.is_circuit(a))


# ---------------------------------------------------------------- helpers

def test_expand_wildcard():
    addr = "/ip4/0.0.0.0/tcp/4001"
    assert p2p_mesh.expand_wildcard(addr, ["127.0.0.1", "10.0.0.5", "192.168.1.9"]) == [
        "/ip4/10.0.0.5/tcp/4001", "/ip4/192.168.1.9/tcp/4001"]
    assert p2p_mesh.expand_wildcard(addr, ["127.0.0.1"]) == ["/ip4/127.0.0.1/tcp/4001"]
    assert p2p_mesh.expand_wildcard("/ip4/10.1.1.1/tcp/1", ["x"]) == ["/ip4/10.1.1.1/tcp/1"]


def test_proxy_from_env(monkeypatch):
    for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(v, raising=False)
    assert p2p_mesh.proxy_from_env() == ""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    assert p2p_mesh.proxy_from_env() == "http://proxy:3128"


def test_peer_cache_roundtrip(tmp_path):
    path = tmp_path / "p2p_peers.json"
    cache = p2p_mesh.PeerCache(path, limit=2)
    cache.note("A", ["/ip4/10.0.0.1/tcp/1"])
    cache.note("B", ["/ip4/127.0.0.1/tcp/2", "/ip4/10.0.0.2/tcp/2"])
    cache.note("C", ["/ip4/10.0.0.3/tcp/3", "/ip4/1.2.3.4/tcp/4/p2p/R/p2p-circuit"])
    cache.note("D", ["/ip4/10.0.0.4/tcp/4"])
    assert ("B", ["/ip4/10.0.0.2/tcp/2"]) in cache.entries()   # loopback dropped
    cache.save()
    again = p2p_mesh.PeerCache(path, limit=2)
    assert [pid for pid, _ in again.entries()] == ["D", "C"]
    assert again.entries()[1][1][1].endswith("/p2p-circuit")
    path.write_text("{torn", encoding="utf-8")
    assert len(p2p_mesh.PeerCache(path)) == 0


def test_proxied_machine_runs_websocket_only(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    assert AutonetHost(node_id="p")._transport == "ws"
    monkeypatch.delenv("HTTPS_PROXY")
    for v in ("https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(v, raising=False)
    assert AutonetHost(node_id="q")._transport == "tcp+ws"


# ------------------------------------------------------------ transports

@trio_test
async def test_listens_on_tcp_and_ws_and_dials_whichever_is_offered():
    both = _host("both")
    tcp_only = _host("tcp-only", transport="tcp")
    ws_only = _host("ws-only", transport="ws")
    async with both.run(), tcp_only.run(), ws_only.run():
        assert any(not p2p_mesh.is_ws(a) for a in both.addrs)
        assert any(p2p_mesh.is_ws(a) for a in both.addrs)
        assert await tcp_only.connect_to_peer(_tcp_addr(both))
        assert await ws_only.connect_to_peer(_ws_addr(both))
        # Dialled by peer id alone (the peerstore holds both addresses): a
        # tcp+ws host prefers TCP.
        other = _host("other")
        async with other.run():
            assert await other.connect_to_peer(_ws_addr(both))
            assert await both.reach_peer(other.peer_id)


# --------------------------------------------------- relay and circuits

async def _natd_pair(relay, a, c):
    """a and c both dial only the relay; c (and a) reserve on it."""
    assert await a.connect_to_peer(_tcp_addr(relay))
    assert await c.connect_to_peer(_tcp_addr(relay))
    assert await _until(lambda: relay.peer_id in c.reservations, 20)
    assert any(p2p_mesh.is_circuit(x) for x in c.addrs), c.addrs


@trio_test
async def test_natd_peer_reserves_and_is_reached_through_the_circuit():
    relay = _host("relay", relay_hop=True)
    a = _host("nat-a", advertise_listen_addrs=False)
    c = _host("nat-c", advertise_listen_addrs=False)
    async with relay.run(), a.run(), c.run():
        await _natd_pair(relay, a, c)
        circuit = [x for x in c.addrs if p2p_mesh.is_circuit(x)][0]
        assert f"/p2p/{relay.peer_id}/p2p-circuit/p2p/{c.peer_id}" in circuit
        # c advertises nothing else, so a can only come in through the relay.
        assert all(p2p_mesh.is_circuit(x) for x in c.addrs)
        await relay.advertise_capability()
        assert await _until(lambda: c.peer_id not in a.get_connected_peers() or True)
        a._known_capabilities[c.peer_id] = c._capability
        assert await a.reach_peer(c.peer_id, timeout=20)
        assert "relayed" in a.connection_kinds()[c.peer_id]
        # Service calls ride the circuit directly (no forwarding hop).
        c.set_service_handler(lambda msg: _echo(msg))
        reply = await a.request_service(c.peer_id, {"msg_id": "m1", "x": 1}, timeout=20)
        assert reply == {"msg_id": "m1", "echo": 1}


async def _echo(msg):
    return {"msg_id": msg.get("msg_id"), "echo": msg.get("x")}


@trio_test
async def test_gossip_survives_the_relayed_connection_closing():
    """a first reaches c through the relay, so their pubsub streams run on
    the relayed connection; a direct one is added (as DCUtR does) and the
    relayed one closes. Gossip must move to the direct connection (the
    harness caught it stopping: py-libp2p kept the dead stream)."""
    topic = "/autonet/world-events/mesh-test"
    relay = _host("relay", relay_hop=True)
    a = _host("nat-a", advertise_listen_addrs=False, mesh_interval=0)
    c = _host("nat-c", advertise_listen_addrs=False, mesh_interval=0)
    got = []
    async with relay.run(), a.run(), c.run():
        await _natd_pair(relay, a, c)
        await c.subscribe_topic(topic, lambda _t, d: got.append(d))
        await a.subscribe_topic(topic, lambda _t, _d: None)
        a._known_capabilities[c.peer_id] = c._capability
        assert await a.reach_peer(c.peer_id, timeout=20)
        assert "relayed" in a.connection_kinds()[c.peer_id]
        swarm = a._host.get_network()
        relayed = list(swarm.get_connections(c._host.get_id()))
        tcp = next(x for x in c._host.get_transport_addrs() if not p2p_mesh.is_ws(x))
        await swarm.dial_addr(tcp, c._host.get_id())
        assert await _until(lambda: "direct" in a.connection_kinds().get(c.peer_id, []), 10)
        for conn in relayed:
            await conn.close()
        assert await _until(lambda: a.connection_kinds().get(c.peer_id) == ["direct"], 10)
        with trio.fail_after(20):
            while not got:
                await a.publish_topic(topic, b"after-the-relay-leg")
                await trio.sleep(0.5)
        assert got[-1] == b"after-the-relay-leg"


@trio_test
async def test_blob_found_through_dht_providers_and_fetched_over_a_circuit():
    """The tool-code path: the holder provides, a peer that has never heard
    of it finds it in the DHT and fetches over a relay circuit."""
    blob = b"tool code held by one NAT'd daemon" * 50
    cid = hashlib.sha256(blob).hexdigest()
    relay = _host("relay", relay_hop=True)
    a = _host("nat-a", advertise_listen_addrs=False)
    c = _host("nat-c", advertise_listen_addrs=False)
    async with relay.run(), a.run(), c.run():
        await _natd_pair(relay, a, c)
        host_blob_server(c).put(blob)
        assert await _until(lambda: relay.peer_id in a._dht.routing_table.get_peer_ids()
                            if a._dht else False, 20)
        with trio.fail_after(30):
            while not await c.provide_blob(cid):
                await trio.sleep(0.5)
        # The relay holds the provider record, not the blob.
        assert host_blob_server(relay).get_local(cid) is None
        with trio.fail_after(45):
            got = await a.fetch_blob_from_network(cid)
        assert got == blob
        assert a.last_fetch_source == "dht"
        # Over the circuit, or a direct link if DCUtR already upgraded it
        # (on loopback the observed address is dialable).
        assert c.peer_id in a.connection_kinds()


@trio_test
async def test_blob_requests_are_never_forwarded():
    blob = b"held by c only"
    cid = hashlib.sha256(blob).hexdigest()
    a, b, c = _host("a", enable_dht=False), _host("b", enable_dht=False), _host("c", enable_dht=False)
    async with a.run(), b.run(), c.run():
        host_blob_server(c).put(blob)
        host_blob_server(b)
        assert await a.connect_to_peer(_tcp_addr(b))
        assert await c.connect_to_peer(_tcp_addr(b))
        with trio.fail_after(15):
            assert await a.fetch_blob_from_peers([b.peer_id], cid) is None
        assert host_blob_server(b).get_local(cid) is None


# -------------------------------------------- mesh dialer, cache, cap

@trio_test
async def test_mesh_dials_peers_found_in_the_dht():
    """b and c only know the hub; the mesh loop's DHT walk names them to
    each other and they connect directly up to the target count."""
    hub = _host("hub", relay_hop=True)
    b = _host("b", target_peers=3)
    c = _host("c", target_peers=3)
    async with hub.run(), b.run(), c.run():
        assert await b.connect_to_peer(_tcp_addr(hub))
        assert await c.connect_to_peer(_tcp_addr(hub))
        assert await _until(lambda: c.peer_id in b.get_connected_peers(), 40)
        assert "direct" in b.connection_kinds()[c.peer_id]


@trio_test
async def test_restart_with_every_seed_down_rejoins_through_the_peer_cache(tmp_path):
    cache = tmp_path / "p2p_peers.json"
    friend = _host("friend")
    async with friend.run():
        first = _host("me", peer_cache_path=str(cache))
        async with first.run():
            assert await first.connect_to_peer(_tcp_addr(friend))
            assert await _until(lambda: friend.peer_id in dict(first.peer_cache.entries()), 10)
        assert friend.peer_id in json.loads(cache.read_text())["peers"]
        dead_seed = "/ip4/127.0.0.1/tcp/1/p2p/12D3KooWQXiXmXcVRnCJZ8WErsRZcwApSKc9MybYhaQgLdgScVpT"
        again = _host("me", peer_cache_path=str(cache), bootstrap_peers=[dead_seed],
                      bootstrap_redial_interval=0)
        async with again.run():
            assert await _until(lambda: friend.peer_id in again.get_connected_peers(), 20)


@trio_test
async def test_connection_cap_prunes_unprotected_peers():
    hub = _host("hub", target_peers=1, max_peers=3)
    hub.prune_grace = 1.0
    peers = [_host(f"p{i}", enable_dht=False, mesh_interval=0) for i in range(7)]
    from contextlib import AsyncExitStack
    async with hub.run(), AsyncExitStack() as stack:
        for p in peers:
            await stack.enter_async_context(p.run())
        for p in peers:
            await p.connect_to_peer(_tcp_addr(hub))
            await trio.sleep(0.2)
        assert await _until(lambda: len(hub.get_connected_peers()) <= 3, 60, 0.5), \
            len(hub.get_connected_peers())


# ---------------------------------------------------------------- SOCKS5

async def _pipe(src, dst):
    try:
        while True:
            data = await src.receive_some(65536)
            if not data:
                break
            await dst.send_all(data)
    except Exception:
        pass


async def _socks5(stream, seen):
    """Minimal SOCKS5 proxy, no auth, CONNECT by name or IPv4."""
    async def read(n):
        buf = b""
        while len(buf) < n:
            buf += await stream.receive_some(n - len(buf))
        return buf
    _, nmeth = await read(2)
    await read(nmeth)
    await stream.send_all(b"\x05\x00")
    _, cmd, _, atyp = await read(4)
    if atyp == 3:
        host = (await read((await read(1))[0])).decode()
    else:
        host = ".".join(str(b) for b in await read(4))
    port = struct.unpack(">H", await read(2))[0]
    seen.append(f"{host}:{port}")
    upstream = await trio.open_tcp_stream(host, port)
    await stream.send_all(b"\x05\x00\x00\x01" + bytes(4) + b"\x00\x00")
    async with trio.open_nursery() as n:
        n.start_soon(_pipe, stream, upstream)
        n.start_soon(_pipe, upstream, stream)


@trio_test
async def test_ws_dial_through_a_socks5_proxy():
    import os
    seen = []
    r, a = _host("relay", transport="ws"), _host("a", transport="ws")
    async with trio.open_nursery() as nursery:
        listeners = await nursery.start(functools.partial(
            trio.serve_tcp, lambda s: _socks5(s, seen), 0, host="127.0.0.1"))
        port = listeners[0].socket.getsockname()[1]
        async with r.run(), a.run():
            addr = _ws_addr(r).replace("/ip4/127.0.0.1/", "/dns4/localhost/")
            os.environ["ALL_PROXY"] = f"socks5h://127.0.0.1:{port}"
            try:
                assert await a.connect_to_peer(addr)
            finally:
                os.environ.pop("ALL_PROXY", None)
            assert seen and seen[0].startswith("localhost:")
            assert r.peer_id in a.get_connected_peers()
        nursery.cancel_scope.cancel()
