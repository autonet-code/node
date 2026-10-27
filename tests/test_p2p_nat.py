"""
NAT and real-network behaviour of the P2P layer: announce addresses,
/dns4 resolution, bootstrap redial, parallel blob fetch and the one-hop
blob relay. Real localhost libp2p hosts, no network mocking.

Run: python -m pytest tests/test_p2p_nat.py -v
"""

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from nodes.common.blob_resolver import host_blob_server
from nodes.common.config import load_config
from nodes.common import p2p as p2p_mod
from nodes.common.p2p import AutonetHost, resolve_dns_multiaddr


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, **kw):
    return AutonetHost(node_id=name, listen_host="127.0.0.1", **kw)


def test_resolve_dns_multiaddr():
    assert resolve_dns_multiaddr("/dns4/localhost/tcp/4001") == "/ip4/127.0.0.1/tcp/4001"
    assert resolve_dns_multiaddr("/ip4/1.2.3.4/tcp/1") == "/ip4/1.2.3.4/tcp/1"
    bad = "/dns4/no-such-host.invalid/tcp/1"
    assert resolve_dns_multiaddr(bad) == bad


def test_env_lists(monkeypatch):
    monkeypatch.setenv("AUTONET_P2P_BOOTSTRAP", "/ip4/1.2.3.4/tcp/1/p2p/A, /ip4/5.6.7.8/tcp/2/p2p/B")
    monkeypatch.setenv("AUTONET_P2P_ANNOUNCE", "/dns4/hub.example/tcp/4101")
    monkeypatch.setenv("AUTONET_P2P_HOST", "127.0.0.1")
    cfg = load_config(os.devnull)
    assert cfg.p2p.bootstrap_peers == ["/ip4/1.2.3.4/tcp/1/p2p/A", "/ip4/5.6.7.8/tcp/2/p2p/B"]
    assert cfg.p2p.announce_addrs == ["/dns4/hub.example/tcp/4101"]
    assert cfg.p2p.listen_host == "127.0.0.1"


@trio_test
async def test_announce_replaces_advertised_addrs():
    host = _host("nat-0", announce_addrs=["/ip4/203.0.113.7/tcp/4101"])
    async with host.run():
        expected = f"/ip4/203.0.113.7/tcp/4101/p2p/{host.peer_id}"
        assert host.addrs == [expected]
        assert host._capability.listen_addrs == [expected]


class _FakeDNS:
    """Stands in for resolve_dns_multiaddr: maps /dns4/<name> to an ip."""

    def __init__(self, **names):
        self.names = names

    def __call__(self, addr):
        parts = addr.split("/")
        if len(parts) > 2 and parts[1] == "dns4" and parts[2] in self.names:
            parts[1], parts[2] = "ip4", self.names[parts[2]]
        return "/".join(parts)


@trio_test
async def test_redial_follows_bootstrap_dns_change():
    """The bootstrap name first points at a dead address; once it points
    at the live peer the redial loop connects there."""
    dns = _FakeDNS(boot="127.0.0.2")
    real, p2p_mod.resolve_dns_multiaddr = p2p_mod.resolve_dns_multiaddr, dns
    try:
        late = _host("late")
        async with late.run():
            port = late.addrs[0].split("/tcp/")[1].split("/")[0]
            boot = f"/dns4/boot/tcp/{port}/p2p/{late.peer_id}"
            early = _host("early", bootstrap_peers=[boot], bootstrap_redial_interval=0.3)
            async with early.run():
                await trio.sleep(0.5)
                assert late.peer_id not in early.get_connected_peers()
                dns.names["boot"] = "127.0.0.1"
                with trio.move_on_after(15):
                    while late.peer_id not in early.get_connected_peers():
                        await trio.sleep(0.1)
                assert late.peer_id in early.get_connected_peers()
    finally:
        p2p_mod.resolve_dns_multiaddr = real


@trio_test
async def test_announce_dns_change_is_advertised():
    dns = _FakeDNS(hub="203.0.113.7")
    real, p2p_mod.resolve_dns_multiaddr = p2p_mod.resolve_dns_multiaddr, dns
    try:
        moving = _host("moving", announce_addrs=["/dns4/hub/tcp/4101"],
                       bootstrap_redial_interval=0.3)
        watcher = _host("watcher")
        async with moving.run(), watcher.run():
            assert moving.addrs == [f"/ip4/203.0.113.7/tcp/4101/p2p/{moving.peer_id}"]
            assert await moving.connect_to_peer(watcher.addrs[0])
            dns.names["hub"] = "203.0.113.8"
            new = f"/ip4/203.0.113.8/tcp/4101/p2p/{moving.peer_id}"
            with trio.move_on_after(5):
                while True:
                    cap = watcher.known_capabilities.get(moving.peer_id)
                    if cap and cap.listen_addrs == [new]:
                        break
                    await trio.sleep(0.1)
            assert moving.addrs == [new]
            assert watcher.known_capabilities[moving.peer_id].listen_addrs == [new]
    finally:
        p2p_mod.resolve_dns_multiaddr = real


@trio_test
async def test_parallel_fetch_skips_dead_peer():
    blob = b"tool-code-blob"
    cid = hashlib.sha256(blob).hexdigest()
    holder, asker = _host("holder"), _host("asker")
    async with holder.run(), asker.run():
        host_blob_server(holder).put(blob)
        await asker.connect_to_peer(holder.addrs[0])
        dead = "12D3KooWDeadBeefDeadBeefDeadBeefDeadBeefDeadBeefDeadBe"
        with trio.fail_after(8):
            got = await asker.fetch_blob_from_peers(
                [dead, holder.peer_id], cid, timeout_per_peer=5.0)
        assert got == blob


@trio_test
async def test_two_hop_relay_between_nat_peers():
    """A and C both dial out to B only; A fetches a blob only C holds."""
    blob = b"carry-bundle-held-by-a-natd-peer"
    cid = hashlib.sha256(blob).hexdigest()
    a, b, c = _host("nat-a"), _host("hub-b"), _host("nat-c")
    async with a.run(), b.run(), c.run():
        host_blob_server(c).put(blob)
        host_blob_server(b)  # installs B's relay cache
        assert await a.connect_to_peer(b.addrs[0])
        assert await c.connect_to_peer(b.addrs[0])
        assert c.peer_id not in a.get_connected_peers()
        with trio.fail_after(10):
            got = await a.fetch_blob_from_peers([b.peer_id], cid)
        assert got == blob
        # B cached what it relayed.
        assert host_blob_server(b).get_local(cid) == blob
        # A request that allows no hops is not forwarded.
        other = b"second-blob"
        host_blob_server(c).put(other)
        with trio.fail_after(10):
            miss = await a.fetch_blob_from_peers(
                [b.peer_id], hashlib.sha256(other).hexdigest(), hops=0)
        assert miss is None


@trio_test
async def test_reach_check_reports_observed_ip_and_reachability():
    asker, peer = _host("asker"), _host("peer")
    async with asker.run(), peer.run():
        assert await asker.connect_to_peer(peer.addrs[0])
        port = asker._listen_port_actual()
        with trio.fail_after(10):
            open_ = await asker.check_reachability(peer._host.get_id(), port)
        assert open_ == {"ip": "127.0.0.1", "reachable": True}
        # A port nobody listens on is reported unreachable.
        with trio.fail_after(10):
            closed = await asker.check_reachability(peer._host.get_id(), 1)
        assert closed == {"ip": "127.0.0.1", "reachable": False}


@trio_test
async def test_auto_reach_does_not_announce_private_ip():
    """A LAN/loopback observation is reachable but not announced."""
    asker = _host("asker", auto_reach=True, reach_interval=0.3)
    peer = _host("peer")
    async with asker.run(), peer.run():
        listen = list(asker.addrs)
        assert await asker.connect_to_peer(peer.addrs[0])
        await trio.sleep(1.0)
        assert asker.addrs == listen


def test_is_public_ip():
    assert p2p_mod._is_public_ip("188.25.100.16")
    for ip in ("192.168.3.9", "10.241.68.122", "127.0.0.1", "100.64.0.1", "nonsense"):
        assert not p2p_mod._is_public_ip(ip)


def test_threadsafe_publish_never_waits_for_the_trio_thread():
    """A thread holding a lock that the trio thread is waiting on can still
    publish (the world lock / gossip ingest deadlock): the publish is
    queued, and it goes out once the trio thread is free."""
    import threading

    host = _host("pub")
    sent = []

    async def _publish(topic, data):
        sent.append((topic, data))
        return True
    host.publish_topic = _publish
    lock = threading.Lock()
    ready = threading.Event()

    async def _loop():
        host._trio_token = trio.lowlevel.current_trio_token()
        async with trio.open_nursery() as nursery:
            host._nursery = nursery
            ready.set()
            # Blocks the whole trio thread, as a topic handler waiting on
            # the world lock does.
            lock.acquire()
            lock.release()
            with trio.fail_after(5):
                while not sent:
                    await trio.sleep(0.01)

    lock.acquire()
    t = threading.Thread(target=trio.run, args=(_loop,))
    t.start()
    assert ready.wait(5)
    done = threading.Event()
    threading.Thread(target=lambda: (host.publish_topic_threadsafe("t", b"x"),
                                     done.set())).start()
    assert done.wait(2), "publish_topic_threadsafe waited for the trio thread"
    lock.release()
    t.join(10)
    assert sent == [("t", b"x")]
