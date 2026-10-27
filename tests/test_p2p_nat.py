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


@trio_test
async def test_bootstrap_redial_reaches_late_peer():
    """The startup dial misses a peer that is not up yet; the redial loop
    connects once it is."""
    dead = "/ip4/127.0.0.1/tcp/1/p2p/12D3KooWDeadBeefDeadBeefDeadBeefDeadBeefDeadBeefDeadBe"
    early = _host("early", bootstrap_peers=[dead], bootstrap_redial_interval=0.5)
    async with early.run():
        late = _host("late")
        async with late.run():
            # The redial list is read each round; point it at the live peer.
            early._bootstrap_peers = [dead, late.addrs[0]]
            with trio.move_on_after(5):
                while late.peer_id not in early.get_connected_peers():
                    await trio.sleep(0.1)
            assert late.peer_id in early.get_connected_peers()


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
