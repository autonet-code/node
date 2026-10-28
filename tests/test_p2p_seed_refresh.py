"""
A running daemon re-reads its Registry seed list and redials seeds when it
has no peers (issue #42). Real localhost libp2p hosts; the Registry read is
a stand-in callable.

Run: python -m pytest tests/test_p2p_seed_refresh.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from atn import config as atn_config
from atn.config import RPBConfig, registry_seed_source
from nodes.common import p2p as p2p_mod
from nodes.common.p2p import AutonetHost


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _host(name, **kw):
    return AutonetHost(node_id=name, listen_host="127.0.0.1", **kw)


async def _wait_connected(host, peer_id, seconds=15):
    with trio.move_on_after(seconds):
        while peer_id not in host.get_connected_peers():
            await trio.sleep(0.1)
    return peer_id in host.get_connected_peers()


@trio_test
async def test_seed_added_to_the_registry_is_dialled():
    """The daemon joined with no seeds; the DAO then adds a relay."""
    seeds = []
    relay = _host("relay")
    async with relay.run():
        daemon = _host("daemon", seed_source=lambda: list(seeds),
                       seed_refresh_interval=0.3)
        async with daemon.run():
            await trio.sleep(0.5)
            assert relay.peer_id not in daemon.get_connected_peers()
            seeds.append(relay.addrs[0])
            assert await _wait_connected(daemon, relay.peer_id)
            assert daemon._bootstrap_peers == [relay.addrs[0]]


@trio_test
async def test_failed_or_empty_read_keeps_the_seeds():
    reads = iter([None, [], RuntimeError("rpc down")])

    def source():
        r = next(reads)
        if isinstance(r, Exception):
            raise r
        return r

    seed = "/ip4/127.0.0.1/tcp/1/p2p/QmSeed"
    host = _host("keep", bootstrap_peers=[seed], seed_source=source,
                 seed_refresh_interval=0, bootstrap_redial_interval=0)
    host._alone_check_interval = 60  # keep the alone-redial out of the way
    async with host.run():
        for _ in range(3):
            assert await host.refresh_seeds() == []
            assert host._bootstrap_peers == [seed]


class _FakeDNS:
    def __init__(self, **names):
        self.names = names

    def __call__(self, addr):
        parts = addr.split("/")
        if len(parts) > 2 and parts[1] == "dns4" and parts[2] in self.names:
            parts[1], parts[2] = "ip4", self.names[parts[2]]
        return "/".join(parts)


@trio_test
async def test_peerless_daemon_redials_without_the_periodic_redial():
    """With the periodic redial off, a daemon with no peers still redials
    its seeds (here: once the seed's name points at the live relay)."""
    dns = _FakeDNS(boot="127.0.0.2")
    real, p2p_mod.resolve_dns_multiaddr = p2p_mod.resolve_dns_multiaddr, dns
    try:
        relay = _host("relay")
        async with relay.run():
            port = relay.addrs[0].split("/tcp/")[1].split("/")[0]
            boot = f"/dns4/boot/tcp/{port}/p2p/{relay.peer_id}"
            daemon = _host("alone", bootstrap_peers=[boot],
                           bootstrap_redial_interval=0)
            daemon._alone_check_interval = 0.2
            daemon._alone_max_backoff = 0.4
            async with daemon.run():
                await trio.sleep(0.5)
                assert relay.peer_id not in daemon.get_connected_peers()
                dns.names["boot"] = "127.0.0.1"
                assert await _wait_connected(daemon, relay.peer_id)
    finally:
        p2p_mod.resolve_dns_multiaddr = real


def test_registry_seed_source(monkeypatch):
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    an = RPBConfig(rpc_url="http://rpc", dao_registry_address="0xReg",
                   jurisdiction_id="j1")
    calls = []

    def fetch(addr, rpc, jid):
        calls.append((addr, rpc, jid))
        return {"jurisdictions": {jid: {"network": {}, "contracts": {
            "substrate": "0xS", "bootstrap": " /dns4/a/tcp/1/p2p/A, ,/dns4/b/tcp/2/p2p/B"}}}}

    monkeypatch.setattr(atn_config, "_fetch_onchain_registry", fetch)
    read = registry_seed_source(an)
    assert read() == ["/dns4/a/tcp/1/p2p/A", "/dns4/b/tcp/2/p2p/B"]
    assert calls == [("0xReg", "http://rpc", "j1")]

    monkeypatch.setattr(atn_config, "_fetch_onchain_registry", lambda *a: None)
    assert read() is None

    # Not the seed source: no Registry, seeds in config.yaml, URL override.
    assert registry_seed_source(RPBConfig(rpc_url="http://rpc")) is None
    an.explicit_fields = ["bootstrap_peers"]
    assert registry_seed_source(an) is None
    an.explicit_fields = []
    monkeypatch.setenv("ATN_REGISTRY_URL", "https://example/registry.json")
    assert registry_seed_source(an) is None
