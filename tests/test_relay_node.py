"""Relay node (nodes/relay.py): two daemons that are only connected to the
relay still exchange gossip and blobs through it, and its peer id is stable
across restarts.

Run: python -m pytest tests/test_relay_node.py -v
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from nodes import relay
from nodes.common.blob_resolver import host_blob_server
from nodes.common.event_gossip import topic_for_rpb
from nodes.common.p2p import AutonetHost

JURISDICTION = "31337:0x00000000000000000000000000000000000000aa"


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def test_key_seed_is_persisted(tmp_path):
    path = tmp_path / "relay.key"
    first = relay.load_key_seed(path)
    assert len(first) == 32 and relay.load_key_seed(path) == first


@trio_test
async def test_stable_peer_id_from_seed():
    seed = bytes(range(32))
    ids = []
    for _ in range(2):
        h = AutonetHost(node_id="r", listen_host="127.0.0.1", key_seed=seed)
        async with h.run():
            ids.append(h.peer_id)
    assert ids[0] == ids[1]


@trio_test
async def test_gossip_and_blobs_cross_the_relay(tmp_path):
    args = relay.parse_args([
        "--port", "0", "--host", "127.0.0.1", "--no-upnp",
        "--jurisdiction", JURISDICTION, "--data-dir", str(tmp_path / "relay"),
        "--advertise-interval", "0.5"])
    topic = topic_for_rpb(JURISDICTION)
    got = []
    blob = b"tool-code-only-c-holds"
    cid = hashlib.sha256(blob).hexdigest()

    async with trio.open_nursery() as nursery:
        relay_addr = []

        async def _relay():
            # Run the relay and capture its printed address.
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

        a = AutonetHost(node_id="nat-a", listen_host="127.0.0.1")
        c = AutonetHost(node_id="nat-c", listen_host="127.0.0.1")
        async with a.run(), c.run():
            assert await a.connect_to_peer(relay_addr[0])
            assert await c.connect_to_peer(relay_addr[0])
            assert c.peer_id not in a.get_connected_peers()
            await c.subscribe_topic(topic, lambda _t, data: got.append(data))
            await a.subscribe_topic(topic, lambda _t, _d: None)
            # Let the gossipsub meshes form (heartbeat 1 s).
            with trio.fail_after(20):
                while not got:
                    await a.publish_topic(topic, b"event-batch")
                    await trio.sleep(0.5)
            assert got[0] == b"event-batch"

            host_blob_server(c).put(blob)
            relay_pid = relay_addr[0].split("/p2p/")[1]
            with trio.fail_after(15):
                fetched = await a.fetch_blob_from_peers([relay_pid], cid)
            assert fetched == blob
            # The relay kept the blob on disk.
            assert (tmp_path / "relay" / "blobs").exists()
        nursery.cancel_scope.cancel()
