"""Same minimal pubsub test, but using AutonetHost (our wrapper)
instead of raw libp2p new_host(). All in one trio loop, no threads.

If this works, the bug is in our threading bridge.
If it doesn't work, the bug is in AutonetHost.run() itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import trio
from nodes.common.p2p import AutonetHost, NodeCapability

TOPIC = "/test/topic/2"


async def main():
    cap_a = NodeCapability(peer_id="", node_id="A")
    cap_b = NodeCapability(peer_id="", node_id="B")
    host_a = AutonetHost(node_id="A", listen_port=0, listen_host="127.0.0.1", capability=cap_a)
    host_b = AutonetHost(node_id="B", listen_port=0, listen_host="127.0.0.1", capability=cap_b)

    received_b = []

    async with host_a.run():
        async with host_b.run():
            print(f"A peer={host_a.peer_id[:16]}", flush=True)
            print(f"B peer={host_b.peer_id[:16]}", flush=True)

            # Subscribe (this triggers _ensure_gossipsub on both)
            def handler_a(topic, data):
                pass
            def handler_b(topic, data):
                received_b.append(data)
                print(f"[B] received: {data!r}", flush=True)

            await host_a.subscribe_topic(TOPIC, handler_a)
            await host_b.subscribe_topic(TOPIC, handler_b)
            print("both subscribed", flush=True)

            # Now connect
            from multiaddr import Multiaddr
            from libp2p.peer.peerinfo import info_from_p2p_addr
            ma_a = Multiaddr(f"{host_a.addrs[0]}")
            print(f"B dialing {ma_a}", flush=True)
            await host_b.connect_to_peer(str(ma_a))
            print("connected", flush=True)

            # Watch mesh formation
            for i in range(15):
                await trio.sleep(1)
                a_mesh = host_a._gossipsub.mesh.get(TOPIC, set()) if host_a._gossipsub else set()
                b_mesh = host_b._gossipsub.mesh.get(TOPIC, set()) if host_b._gossipsub else set()
                a_pt = host_a._pubsub.peer_topics.get(TOPIC, set()) if host_a._pubsub else set()
                b_pt = host_b._pubsub.peer_topics.get(TOPIC, set()) if host_b._pubsub else set()
                print(
                    f"t={i+1}s A.mesh={len(a_mesh)} B.mesh={len(b_mesh)} "
                    f"A.peer_topics={len(a_pt)} B.peer_topics={len(b_pt)}",
                    flush=True,
                )
                if a_mesh and b_mesh:
                    break

            # A publishes
            print("A publishing", flush=True)
            await host_a.publish_topic(TOPIC, b"hello-from-A-via-AutonetHost")

            # Give time to deliver
            await trio.sleep(3)
            print(f"received_b={received_b}", flush=True)


trio.run(main)
