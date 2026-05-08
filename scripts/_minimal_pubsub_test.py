"""Minimal py-libp2p gossipsub test, no autonet wiring.

Two AutonetHost instances, same trio loop, subscribe to the same
topic, one publishes, the other receives. If this works, the
gossipsub mesh forms fine and our autonet wrapping is the issue.

Strips down everything: no threads, no event-gossip, no notifees
beyond pubsub's own. All in one trio.run().
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import trio
from libp2p.pubsub.gossipsub import GossipSub
from libp2p.pubsub.pubsub import Pubsub
from libp2p.tools.async_service.trio_service import background_trio_service
from libp2p.custom_types import TProtocol
from multiaddr import Multiaddr

from libp2p import new_host

TOPIC = "/test/topic/1"


async def main():
    listen_a = [Multiaddr("/ip4/127.0.0.1/tcp/0")]
    listen_b = [Multiaddr("/ip4/127.0.0.1/tcp/0")]

    host_a = new_host()
    host_b = new_host()

    async with host_a.run(listen_addrs=listen_a):
        async with host_b.run(listen_addrs=listen_b):
            print(f"A peer={host_a.get_id()}", flush=True)
            print(f"A addrs={[str(a) for a in host_a.get_addrs()]}", flush=True)
            print(f"B peer={host_b.get_id()}", flush=True)
            print(f"B addrs={[str(a) for a in host_b.get_addrs()]}", flush=True)

            protocols = [TProtocol("/meshsub/1.1.0"), TProtocol("/meshsub/1.0.0")]

            gs_a = GossipSub(
                protocols=protocols,
                degree=6, degree_low=4, degree_high=12,
                heartbeat_interval=1,
            )
            ps_a = Pubsub(host=host_a, router=gs_a, strict_signing=False)
            gs_b = GossipSub(
                protocols=protocols,
                degree=6, degree_low=4, degree_high=12,
                heartbeat_interval=1,
            )
            ps_b = Pubsub(host=host_b, router=gs_b, strict_signing=False)

            async with background_trio_service(ps_a):
                async with background_trio_service(ps_b):
                    async with background_trio_service(gs_a):
                        async with background_trio_service(gs_b):
                            print("services started", flush=True)

                            # Subscribe FIRST (this is the order our autonet
                            # wiring uses — subscribe on EventGossip construction,
                            # then dial bootstrap peers).
                            sub_a = await ps_a.subscribe(TOPIC)
                            sub_b = await ps_b.subscribe(TOPIC)
                            print("both subscribed BEFORE connect", flush=True)

                            # Now connect B → A
                            from libp2p.peer.peerinfo import info_from_p2p_addr
                            ma_a = Multiaddr(
                                f"{host_a.get_addrs()[0]}/p2p/{host_a.get_id()}"
                            )
                            print(f"B dialing {ma_a}", flush=True)
                            info = info_from_p2p_addr(ma_a)
                            await host_b.connect(info)
                            print("connected", flush=True)

                            # Wait for mesh formation. Heartbeat is 1s.
                            for i in range(10):
                                await trio.sleep(1)
                                a_mesh = gs_a.mesh.get(TOPIC, set())
                                b_mesh = gs_b.mesh.get(TOPIC, set())
                                a_topics = (
                                    list(ps_a.peer_topics.get(TOPIC, set()))
                                    if hasattr(ps_a, "peer_topics") else "?"
                                )
                                print(
                                    f"t={i+1}s a_mesh={len(a_mesh)} b_mesh={len(b_mesh)} "
                                    f"a.peer_topics[{TOPIC}]={len(ps_a.peer_topics.get(TOPIC, set()))} "
                                    f"b.peer_topics[{TOPIC}]={len(ps_b.peer_topics.get(TOPIC, set()))}",
                                    flush=True,
                                )
                                if a_mesh and b_mesh:
                                    break

                            # Try publishing from A
                            print("A publishing", flush=True)
                            await ps_a.publish(TOPIC, b"hello-from-A")

                            # Wait for B to receive
                            with trio.move_on_after(10) as scope:
                                msg = await sub_b.get()
                                print(f"B received: {msg.data!r} from {msg.from_id}", flush=True)
                            if scope.cancelled_caught:
                                print("B did NOT receive within 10s", flush=True)


trio.run(main)
