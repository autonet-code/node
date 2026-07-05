"""Same as minimal pubsub test but each host runs in its own
thread with its own trio loop. Tests whether the cross-thread
bridge in autonet's LibP2PTransport is what's breaking gossipsub.
"""

from __future__ import annotations

import sys
import threading
import time
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


def make_host_thread(name, listen_port, peer_holder, ready_event,
                     received_holder, do_connect_to=None,
                     publish_payload=None):
    """Run one host + pubsub in a dedicated thread/trio loop."""
    def run():
        async def _main():
            listen = [Multiaddr(f"/ip4/127.0.0.1/tcp/{listen_port}")]
            host = new_host()
            async with host.run(listen_addrs=listen):
                addrs = [str(a) for a in host.get_addrs()]
                peer_id = str(host.get_id())
                print(f"[{name}] peer={peer_id[:16]} addrs={addrs}", flush=True)

                protocols = [TProtocol("/meshsub/1.1.0"), TProtocol("/meshsub/1.0.0")]
                gs = GossipSub(
                    protocols=protocols, degree=6, degree_low=4,
                    degree_high=12, heartbeat_interval=1,
                )
                ps = Pubsub(host=host, router=gs, strict_signing=False)

                async with background_trio_service(ps):
                    async with background_trio_service(gs):
                        peer_holder[name] = {
                            "peer_id": peer_id,
                            "addr": addrs[0],
                            "host": host,
                            "ps": ps,
                            "gs": gs,
                            "trio_token": trio.lowlevel.current_trio_token(),
                        }
                        ready_event.set()

                        # Wait until both hosts are ready and dial info populated
                        while "dial_target" not in peer_holder.get(name, {}):
                            await trio.sleep(0.1)

                        target = peer_holder[name].get("dial_target")
                        if target:
                            from libp2p.peer.peerinfo import info_from_p2p_addr
                            info = info_from_p2p_addr(Multiaddr(target))
                            print(f"[{name}] dialing {target[:80]}", flush=True)
                            await host.connect(info)
                            print(f"[{name}] connected", flush=True)

                        # Subscribe
                        sub = await ps.subscribe(TOPIC)
                        print(f"[{name}] subscribed", flush=True)
                        peer_holder[name]["subscribed"] = True

                        # Heartbeat watch
                        for i in range(15):
                            await trio.sleep(1)
                            mesh = gs.mesh.get(TOPIC, set())
                            pt = ps.peer_topics.get(TOPIC, set())
                            print(
                                f"[{name}] t={i+1}s mesh={len(mesh)} "
                                f"peer_topics={len(pt)}",
                                flush=True,
                            )
                            if mesh and pt:
                                break

                        # Publisher publishes
                        if publish_payload:
                            print(f"[{name}] publishing", flush=True)
                            await ps.publish(TOPIC, publish_payload)

                        # All wait for receive or timeout
                        with trio.move_on_after(15) as scope:
                            msg = await sub.get()
                            print(f"[{name}] received: {msg.data!r}", flush=True)
                            received_holder[name] = msg.data
                        if scope.cancelled_caught:
                            print(f"[{name}] DID NOT RECEIVE within 15s", flush=True)

        try:
            trio.run(_main)
        except Exception as e:
            print(f"[{name}] crashed: {e}", flush=True)

    t = threading.Thread(target=run, daemon=True, name=name)
    t.start()
    return t


def main():
    peer_holder = {}
    received_holder = {}
    ready_a = threading.Event()
    ready_b = threading.Event()

    t_a = make_host_thread("A", 0, peer_holder, ready_a, received_holder)
    t_b = make_host_thread("B", 0, peer_holder, ready_b, received_holder,
                           publish_payload=b"hello-from-B")

    ready_a.wait(timeout=10)
    ready_b.wait(timeout=10)

    # Tell B to dial A
    a_dial = f"{peer_holder['A']['addr']}/p2p/{peer_holder['A']['peer_id']}"
    peer_holder["B"]["dial_target"] = a_dial
    peer_holder["A"]["dial_target"] = None  # A waits

    t_a.join(timeout=60)
    t_b.join(timeout=60)

    print("\n=== RESULT ===", flush=True)
    print(f"A received: {received_holder.get('A')!r}", flush=True)
    print(f"B received: {received_holder.get('B')!r}", flush=True)


main()
