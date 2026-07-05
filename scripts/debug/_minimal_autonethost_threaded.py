"""Same as _minimal_autonethost_test but uses AutonetHost from a
thread + threadsafe publish/subscribe from main thread. Mirrors
the cross_machine_smoke.py architecture.
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
from nodes.common.p2p import AutonetHost, NodeCapability

TOPIC = "/test/topic/3"


def make_host(name, listen_port):
    cap = NodeCapability(peer_id="", node_id=name)
    host = AutonetHost(
        node_id=name, listen_port=listen_port, listen_host="127.0.0.1",
        capability=cap,
    )
    received = []

    def _run():
        async def _main():
            async with host.run():
                while True:
                    await trio.sleep(60)
        try:
            trio.run(_main)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[{name}] crashed: {e}", flush=True)

    t = threading.Thread(target=_run, daemon=True, name=name)
    t.start()
    return host, received, t


def main():
    host_a, received_a, t_a = make_host("A", 0)
    host_b, received_b, t_b = make_host("B", 0)

    # Wait for both ready
    while not host_a._ready_event.is_set():
        time.sleep(0.1)
    while not host_b._ready_event.is_set():
        time.sleep(0.1)

    print(f"A peer={host_a.peer_id[:16]} addrs={host_a.addrs}", flush=True)
    print(f"B peer={host_b.peer_id[:16]} addrs={host_b.addrs}", flush=True)

    # Subscribe via threadsafe path
    def handler_a(topic, data):
        received_a.append(data)
        print(f"[A] received: {data!r}", flush=True)
    def handler_b(topic, data):
        received_b.append(data)
        print(f"[B] received: {data!r}", flush=True)

    host_a.subscribe_topic_threadsafe(TOPIC, handler_a)
    host_b.subscribe_topic_threadsafe(TOPIC, handler_b)
    print("both subscribed via threadsafe", flush=True)

    # Connect B → A using a threadsafe trick: schedule connect via from_thread
    a_addr = f"{host_a.addrs[0]}"
    print(f"B dialing {a_addr}", flush=True)

    async def _do_connect():
        ok = await host_b.connect_to_peer(a_addr)
        print(f"B connect result: {ok}", flush=True)

    trio.from_thread.run(_do_connect, trio_token=host_b._trio_token)

    # Watch mesh formation
    for i in range(15):
        time.sleep(1)
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
    host_a.publish_topic_threadsafe(TOPIC, b"hello-from-A-threadsafe")

    time.sleep(3)
    print(f"received_a={received_a}", flush=True)
    print(f"received_b={received_b}", flush=True)


main()
