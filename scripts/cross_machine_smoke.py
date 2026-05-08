"""Cross-machine substrate smoke test.

Run a single daemon with an explicit listen multiaddr + optional
bootstrap multiaddr, all driven by env vars so the same script
runs on both ends:

  Local (bootstrap server):
    SUBSTRATE_LISTEN_PORT=4001 \
    SUBSTRATE_RPB=cross-smoke \
    SUBSTRATE_DATA=/tmp/sub_local \
    python -m scripts.cross_machine_smoke

  EC2 (bootstrap dialer):
    SUBSTRATE_LISTEN_PORT=4001 \
    SUBSTRATE_RPB=cross-smoke \
    SUBSTRATE_DATA=/tmp/sub_ec2 \
    SUBSTRATE_BOOTSTRAP=/ip4/10.241.68.122/tcp/4001/p2p/<peer_id> \
    python -m scripts.cross_machine_smoke

What this exercises:
  - libp2p host on ZeroTier IP (real network, not in-process)
  - WorldService construction
  - EventGossip wired via LibP2PTransport
  - FederatedCloseDriver
  - Manual stdin loop: type "obs <axis>" to submit an observation,
    "close" to close the epoch, "stats" for state, "quit" to stop

No chain side — substrate-only. We're testing whether cross-machine
gossip + federation works end-to-end.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Allow running as a standalone script (python scripts/...) — when
# the script lives outside the autonet package, the package needs
# to be reachable. Two cases:
#   - dev: parent dir contains the source tree
#   - prod: autonet-computer is pip-installed and on sys.path already
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import trio


def main():
    listen_port = int(os.environ.get("SUBSTRATE_LISTEN_PORT", "4001"))
    listen_host = os.environ.get("SUBSTRATE_LISTEN_HOST", "0.0.0.0")
    rpb = os.environ.get("SUBSTRATE_RPB", "cross-smoke")
    data_dir = Path(os.environ.get("SUBSTRATE_DATA", "/tmp/sub_smoke")).expanduser()
    bootstrap_str = os.environ.get("SUBSTRATE_BOOTSTRAP", "").strip()
    bootstrap = [bootstrap_str] if bootstrap_str else []

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] listen={listen_host}:{listen_port}", flush=True)
    print(f"[smoke] rpb={rpb}", flush=True)
    print(f"[smoke] data_dir={data_dir}", flush=True)
    print(f"[smoke] bootstrap={bootstrap or '(none — server mode)'}", flush=True)

    from nodes.common.event_gossip import EventGossip, Keypair, LibP2PTransport
    from nodes.common.federated_close_driver import FederatedCloseDriver
    from nodes.common.p2p import AutonetHost, NodeCapability
    from nodes.common.world_service import WorldService

    svc = WorldService(rpb_address=rpb, data_root=data_dir / "world")

    cap = NodeCapability(peer_id="", node_id=f"smoke-{rpb}")
    host = AutonetHost(
        node_id=f"smoke-{rpb}",
        listen_port=listen_port,
        listen_host=listen_host,
        bootstrap_peers=bootstrap,
        capability=cap,
    )

    gossip_holder = {"ev": None, "driver": None}

    def _run_host():
        async def _main():
            async with host.run():
                print(
                    f"[smoke] peer_id={host.peer_id}",
                    flush=True,
                )
                print(f"[smoke] addrs={host.addrs}", flush=True)
                # Once the host has a trio token, wire EventGossip.
                transport = LibP2PTransport(host)
                kp = Keypair.generate()
                ev = EventGossip(
                    rpb_address=rpb,
                    keypair=kp,
                    transport=transport,
                    world_service=svc,
                )
                gossip_holder["ev"] = ev
                gossip_holder["driver"] = FederatedCloseDriver(
                    gossip=ev, embedding_dim=svc.embedding_dim,
                )
                print(
                    f"[smoke] gossip pubkey={kp.public_key.hex()[:16]}",
                    flush=True,
                )

                if bootstrap:
                    # Give the host a moment to settle, then dial.
                    await trio.sleep(2.0)
                    for ma in bootstrap:
                        try:
                            ok = await host.connect_to_peer(ma)
                            print(
                                f"[smoke] dialed {ma[:60]}... ok={ok}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"[smoke] dial failed: {e}", flush=True)

                # Sit and serve.
                while True:
                    await trio.sleep(60)
        try:
            trio.run(_main)
        except KeyboardInterrupt:
            pass

    host_thread = threading.Thread(target=_run_host, daemon=True)
    host_thread.start()

    # Wait for gossip to wire up
    while gossip_holder["ev"] is None:
        time.sleep(0.5)

    # Autonomous loop: each daemon emits one obs every OBS_INTERVAL
    # and closes the epoch every CLOSE_INTERVAL, printing the
    # federated result. Run for a fixed duration then exit so the
    # test is self-contained.
    obs_interval = float(os.environ.get("SUBSTRATE_OBS_INTERVAL", "3.0"))
    close_interval = float(os.environ.get("SUBSTRATE_CLOSE_INTERVAL", "15.0"))
    duration = float(os.environ.get("SUBSTRATE_DURATION", "60.0"))
    obs_axis = int(os.environ.get("SUBSTRATE_OBS_AXIS", "0"))

    from nodes.common.world_model_substrate.adapter import N_DIMS
    from world_model.generalized import Observation

    agent_id = f"0x{rpb[:8].ljust(8, '0').encode().hex()[:40]}"
    print(f"[smoke] autonomous: obs_interval={obs_interval} close_interval={close_interval} duration={duration} agent={agent_id}", flush=True)

    started = time.time()
    next_obs = started + 2.0      # let peers find each other first
    next_close = started + close_interval
    obs_counter = 0
    epoch_counter = 0

    if svc.current_epoch_id is None:
        svc.open_epoch()

    while time.time() - started < duration:
        now = time.time()
        if now >= next_obs:
            obs_counter += 1
            coords = [0.0] * (N_DIMS + svc.embedding_dim)
            coords[obs_axis % N_DIMS] = 0.5
            for i in range(svc.embedding_dim):
                coords[N_DIMS + i] = 0.01 * ((obs_axis + i) % 7)
            obs = Observation(
                id=f"obs_{rpb}_{obs_counter}",
                coords=tuple(coords),
                label=f"{rpb}_obs_{obs_counter}",
            )
            try:
                svc.submit_observation(obs, agent_id=agent_id)
                ev = gossip_holder["ev"]
                print(
                    f"[smoke] t={now-started:5.1f}s obs#{obs_counter} "
                    f"axis={obs_axis} stats={ev.stats}",
                    flush=True,
                )
            except Exception as e:
                print(f"[smoke] submit failed: {e}", flush=True)
            next_obs = now + obs_interval

        if now >= next_close:
            epoch_counter += 1
            local_close = svc.close_epoch()
            local_close["epoch_id"] = f"smoke_e{epoch_counter}"
            fed = gossip_holder["driver"].run(local_close)
            if fed is None:
                print(
                    f"[smoke] t={now-started:5.1f}s close#{epoch_counter}: no batches",
                    flush=True,
                )
            else:
                print(
                    f"[smoke] t={now-started:5.1f}s close#{epoch_counter} "
                    f"batches={fed.n_batches} is_winner={fed.is_winner} "
                    f"agent_mint={fed.close_result['authoritative_payload'].get('agent_mint', {})}",
                    flush=True,
                )
            svc.open_epoch()
            next_close = now + close_interval

        time.sleep(0.5)

    print("[smoke] duration reached, shutting down", flush=True)
    try:
        if svc.current_epoch_id is not None:
            svc.close_epoch()
        svc.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
