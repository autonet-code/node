"""Relay node: networking only (docs/p2p_networking.md, "Relay nodes").

A seed and meeting point for daemons, light enough for a small server:

  - a stable bootstrap address (its peer id persists in ``<data>/relay.key``);
  - a circuit relay (libp2p circuit relay v2, hop with reservations): NAT'd
    daemons reserve a slot and advertise ``/p2p-circuit`` addresses through
    it, so any peer can reach them; DCUtR then upgrades most of those
    connections to direct ones by hole punching;
  - a DHT server (Kademlia) that daemons find peers and blob providers in;
  - a gossip relay: it subscribes to each served jurisdiction's event topic,
    so the gossipsub mesh forwards events between peers that are not
    connected to each other;
  - a blob keeper: after each close daemons offer it the epoch's state
    blobs (payload, carry-over, agent mints), which it fetches, keeps on
    disk and provides in the DHT (issue #44);
  - a batch log: the event batches it relays are kept for a while and
    served to daemons catching up on an epoch (``/autonet/batches/1.0.0``,
    issue #41);
  - a service relay: a paid service request addressed to a provider the
    relay is connected to is forwarded one hop and the reply piped back
    (``/autonet/service/1.0.0``; bytes only, the relay serves nothing);
  - relayed connections (the circuit relay above) are unlimited unless the
    operator caps them (``--circuit-max-duration``, ``--circuit-max-bytes``,
    ``--max-circuits``; Decision 2026-11-10: relays give bandwidth like a
    torrent seed and are not paid). Service streams ride them when DCUtR
    cannot make a connection direct;
  - the reachability check that tells home daemons whether peers can
    reach them.

No close, no chain, no agents, no model.

Run:
    atn relay --port 4001 --tcp-port 4002         --announce /dns4/hub.example.org/tcp/443/wss,/ip4/203.0.113.7/tcp/4002

``--port`` is the WebSocket port (the origin behind an HTTPS proxy such as
Cloudflare), ``--tcp-port`` the TCP one (default ``--port`` + 1; hole
punching needs the relay to see daemons' real addresses, so expose it
directly when you can).

(or ``python -m nodes.relay``). Without ``--jurisdiction`` it relays the
jurisdiction this install would join (the saved selection, else the
packaged registry); pass ``--jurisdiction <chain_id>:<DAO>`` (repeatable)
to choose.

Every flag has an env twin (``AUTONET_P2P_PORT``, ``AUTONET_P2P_HOST``,
``AUTONET_P2P_ANNOUNCE``, ``AUTONET_P2P_BOOTSTRAP``, ``AUTONET_P2P_TCP_PORT``,
``AUTONET_RELAY_JURISDICTIONS``, ``AUTONET_RELAY_DIR``,
``AUTONET_RELAY_SERVICE_TIMEOUT``, ``AUTONET_RELAY_CIRCUIT_MAX_DURATION``,
``AUTONET_RELAY_CIRCUIT_MAX_BYTES``, ``AUTONET_RELAY_MAX_CIRCUITS``,
``AUTONET_RELAY_NO_CIRCUITS``). It prints its full
multiaddr on start: that is the address to publish as a bootstrap seed.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("relay")


def _csv(value: str) -> List[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def load_key_seed(path: Path) -> bytes:
    """The relay's ed25519 seed, made on first start and kept so its peer
    id (and therefore its bootstrap address) never changes."""
    if path.exists():
        seed = bytes.fromhex(path.read_text(encoding="utf-8").strip())
        if len(seed) == 32:
            return seed
        raise ValueError(f"{path} does not hold a 32-byte hex seed")
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.token_bytes(32)
    path.write_text(seed.hex(), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return seed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    env = os.environ.get
    ap = argparse.ArgumentParser(prog="atn relay",
                                 description="Autonet relay node (networking only)")
    ap.add_argument("--port", type=int, default=int(env("AUTONET_P2P_PORT", "4001")),
                    help="WebSocket port (behind an HTTPS proxy: its origin)")
    ap.add_argument("--tcp-port", type=int, default=int(env("AUTONET_P2P_TCP_PORT", "0")),
                    help="TCP port (default --port + 1)")
    ap.add_argument("--host", default=env("AUTONET_P2P_HOST", "0.0.0.0"))
    ap.add_argument("--announce", default=env("AUTONET_P2P_ANNOUNCE", ""),
                    help="comma list of public multiaddrs to advertise")
    ap.add_argument("--bootstrap", default=env("AUTONET_P2P_BOOTSTRAP", ""),
                    help="comma list of other seeds to stay connected to")
    ap.add_argument("--jurisdiction", action="append", default=None,
                    help="jurisdiction key <chain_id>:<DAO address> to relay "
                         "(repeatable; env AUTONET_RELAY_JURISDICTIONS)")
    ap.add_argument("--data-dir", default=env("AUTONET_RELAY_DIR",
                                              str(Path.home() / ".autonet-relay")))
    ap.add_argument("--transport", choices=("tcp+ws", "tcp", "ws"),
                    default=env("AUTONET_P2P_TRANSPORT", "tcp+ws"),
                    help="listeners to run (default both; ws: WebSocket only)")
    ap.add_argument("--no-upnp", action="store_true")
    ap.add_argument("--advertise-interval", type=float, default=60.0)
    ap.add_argument("--service-timeout", type=float,
                    default=float(env("AUTONET_RELAY_SERVICE_TIMEOUT", "600")),
                    help="longest wait (s) for a provider's reply to a "
                         "forwarded service request")
    # Relayed connections are unlimited unless the operator caps them (a
    # relay gives bandwidth like a torrent seed; circuit relay v2 stacks
    # default to tight caps, go-libp2p's 2 minutes and 128 KiB).
    ap.add_argument("--circuit-max-duration", type=float,
                    default=float(env("AUTONET_RELAY_CIRCUIT_MAX_DURATION", "0")),
                    help="seconds one relayed connection may stay open (0 = no limit)")
    ap.add_argument("--circuit-max-bytes", type=int,
                    default=int(env("AUTONET_RELAY_CIRCUIT_MAX_BYTES", "0")),
                    help="bytes one relayed connection may carry, both ways "
                         "(0 = no limit)")
    ap.add_argument("--max-circuits", type=int,
                    default=int(env("AUTONET_RELAY_MAX_CIRCUITS", "0")),
                    help="relayed connections open at once to one daemon "
                         "(0 = the default, 64)")
    ap.add_argument("--no-circuits", action="store_true",
                    default=env("AUTONET_RELAY_NO_CIRCUITS", "") not in ("", "0"),
                    help="do not relay connections or hold reservations "
                         "(gossip, blobs and service requests are still relayed)")
    args = ap.parse_args(argv)
    if args.jurisdiction is None:
        args.jurisdiction = _csv(env("AUTONET_RELAY_JURISDICTIONS", ""))
    if not args.jurisdiction:
        try:
            from atn.config import boot_jurisdiction_key
            key = boot_jurisdiction_key()
        except Exception:  # noqa: BLE001 — the relay runs without atn too
            key = ""
        if key:
            args.jurisdiction = [key]
    return args


async def run(args: argparse.Namespace) -> None:
    import trio
    from .common.blob_resolver import host_blob_server
    from .common.blob_store import BlobStore
    from .common.event_gossip import (
        BatchLog, SignedBatch, _verify, topic_for_rpb, topic_for_services)
    from .common.p2p import AutonetHost, NodeCapability

    data = Path(args.data_dir).expanduser()
    host = AutonetHost(
        node_id="relay",
        # --port is the WebSocket port (the TCP one with --transport tcp);
        # --tcp-port the TCP one next to it, default --port + 1.
        listen_port=args.port,
        tcp_port=args.tcp_port,
        listen_host=args.host,
        bootstrap_peers=_csv(args.bootstrap),
        announce_addrs=_csv(args.announce),
        enable_upnp=not args.no_upnp,
        auto_reach=not args.announce,
        key_seed=load_key_seed(data / "relay.key"),
        transport=args.transport,
        capability=NodeCapability(peer_id="", node_id="relay", roles=["relay"]),
        relay_hop=not args.no_circuits,
        peer_cache_path=str(data / "relay_peers.json"),
        # A relay is a meeting point, not a mesh member: it dials nobody
        # but its seeds and holds as many connections as daemons make.
        target_peers=1,
        max_peers=int(os.environ.get("AUTONET_RELAY_MAX_PEERS", "2000")),
    )
    # Service requests: forwarding is built into the host (it has no service
    # handler of its own, so anything addressed to the relay itself is refused).
    host.service_relay_timeout = args.service_timeout
    host.relay_limits = {
        "duration": max(0, int(args.circuit_max_duration or 0)),
        "bytes": max(0, int(args.circuit_max_bytes or 0)),
        "circuits": max(0, int(args.max_circuits or 0)),
    }
    if args.no_circuits:
        logger.info("not relaying connections (--no-circuits)")
    else:
        caps = {k: v for k, v in host.relay_limits.items() if v}
        logger.info("relaying connections (%s)",
                    ", ".join(f"{k} {v}" for k, v in caps.items()) or "no limit")
    async with host.run():
        # Blobs: serve what is on disk, and keep the blobs daemons offer
        # after a close (the anchored state, issue #44), so it outlives the
        # daemons that computed it. Kept blobs are provided in the DHT.
        store = BlobStore(data_dir=str(data / "blobs"))
        server = host_blob_server(host)
        server.add_source(store.get_bytes_local)
        host.set_blob_offer_store(store.has, store.add_bytes)
        kept = sorted(p.name for p in (data / "blobs").iterdir()
                      if len(p.name) == 64) if (data / "blobs").is_dir() else []
        host.provide_blobs_threadsafe(kept[-2000:])

        def keep(log: BatchLog):
            def on_batch(_topic: str, data: bytes) -> None:
                try:
                    signed = SignedBatch.from_wire(data)
                except Exception:  # noqa: BLE001 — not a batch, nothing to keep
                    return
                b = signed.batch
                if _verify(b.sender_pubkey, b.canonical_payload(), signed.signature):
                    log.add(b.content_hash(), b.timestamp, data)
            return on_batch

        for key in args.jurisdiction:
            topic = topic_for_rpb(key)
            log = BatchLog()
            host.set_batch_source(topic, log.serve)
            await host.subscribe_topic(topic, keep(log))
            # Service specs (atn/service_catalog.py): forwarded only, so
            # daemons that reach each other through this relay see them.
            await host.subscribe_topic(topic_for_services(key),
                                       lambda _t, _d: None)
            logger.info("relaying gossip for jurisdiction %s", key)
        if not args.jurisdiction:
            logger.warning("no --jurisdiction given: blobs and reach checks "
                           "only, no gossip relay")

        for addr in host.addrs:
            print(f"relay address: {addr}", flush=True)
        while True:
            await host.advertise_capability()
            logger.info("relay: %d peers connected", len(host.get_connected_peers()))
            await trio.sleep(args.advertise_interval)


def main(argv: Optional[List[str]] = None) -> int:
    import trio
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    try:
        trio.run(run, parse_args(argv))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
