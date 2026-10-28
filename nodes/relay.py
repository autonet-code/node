"""Relay node: networking only (docs/p2p_networking.md, "Relay nodes").

A seed and meeting point for daemons, light enough for a small server:

  - a stable bootstrap address (its peer id persists in ``<data>/relay.key``);
  - a gossip relay: it subscribes to each served jurisdiction's event topic,
    so the gossipsub mesh forwards events between peers that are not
    connected to each other (NAT'd daemons dial out to the relay);
  - a blob relay: requests it cannot answer are forwarded one hop and the
    blobs it relays are cached on disk;
  - a service relay: a paid service request addressed to a provider the
    relay is connected to is forwarded one hop and the reply piped back
    (``/autonet/service/1.0.0``; bytes only, the relay serves nothing);
  - the reachability check that tells home daemons whether peers can
    reach them.

No close, no chain, no agents, no model.

Run:
    atn relay --port 4001 --announce /dns4/hub.example.org/tcp/4001

(or ``python -m nodes.relay``). Without ``--jurisdiction`` it relays the
jurisdiction this install would join (the saved selection, else the
packaged registry); pass ``--jurisdiction <chain_id>:<DAO>`` (repeatable)
to choose.

Every flag has an env twin (``AUTONET_P2P_PORT``, ``AUTONET_P2P_HOST``,
``AUTONET_P2P_ANNOUNCE``, ``AUTONET_P2P_BOOTSTRAP``,
``AUTONET_RELAY_JURISDICTIONS``, ``AUTONET_RELAY_DIR``,
``AUTONET_RELAY_SERVICE_TIMEOUT``). It prints its full
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
    ap.add_argument("--port", type=int, default=int(env("AUTONET_P2P_PORT", "4001")))
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
    ap.add_argument("--transport", choices=("tcp", "ws"),
                    default=env("AUTONET_P2P_TRANSPORT", "tcp"),
                    help="ws: serve WebSocket, e.g. behind an HTTPS proxy as wss on 443")
    ap.add_argument("--no-upnp", action="store_true")
    ap.add_argument("--advertise-interval", type=float, default=60.0)
    ap.add_argument("--service-timeout", type=float,
                    default=float(env("AUTONET_RELAY_SERVICE_TIMEOUT", "600")),
                    help="longest wait (s) for a provider's reply to a "
                         "forwarded service request")
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
    from .common.event_gossip import topic_for_rpb
    from .common.p2p import AutonetHost, NodeCapability

    data = Path(args.data_dir).expanduser()
    host = AutonetHost(
        node_id="relay",
        listen_port=args.port,
        listen_host=args.host,
        bootstrap_peers=_csv(args.bootstrap),
        announce_addrs=_csv(args.announce),
        enable_upnp=not args.no_upnp,
        auto_reach=not args.announce,
        key_seed=load_key_seed(data / "relay.key"),
        transport=args.transport,
        capability=NodeCapability(peer_id="", node_id="relay", roles=["relay"]),
    )
    # Service requests: forwarding is built into the host (it has no service
    # handler of its own, so anything addressed to the relay itself is refused).
    host.service_relay_timeout = args.service_timeout
    async with host.run():
        # Blobs: serve what is on disk, and keep what we relay on disk
        # (replacing the default in-memory relay cache).
        store = BlobStore(data_dir=str(data / "blobs"))
        server = host_blob_server(host)
        server.add_source(store.get_bytes_local)
        host.set_blob_relay_sink(lambda _cid, blob: store.add_bytes(blob))

        for key in args.jurisdiction:
            await host.subscribe_topic(topic_for_rpb(key), lambda _t, _d: None)
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
