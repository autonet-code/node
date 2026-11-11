#!/usr/bin/env python3
"""One mesh node for the NAT harness (scripts/mesh_nat_harness.py).

The daemon's p2p layer as it runs in ``atn``: an AutonetHost (both
transports, circuit relay v2 client, DCUtR, DHT, mesh loop, peer cache)
with a disk blob store served to peers and the event topic of one
jurisdiction. In place of the agent runtime it takes commands on a local
control socket (``mesh_ctl.py``):

  status                 peer id, addresses, connections (direct/relayed and
                         the remote address), reservations, DHT mode/size
  publish_tool {name}    store a manifest and its code, provide both in the
                         DHT, then gossip the registration (the order
                         ToolStore uses), like a daemon publishing a tool
  gossip {text}          publish an event on the topic
  received               events seen on the topic, and the tool fetches
                         this node made when a registration arrived
  fetch {cid}            find the cid's providers in the DHT and fetch from
                         them only
  session {peer, bytes}  open a service session to the peer and echo bytes
                         through it (its path, its connection's remote
                         address, whether every byte came back)

On every tool registration it hears, the node looks the manifest and code
up in the DHT and fetches them from the providers, as a daemon pulls a
remote manifest for discovery and its code on adoption.

Env: MESH_NAME, MESH_PORT (ws), MESH_TCP_PORT, MESH_SEEDS (comma list),
MESH_TOPIC_KEY, MESH_DATA (default /data), MESH_CTL_PORT (default 9000).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import trio

sys.path.insert(0, os.environ.get("MESH_REPO", "/repo"))

from nodes.common.blob_resolver import host_blob_server  # noqa: E402
from nodes.common.blob_store import BlobStore  # noqa: E402
from nodes.common.event_gossip import topic_for_rpb  # noqa: E402
from nodes.common.p2p import AutonetHost, NodeCapability  # noqa: E402
from nodes.relay import load_key_seed  # noqa: E402

NAME = os.environ.get("MESH_NAME", "node")
DATA = Path(os.environ.get("MESH_DATA", "/data"))
TOPIC = topic_for_rpb(os.environ.get("MESH_TOPIC_KEY", "31337:0xmesh"))

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format=f"%(asctime)s [{NAME}] [%(name)s] %(levelname)s: %(message)s")
for noisy in ("libp2p", "multiaddr", "trio_websocket"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("mesh-node")


class _FreeEcho:
    """A free service session that echoes every byte (the transport under
    test is the mesh's, not the payment's: no ledger, so nothing is charged
    and no voucher is asked for)."""

    async def accept(self, open_msg: dict):
        from nodes.common.service_session import Accept
        return Accept(reply={"ok": True, "result": {
            "request_id": open_msg.get("request_id"), "prices": {}}})

    async def check_voucher(self, open_msg: dict, frame: dict):
        from nodes.common.service_session import VoucherCheck
        return VoucherCheck(ok=False, error="a free session")

    async def serve(self, session) -> None:
        while True:
            data = await session.receive()
            if not data:
                return
            await session.send(data)

    async def ended(self, session) -> None:
        return None


class Node:
    def __init__(self, host: AutonetHost, store: BlobStore):
        self.host = host
        self.store = store
        self.received: list = []
        self.fetches: list = []

    def put(self, blob: bytes) -> str:
        cid = self.store.add_bytes(blob)
        return cid

    def status(self) -> dict:
        h = self.host
        conns = {}
        for pid, conns_ in h.host.get_network().get_connections_map().items():
            entries = []
            for c in conns_:
                try:
                    addrs = [str(a) for a in c.get_transport_addresses()]
                except Exception:
                    addrs = []
                remote = None
                try:
                    remote = c.muxed_conn.get_remote_address()
                except Exception:
                    pass
                kind = "relayed" if addrs and all("/p2p-circuit" in a for a in addrs) else "direct"
                entries.append({"kind": kind, "addrs": addrs, "remote": list(remote) if remote else None})
            conns[str(pid)] = entries
        caps = {pid: cap.node_id for pid, cap in h.known_capabilities.items()}
        return {"name": NAME, "peer_id": h.peer_id, "addrs": h.addrs,
                "transport": h._transport, "reservations": h.reservations,
                "dht_mode": h.dht_mode,
                "dht_peers": h._dht.get_routing_table_size() if h._dht else 0,
                "connections": conns, "names": caps,
                "provided": len(h.provided_blobs), "peer_cache": len(h.peer_cache)}

    async def fetch_via_dht(self, cid: str, timeout: float = 60.0) -> dict:
        t0 = time.time()
        out = {"cid": cid, "ok": False, "providers": [], "via": None}
        with trio.move_on_after(timeout):
            while True:
                providers = await self.host.find_blob_providers(cid)
                out["providers"] = [str(p.peer_id) for p in providers]
                if providers:
                    data = await self.host.fetch_blob_from_peers(
                        [p.peer_id for p in providers], cid, timeout_per_peer=20.0)
                    if data is not None:
                        self.store.add_bytes(data)
                        out["ok"] = True
                        kinds = self.host.connection_kinds()
                        out["via"] = {p: kinds.get(p) for p in out["providers"]}
                        break
                await trio.sleep(2.0)
        out["seconds"] = round(time.time() - t0, 1)
        return out

    async def publish_tool(self, name: str) -> dict:
        code = f"print('tool {name} from {NAME} at {time.time()}')\n".encode()
        code_cid = self.put(code)
        manifest = json.dumps({"name": name, "author": NAME, "code_digest": code_cid},
                              sort_keys=True).encode()
        manifest_cid = self.put(manifest)
        # The ToolStore order: provide the blobs, then send the registration.
        provided = await self.host.provide_blobs([manifest_cid, code_cid], timeout=20.0)
        event = json.dumps({"type": "tool_registered", "name": name, "from": NAME,
                            "manifest": manifest_cid, "code": code_cid}).encode()
        await self.host.publish_topic(TOPIC, event)
        return {"manifest": manifest_cid, "code": code_cid, "provided": provided}

    def on_event(self, _topic: str, data: bytes) -> None:
        try:
            event = json.loads(data.decode())
        except Exception:
            return
        event["at"] = time.time()
        self.received.append(event)
        if event.get("type") == "tool_registered" and self.host._nursery is not None:
            self.host._nursery.start_soon(self._pull_tool, event)

    async def _pull_tool(self, event: dict) -> None:
        results = []
        for key in ("manifest", "code"):
            results.append(await self.fetch_via_dht(event[key]))
        self.fetches.append({"tool": event["name"], "from": event["from"],
                             "ok": all(r["ok"] for r in results), "results": results})
        log.info("fetched tool %s from %s via the DHT: %s", event["name"], event["from"],
                 [(r["ok"], r["providers"][:3], r["seconds"]) for r in results])

    async def session(self, peer: str, nbytes: int, timeout: float = 120.0) -> dict:
        """Open a service session to ``peer`` (docs/services_market.md,
        Decision 2026-11-12) and echo ``nbytes`` through it: the path it
        took, the remote address of the connection, and whether every byte
        came back."""
        t0 = time.time()
        out = {"peer": peer, "ok": False, "path": None, "bytes": nbytes}
        opening = {"spec_digest": "00" * 32, "request_id": f"h-{int(t0 * 1000)}",
                   "args": {}}
        payload = bytes(range(256)) * (nbytes // 256 + 1)
        payload = payload[:nbytes]
        try:
            with trio.fail_after(timeout):
                async with self.host.service_session(peer, opening, timeout=60) as st:
                    out["path"] = st.path
                    back = bytearray()

                    async def reader():
                        while len(back) < nbytes:
                            chunk = await st.receive()
                            if not chunk:
                                return
                            back.extend(chunk)
                    async with trio.open_nursery() as n:
                        n.start_soon(reader)
                        for off in range(0, nbytes, 60000):
                            await st.send(payload[off:off + 60000])
                    out["ok"] = bytes(back) == payload
        except BaseException as e:  # noqa: BLE001 (reported to the harness)
            out["error"] = repr(e)[:300]
        out["seconds"] = round(time.time() - t0, 1)
        try:
            from libp2p.peer.id import ID
            remote = []
            for c in self.host.host.get_network().get_connections(ID.from_base58(peer)):
                try:
                    remote.append(list(c.muxed_conn.get_remote_address()))
                except Exception:
                    pass
            out["remote"] = remote
        except Exception:
            out["remote"] = []
        return out

    async def command(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "status":
            return self.status()
        if cmd == "session":
            return await self.session(str(req.get("peer") or ""),
                                      int(req.get("bytes") or 1 << 20))
        if cmd == "publish_tool":
            return await self.publish_tool(str(req.get("name") or "tool"))
        if cmd == "gossip":
            ok = await self.host.publish_topic(TOPIC, json.dumps(
                {"type": "note", "text": req.get("text", ""), "from": NAME}).encode())
            return {"ok": ok}
        if cmd == "received":
            return {"received": self.received, "fetches": self.fetches}
        if cmd == "fetch":
            return await self.fetch_via_dht(str(req.get("cid")))
        return {"error": f"unknown command {cmd!r}"}


async def serve_ctl(node: Node, port: int) -> None:
    async def handle(stream):
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = await stream.receive_some(65536)
            if not chunk:
                return
            buf += chunk
        try:
            reply = await node.command(json.loads(buf.decode()))
        except Exception as e:  # noqa: BLE001
            reply = {"error": repr(e)}
        await stream.send_all(json.dumps(reply).encode() + b"\n")
        await stream.aclose()
    await trio.serve_tcp(handle, port, host="127.0.0.1")


async def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in os.environ.get("MESH_SEEDS", "").split(",") if s.strip()]
    host = AutonetHost(
        node_id=NAME,
        listen_port=int(os.environ.get("MESH_PORT", "4001")),
        tcp_port=int(os.environ.get("MESH_TCP_PORT", "0")),
        bootstrap_peers=seeds,
        bootstrap_redial_interval=15,
        enable_upnp=False,
        auto_reach=True,
        reach_interval=60,
        key_seed=load_key_seed(DATA / "p2p.key"),
        capability=NodeCapability(peer_id="", node_id=NAME),
        peer_cache_path=str(DATA / "p2p_peers.json"),
        mesh_interval=float(os.environ.get("MESH_INTERVAL", "5")),
        target_peers=8,
        provide_interval=120,
    )
    store = BlobStore(data_dir=str(DATA / "blobs"))
    async with host.run():
        server = host_blob_server(host)
        server.add_source(store.get_bytes_local)
        node = Node(host, store)
        host.set_service_session_handler(_FreeEcho())
        await host.subscribe_topic(TOPIC, node.on_event)
        log.info("mesh node %s up: %s", host.peer_id, host.addrs)
        async with trio.open_nursery() as n:
            n.start_soon(serve_ctl, node, int(os.environ.get("MESH_CTL_PORT", "9000")))
            while True:
                await host.advertise_capability()
                await trio.sleep(15)


if __name__ == "__main__":
    trio.run(main)
