"""Helpers for the P2P mesh (docs/p2p_networking.md): addresses, the proxy
check and the peer cache. ``nodes/common/p2p.py`` (AutonetHost) uses them;
they hold no libp2p state of their own.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Circuit relay v2 protocol ids (py-libp2p 0.8.1, libp2p spec).
HOP_PROTOCOL = "/libp2p/circuit/relay/0.2.0/hop"
STOP_PROTOCOL = "/libp2p/circuit/relay/0.2.0/stop"
# py-libp2p 0.8.1 hard-codes the Kademlia protocol id (the prefix argument
# is ignored), so our DHT speaks /ipfs/kad/1.0.0. It stays our own network
# because daemons only ever dial Autonet peers.
KAD_PROTOCOL = "/ipfs/kad/1.0.0"

_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
               "ALL_PROXY", "all_proxy")


def proxy_from_env() -> str:
    """The egress proxy URL this machine must dial through, or ''."""
    for name in _PROXY_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def is_circuit(addr: object) -> bool:
    return "/p2p-circuit" in str(addr)


def is_ws(addr: object) -> bool:
    s = str(addr)
    return "/ws" in s or "/wss" in s


def addr_ip(addr: object) -> str:
    """The /ip4 or /ip6 value of a multiaddr string, or ''."""
    parts = str(addr).split("/")
    for i, p in enumerate(parts[:-1]):
        if p in ("ip4", "ip6"):
            return parts[i + 1]
    return ""


def is_loopback_addr(addr: object) -> bool:
    ip = addr_ip(addr)
    try:
        return bool(ip) and ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


def local_ipv4s() -> List[str]:
    """This machine's IPv4 interface addresses, loopback last."""
    found: List[str] = []
    try:
        import psutil
        import socket
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and a.address not in found:
                    found.append(a.address)
    except Exception:  # noqa: BLE001 - psutil missing: loopback only
        pass
    found = [ip for ip in found if not ip.startswith("169.254.")]
    found.sort(key=lambda ip: ip.startswith("127."))
    return found or ["127.0.0.1"]


def expand_wildcard(addr: str, ips: Optional[List[str]] = None) -> List[str]:
    """``/ip4/0.0.0.0/...`` as one address per interface IP. Loopback is
    kept only when the machine has nothing else, since a remote peer that
    dials it reaches itself."""
    if "/ip4/0.0.0.0/" not in addr:
        return [addr]
    ips = ips if ips is not None else local_ipv4s()
    routable = [ip for ip in ips if not ip.startswith("127.")] or ips
    return [addr.replace("/ip4/0.0.0.0/", f"/ip4/{ip}/", 1) for ip in routable]


class PeerCache:
    """Peers this daemon was connected to, kept in the data dir so a
    restart with every seed down can still rejoin (``p2p_peers.json``).

    One entry per peer id: its addresses (circuit addresses included) and
    when it was last connected. The newest ``limit`` entries are kept.
    """

    def __init__(self, path: Optional[Path], limit: int = 64):
        self.path = Path(path) if path else None
        self.limit = limit
        self._peers: Dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for pid, entry in (data.get("peers") or {}).items():
                addrs = [str(a) for a in (entry.get("addrs") or []) if a]
                if pid and addrs:
                    self._peers[str(pid)] = {
                        "addrs": addrs,
                        "last_seen": float(entry.get("last_seen") or 0),
                    }
        except Exception as e:  # noqa: BLE001 - a torn cache is just empty
            logger.warning("peer cache %s unreadable (%s); starting empty", self.path, e)

    def note(self, peer_id: str, addrs: Iterable[str]) -> None:
        addrs = [str(a) for a in addrs if a]
        # Loopback only when that is all a peer has (a peer on this machine).
        addrs = [a for a in addrs if not is_loopback_addr(a)] or addrs
        if not peer_id or not addrs:
            return
        old = self._peers.get(peer_id)
        now = time.time()
        if old and old["addrs"] == addrs and now - old["last_seen"] < 300:
            return
        self._peers[peer_id] = {"addrs": addrs, "last_seen": now}
        self._dirty = True

    def forget(self, peer_id: str) -> None:
        if self._peers.pop(peer_id, None) is not None:
            self._dirty = True

    def entries(self) -> List[tuple]:
        """``(peer_id, addrs)`` newest first."""
        items = sorted(self._peers.items(), key=lambda kv: -kv[1]["last_seen"])
        return [(pid, list(e["addrs"])) for pid, e in items]

    def __len__(self) -> int:
        return len(self._peers)

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        items = sorted(self._peers.items(), key=lambda kv: -kv[1]["last_seen"])
        self._peers = dict(items[: self.limit])
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"peers": self._peers}, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            logger.debug("peer cache save failed: %s", e)
