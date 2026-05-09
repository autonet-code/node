"""On-chain peer attribution: libp2p PeerId → on-chain agent address.

Phase 12 — replaces the bootstrap-multiaddr-only peer model with one
where the chain is the source of truth for *who exists* and the
gossip layer handles *where they are right now*.

Workflow
--------

1. Daemon starts up → reads ``Substrate.sol``'s registered-agent list.
2. For each agent, reads its libp2p PeerId from ``getAgentPeerId``.
3. Builds a ``PeerId → agent_address`` map (and the reverse).
4. Connects to the configured/community bootstrap peer(s).
5. Joins gossip topics. Existing peers gossip their connections; the
   new peer learns about everyone via libp2p's natural peer-exchange.
6. As gossipped peers come online, the attribution map says which
   on-chain agent each PeerId corresponds to. PeerIds that aren't in
   the map are unregistered and can be ignored for consensus.
7. Periodic refresh picks up newly-registered agents.

What this replaces
------------------

The pre-Phase-12 model required every daemon to know every other
daemon's multiaddr by static config. The new model only requires
knowing one bootstrap multiaddr — the rest of the network is
discovered via gossip, with on-chain attribution telling you which
of the discovered peers are real agents vs. random nodes.

What this does NOT do (yet)
---------------------------

DHT-based PeerId-to-multiaddr resolution. py-libp2p's Kademlia DHT
module is in alpha and unused in our setup; tracked in #73.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


# Community bootstrap peers — well-known, daemons fall back to these
# when no user-configured bootstrap_peers are present in autonet.yaml.
# Format: libp2p multiaddrs.
#
# These are the seeds that let a fresh ``pip install autonet-computer``
# join the network without manual setup. As the network grows, the
# community can publish more bootstrap nodes; users override locally
# via ``p2p.bootstrap_peers`` in autonet.yaml.
COMMUNITY_BOOTSTRAP_PEERS: list[str] = [
    # Will be populated as we publish stable bootstrap nodes.
    # For now this is empty — users must configure bootstrap_peers
    # explicitly until we have at least one always-on node to publish.
]


@dataclass
class AttributedPeer:
    """A libp2p peer with its on-chain identity attached."""
    peer_id: str            # libp2p PeerId (base58 string)
    agent_address: str      # 0x... on-chain agent address
    lineage_hash: str       # bytes32 hex
    registered_at: int      # block timestamp


class PeerAttribution:
    """Maintains a libp2p-PeerId ↔ on-chain-agent mapping.

    Reads ``Substrate.sol`` periodically and exposes lookups in both
    directions. Thread-safe; refresh runs synchronously when called.
    """

    def __init__(
        self,
        contract: Any,
        *,
        refresh_interval_s: float = 300.0,
    ):
        """``contract`` is a ``web3`` contract instance bound to the
        deployed Substrate.sol address. ``refresh_interval_s`` is the
        TTL after which a ``maybe_refresh()`` call will re-read the
        chain state.
        """
        self._contract = contract
        self._refresh_interval_s = refresh_interval_s
        self._lock = threading.RLock()
        self._by_peer_id: dict[str, AttributedPeer] = {}
        self._by_address: dict[str, AttributedPeer] = {}
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> int:
        """Force-read the chain state and rebuild the attribution map.

        Returns the number of attributed peers after refresh.
        """
        with self._lock:
            try:
                count = self._contract.functions.registeredAgentCount().call()
            except Exception as e:
                logger.warning("registeredAgentCount() failed: %s", e)
                return len(self._by_peer_id)

            new_by_peer: dict[str, AttributedPeer] = {}
            new_by_addr: dict[str, AttributedPeer] = {}
            for i in range(count):
                try:
                    addr = self._contract.functions.getRegisteredAgent(i).call()
                except Exception as e:
                    logger.debug("getRegisteredAgent(%d) failed: %s", i, e)
                    continue
                try:
                    peer_id_bytes = self._contract.functions.getAgentPeerId(addr).call()
                    record = self._contract.functions.agents(addr).call()
                    lineage, registered_at, active, _, _ = record
                except Exception as e:
                    logger.debug("agent metadata read failed for %s: %s", addr, e)
                    continue
                if not active or not peer_id_bytes:
                    continue

                peer_id_str = self._decode_peer_id(peer_id_bytes)
                if not peer_id_str:
                    continue

                attr = AttributedPeer(
                    peer_id=peer_id_str,
                    agent_address=addr,
                    lineage_hash=lineage.hex() if isinstance(lineage, bytes) else str(lineage),
                    registered_at=int(registered_at),
                )
                new_by_peer[peer_id_str] = attr
                new_by_addr[addr.lower()] = attr

            self._by_peer_id = new_by_peer
            self._by_address = new_by_addr
            self._last_refresh = time.time()
            logger.info("Peer attribution refreshed: %d agents", len(new_by_peer))
            return len(new_by_peer)

    def maybe_refresh(self) -> None:
        """Refresh only if the cached state is older than the TTL."""
        if time.time() - self._last_refresh > self._refresh_interval_s:
            self.refresh()

    @staticmethod
    def _decode_peer_id(peer_id_bytes: bytes) -> str:
        """Decode the on-chain bytes back into a libp2p PeerId string.

        On-chain ``peerId`` is the raw bytes of the libp2p PeerId
        multihash. Convert back to base58 for comparison with
        ``host.get_id()`` output.
        """
        if not peer_id_bytes:
            return ""
        try:
            import base58  # type: ignore
            return base58.b58encode(peer_id_bytes).decode("ascii")
        except ImportError:
            # Fallback: assume the bytes are already a UTF-8 PeerId string
            # (the synthetic test path). Will not match real libp2p ids.
            try:
                return peer_id_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return ""

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def agent_for_peer(self, peer_id: str) -> Optional[AttributedPeer]:
        """Return the on-chain agent attribution for a libp2p PeerId,
        or ``None`` if the PeerId isn't registered."""
        with self._lock:
            return self._by_peer_id.get(peer_id)

    def peer_for_agent(self, agent_address: str) -> Optional[AttributedPeer]:
        """Return the libp2p PeerId attribution for an on-chain agent
        address, or ``None`` if the address isn't registered."""
        with self._lock:
            return self._by_address.get(agent_address.lower())

    def is_registered_peer(self, peer_id: str) -> bool:
        """True iff a libp2p peer is attributed to a registered agent."""
        with self._lock:
            return peer_id in self._by_peer_id

    def all_peers(self) -> list[AttributedPeer]:
        """Snapshot of all currently-attributed peers."""
        with self._lock:
            return list(self._by_peer_id.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_peer_id)


def resolve_bootstrap_peers(user_configured: list[str]) -> list[str]:
    """Resolve the effective bootstrap peer list.

    Returns the user-configured list when non-empty; otherwise falls
    back to the community defaults. Daemons should call this instead
    of reading ``cfg.p2p.bootstrap_peers`` directly so a fresh install
    with empty config still has somewhere to dial.
    """
    if user_configured:
        return list(user_configured)
    return list(COMMUNITY_BOOTSTRAP_PEERS)
