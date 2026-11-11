"""
P2P Communication Layer for Autonet (py-libp2p 0.8.1).

The network is a mesh (docs/p2p_networking.md):

- every host listens on TCP and WebSocket at once and dials whichever a
  peer offers (TCP preferred; ws/wss for proxied machines and relays
  behind an HTTPS proxy such as Cloudflare);
- relays run circuit relay v2 (hop, with reservations); a NAT'd daemon
  reserves a slot on the relays it is connected to and advertises
  ``/p2p-circuit`` addresses, so anyone can reach it through a relay;
- DCUtR upgrades a relayed connection to a direct TCP one by hole punching;
- a Kademlia DHT on every daemon finds peers and blob providers: whoever
  holds a blob a tool needs announces it (``provide``) and fetchers dial
  the providers they find, directly or through a circuit;
- a peer cache in the data dir lets a restart rejoin with every seed down;
  the mesh loop dials DHT-found and capability peers up to a target count
  and the connection manager prunes above a cap.

Gossip (GossipSub), capability advertisement, service calls, batch
catch-up, blob offers and the reach check ride on the same connections.
Uses trio as the async runtime (required by py-libp2p).

Usage:
    from nodes.common.p2p import AutonetHost
    host = AutonetHost(node_id="daemon", listen_port=4001)
    async with host.run():
        await host.publish_topic(topic, data)
"""

import hashlib
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from . import p2p_mesh

try:
    import trio
    from multiaddr import Multiaddr

    import libp2p
    from libp2p import new_host
    from libp2p.custom_types import TProtocol
    from libp2p.abc import IHost
    from libp2p.peer.id import ID as PeerID
    from libp2p.peer.peerinfo import PeerInfo, info_from_p2p_addr
    from libp2p.host.ping import PingService
    from libp2p.pubsub.gossipsub import GossipSub
    from libp2p.pubsub.pubsub import Pubsub
    from libp2p.tools.anyio_service import TrioManager, background_trio_service

    _P2P_AVAILABLE = True
except ImportError:
    _P2P_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Framed I/O helpers
# =============================================================================


# Noise protocol has a 65535-byte message limit. Yamux adds a 12-byte
# header per frame. Keep chunks well under the limit.
_MAX_WRITE_CHUNK = 65000


async def _write_framed(stream, data: bytes):
    """Write a length-prefixed frame: [4-byte big-endian length][payload].

    Chunks writes at 65535 bytes to respect libp2p stream limits.
    """
    header = struct.pack(">I", len(data))
    await stream.write(header)
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + _MAX_WRITE_CHUNK]
        await stream.write(chunk)
        offset += len(chunk)
    await stream.close()


# A peer announces each frame's length; refuse it at the header instead of
# reading whatever is announced. Blob replies (anchored state included) may
# be large; every other message is small.
MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_BLOB_FRAME_BYTES = 256 * 1024 * 1024


def _check_length(length: int, max_len: int) -> None:
    if length > max_len:
        raise ValueError(f"frame of {length} bytes exceeds the {max_len}-byte cap")


async def _read_framed(stream, max_len: int = MAX_FRAME_BYTES) -> bytes:
    """Read a length-prefixed frame and return the payload."""
    header = await stream.read(4)
    if len(header) < 4:
        raise ValueError("Truncated frame header")
    length = struct.unpack(">I", header)[0]
    _check_length(length, max_len)
    if length == 0:
        await stream.close()
        return b""
    # Read in chunks
    parts = []
    remaining = length
    while remaining > 0:
        chunk = await stream.read(min(remaining, _MAX_WRITE_CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    await stream.close()
    return b"".join(parts)


async def _write_frame_no_close(stream, data: bytes):
    """Write a length-prefixed frame without closing the stream (for request-response)."""
    header = struct.pack(">I", len(data))
    await stream.write(header)
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + _MAX_WRITE_CHUNK]
        await stream.write(chunk)
        offset += len(chunk)


async def _read_frame_no_close(stream, max_len: int = MAX_FRAME_BYTES) -> bytes:
    """Read a length-prefixed frame without closing the stream (for request-response)."""
    header = await stream.read(4)
    if len(header) < 4:
        raise ValueError("Truncated frame header")
    length = struct.unpack(">I", header)[0]
    _check_length(length, max_len)
    if length == 0:
        return b""
    parts = []
    remaining = length
    while remaining > 0:
        chunk = await stream.read(min(remaining, _MAX_WRITE_CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


# =============================================================================
# Protocol IDs
# =============================================================================

WEIGHTS_PROTOCOL = TProtocol("/autonet/weights/1.0.0")
ACTIVATIONS_PROTOCOL = TProtocol("/autonet/activations/1.0.0")
CAPABILITY_PROTOCOL = TProtocol("/autonet/capability/1.0.0")
def _pruned(content_hash: str) -> bool:
    """A pruned tool's blob (Decision 2026-11-10): never announced or fetched."""
    from .pruned_tools import is_pruned
    return is_pruned(content_hash)


BLOB_PROTOCOL = TProtocol("/autonet/blob/1.0.0")
# A blob request frame is "<hash>" (a "|<hops>" suffix from older daemons is
# ignored). A peer answers from what it holds and never forwards: a fetcher
# finds the holders through the DHT (``provide`` / ``find_providers``) and
# dials them, directly or through a relay circuit.
EMBEDDING_PROTOCOL = TProtocol("/autonet/embeddings/1.0.0")
INFERENCE_REQUEST_PROTOCOL = TProtocol("/rpb/inference/1.0.0")
# Robot poll rail (docs/robots.md, 2026-10-25): polls, link pushes and
# receipts between daemons, one JSON request and one JSON response a stream.
ROBOT_PROTOCOL = TProtocol("/autonet/robot/1.0.0")
# Paid service calls (docs/services_market.md, docs/p2p_networking.md): one
# JSON envelope {"target": <peer id>, "hops": n, "timeout": s, "msg":
# <service_request>} and one JSON reply a stream. A peer that is not the
# target forwards it to the target, but only if it is connected to it and
# the envelope allows a hop, so a NAT'd provider is reachable through a
# relay both sides dialled out to. A missing "hops" means direct only.
SERVICE_PROTOCOL = TProtocol("/autonet/service/1.0.0")
SERVICE_RELAY_MAX_HOPS = 1
# Reachability check: we send our listen port, the peer answers with the IP
# it sees us on and whether a fresh TCP connect to that IP and port worked.
# Only the requester's own observed IP is ever dialled.
REACH_PROTOCOL = TProtocol("/autonet/reach/1.0.0")
# Batch catch-up (issue #41): {topic, since, have} in, {batches} out, the
# batches a peer (daemon or relay) holds for that topic signed after
# ``since``. Not forwarded: relays answer from their own log.
BATCHES_PROTOCOL = TProtocol("/autonet/batches/1.0.0")
# Blob offers (issue #44): {"cids": [...]} in, {"kept": n} out. The
# offering peer names blobs it holds; a peer that keeps blobs for others (a
# relay) fetches the ones it lacks back from the offerer, direct only, and
# stores them, so the state an anchor commits outlives the daemons that
# computed it. Everyone else answers {"kept": 0}.
BLOB_OFFER_PROTOCOL = TProtocol("/autonet/blob-offer/1.0.0")
BLOB_OFFER_MAX_CIDS = 16
# Peers that speak one of these are Autonet peers worth a mesh dial.
KAD_OR_AUTONET_PROTOCOLS = {"/ipfs/kad/1.0.0", str(BLOB_PROTOCOL), str(SERVICE_PROTOCOL)}

# What a peer knows of another (issue: service streams): {"peer": id} in,
# {"peer", "connected", "reachable", "addrs", "roles", "timestamp"} out,
# from the capability that peer sent it (kept under its own peer id). A
# buyer asks its relays, so it can dial a reachable provider directly and
# pick a relay that is connected to a NAT'd one.
PEER_INFO_PROTOCOL = TProtocol("/autonet/peer-info/1.0.0")
# How long a relay's answer about a peer is trusted for ``peer_reachable``,
# and the least time between two background lookups of one peer.
PEER_INFO_TTL = 600.0
PEER_INFO_RETRY = 120.0
GUILD_GOSSIP_TOPIC_PREFIX = "/autonet/guild/"
PROVIDER_REGISTRY_TOPIC = "/rpb/provider-registry/"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class AgentAdvertisement:
    """Metadata for a single agent running on this node."""
    address: str                                           # on-chain wallet address
    name: str = ""
    description: str = ""
    agent_type: str = ""                                   # general, explore, implement, etc.
    model: str = ""                                        # e.g. "claude-opus-4-6"
    is_root: bool = False
    parent_address: str = ""
    registered_on_chain: bool = False
    # Phase 7.5b: per-agent substrate-backed inference advertisement.
    # Same shape as NodeCapability.inference but scoped to one agent
    # so a daemon hosting multiple agents can advertise different
    # offers per agent. Empty dict = this agent doesn't serve
    # inference. See discover_inference_providers for filtering.
    inference: Dict = field(default_factory=dict)


@dataclass
class InferenceProviderMatch:
    """A single agent on a remote peer offering inference. Phase 7.5b
    discovery returns these instead of NodeCapability so that callers
    can talk to a specific agent's offer rather than a daemon-wide
    blob.

    Attributes
    ----------
    peer_id:        libp2p peer id of the daemon hosting the agent
    agent_address: 0x address of the agent (used for payment routing
                    via Substrate.payForService)
    renderer_model: e.g. "qwen3:4b"; same field as in the inference dict
    price_atn:      per-probe price scaled like the on-chain side
    inference:      full inference dict (carries any extras)
    capability:     the originating NodeCapability (peer-level info,
                    listen_addrs, etc.)
    latency_ms:     EMA RTT to the daemon, or None if unmeasured
    """
    peer_id: str
    agent_address: str
    renderer_model: str
    price_atn: int
    inference: Dict
    capability: "NodeCapability"
    latency_ms: Optional[float] = None


@dataclass
class ModelState:
    """VL-JEPA model state — gossipped with NodeCapability.

    Every node that runs training cycles populates this from the on-chain
    RPB state and local training metrics.  Nodes that don't train leave
    this as defaults (version=0) and rely on peers for the latest state.
    """
    # On-chain (from RPB contract via discover_jurisdiction)
    model_version: int = 0                                # RPB.modelVersion()
    model_hash: str = ""                                  # RPB.currentModelHash() hex
    architecture_hash: str = ""                           # RPB.modelArchitectureHash() hex
    current_epoch: int = 0                                # RPB.currentEpoch()
    total_training_tokens: int = 0                        # RPB.totalTrainingTokens()
    training_reward_pool: int = 0                         # RPB.trainingRewardPool() (wei)
    total_inference_revenue: int = 0                      # RPB.totalInferenceRevenue() (wei)
    # Local training metrics (from last training cycle)
    local_cycles_completed: int = 0
    last_loss: float = 0.0
    last_cosine_similarity: float = 0.0
    architecture: str = ""                                # "text_jepa", "vl_jepa", etc.
    param_count: int = 0                                  # total model parameters
    # Latest weight delta (blob store CID, for aggregator discovery)
    latest_delta_cid: str = ""
    # Aggregate (computed from peer gossip)
    known_contributors: int = 0                           # unique peers with cycles > 0


@dataclass
class NodeCapability:
    """A node's advertised capabilities."""
    peer_id: str
    node_id: str
    roles: List[str] = field(default_factory=list)       # e.g. ["solver", "aggregator", "inference-provider"]
    gpu_type: str = ""                                    # e.g. "RTX 4090"
    gpu_memory_mb: int = 0
    bandwidth_mbps: float = 0.0
    modules_hosted: List[str] = field(default_factory=list)  # module IDs
    listen_addrs: List[str] = field(default_factory=list)
    agents: List[Dict] = field(default_factory=list)      # list of AgentAdvertisement dicts
    model_state: Dict = field(default_factory=dict)       # ModelState as dict
    # Inference advertisement. Loose dict so the wire format can evolve.
    # Set when this node serves /rpb/inference/1.0.0. Empty dict means
    # "this node is not advertising inference."
    # Conventional keys (callers may add others):
    #   - "renderer_model": str  (e.g. "qwen3:4b")
    #   - "price_atn":      int  (ATN per probe, scaled to uint256
    #                              like the chain side)
    #   - "agent_address":  str  (0x address that should receive
    #                              payForService funds)
    #   - "schema":         int  (1, for forward-compat)
    inference: Dict = field(default_factory=dict)
    # Whether this node announces an address peers can dial (configured
    # announce addresses, or one the reach check confirmed). A NAT'd node
    # is reached only over connections it opened, or through a relay.
    reachable: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "NodeCapability":
        d = json.loads(data.decode("utf-8"))
        # Keys this build does not know are dropped, not fatal.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GuildMembership:
    """Minimal guild abstraction for gossip/routing (no on-chain contracts yet)."""
    guild_id: str
    member_peer_ids: Set[str] = field(default_factory=set)
    modules: List[str] = field(default_factory=list)  # modules this guild trains

    @property
    def topic(self) -> str:
        return f"{GUILD_GOSSIP_TOPIC_PREFIX}{self.guild_id}"


@dataclass
class PeerLatency:
    """Latency tracking for a single peer."""
    peer_id: str
    rtt_ms: float = 0.0
    ema_rtt_ms: float = 0.0
    samples: int = 0
    last_ping: float = 0.0
    reachable: bool = True


# =============================================================================
# PeerLatencyTracker (Story 4.2)
# =============================================================================


class PeerLatencyTracker:
    """
    Tracks RTT to known peers using libp2p ping.
    Uses exponential moving average (EMA) for smoothing.
    """

    def __init__(self, host: IHost, ema_alpha: float = 0.3):
        self._host = host
        self._ping = PingService(host)
        self._alpha = ema_alpha
        self._peers: Dict[str, PeerLatency] = {}

    async def ping_peer(self, peer_id: PeerID) -> Optional[float]:
        """Ping a peer and return RTT in milliseconds. None if unreachable."""
        pid_str = str(peer_id)
        try:
            start = time.monotonic()
            await self._ping.ping(peer_id)
            rtt_ms = (time.monotonic() - start) * 1000.0

            if pid_str not in self._peers:
                self._peers[pid_str] = PeerLatency(peer_id=pid_str)

            entry = self._peers[pid_str]
            entry.rtt_ms = rtt_ms
            entry.samples += 1
            entry.last_ping = time.time()
            entry.reachable = True

            # EMA update
            if entry.samples == 1:
                entry.ema_rtt_ms = rtt_ms
            else:
                entry.ema_rtt_ms = (
                    self._alpha * rtt_ms + (1 - self._alpha) * entry.ema_rtt_ms
                )

            return rtt_ms
        except Exception as e:
            logger.debug(f"Ping failed for {pid_str[:16]}: {e}")
            if pid_str in self._peers:
                self._peers[pid_str].reachable = False
            return None

    async def ping_all(self, peer_ids: List[PeerID]):
        """Ping all peers concurrently."""
        async with trio.open_nursery() as nursery:
            for pid in peer_ids:
                nursery.start_soon(self.ping_peer, pid)

    def select_fastest_peers(self, n: int) -> List[PeerLatency]:
        """Return the n lowest-latency reachable peers."""
        reachable = [p for p in self._peers.values() if p.reachable and p.samples > 0]
        reachable.sort(key=lambda p: p.ema_rtt_ms)
        return reachable[:n]

    def select_peers_for_route(
        self, module_ids: List[str], capabilities: Dict[str, NodeCapability]
    ) -> List[Tuple[str, str]]:
        """
        For inference pipeline: pick lowest-latency peer hosting each module.

        Returns list of (module_id, peer_id) pairs in order.
        """
        route = []
        for module_id in module_ids:
            candidates = []
            for pid, cap in capabilities.items():
                if module_id in cap.modules_hosted:
                    latency = self._peers.get(pid)
                    if latency and latency.reachable:
                        candidates.append((pid, latency.ema_rtt_ms))

            if candidates:
                candidates.sort(key=lambda x: x[1])
                route.append((module_id, candidates[0][0]))
            else:
                route.append((module_id, ""))  # No peer found
        return route

    def get_latency(self, peer_id: str) -> Optional[PeerLatency]:
        return self._peers.get(peer_id)

    @property
    def all_latencies(self) -> Dict[str, PeerLatency]:
        return dict(self._peers)


# =============================================================================
# AutonetHost (Stories 4.1 - 4.7)
# =============================================================================


def _is_public_ip(ip: str) -> bool:
    """A LAN, loopback or tunnel address seen by a peer is not worth
    announcing: the listen addresses already cover those peers."""
    import ipaddress
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def resolve_dns_multiaddr(addr: str) -> str:
    """Rewrite a leading /dns4, /dns6 or /dns component to /ip4 or /ip6.

    Returns the address unchanged when it has no such component or the
    name does not resolve.
    """
    parts = addr.split("/")
    if len(parts) < 3 or parts[1] not in ("dns", "dns4", "dns6"):
        return addr
    import socket
    family = {"dns4": socket.AF_INET, "dns6": socket.AF_INET6}.get(parts[1], 0)
    try:
        info = socket.getaddrinfo(parts[2], None, family)
    except OSError as e:
        logger.warning(f"Could not resolve {parts[2]} in {addr}: {e}")
        return addr
    if not info:
        return addr
    fam, ip = info[0][0], info[0][4][0]
    parts[1] = "ip6" if fam == socket.AF_INET6 else "ip4"
    parts[2] = ip
    return "/".join(parts)


TRANSPORTS = ("tcp", "ws", "tcp+ws")

# Circuit relay v2 limits on a relay (hop). A reservation lasts an hour and
# is refreshed. A relayed connection is unlimited unless the relay's
# operator caps it (Decision 2026-11-10: relays give bandwidth like a
# torrent seed, and a service stream may be a video call): no duration cap,
# no data cap. py-libp2p 0.8.1 reads a data limit of 0 as "no shared
# budget", but its per-reservation fallback then allows 0 bytes, so "no
# cap" is a byte count no circuit reaches. ``AutonetHost.relay_limits``
# holds the operator's caps (``atn relay --circuit-max-*``).
RELAY_RESERVATION_TTL = 3600
RELAY_UNLIMITED_BYTES = 1 << 62
RELAY_CIRCUIT_DURATION = 0            # seconds; 0 = no limit
RELAY_CIRCUIT_BYTES = RELAY_UNLIMITED_BYTES
RELAY_MAX_RESERVATIONS = 1024
RELAY_MAX_CIRCUITS_PER_PEER = 64
# How long a dial waits on the direct addresses before it also tries the
# peer's circuit addresses (both race; the first connection wins).
CIRCUIT_DIAL_DELAY = 2.0


class _ConnNotifee:
    """Swarm notifee: hands every new connection to the host (DHT feed,
    pubsub nudge, peer cache). Implements py-libp2p's INotifee shape."""

    def __init__(self, host: "AutonetHost"):
        self._host = host

    async def opened_stream(self, network, stream) -> None:
        pass

    async def closed_stream(self, network, stream) -> None:
        pass

    async def connected(self, network, conn) -> None:
        self._host._on_connected(conn)

    async def disconnected(self, network, conn) -> None:
        self._host._on_disconnected(conn)

    async def listen(self, network, multiaddr) -> None:
        pass

    async def listen_close(self, network, multiaddr) -> None:
        pass


class AutonetHost:
    """
    Main P2P host for Autonet nodes.

    Wraps a py-libp2p 0.8.1 host and runs the mesh around it: TCP and
    WebSocket listeners, circuit relay v2 (hop on relays, reservations on
    NAT'd daemons), DCUtR hole punching, a Kademlia DHT for peer and blob
    provider discovery, the peer cache, the mesh dialer and the connection
    cap. See docs/p2p_networking.md.

    P2P dependencies (libp2p, trio) ship with the core install.
    """

    def __init__(
        self,
        node_id: str,
        listen_port: int = 0,
        listen_host: str = "0.0.0.0",
        bootstrap_peers: Optional[List[str]] = None,
        enable_quic: bool = False,
        enable_upnp: bool = False,
        capability: Optional[NodeCapability] = None,
        announce_addrs: Optional[List[str]] = None,
        bootstrap_redial_interval: float = 30.0,
        auto_reach: bool = False,
        reach_interval: float = 300.0,
        key_seed: Optional[bytes] = None,
        transport: str = "tcp+ws",
        seed_source: Optional[Callable[[], Optional[List[str]]]] = None,
        seed_refresh_interval: float = 300.0,
        tcp_port: int = 0,
        relay_hop: bool = False,
        enable_dht: bool = True,
        peer_cache_path: Optional[str] = None,
        target_peers: int = 8,
        max_peers: int = 48,
        mesh_interval: float = 15.0,
        provide_interval: float = 1800.0,
        max_reservations: int = 2,
        advertise_listen_addrs: bool = True,
    ):
        self.node_id = node_id
        self.logger = logging.getLogger(f"P2P[{node_id}]")
        # ``listen_port`` is the port of the WebSocket listener (of the TCP
        # one with transport "tcp"): the port Registry seeds and the data
        # dir's p2p.port have always named. With both transports TCP sits on
        # ``tcp_port`` (0 = listen_port + 1 when listen_port is set, else
        # random).
        self._listen_port = listen_port
        self._tcp_port = tcp_port
        self._listen_host = listen_host
        self._bootstrap_peers = bootstrap_peers or []
        self._announce_addrs = announce_addrs or []
        self._announce_resolved: List[Multiaddr] = []
        # Without configured announce addrs, auto_reach finds our public
        # address (UPnP mapping and/or a peer's observation) and announces
        # it once a peer confirms it can connect to it.
        self._auto_reach = auto_reach
        self._reach_interval = reach_interval
        # A 32-byte ed25519 seed gives a stable peer id across restarts (a
        # seed node's bootstrap address embeds it); None = a fresh id.
        self._key_seed = key_seed
        # "tcp+ws" (default): listen on both, dial whichever a peer offers.
        # A machine whose only egress is an HTTP(S)/SOCKS proxy cannot dial
        # TCP, so it runs WebSocket only and dials through the proxy
        # (nodes/common/ws_proxy.py). "both" is an alias of "tcp+ws".
        transport = "tcp+ws" if transport in ("both", "ws+tcp") else transport
        if transport not in TRANSPORTS:
            raise ValueError(f"unknown p2p transport {transport!r}")
        self._proxy = p2p_mesh.proxy_from_env()
        if self._proxy and "tcp" in transport:
            if transport == "tcp":
                self.logger.warning("egress proxy set but transport is tcp: "
                                    "TCP dials will not go through it")
            else:
                transport = "ws"
                self.logger.info("egress proxy set: WebSocket only, dialled "
                                 "through the proxy")
        self._transport = transport
        if "ws" in transport:
            # HTTP CONNECT and SOCKS5 proxies for the ws dialer, and the
            # per-connection task groups (ws_proxy._guarded).
            from .ws_proxy import install as _install_ws_proxy
            _install_ws_proxy()
        self._bootstrap_redial_interval = bootstrap_redial_interval
        # The seed list is re-read from ``seed_source`` (the jurisdiction's
        # Registry, blocking; run off the trio thread) every
        # ``seed_refresh_interval``, so a relay the DAO adds later is
        # dialled without a restart. None or <= 0 = the list given here.
        self._seed_source = seed_source
        self._seed_refresh_interval = seed_refresh_interval
        # While no peer is connected the seeds (and cached peers) are
        # redialled at once and then with backoff from this interval up to
        # ``_alone_max_backoff``.
        self._alone_check_interval = 5.0
        self._alone_max_backoff = 120.0
        self._enable_quic = enable_quic
        self._enable_upnp = enable_upnp
        # Mesh (docs/p2p_networking.md, "Mesh").
        self._relay_hop = relay_hop
        self._enable_dht = enable_dht
        self._peer_cache = p2p_mesh.PeerCache(peer_cache_path)
        self._target_peers = max(1, int(target_peers))
        self._max_peers = max(self._target_peers, int(max_peers))
        self._mesh_interval = mesh_interval
        self._provide_interval = provide_interval
        self._max_reservations = max(0, int(max_reservations))
        # False advertises circuit (and announced) addresses only: a test
        # and harness knob that makes a loopback host behave as NAT'd.
        self._advertise_listen_addrs = advertise_listen_addrs
        # New connections are not pruned for this long (seconds).
        self.prune_grace = 30.0

        # Handlers registered by user code
        self._weight_handler: Optional[Callable] = None
        self._activation_handler: Optional[Callable] = None
        self._guild_handlers: Dict[str, Callable] = {}
        self._blob_handler: Optional[Callable] = None       # async (hash: str) -> Optional[bytes]
        self._blob_relay_timeout = 5.0
        self._embedding_handler: Optional[Callable] = None  # async (peer_id, metadata, tensor_bytes)
        self._service_handler: Optional[Callable] = None    # async (msg: dict) -> dict, in trio
        self._batch_sources: Dict[str, Callable] = {}       # topic -> serve(request) -> dict
        self._blob_offer_store: Optional[Tuple[Callable, Callable]] = None  # (has(cid), put(bytes)), relays only
        # Longest a forwarding peer waits for the target's reply (the
        # requester's own timeout is honoured when it is shorter).
        self.service_relay_timeout = 600.0
        # The relay operator's caps on relayed connections, set before run()
        # (atn relay): {"duration": s, "bytes": n, "circuits": n per daemon};
        # a missing or 0 value keeps the default (no limit, see
        # RELAY_CIRCUIT_*).
        self.relay_limits: Dict[str, int] = {}
        self._service_session_handler: Optional[Any] = None
        self._peer_info_cache: Dict[str, Tuple[float, dict]] = {}
        self._peer_info_due: Dict[str, float] = {}

        # State (initialized in run())
        self._host: Optional[IHost] = None
        self._latency_tracker: Optional[PeerLatencyTracker] = None
        self._gossipsub: Optional[GossipSub] = None
        self._pubsub: Optional[Pubsub] = None
        self._guild_subscriptions: Dict[str, Any] = {}
        self._known_capabilities: Dict[str, NodeCapability] = {}
        self._guilds: Dict[str, GuildMembership] = {}
        self._circuit_protocol: Any = None
        self._circuit: Any = None          # CircuitV2Transport (dials circuits)
        self._dcutr: Any = None
        self._dht: Any = None
        self._dht_mode = ""
        self._reservations: Dict[str, float] = {}   # relay peer id -> expiry
        self._provided: Set[str] = set()            # blob cids we provide
        self._provided_at: Dict[str, float] = {}
        self._dial_failures: Dict[str, Tuple[int, float]] = {}  # pid -> (n, retry_at)
        self._seed_ids: Set[str] = set()
        self.last_fetch_source = ""
        # Generic topic gossip (Phase 10.2). Reader tasks for arbitrary
        # gossipsub topics live in the host's main nursery; cross-thread
        # callers schedule publishes via the captured trio token.
        self._nursery: Optional["trio.Nursery"] = None
        self._trio_token: Optional["trio.lowlevel.TrioToken"] = None
        self._topic_subscriptions: Dict[str, Any] = {}
        self._topic_handlers: Dict[str, List[Callable[[str, bytes], None]]] = {}
        # Set inside run() once the trio token is captured so cross-thread
        # callers can wait for "host is ready to accept publishes".
        import threading as _threading
        self._ready_event = _threading.Event()

        # Our capability
        self._capability = capability or NodeCapability(
            peer_id="", node_id=node_id
        )
        if relay_hop and "relay" not in self._capability.roles:
            self._capability.roles.append("relay")

        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _listen_multiaddrs(self) -> List[Multiaddr]:
        h = self._listen_host
        if self._transport == "tcp":
            return [Multiaddr(f"/ip4/{h}/tcp/{self._listen_port}")]
        ws = Multiaddr(f"/ip4/{h}/tcp/{self._listen_port}/ws")
        if self._transport == "ws":
            return [ws]
        tcp = self._tcp_port or (self._listen_port + 1 if self._listen_port else 0)
        return [Multiaddr(f"/ip4/{h}/tcp/{tcp}"), ws]

    @asynccontextmanager
    async def run(self):
        """
        Start the P2P host and all services.

        Usage:
            async with host.run():
                # host is now listening and discoverable
                await host.publish_topic(...)
        """
        from libp2p.network.config import ConnectionConfig, RetryConfig

        listen_addrs = self._listen_multiaddrs()
        key_pair = None
        if self._key_seed is not None:
            from libp2p.crypto.ed25519 import create_new_key_pair
            key_pair = create_new_key_pair(self._key_seed)
        # The connection manager: prune above ``max_peers`` (seeds and
        # relays are protected); our mesh loop does the dialling, so the
        # library's auto-connector is stopped once the swarm runs.
        conn_cfg = ConnectionConfig(
            max_connections=self._max_peers * 2 + 8,
            min_connections=0,
            low_watermark=0,
            high_watermark=self._max_peers,
            dial_timeout=8.0,
            grace_period=self.prune_grace,
        )
        self._host = new_host(
            key_pair=key_pair,
            listen_addrs=listen_addrs,
            enable_quic=self._enable_quic,
            # UPnP is handled in _auto_reach_loop.
            enable_upnp=False,
            connection_config=conn_cfg,
        )
        # Identify, signed peer records, DHT provider records and the relay
        # all read get_addrs. py-libp2p's addrs_factory cannot carry circuit
        # addresses (it strips everything from the first /p2p/ on), so the
        # host instance gets ours.
        self._host.get_addrs = self._get_addrs
        swarm = self._host.get_network()
        # One quick retry per address: a peer that is down should not hold
        # a dial for half a minute before its circuit is tried.
        swarm.retry_config = RetryConfig(max_retries=1, initial_delay=0.2, max_delay=1.0)
        self._install_dial_fallback(swarm)
        swarm.register_notifee(_ConnNotifee(self))

        async with self._host.run(listen_addrs=listen_addrs):
            self._running = True
            try:
                await swarm.auto_connector.stop()
            except Exception:  # noqa: BLE001 - absent in a future version
                pass
            peer_id = str(self._host.get_id())
            if self._announce_addrs:
                self._announce_resolved = self._resolve_announce_addrs()
            addrs = self._host.get_addrs()

            # Update our capability with actual peer ID and addresses
            self._capability.peer_id = peer_id
            self._capability.listen_addrs = [str(a) for a in addrs]
            self._capability.reachable = bool(self._announce_resolved)

            self.logger.info(
                f"P2P host started: {peer_id[:16]}... "
                f"listening on {[str(a) for a in addrs]}"
            )

            # Initialize subsystems
            self._latency_tracker = PeerLatencyTracker(self._host)
            self._setup_stream_handlers()
            # Note: GossipSub is initialized lazily in join_guild() /
            # subscribe_topic().

            # Capture the trio token + open a long-lived nursery so
            # background tasks (topic readers, etc.) can run for the
            # lifetime of the host. Cross-thread callers schedule
            # publishes via the captured token.
            self._trio_token = trio.lowlevel.current_trio_token()
            try:
                async with trio.open_nursery() as nursery:
                    self._nursery = nursery
                    await self._start_mesh_services()
                    refresh = (self._seed_source is not None
                               and self._seed_refresh_interval > 0)
                    if self._bootstrap_peers:
                        nursery.start_soon(self._dial_seeds, list(self._bootstrap_peers))
                    if ((self._bootstrap_peers or self._announce_addrs or refresh)
                            and self._bootstrap_redial_interval > 0):
                        nursery.start_soon(self._maintain_addrs)
                    if refresh:
                        nursery.start_soon(self._refresh_seeds_loop)
                    if self._bootstrap_peers or refresh or len(self._peer_cache):
                        nursery.start_soon(self._redial_when_alone)
                    if not self._announce_addrs and (self._auto_reach or self._enable_upnp):
                        nursery.start_soon(self._auto_reach_loop)
                    if self._mesh_interval > 0:
                        nursery.start_soon(self._mesh_loop)
                    self._ready_event.set()
                    try:
                        yield self
                    finally:
                        # Cancel background tasks (topic readers).
                        nursery.cancel_scope.cancel()
            finally:
                self._peer_cache.save()
                self._ready_event.clear()
                self._nursery = None
                self._trio_token = None
                self._running = False
                self.logger.info("P2P host shutting down")

    async def _start_service(self, service: Any) -> Any:
        """Run a py-libp2p Service for the host's lifetime."""
        manager = TrioManager(service)
        service._manager = manager
        self._nursery.start_soon(manager.run)
        await manager.wait_started()
        return manager

    async def _start_mesh_services(self) -> None:
        """Circuit relay v2, DCUtR and the DHT."""
        from libp2p.relay.circuit_v2.config import RelayConfig, RelayRole
        from libp2p.relay.circuit_v2.protocol import CircuitV2Protocol
        from libp2p.relay.circuit_v2.resources import RelayLimits
        from libp2p.relay.circuit_v2.transport import CircuitV2Transport

        caps = self.relay_limits or {}

        def cap(key: str, default: int) -> int:
            try:
                value = int(caps.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            return value if value > 0 else default
        limits = RelayLimits(
            duration=cap("duration", RELAY_CIRCUIT_DURATION),
            data=cap("bytes", RELAY_CIRCUIT_BYTES),
            max_circuit_conns=cap("circuits", RELAY_MAX_CIRCUITS_PER_PEER),
            max_reservations=RELAY_MAX_RESERVATIONS,
            reservation_ttl=RELAY_RESERVATION_TTL)
        self._circuit_protocol = CircuitV2Protocol(
            self._host, limits=limits, allow_hop=self._relay_hop)
        await self._start_service(self._circuit_protocol)
        # STOP only: we reserve on relays ourselves (_maintain_reservations),
        # not on every relay a dial happens to cross.
        self._circuit = CircuitV2Transport(
            self._host, self._circuit_protocol,
            RelayConfig(roles=RelayRole.STOP, limits=limits))
        if "tcp" in self._transport and not self._relay_hop:
            # Hole punching is TCP only (WebSocket cannot be punched).
            from libp2p.relay.circuit_v2.dcutr import DCUtRProtocol
            self._dcutr = DCUtRProtocol(self._host)
            await self._start_service(self._dcutr)
        if self._enable_dht:
            from libp2p.kad_dht.kad_dht import DHTMode, KadDHT
            mode = DHTMode.SERVER if self._dht_should_serve() else DHTMode.CLIENT
            # Subnet diversity off: NAT'd daemons share their relay's IP in
            # the circuit addresses they advertise.
            self._dht = KadDHT(self._host, mode, enable_random_walk=False,
                               max_peers_per_subnet=0)
            self._dht_mode = mode.value
            await self._start_service(self._dht)

    # =========================================================================
    # Addresses
    # =========================================================================

    def _get_addrs(self) -> List[Multiaddr]:
        """``_advertised_addrs`` with our ``/p2p/<peer id>`` appended."""
        own = f"/p2p/{self._host.get_id()}"
        out = []
        for a in self._advertised_addrs():
            s_ = str(a)
            try:
                out.append(Multiaddr(s_ if s_.endswith(own) else s_ + own))
            except Exception:
                continue
        return out

    def _advertised_addrs(self, candidates: Optional[List[Multiaddr]] = None) -> List[Multiaddr]:
        """What identify, signed peer records, DHT provider records and our
        capability carry: announced addresses
        (configured, or confirmed by the reach check) else our listen
        addresses with wildcards expanded, then a ``/p2p-circuit`` address
        per relay we hold a reservation on. Observed addresses are not
        advertised: behind NAT they name the wrong port."""
        out: List[str] = []
        if self._announce_resolved:
            out.extend(str(a) for a in self._announce_resolved)
        elif self._advertise_listen_addrs and self._host is not None:
            listen = [str(a) for a in self._host.get_transport_addrs()]
            # TCP first: DCUtR binds its hole-punch dials to the first one.
            listen.sort(key=p2p_mesh.is_ws)
            for a in listen:
                out.extend(p2p_mesh.expand_wildcard(a))
        out.extend(self._circuit_addr_strs())
        result, seen = [], set()
        for a in out:
            if a in seen:
                continue
            seen.add(a)
            try:
                result.append(Multiaddr(a))
            except Exception:
                continue
        return result

    def _circuit_addr_strs(self) -> List[str]:
        """``<relay addr>/p2p/<relay>/p2p-circuit`` for every live
        reservation, the relay's most dialable address first."""
        if self._host is None:
            return []
        now = time.time()
        out = []
        for rid, expires in list(self._reservations.items()):
            if expires <= now:
                continue
            addr = self._relay_dial_addr(rid)
            if addr:
                out.append(f"{addr}/p2p/{rid}/p2p-circuit")
        return out

    def _relay_dial_addr(self, relay_id: str) -> str:
        """The address others should dial a relay on: the seed address we
        were given for it, else the best one it advertised (public before
        private, never loopback unless that is all it has)."""
        for seed in self._bootstrap_peers:
            if seed.endswith(f"/p2p/{relay_id}"):
                return seed.split("/p2p/")[0]
        try:
            addrs = [str(a).split("/p2p/")[0]
                     for a in self._host.get_peerstore().addrs(PeerID.from_base58(relay_id))
                     if not p2p_mesh.is_circuit(a)]
        except Exception:
            return ""

        def rank(a: str) -> tuple:
            ip = p2p_mesh.addr_ip(a)
            return (p2p_mesh.is_loopback_addr(a), not _is_public_ip(ip) if ip else False,
                    p2p_mesh.is_ws(a))
        addrs.sort(key=rank)
        return addrs[0] if addrs else ""

    def _resolve(self, addr: str) -> str:
        """TCP dials only IPs, so names in TCP addresses resolve here.
        WebSocket addresses keep their names: TLS (SNI) and the HTTP Host
        header must carry them (an HTTPS proxy in front of a relay routes on
        them), and behind an HTTP proxy the proxy resolves them."""
        return addr if p2p_mesh.is_ws(addr) else resolve_dns_multiaddr(addr)

    def _resolve_announce_addrs(self) -> List[Multiaddr]:
        """The announce addresses with DNS names resolved, /p2p stripped."""
        announce = []
        for addr in self._announce_addrs:
            addr = self._resolve(addr).split("/p2p/")[0]
            try:
                announce.append(Multiaddr(addr))
            except Exception as e:
                self.logger.warning(f"Ignoring bad announce address {addr}: {e}")
        return announce

    async def _addrs_changed(self) -> None:
        """Our advertised addresses changed: refresh the capability, tell
        connected peers (identify push) and re-provide what we hold so the
        provider records carry the new addresses."""
        self._capability.listen_addrs = self.addrs
        try:
            await self._host._push_identify_to_all_peers()
        except Exception as e:  # noqa: BLE001 - best effort
            self.logger.debug("identify push failed: %s", e)
        await self.advertise_capability()
        if self._provided and self._nursery is not None:
            self._nursery.start_soon(self._reprovide, True)

    # =========================================================================
    # Dialling: direct addresses first, circuit addresses in the race
    # =========================================================================

    def _install_dial_fallback(self, swarm: Any) -> None:
        """Every dial (host.connect, new_stream, the DHT, gossip) reaches a
        peer through a relay circuit when its direct addresses fail.

        py-libp2p's Swarm.dial_peer skips /p2p-circuit addresses, so the
        swarm instance gets a dial_peer that races the original (direct)
        dial against the peer's circuit addresses, the circuit starting
        ``CIRCUIT_DIAL_DELAY`` later or as soon as the direct dial fails."""
        original = swarm.dial_peer
        in_flight: Dict[str, Any] = {}

        async def dial_peer(peer_id: Any) -> List[Any]:
            live = [c for c in swarm.get_connections(peer_id) if not c.is_closed]
            if live:
                return live
            key = str(peer_id)
            if key in in_flight:   # one dial per peer: join the running one
                await in_flight[key].wait()
                live = [c for c in swarm.get_connections(peer_id) if not c.is_closed]
                if live:
                    return live
            done = in_flight[key] = trio.Event()
            try:
                return await _dial(peer_id)
            except Exception:
                # Kademlia drops peers that do not answer: without this a
                # dead peer (a relay that went down, a daemon reachable only
                # through it) stalls every lookup until the hourly refresh.
                self._dht_forget(peer_id)
                raise
            finally:
                in_flight.pop(key, None)
                done.set()

        async def _dial(peer_id: Any) -> List[Any]:
            try:
                known = list(self._host.get_peerstore().addrs(peer_id))
            except Exception:
                known = []
            circuits = [a for a in known if p2p_mesh.is_circuit(a)
                        and not str(a).startswith(f"/p2p/{peer_id}")]
            if not circuits or self._circuit is None:
                return await original(peer_id)
            if not any(not p2p_mesh.is_circuit(a) for a in known):
                return [await self._dial_circuits(peer_id, circuits)]
            result: List[Any] = []
            errors: List[BaseException] = []
            direct_failed = trio.Event()
            async with trio.open_nursery() as n:
                async def go_direct():
                    try:
                        result.extend(await original(peer_id))
                        n.cancel_scope.cancel()
                    except Exception as e:
                        errors.append(e)
                        direct_failed.set()

                async def go_circuit():
                    with trio.move_on_after(CIRCUIT_DIAL_DELAY):
                        await direct_failed.wait()
                    try:
                        result.append(await self._dial_circuits(peer_id, circuits))
                        n.cancel_scope.cancel()
                    except Exception as e:
                        errors.append(e)
                n.start_soon(go_direct)
                n.start_soon(go_circuit)
            if result:
                return result
            from libp2p.network.exceptions import SwarmException
            raise SwarmException(f"unable to connect to {peer_id}: "
                                 + "; ".join(str(e) for e in errors))

        swarm.dial_peer = dial_peer

    async def _dial_circuits(self, peer_id: Any, circuits: List[Any]) -> Any:
        """A connection to ``peer_id`` through one of its relays (relays we
        are connected to first)."""
        connected = set(self.get_connected_peers())

        def relay_of(a: Any) -> str:
            return str(a).split("/p2p-circuit")[0].rsplit("/p2p/", 1)[-1]
        ordered = sorted(circuits, key=lambda a: relay_of(a) not in connected)
        last: Optional[BaseException] = None
        for addr in ordered:
            maddr = addr if isinstance(addr, Multiaddr) else Multiaddr(str(addr))
            if not str(maddr).endswith(f"/p2p/{peer_id}"):
                maddr = maddr.encapsulate(Multiaddr(f"/p2p/{peer_id}"))
            try:
                with trio.fail_after(20):
                    conn = await self._circuit.dial(maddr)
                self.logger.debug("connected to %s through relay %s",
                                  str(peer_id)[:16], relay_of(addr)[:16])
                return conn
            except Exception as e:  # noqa: BLE001 - try the next relay
                last = e
        raise ConnectionError(f"no relay circuit to {str(peer_id)[:16]}: {last}")

    def connection_kinds(self) -> Dict[str, List[str]]:
        """peer id -> the kind of each connection to it ("direct" or
        "relayed"), for status pages and the NAT harness."""
        out: Dict[str, List[str]] = {}
        if self._host is None:
            return out
        for pid, conns in self._host.get_network().get_connections_map().items():
            kinds = []
            for c in conns:
                try:
                    addrs = [str(a) for a in c.get_transport_addresses()]
                except Exception:
                    addrs = []
                kinds.append("relayed" if addrs and all(p2p_mesh.is_circuit(a) for a in addrs)
                             else "direct")
            out[str(pid)] = kinds
        return out

    def direct_peers(self) -> List[str]:
        return [p for p, kinds in self.connection_kinds().items() if "direct" in kinds]

    # =========================================================================
    # Connections: DHT feed, peer cache, relay reservations, the mesh loop
    # =========================================================================

    def _on_connected(self, conn: Any) -> None:
        """A new connection (either direction): after identify has run,
        feed the peer to the DHT and nudge pubsub."""
        nursery = self._nursery
        if nursery is None:
            return
        try:
            pid = conn.muxed_conn.peer_id
        except Exception:
            return
        nursery.start_soon(self._after_connect, pid)

    def _on_disconnected(self, conn: Any) -> None:
        """A connection closed: if our pubsub stream to that peer ran on it
        and the peer is still connected another way, reopen the stream on
        a live connection now; a peer we reached only through a relay
        leaves the DHT routing table when its last connection goes."""
        nursery = self._nursery
        try:
            pid = conn.muxed_conn.peer_id
        except Exception:
            return
        if nursery is None:
            return

        try:
            relayed = all(p2p_mesh.is_circuit(a) for a in conn.get_transport_addresses())
        except Exception:
            relayed = False

        async def reopen() -> None:
            await trio.sleep(0)   # let the swarm drop the closed connection
            if self._pubsub is not None and self._drop_stale_pubsub_stream(pid):
                await self._nudge_pubsub_peer(pid)
            if relayed and not self._host.get_network().get_connections(pid):
                # Reachable only through a relay that is gone: out of the
                # routing table until it connects again, so lookups do not
                # wait on it.
                self._dht_forget(pid)
        nursery.start_soon(reopen)

    async def _after_connect(self, pid: Any) -> None:
        await trio.sleep(1.5)  # identify fills the peerstore protocols
        if pid not in self._host.get_connected_peers():
            return
        await self._dht_add(pid)
        await self._nudge_pubsub_peer(pid)
        if self._max_reservations and not self._relay_hop and self._is_relay(pid):
            await self._maintain_reservations()

    def _peer_protocols(self, pid: Any) -> List[str]:
        try:
            return [str(p) for p in (self._host.get_peerstore().get_protocols(pid) or [])]
        except Exception:
            return []

    def _is_relay(self, pid: Any) -> bool:
        if p2p_mesh.HOP_PROTOCOL in self._peer_protocols(pid):
            return True
        cap = self._known_capabilities.get(str(pid))
        return bool(cap and "relay" in (cap.roles or []))

    def _dht_forget(self, pid: Any) -> None:
        if self._dht is None:
            return
        try:
            self._dht.routing_table.remove_peer(pid)
        except Exception:  # noqa: BLE001
            pass

    async def _dht_add(self, pid: Any) -> None:
        if self._dht is None:
            return
        try:
            if p2p_mesh.KAD_PROTOCOL in self._peer_protocols(pid):
                await self._dht.add_peer(pid)
        except Exception as e:  # noqa: BLE001 - routing table refused it
            self.logger.debug("dht add_peer %s: %s", str(pid)[:16], e)

    def _dht_should_serve(self) -> bool:
        """DHT server mode when peers can reach us: a relay, an announced
        or confirmed public address, a live relay reservation (reachable
        through the circuit) or a direct link to another daemon (it can
        query us over it). Client mode otherwise."""
        if self._relay_hop or self._announce_resolved:
            return True
        if any(exp > time.time() for exp in self._reservations.values()):
            return True
        if self._host is not None:
            relays = {str(p) for p in self._host.get_connected_peers() if self._is_relay(p)}
            if any(p not in relays for p in self.direct_peers()):
                return True
        return False

    async def _update_dht_mode(self) -> None:
        if self._dht is None:
            return
        from libp2p.kad_dht.kad_dht import DHTMode
        want = DHTMode.SERVER if self._dht_should_serve() else DHTMode.CLIENT
        if want.value != self._dht_mode:
            await self._dht.switch_mode(want)
            self._dht_mode = want.value
            self.logger.info("DHT mode: %s", want.value.lower())
            # Peers learn we (no longer) speak the DHT protocol from identify.
            try:
                await self._host._push_identify_to_all_peers()
            except Exception:  # noqa: BLE001
                pass
            if want == DHTMode.SERVER:
                for pid in self._host.get_connected_peers():
                    await self._dht_add(pid)

    @property
    def dht_mode(self) -> str:
        return self._dht_mode.lower()

    async def _maintain_reservations(self) -> None:
        """Hold a circuit relay v2 reservation on up to ``max_reservations``
        connected relays, refreshed before they expire; reservations on
        relays we lost are dropped. A change of the set changes our
        advertised circuit addresses."""
        if self._circuit is None or self._relay_hop or not self._max_reservations:
            return
        from libp2p.relay.circuit_v2.protocol import PROTOCOL_ID as HOP
        now = time.time()
        before = sorted(rid for rid, exp in self._reservations.items() if exp > now)
        connected = {str(p): p for p in self._host.get_connected_peers()}
        for rid in list(self._reservations):
            if rid not in connected or self._reservations[rid] <= now:
                self._reservations.pop(rid, None)
        relays = [p for s, p in connected.items() if self._is_relay(p)]
        relays.sort(key=lambda p: str(p) not in self._seed_ids)
        for pid in relays:
            rid = str(pid)
            held = self._reservations.get(rid, 0)
            if held - now > RELAY_RESERVATION_TTL / 4:
                continue
            if rid not in self._reservations and len(self._reservations) >= self._max_reservations:
                continue
            try:
                with trio.fail_after(15):
                    stream = await self._host.new_stream(pid, [HOP])
                    ok = await self._circuit._make_reservation(stream, pid)
                    await stream.close()
            except Exception as e:  # noqa: BLE001
                self.logger.debug("reservation on %s failed: %s", rid[:16], e)
                ok = False
            if ok:
                expires = float(self._circuit._reservations.get(pid) or 0)
                self._reservations[rid] = expires or now + RELAY_RESERVATION_TTL
                self._protect(pid, "relay")
        after = sorted(rid for rid, exp in self._reservations.items() if exp > now)
        if after != before:
            self.logger.info("relay reservations: %s", [r[:16] for r in after] or "none")
            await self._update_dht_mode()
            await self._addrs_changed()

    @property
    def reservations(self) -> List[str]:
        now = time.time()
        return [rid for rid, exp in self._reservations.items() if exp > now]

    def _protect(self, pid: Any, tag: str) -> None:
        try:
            self._host.get_network().protect(pid, f"autonet-{tag}")
        except Exception:
            pass

    async def _mesh_loop(self) -> None:
        """Every ``mesh_interval``: keep relay reservations, feed the DHT,
        remember connected peers, dial towards ``target_peers`` from
        capability gossip, the DHT and the peer cache, and re-provide."""
        rounds = 0
        while True:
            await trio.sleep(self._mesh_interval if rounds else 2.0)
            rounds += 1
            try:
                await self._mesh_round(walk=(rounds % 4 == 1))
            except trio.Cancelled:
                raise
            except Exception as e:  # noqa: BLE001 - next round tries again
                self.logger.debug("mesh round failed: %s", e)

    async def _mesh_round(self, walk: bool = False) -> None:
        await self._maintain_reservations()
        connected = self._host.get_connected_peers()
        for pid in connected:
            await self._dht_add(pid)
            try:
                addrs = [str(a) for a in self._host.get_peerstore().addrs(pid)]
            except Exception:
                addrs = []
            self._peer_cache.note(str(pid), addrs)
        self._peer_cache.save()
        await self._update_dht_mode()
        if walk and self._dht is not None and self._dht.get_routing_table_size():
            # A lookup for a random key: the replies name peers (with their
            # addresses, circuit ones included) we did not know.
            import secrets
            with trio.move_on_after(20):
                try:
                    await self._dht.peer_routing.find_closest_peers_network(
                        secrets.token_bytes(32))
                except Exception as e:  # noqa: BLE001
                    self.logger.debug("dht walk failed: %s", e)
        n_connected = len(self._host.get_connected_peers())
        if n_connected < self._target_peers:
            await self._dial_candidates(self._target_peers - n_connected)
        elif n_connected > self._max_peers:
            # py-libp2p prunes only when a connection is added; once the
            # grace period of the newest ones is over, prune here.
            try:
                await self._host.get_network().connection_pruner.maybe_prune_connections()
            except Exception as e:  # noqa: BLE001
                self.logger.debug("prune failed: %s", e)
        if self._provide_interval > 0:
            await self._reprovide(False)

    def _mesh_candidates(self) -> List[Tuple[str, List[str]]]:
        """Peers worth dialling, as (peer id, addresses): capability gossip,
        the DHT routing table, the peerstore, then the peer cache."""
        own = self.peer_id
        connected = set(self.get_connected_peers())
        out: Dict[str, List[str]] = {}

        def add(pid: str, addrs: List[str]) -> None:
            if pid and pid != own and pid not in connected and pid not in out:
                out[pid] = [a.split(f"/p2p/{pid}")[0] if a.endswith(f"/p2p/{pid}") else a
                            for a in addrs]
        for cap in list(self._known_capabilities.values()):
            add(cap.peer_id, list(cap.listen_addrs or []))
        if self._dht is not None:
            try:
                for info in self._dht.routing_table.get_peer_infos():
                    add(str(info.peer_id), [str(a) for a in info.addrs])
            except Exception:
                pass
        try:
            ps = self._host.get_peerstore()
            for pid in ps.peers_with_addrs():
                if KAD_OR_AUTONET_PROTOCOLS.intersection(self._peer_protocols(pid)):
                    add(str(pid), [str(a) for a in ps.addrs(pid)])
        except Exception:
            pass
        for pid, addrs in self._peer_cache.entries():
            add(pid, addrs)
        return list(out.items())

    async def _dial_candidates(self, want: int) -> int:
        """Dial up to ``want`` new peers concurrently; failures back off
        (30 s doubling to 30 min per peer)."""
        now = time.time()
        picks = []
        for pid, addrs in self._mesh_candidates():
            fails, retry_at = self._dial_failures.get(pid, (0, 0.0))
            if retry_at > now:
                continue
            picks.append((pid, addrs))
            if len(picks) >= want:
                break
        if not picks:
            return 0
        made = [0]

        async def one(pid: str, addrs: List[str]) -> None:
            try:
                peer = PeerID.from_base58(pid)
                maddrs = []
                for a in addrs:
                    try:
                        maddrs.append(Multiaddr(a))
                    except Exception:
                        pass
                if maddrs:
                    self._host.get_peerstore().add_addrs(peer, maddrs, 3600)
                with trio.fail_after(30):
                    await self._host.connect(PeerInfo(peer, []))
                made[0] += 1
                self._dial_failures.pop(pid, None)
                await self._nudge_pubsub_peer(peer)
            except Exception as e:  # noqa: BLE001
                fails = self._dial_failures.get(pid, (0, 0.0))[0] + 1
                self._dial_failures[pid] = (fails, time.time() + min(30 * 2 ** (fails - 1), 1800))
                self.logger.debug("mesh dial %s failed: %s", pid[:16], e)
        async with trio.open_nursery() as n:
            for pid, addrs in picks:
                n.start_soon(one, pid, addrs)
        if made[0]:
            self.logger.info("mesh: dialled %d new peer(s); %d connected",
                             made[0], len(self.get_connected_peers()))
        return made[0]

    @property
    def peer_cache(self) -> "p2p_mesh.PeerCache":
        return self._peer_cache

    # =========================================================================
    # Seeds
    # =========================================================================

    async def _maintain_addrs(self) -> None:
        """Track address changes, every ``bootstrap_redial_interval``.

        Announce names are resolved again; if the result changed (a
        dynamic-DNS or tunnel name now points elsewhere) we advertise the
        new address at once. Bootstrap peers we are not connected to are
        redialled at their freshly resolved address, so a daemon that
        started first, lost its link, or whose bootstrap peer moved would
        not stay cut off.
        """
        while True:
            await trio.sleep(self._bootstrap_redial_interval)
            if self._announce_addrs:
                fresh = await trio.to_thread.run_sync(self._resolve_announce_addrs)
                if fresh and [str(a) for a in fresh] != [str(a) for a in self._announce_resolved]:
                    self.logger.info(f"Announce address changed to {[str(a) for a in fresh]}")
                    self._announce_resolved = fresh
                    self._capability.reachable = bool(fresh)
                    await self._addrs_changed()
            await self._dial_seeds(list(self._bootstrap_peers))

    async def _dial_seeds(self, addrs: List[str]) -> None:
        """Dial the seeds in ``addrs`` we are not connected to, each at its
        freshly resolved address, concurrently. Seeds are protected from
        connection pruning."""
        connected = set(self.get_connected_peers())

        async def one(addr: str) -> None:
            pid = addr.split("/p2p/")[-1] if "/p2p/" in addr else ""
            if not pid or pid == self.peer_id:
                return
            self._seed_ids.add(pid)
            if pid in connected:
                return
            addr = await trio.to_thread.run_sync(self._resolve, addr)
            # Drop cached addresses so a peer that moved is dialled
            # only at its current address, not the stale one first.
            try:
                self._host.get_peerstore().clear_addrs(PeerID.from_base58(pid))
            except Exception:
                pass
            with trio.move_on_after(15):
                if await self.connect_to_peer(addr):
                    self._protect(PeerID.from_base58(pid), "seed")
        async with trio.open_nursery() as n:
            for addr in addrs:
                n.start_soon(one, addr)

    async def refresh_seeds(self) -> List[str]:
        """Re-read the seed list from ``seed_source`` and dial the new ones.

        A failed read or an empty list keeps the seeds we have: a daemon
        must not cut itself off because the RPC hiccuped or the key was
        cleared. Returns the seeds that were added.
        """
        if self._seed_source is None:
            return []
        try:
            fresh = await trio.to_thread.run_sync(self._seed_source)
        except Exception as e:
            self.logger.debug(f"Seed list read failed: {e}")
            return []
        fresh = list(dict.fromkeys(a.strip() for a in (fresh or []) if a and a.strip()))
        if not fresh:
            return []
        added = [a for a in fresh if a not in self._bootstrap_peers]
        if fresh != self._bootstrap_peers:
            self.logger.info(f"Seed list changed: {fresh}")
            self._bootstrap_peers = fresh
        if added:
            await self._dial_seeds(added)
        return added

    async def _refresh_seeds_loop(self) -> None:
        """Re-read the seed list every ``seed_refresh_interval``."""
        while True:
            await trio.sleep(self._seed_refresh_interval)
            await self.refresh_seeds()

    async def _redial_when_alone(self) -> None:
        """Redial the seeds and the cached peers as soon as no peer is
        connected.

        The redial in _maintain_addrs waits its interval (or is off); a
        daemon whose last link dropped should not sit cut off that long.
        The peer cache is what lets a restart rejoin when every seed is
        down. Repeated failures back off so dead peers are not hammered.
        """
        backoff = self._alone_check_interval
        while True:
            if self.get_connected_peers() or not (self._bootstrap_peers or len(self._peer_cache)):
                backoff = self._alone_check_interval
                await trio.sleep(self._alone_check_interval)
                continue
            async with trio.open_nursery() as n:
                n.start_soon(self._dial_seeds, list(self._bootstrap_peers))
                n.start_soon(self._dial_cached_peers)
            if self.get_connected_peers():
                continue
            await trio.sleep(backoff)
            backoff = min(backoff * 2, self._alone_max_backoff)

    async def _dial_cached_peers(self, limit: int = 16) -> int:
        """Dial the newest cached peers (circuit addresses included)."""
        entries = [(pid, addrs) for pid, addrs in self._peer_cache.entries()
                   if pid != self.peer_id][:limit]
        made = [0]

        async def one(pid: str, addrs: List[str]) -> None:
            try:
                peer = PeerID.from_base58(pid)
                self._host.get_peerstore().add_addrs(
                    peer, [Multiaddr(a) for a in addrs], 3600)
                with trio.fail_after(20):
                    await self._host.connect(PeerInfo(peer, []))
                made[0] += 1
                await self._nudge_pubsub_peer(peer)
            except Exception as e:  # noqa: BLE001
                self.logger.debug("cached peer %s unreachable: %s", pid[:16], e)
        async with trio.open_nursery() as n:
            for pid, addrs in entries:
                n.start_soon(one, pid, addrs)
        if made[0]:
            self.logger.info("rejoined through %d cached peer(s)", made[0])
        return made[0]

    # =========================================================================
    # Reachability: UPnP and the reach check
    # =========================================================================

    def _listen_ports(self) -> Dict[str, int]:
        """{"tcp": port, "ws": port} actually bound."""
        out: Dict[str, int] = {}
        for addr in self._host.get_transport_addrs():
            port = addr.value_for_protocol("tcp")
            if port:
                out["ws" if p2p_mesh.is_ws(addr) else "tcp"] = int(port)
        return out

    def _listen_port_actual(self) -> int:
        ports = self._listen_ports()
        return ports.get("tcp") or ports.get("ws") or 0

    async def _set_auto_announce(self, addrs: Optional[List[str]]) -> None:
        """Announce ``addrs`` (or go back to the listen addrs on None/[])
        and tell peers if that changed."""
        if isinstance(addrs, str):
            addrs = [addrs]
        fresh = [Multiaddr(a) for a in (addrs or [])]
        if [str(a) for a in fresh] == [str(a) for a in self._announce_resolved]:
            return
        self.logger.info(
            f"Reachable at {[str(a) for a in fresh]}" if fresh
            else "No longer reachable; announcing listen addrs"
        )
        self._announce_resolved = fresh
        self._capability.reachable = bool(fresh)
        await self._update_dht_mode()
        await self._addrs_changed()

    async def _auto_reach_loop(self) -> None:
        """Find and confirm our public addresses, every ``reach_interval``.

        With UPnP on, the router maps our listen ports and reports its
        public IP. Either way a connected peer is asked what IP it sees
        us on and whether it can connect to us there, once per listen port;
        we announce ``/ip4/<ip>/tcp/<port>`` (and ``.../ws`` for the
        WebSocket port) only for the ports it can reach. Repeating the
        check catches a changed public IP. A daemon nobody can reach keeps
        working NAT'd: it dials out and is reachable through its relays.
        """
        ports = self._listen_ports()
        upnp = None
        if self._enable_upnp and ports:
            try:
                from libp2p.discovery.upnp.upnp import UpnpManager
                upnp = UpnpManager()
                if not await upnp.discover():
                    upnp = None
                else:
                    for p in ports.values():
                        await upnp.add_port_mapping(p, "TCP")
            except Exception as e:  # miniupnpc missing or the router said no
                self.logger.info(f"UPnP unavailable: {e}")
                upnp = None
        try:
            while True:
                if upnp is not None:
                    # Leases expire and routers reboot: renew each round.
                    for p in ports.values():
                        await upnp.add_port_mapping(p, "TCP")
                peers = [p for p in self._host.get_connected_peers()
                         if str(p) in self.direct_peers()]
                verdicts: Dict[str, Dict] = {}
                for pid in peers[:3]:
                    for kind, port in ports.items():
                        if kind in verdicts:
                            continue
                        with trio.move_on_after(15):
                            v = await self.check_reachability(pid, port)
                            if v is not None:
                                verdicts[kind] = v
                    if len(verdicts) == len(ports):
                        break
                if not verdicts:
                    await trio.sleep(2.0)  # no peer answered yet; retry soon
                    continue
                announce = []
                for kind in ("tcp", "ws"):
                    v = verdicts.get(kind)
                    ip = (v or {}).get("ip") or ""
                    if v and v.get("reachable") and _is_public_ip(ip):
                        proto = "ip6" if ":" in ip else "ip4"
                        suffix = "/ws" if kind == "ws" else ""
                        announce.append(f"/{proto}/{ip}/tcp/{ports[kind]}{suffix}")
                await self._set_auto_announce(announce)
                await trio.sleep(self._reach_interval)
        finally:
            if upnp is not None:
                with trio.CancelScope(shield=True), trio.move_on_after(3):
                    for p in ports.values():
                        await upnp.remove_port_mapping(p, "TCP")

    async def check_reachability(self, peer_id: Any, port: int) -> Optional[Dict]:
        """Ask ``peer_id`` whether it can connect to us on ``port``.
        Returns {"ip": observed ip, "reachable": bool} or None on failure."""
        try:
            stream = await self._host.new_stream(peer_id, [REACH_PROTOCOL])
            await _write_frame_no_close(stream, json.dumps({"port": int(port)}).encode("utf-8"))
            data = await _read_frame_no_close(stream)
            await stream.close()
            return json.loads(data.decode("utf-8")) if data else None
        except Exception as e:
            self.logger.debug(f"Reachability check via {str(peer_id)[:16]} failed: {e}")
            return None

    async def _handle_reach_stream(self, stream):
        """Tell the requester its observed IP and whether we can connect
        to it on the port it named. We dial only that observed IP, and
        never answer over a relayed connection (we would see the relay)."""
        try:
            req = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            port = int(req.get("port", 0))
            remote = stream.get_remote_address()
            ip = remote[0] if remote else ""
            reachable = False
            if ip and 0 < port < 65536:
                with trio.move_on_after(5):
                    try:
                        conn = await trio.open_tcp_stream(ip, port)
                        await conn.aclose()
                        reachable = True
                    except OSError:
                        pass
            await _write_framed(stream, json.dumps({"ip": ip, "reachable": reachable}).encode("utf-8"))
        except Exception as e:
            self.logger.debug(f"Error handling reach stream: {e}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def peer_id(self) -> str:
        if self._host:
            return str(self._host.get_id())
        return ""

    def connected_relays(self) -> List[str]:
        """Connected peers that advertised the relay role."""
        return [p for p in self.get_connected_peers()
                if "relay" in (getattr(self._known_capabilities.get(p), "roles", None) or [])]

    @property
    def addrs(self) -> List[str]:
        if self._host:
            return [str(a) for a in self._host.get_addrs()]
        return []

    @property
    def latency_tracker(self) -> Optional[PeerLatencyTracker]:
        return self._latency_tracker

    @property
    def known_capabilities(self) -> Dict[str, NodeCapability]:
        return dict(self._known_capabilities)

    @property
    def host(self) -> Optional[IHost]:
        return self._host

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def _reconcile_pubsub_peers(self) -> None:
        """Periodically check that every connected libp2p peer is
        registered with pubsub. PubsubNotifee.connected delivery has
        been observed to lag on real networks; this loop catches the
        gap.

        Runs every 2s for the lifetime of the host nursery.
        """
        while True:
            try:
                if self._pubsub is not None and self._host is not None:
                    for pid in list(self._pubsub.peers.keys()):
                        if self._drop_stale_pubsub_stream(pid):
                            self.logger.debug(f"pubsub stream to {pid} moved off a closed connection")
                    connected = list(self._host.get_connected_peers())
                    known = set(self._pubsub.peers.keys())
                    for pid in connected:
                        if pid not in known:
                            self.logger.debug(
                                f"reconciling pubsub peer {pid} (was missing)"
                            )
                            await self._pubsub._handle_new_peer_safe(pid)
            except Exception as e:
                self.logger.debug(f"_reconcile_pubsub_peers iter: {e}")
            await trio.sleep(2.0)

    def _drop_stale_pubsub_stream(self, pid: Any) -> bool:
        """Forget our pubsub stream to ``pid`` when the connection it ran
        on has closed while another connection to the peer lives on (a
        relayed connection that DCUtR replaced with a direct one, or one
        whose relay went down). py-libp2p keeps the dead stream as long as
        the peer is connected at all, so gossip to that peer stopped. The
        caller opens a new stream on a live connection. Subscriptions the
        peer told us about are kept."""
        ps = self._pubsub
        stream = ps.peers.get(pid) if ps is not None else None
        if stream is None:
            return False
        muxed = getattr(stream, "muxed_conn", None)
        live = [c.muxed_conn for c in self._host.get_network().get_connections(pid)
                if not getattr(c, "is_closed", False) and getattr(c, "muxed_conn", None) is not None]
        if muxed is None or not live or any(m is muxed for m in live):
            return False
        ps.peers.pop(pid, None)
        queue = getattr(ps, "peer_queues", {}).pop(pid, None)
        if queue is not None:
            try:
                queue.close()
            except Exception:
                pass
        try:
            ps.router.remove_peer(pid)
        except Exception:
            pass
        return True

    async def _nudge_pubsub_peer(self, peer_id: Any) -> None:
        """Manually invoke pubsub's new-peer handler.

        py-libp2p's PubsubNotifee.connected() is supposed to fire on
        every new connection (initiator or dialed-to), enqueueing the
        peer for ``handle_peer_queue`` to consume. Empirically this
        delivery can lag tens of seconds (or never fire) on real
        networks like ZeroTier — observed in Phase 10.6 cross-machine
        smoke. As a workaround, after we explicitly connect to a peer,
        we directly call ``_handle_new_peer_safe`` to skip the queue
        and immediately negotiate the meshsub stream.
        """
        if self._pubsub is None:
            return
        try:
            await self._pubsub._handle_new_peer_safe(peer_id)
        except Exception as e:
            self.logger.debug(f"_nudge_pubsub_peer({peer_id}): {e}")

    async def connect_to_peer(self, multiaddr_str: str) -> bool:
        """Connect to a peer by multiaddr string."""
        if not self._host:
            return False
        try:
            maddr = Multiaddr(multiaddr_str)
            info = info_from_p2p_addr(maddr)
            # An explicit address is a fresh reason to dial: py-libp2p
            # refuses a peer for a minute after all its addresses failed.
            self._forget_dial_failure(info.peer_id)
            await self._host.connect(info)
            self.logger.info(f"Connected to {str(info.peer_id)[:16]}...")
            # Nudge pubsub to register the peer immediately. Workaround
            # for Phase 10.6 cross-machine smoke finding: PubsubNotifee
            # connected delivery can lag on real networks.
            await self._nudge_pubsub_peer(info.peer_id)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to connect to {multiaddr_str}: {e}")
            return False

    def _forget_dial_failure(self, peer_id: Any) -> None:
        try:
            self._host.get_network()._negative_peer_cache.evict(str(peer_id))
        except Exception:
            pass

    async def connect_to_info(self, peer_info: PeerInfo) -> bool:
        """Connect to a peer by PeerInfo."""
        if not self._host:
            return False
        try:
            await self._host.connect(peer_info)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to connect: {e}")
            return False

    def get_connected_peers(self) -> List[str]:
        """Return connected peer IDs."""
        if not self._host:
            return []
        return [str(pid) for pid in self._host.get_connected_peers()]

    def topic_peer_count(self, topic: str) -> int:
        """Peers known to be subscribed to ``topic`` (any thread)."""
        pubsub = self._pubsub
        if pubsub is None:
            return 0
        try:
            return len(pubsub.peer_topics.get(topic, ()) or ())
        except Exception:
            return 0

    # =========================================================================
    # Story 4.3: Weight Delta Transfer
    # =========================================================================

    def set_weight_handler(self, handler: Callable):
        """Set handler for incoming weight deltas: handler(peer_id, data_hash, data)"""
        self._weight_handler = handler

    async def send_weight_delta(
        self, target_peer_id: PeerID, delta_bytes: bytes
    ) -> bool:
        """
        Send weight delta directly to an aggregator peer.

        The delta bytes are prefixed with a 32-byte SHA-256 hash for
        integrity verification on the receiver side.
        """
        if not self._host:
            return False
        try:
            content_hash = hashlib.sha256(delta_bytes).digest()
            # Payload: [32-byte hash][delta_bytes]
            payload = content_hash + delta_bytes

            stream = await self._host.new_stream(
                target_peer_id, [WEIGHTS_PROTOCOL]
            )
            await _write_framed(stream, payload)

            self.logger.info(
                f"Sent weight delta ({len(delta_bytes)} bytes) "
                f"to {str(target_peer_id)[:16]}..."
            )
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send weight delta: {e}")
            return False

    async def _handle_weight_stream(self, stream):
        """Handle incoming weight delta stream."""
        try:
            data = await _read_framed(stream)

            if len(data) < 32:
                self.logger.warning("Received truncated weight delta")
                return

            received_hash = data[:32]
            payload = data[32:]
            actual_hash = hashlib.sha256(payload).digest()

            if received_hash != actual_hash:
                self.logger.warning(
                    f"Weight delta hash mismatch: "
                    f"expected {received_hash.hex()[:16]}, "
                    f"got {actual_hash.hex()[:16]}"
                )
                return

            remote_peer = str(stream.muxed_conn.peer_id)
            self.logger.info(
                f"Received weight delta ({len(payload)} bytes) "
                f"from {remote_peer[:16]}..."
            )

            if self._weight_handler:
                await self._weight_handler(
                    remote_peer, actual_hash.hex(), payload
                )
        except Exception as e:
            self.logger.warning(f"Error handling weight stream: {e}")

    # =========================================================================
    # Story 4.4: Activation Relay
    # =========================================================================

    def set_activation_handler(self, handler: Callable):
        """Set handler for incoming activations: handler(peer_id, metadata, tensor_bytes)"""
        self._activation_handler = handler

    async def send_activation(
        self,
        target_peer_id: PeerID,
        tensor_bytes: bytes,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Forward activation tensor to the next module's host.

        Frame format:
            [4-byte metadata_len][metadata_json][tensor_bytes]
        """
        if not self._host:
            return False
        try:
            meta_bytes = json.dumps(metadata or {}).encode("utf-8")
            meta_len = struct.pack(">I", len(meta_bytes))
            # Inner payload: [4-byte meta_len][meta_json][tensor_bytes]
            payload = meta_len + meta_bytes + tensor_bytes

            stream = await self._host.new_stream(
                target_peer_id, [ACTIVATIONS_PROTOCOL]
            )
            await _write_framed(stream, payload)

            self.logger.debug(
                f"Sent activation ({len(tensor_bytes)} bytes) "
                f"to {str(target_peer_id)[:16]}..."
            )
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send activation: {e}")
            return False

    async def _handle_activation_stream(self, stream):
        """Handle incoming activation relay stream."""
        try:
            data = await _read_framed(stream)

            if len(data) < 4:
                return

            meta_len = struct.unpack(">I", data[:4])[0]
            meta_bytes = data[4:4 + meta_len]
            tensor_bytes = data[4 + meta_len:]

            metadata = json.loads(meta_bytes.decode("utf-8"))
            remote_peer = str(stream.muxed_conn.peer_id)

            self.logger.debug(
                f"Received activation ({len(tensor_bytes)} bytes) "
                f"from {remote_peer[:16]}..."
            )

            if self._activation_handler:
                await self._activation_handler(
                    remote_peer, metadata, tensor_bytes
                )
        except Exception as e:
            self.logger.warning(f"Error handling activation stream: {e}")

    # =========================================================================
    # Story 4.5: Guild-local Gossip (GossipSub)
    # =========================================================================

    async def _ensure_gossipsub(self):
        """Lazily initialize and start GossipSub for guild gossip."""
        if self._pubsub is not None:
            return  # Already initialized
        if not self._host:
            return

        protocols = [TProtocol("/meshsub/1.1.0"), TProtocol("/meshsub/1.0.0")]
        # heartbeat_interval=1 is the libp2p default and is required
        # for mesh formation in any reasonable timeframe. The previous
        # value of 120s meant no GRAFT for 2 minutes, so messages
        # silently failed to propagate during that window. Phase 10.2
        # cross-machine smoke test caught this.
        self._gossipsub = GossipSub(
            protocols=protocols,
            degree=6,
            degree_low=4,
            degree_high=12,
            heartbeat_interval=1,
            # py-libp2p 0.8.1 rate-limits each SENDING peer per topic; a
            # relay forwards every daemon's batches, so 10/s would drop
            # gossip as soon as a few dozen daemons publish together.
            max_messages_per_topic_per_second=500.0,
        )
        self._pubsub = Pubsub(
            host=self._host,
            router=self._gossipsub,
            strict_signing=False,
        )
        # Both Pubsub AND GossipSub are async-services. Each has to be
        # run via its own manager — Pubsub.run() spawns handle_peer_queue
        # (notifee → add_peer); GossipSub.run() spawns heartbeat (mesh
        # GRAFT/PRUNE/IHAVE). Without these, peers connect but the
        # gossipsub mesh never forms. Phase 10.6.
        #
        # We use TrioManager.run_service via start_soon on the host's
        # nursery rather than background_trio_service so the managers'
        # lifetimes are bounded by the host (cancelled when the host
        # nursery is cancelled).
        if self._nursery is not None:
            self._pubsub_manager = await self._start_service(self._pubsub)
            self._gossipsub_manager = await self._start_service(self._gossipsub)
            # Pubsub-peer reconciliation. PubsubNotifee.connected
            # delivery can lag (or fail) on real networks; periodically
            # check that every connected peer is registered with
            # pubsub and nudge any that aren't. Cheap to run.
            self._nursery.start_soon(self._reconcile_pubsub_peers)

    async def join_guild(
        self,
        guild: GuildMembership,
        handler: Optional[Callable] = None,
    ):
        """
        Join a guild's gossip topic.

        Args:
            guild: Guild membership info
            handler: Callback for incoming messages: handler(guild_id, peer_id, data)
        """
        await self._ensure_gossipsub()
        if not self._pubsub:
            return

        self._guilds[guild.guild_id] = guild
        guild.member_peer_ids.add(self.peer_id)

        if handler:
            self._guild_handlers[guild.guild_id] = handler

        sub = await self._pubsub.subscribe(guild.topic)
        self._guild_subscriptions[guild.guild_id] = sub
        self.logger.info(f"Joined guild gossip: {guild.guild_id}")

    async def publish_guild_message(
        self, guild_id: str, data: bytes
    ) -> bool:
        """Publish a message to a guild's gossip topic."""
        if not self._pubsub or guild_id not in self._guilds:
            return False
        try:
            topic = self._guilds[guild_id].topic
            await self._pubsub.publish(topic, data)
            self.logger.debug(f"Published to guild {guild_id}: {len(data)} bytes")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to publish to guild {guild_id}: {e}")
            return False

    async def leave_guild(self, guild_id: str):
        """Leave a guild's gossip topic."""
        if guild_id in self._guild_subscriptions:
            # Unsubscribe
            sub = self._guild_subscriptions.pop(guild_id)
            if hasattr(sub, 'unsubscribe'):
                await sub.unsubscribe()
        self._guilds.pop(guild_id, None)
        self._guild_handlers.pop(guild_id, None)
        self.logger.info(f"Left guild: {guild_id}")

    # =========================================================================
    # Generic gossipsub topics (Phase 10.2)
    # =========================================================================

    async def publish_topic(self, topic: str, data: bytes) -> bool:
        """Publish bytes to an arbitrary gossipsub topic.

        Async — must be called from inside the host's trio loop. For
        cross-thread callers, use ``publish_topic_threadsafe``.
        """
        await self._ensure_gossipsub()
        if not self._pubsub:
            return False
        try:
            await self._pubsub.publish(topic, data)
            return True
        except Exception as e:
            self.logger.warning(f"publish_topic({topic}): {e}")
            return False

    def publish_topic_threadsafe(self, topic: str, data: bytes) -> None:
        """Schedule a topic publish from a non-trio thread.

        Returns immediately. Failures are logged, not raised.
        """
        token = self._trio_token
        nursery = self._nursery
        if token is None or nursery is None:
            self.logger.debug("publish_topic_threadsafe before host run; dropping")
            return

        async def _do():
            await self.publish_topic(topic, data)

        # Never wait for the trio thread: callers publish while holding the
        # world lock, and the trio thread's topic reader takes that lock to
        # ingest peer events, so a blocking hand-off deadlocks the daemon.
        # run_sync_soon only queues; callbacks run in order on the loop.
        try:
            token.run_sync_soon(nursery.start_soon, _do)
        except Exception as e:
            self.logger.debug(f"publish_topic_threadsafe failed: {e}")

    async def subscribe_topic(
        self,
        topic: str,
        handler: Callable[[str, bytes], None],
    ) -> None:
        """Subscribe to a gossipsub topic. ``handler(topic, data)`` is
        invoked (synchronously, in the host's trio task) for every
        inbound message on that topic. Must be called from inside the
        host's trio loop.

        The reader task lives in the host's main nursery, so it stops
        when the host shuts down.
        """
        await self._ensure_gossipsub()
        if not self._pubsub or self._nursery is None:
            return

        self._topic_handlers.setdefault(topic, []).append(handler)
        if topic in self._topic_subscriptions:
            return  # reader already running
        sub = await self._pubsub.subscribe(topic)
        self._topic_subscriptions[topic] = sub
        self._nursery.start_soon(self._topic_reader, topic, sub)

    def subscribe_topic_threadsafe(
        self,
        topic: str,
        handler: Callable[[str, bytes], None],
    ) -> None:
        """Cross-thread version of ``subscribe_topic``."""
        token = self._trio_token
        if token is None:
            self.logger.debug("subscribe_topic_threadsafe before host run; dropping")
            return
        import trio

        async def _do():
            await self.subscribe_topic(topic, handler)

        try:
            trio.from_thread.run(_do, trio_token=token)
        except Exception as e:
            self.logger.debug(f"subscribe_topic_threadsafe failed: {e}")

    async def _topic_reader(self, topic: str, sub: Any) -> None:
        """Pump messages from a pubsub subscription to its handlers.

        One reader per topic; multiple handlers per topic supported.
        """
        try:
            while True:
                msg = await sub.get()
                # Skip messages we sent ourselves (gossipsub echoes).
                try:
                    if msg.from_id == self._host.get_id():
                        continue
                except AttributeError:
                    pass
                data = bytes(getattr(msg, "data", b""))
                for h in list(self._topic_handlers.get(topic, [])):
                    try:
                        h(topic, data)
                    except Exception as e:
                        self.logger.warning(
                            f"topic handler raised on {topic}: {e}"
                        )
        except Exception as e:
            # Cancellation comes through here on shutdown.
            self.logger.debug(f"topic reader for {topic} ended: {e}")

    # =========================================================================
    # Story 4.6: Cross-guild routing for inference pipeline
    # =========================================================================

    def select_peers_for_route(
        self, module_ids: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Route an inference request across guilds hosting different modules.

        Uses the latency tracker to pick optimal path through the module
        pipeline, considering which guild hosts each module.

        Returns:
            List of (module_id, peer_id) pairs in pipeline order.
        """
        if not self._latency_tracker:
            return [(mid, "") for mid in module_ids]

        return self._latency_tracker.select_peers_for_route(
            module_ids, self._known_capabilities
        )

    def get_guild_for_module(self, module_id: str) -> Optional[GuildMembership]:
        """Find which guild hosts a given module."""
        for guild in self._guilds.values():
            if module_id in guild.modules:
                return guild
        return None

    # =========================================================================
    # Story 4.7: Node Capability Advertisement
    # =========================================================================

    async def advertise_capability(self, capability: Optional[NodeCapability] = None):
        """
        Advertise this node's capability to connected peers.

        Sends capability info over the capability protocol to all
        connected peers. In production, this would go through DHT put.
        """
        cap = capability or self._capability
        cap.timestamp = time.time()
        cap_bytes = cap.to_bytes()

        if not self._host:
            return

        for pid in self._host.get_connected_peers():
            try:
                stream = await self._host.new_stream(
                    pid, [CAPABILITY_PROTOCOL]
                )
                await _write_framed(stream, cap_bytes)
            except Exception as e:
                self.logger.debug(
                    f"Failed to advertise to {str(pid)[:16]}: {e}"
                )

        self.logger.info(
            f"Advertised capability to {len(self._host.get_connected_peers())} peers"
        )

    async def _handle_capability_stream(self, stream):
        """Handle incoming capability advertisement.

        Kept under the peer id the stream comes from (Noise-authenticated),
        never the one the payload names: a peer can describe only itself, so
        it cannot claim the relay role for another peer or strip it from a
        real relay."""
        try:
            data = await _read_framed(stream)
            cap = NodeCapability.from_bytes(data)
            try:
                sender = str(stream.muxed_conn.peer_id)
            except Exception:
                sender = ""
            if not sender:
                return
            cap.peer_id = sender
            self._known_capabilities[sender] = cap

            self.logger.debug(
                f"Received capability from {cap.peer_id[:16]}...: "
                f"roles={cap.roles}, modules={cap.modules_hosted}"
            )
        except Exception as e:
            self.logger.warning(f"Error handling capability: {e}")

    def find_capable_peers(
        self,
        role: Optional[str] = None,
        min_gpu_memory: int = 0,
        module_id: Optional[str] = None,
    ) -> List[NodeCapability]:
        """
        Find peers matching capability criteria.

        Args:
            role: Required role (e.g., "solver", "aggregator")
            min_gpu_memory: Minimum GPU memory in MB
            module_id: Must host this module

        Returns:
            List of matching NodeCapability objects.
        """
        results = []
        for cap in self._known_capabilities.values():
            if role and role not in cap.roles:
                continue
            if min_gpu_memory > 0 and cap.gpu_memory_mb < min_gpu_memory:
                continue
            if module_id and module_id not in cap.modules_hosted:
                continue
            results.append(cap)
        return results

    def update_capability(self, **kwargs):
        """Update this node's capability fields."""
        for key, value in kwargs.items():
            if hasattr(self._capability, key):
                setattr(self._capability, key, value)

    def update_model_state(
        self,
        *,
        model_version: int = 0,
        model_hash: str = "",
        architecture_hash: str = "",
        current_epoch: int = 0,
        total_training_tokens: int = 0,
        training_reward_pool: int = 0,
        total_inference_revenue: int = 0,
        local_cycles_completed: int = 0,
        last_loss: float = 0.0,
        last_cosine_similarity: float = 0.0,
        architecture: str = "",
        param_count: int = 0,
        latest_delta_cid: str = "",
    ):
        """Update this node's model state for the next capability advertisement.

        Called after training cycles complete and/or after on-chain state refresh.
        The model state is included in NodeCapability gossip so all peers
        can surface VL-JEPA stats without hitting the chain individually.
        """
        state = ModelState(
            model_version=model_version,
            model_hash=model_hash,
            architecture_hash=architecture_hash,
            current_epoch=current_epoch,
            total_training_tokens=total_training_tokens,
            training_reward_pool=training_reward_pool,
            total_inference_revenue=total_inference_revenue,
            local_cycles_completed=local_cycles_completed,
            last_loss=last_loss,
            last_cosine_similarity=last_cosine_similarity,
            architecture=architecture,
            param_count=param_count,
            latest_delta_cid=latest_delta_cid,
            known_contributors=self.get_network_contributor_count(),
        )
        self._capability.model_state = asdict(state)

    def get_network_model_state(self) -> Dict[str, Any]:
        """Aggregate model state from all known peers.

        Returns the highest model version seen across the network,
        plus contributor count and aggregate training stats.  This is
        the view a node surfaces to its UI without any chain queries.
        """
        best_version = 0
        best_state: Dict[str, Any] = {}
        contributors = 0
        total_local_cycles = 0

        # Include self
        self_state = self._capability.model_state
        if self_state:
            best_version = self_state.get("model_version", 0)
            best_state = dict(self_state)
            if self_state.get("local_cycles_completed", 0) > 0:
                contributors += 1
                total_local_cycles += self_state["local_cycles_completed"]

        # Merge from peers
        for cap in self._known_capabilities.values():
            ms = cap.model_state
            if not ms:
                continue
            pv = ms.get("model_version", 0)
            if pv > best_version:
                best_version = pv
                best_state = dict(ms)
            if ms.get("local_cycles_completed", 0) > 0:
                contributors += 1
                total_local_cycles += ms["local_cycles_completed"]

        if best_state:
            best_state["known_contributors"] = contributors
            best_state["network_total_cycles"] = total_local_cycles
        return best_state

    def get_network_contributor_count(self) -> int:
        """Count unique peers (including self) that have completed training cycles."""
        count = 0
        self_state = self._capability.model_state
        if self_state and self_state.get("local_cycles_completed", 0) > 0:
            count += 1
        for cap in self._known_capabilities.values():
            ms = cap.model_state
            if ms and ms.get("local_cycles_completed", 0) > 0:
                count += 1
        return count

    # =========================================================================
    # Blob Protocol (content-addressed P2P fetch)
    # =========================================================================

    def set_blob_handler(self, handler: Callable):
        """Set handler for blob requests: async handler(content_hash: str) -> Optional[bytes]"""
        self._blob_handler = handler

    async def fetch_blob(
        self,
        target_peer_id: PeerID,
        content_hash: str,
        timeout: float = 30.0,
        hops: int = 0,
    ) -> Optional[bytes]:
        """
        Request a blob from a peer by content hash (dialling it, directly
        or through a relay circuit, if not connected). ``hops`` is accepted
        for older callers and ignored: peers never forward.

        Returns the blob bytes if found and integrity-verified, None otherwise.
        """
        if not self._host:
            return None
        if isinstance(target_peer_id, str):
            try:
                target_peer_id = PeerID.from_base58(target_peer_id)
            except Exception:
                return None
        try:
            with trio.fail_after(timeout):
                stream = await self._host.new_stream(target_peer_id, [BLOB_PROTOCOL])
                # Write request: hash as UTF-8 (no stream close: waiting for response)
                await _write_frame_no_close(stream, content_hash.encode("utf-8"))
                data = await _read_frame_no_close(stream, MAX_BLOB_FRAME_BYTES)
            await stream.close()

            if data:
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash == content_hash:
                    self.logger.debug(
                        f"Fetched blob {content_hash[:16]}... "
                        f"({len(data)} bytes) from {str(target_peer_id)[:16]}..."
                    )
                    return data
                self.logger.warning(
                    f"Blob hash mismatch from {str(target_peer_id)[:16]}: "
                    f"expected {content_hash[:16]}, got {actual_hash[:16]}"
                )
            return None
        except trio.Cancelled:
            raise
        except BaseException as e:
            self.logger.debug(
                f"Failed to fetch blob {content_hash[:16]} "
                f"from {str(target_peer_id)[:16]}: {e}"
            )
            return None

    async def _handle_blob_stream(self, stream):
        """Handle incoming blob request: read content hash, write blob bytes
        (an empty frame on a miss). Never forwarded."""
        try:
            hash_bytes = await _read_frame_no_close(stream)
            content_hash = hash_bytes.decode("utf-8").strip().partition("|")[0]

            blob_data: Optional[bytes] = None
            if self._blob_handler:
                try:
                    blob_data = await self._blob_handler(content_hash)
                except Exception as e:
                    self.logger.debug(f"Blob handler error for {content_hash[:16]}: {e}")

            response = blob_data if blob_data is not None else b""
            await _write_framed(stream, response)

            self.logger.debug(
                f"Blob request {content_hash[:16]}: "
                f"{'served' if blob_data else 'not found'} ({len(response)} bytes)"
            )
        except Exception as e:
            self.logger.warning(f"Error handling blob stream: {e}")

    async def fetch_blob_from_any_peer(
        self,
        content_hash: str,
        timeout_per_peer: float = 10.0,
    ) -> Optional[bytes]:
        """Fetch a blob from the network: its DHT providers and our
        connected peers, concurrently (see ``fetch_blob_from_network``)."""
        return await self.fetch_blob_from_network(
            content_hash, timeout_per_peer=timeout_per_peer)

    async def fetch_blob_from_peers(
        self,
        peer_ids: List[Any],
        content_hash: str,
        timeout_per_peer: float = 10.0,
        hops: int = 0,
    ) -> Optional[bytes]:
        """Ask the given peers (PeerIDs or base58 strings) concurrently;
        the first integrity-verified answer wins and cancels the rest."""
        if not self._host or not peer_ids:
            return None
        result: List[Optional[bytes]] = [None]

        async def try_peer(pid, cancel_scope):
            with trio.move_on_after(timeout_per_peer):
                data = await self.fetch_blob(pid, content_hash, timeout=timeout_per_peer)
                if data is not None and result[0] is None:
                    result[0] = data
                    cancel_scope.cancel()

        own = self.peer_id
        async with trio.open_nursery() as nursery:
            seen = set()
            for pid in peer_ids:
                if isinstance(pid, str):
                    try:
                        pid = PeerID.from_base58(pid)
                    except Exception:
                        continue
                if str(pid) == own or str(pid) in seen:
                    continue
                seen.add(str(pid))
                nursery.start_soon(try_peer, pid, nursery.cancel_scope)

        return result[0]

    async def find_blob_providers(self, content_hash: str, timeout: float = 20.0) -> List[Any]:
        """Peers the DHT names as holders of ``content_hash`` (PeerInfo,
        deduplicated, ourselves left out). Their addresses, circuit ones
        included, go into the peerstore so a dial can reach them."""
        if self._dht is None:
            return []
        infos: List[Any] = []
        try:
            with trio.fail_after(timeout):
                infos = await self._dht.find_providers(content_hash)
        except trio.Cancelled:
            raise
        except BaseException as e:
            self.logger.debug("find_providers %s: %s", content_hash[:16], e)
            return []
        own, out, seen = self.peer_id, [], set()
        for info in infos or []:
            pid = str(info.peer_id)
            if pid == own or pid in seen:
                continue
            seen.add(pid)
            if info.addrs:
                try:
                    self._host.get_peerstore().add_addrs(info.peer_id, list(info.addrs), 3600)
                except Exception:
                    pass
            out.append(info)
        return out

    async def fetch_blob_from_network(
        self,
        content_hash: str,
        peer_ids: Optional[List[Any]] = None,
        timeout_per_peer: float = 10.0,
        timeout: float = 45.0,
    ) -> Optional[bytes]:
        """Fetch a blob wherever it is: ask the connected peers (and any
        ``peer_ids`` given) at once, and meanwhile look its providers up in
        the DHT and ask them (dialled directly or through a relay circuit).
        The first answer whose sha256 matches wins. Returns None when
        nobody reachable holds it within ``timeout``, and for a pruned tool's
        blob (Decision 2026-11-10)."""
        if not self._host or _pruned(content_hash):
            return None
        result: List[Optional[bytes]] = [None]
        asked: Set[str] = set()
        self.last_fetch_source = ""

        async def ask(peers: List[Any], scope, source: str) -> None:
            fresh = [p for p in peers if str(p) not in asked]
            asked.update(str(p) for p in fresh)
            data = await self.fetch_blob_from_peers(
                fresh, content_hash, timeout_per_peer=timeout_per_peer)
            if data is not None and result[0] is None:
                result[0] = data
                self.last_fetch_source = source
                scope.cancel()

        async def via_dht(scope) -> None:
            providers = await self.find_blob_providers(content_hash)
            if providers:
                self.logger.debug("blob %s: %d provider(s) in the DHT",
                                  content_hash[:16], len(providers))
                await ask([p.peer_id for p in providers], scope, "dht")

        with trio.move_on_after(timeout):
            async with trio.open_nursery() as n:
                near = list(self._host.get_connected_peers()) + list(peer_ids or [])
                n.start_soon(ask, near, n.cancel_scope, "neighbour")
                n.start_soon(via_dht, n.cancel_scope)
        return result[0]

    # =========================================================================
    # Providing blobs (DHT provider records)
    # =========================================================================

    async def provide_blob(self, content_hash: str, timeout: float = 30.0) -> bool:
        """Announce in the DHT that we hold ``content_hash`` and keep
        re-announcing it (``provide_interval``). True when at least one
        DHT peer took the record (False with no DHT peers yet: the next
        mesh round retries)."""
        content_hash = str(content_hash).strip().lower()
        if len(content_hash) != 64 or _pruned(content_hash):
            return False      # a pruned tool's blob is never announced
        self._provided.add(content_hash)
        if self._dht is None:
            return False
        ok = False
        try:
            with trio.fail_after(timeout):
                ok = bool(await self._dht.provide(content_hash))
        except trio.Cancelled:
            raise
        except BaseException as e:
            self.logger.debug("provide %s: %s", content_hash[:16], e)
        if ok:
            self._provided_at[content_hash] = time.time()
        return ok

    async def provide_blobs(self, cids: List[str], timeout: float = 30.0) -> int:
        """``provide_blob`` for each, concurrently; returns how many took."""
        done = [0]

        async def one(cid: str) -> None:
            if await self.provide_blob(cid, timeout=timeout):
                done[0] += 1
        async with trio.open_nursery() as n:
            for cid in dict.fromkeys(c for c in cids if c):
                n.start_soon(one, cid)
        return done[0]

    def provide_blobs_blocking(self, cids: List[str], timeout: float = 20.0) -> int:
        """From any thread but the host's: provide and wait (at most
        ``timeout``). Used before a tool's registration event is sent, so
        peers that react to it can already find its blobs. 0 when the host
        is not running (the blobs are provided once it is)."""
        token = self._trio_token
        cids = [str(c).lower() for c in cids if c]
        if token is None or not cids:
            self._provided.update(cids)
            return 0

        async def _do():
            with trio.move_on_after(timeout):
                return await self.provide_blobs(cids, timeout=timeout)
            return 0
        try:
            return int(trio.from_thread.run(_do, trio_token=token) or 0)
        except Exception as e:  # noqa: BLE001 - registration goes on regardless
            self.logger.debug("provide_blobs_blocking failed: %s", e)
            return 0

    def provide_blobs_threadsafe(self, cids: List[str]) -> None:
        """Queue ``provide_blobs`` from any thread without waiting."""
        token, nursery = self._trio_token, self._nursery
        cids = [str(c).lower() for c in cids if c]
        if not cids:
            return
        if token is None or nursery is None:
            self._provided.update(cids)
            return
        try:
            token.run_sync_soon(nursery.start_soon, self.provide_blobs, cids)
        except Exception as e:  # noqa: BLE001
            self.logger.debug("provide_blobs_threadsafe failed: %s", e)

    @property
    def provided_blobs(self) -> List[str]:
        return sorted(self._provided)

    async def _reprovide(self, force: bool = False) -> None:
        """Re-announce provided blobs whose record is older than
        ``provide_interval`` (all of them on ``force``, e.g. after our
        addresses changed), and the ones never announced yet."""
        if self._dht is None or not self._provided:
            return
        # Blobs pruned since they were announced stop being re-announced.
        gone = [c for c in self._provided if _pruned(c)]
        for c in gone:
            self._provided.discard(c)
            self._provided_at.pop(c, None)
        now = time.time()
        due = [c for c in self._provided
               if force or now - self._provided_at.get(c, 0) >= self._provide_interval]
        if due:
            await self.provide_blobs(due[:256])

    # =========================================================================
    # RPB Inference Protocol (/rpb/inference/1.0.0)
    # =========================================================================

    async def request_inference(self, target_peer_id: str, request: dict) -> dict:
        """Send an inference request to a serving peer.

        Uses the /rpb/inference/1.0.0 protocol. The serving peer dispatches
        the request through whatever handler it registered via
        ``set_inference_handler``.

        Args:
            target_peer_id: Peer ID of the serving peer.
            request: Dict with messages, system, model, max_tokens, tools,
                temperature.

        Returns:
            Dict with text, tool_calls, stop_reason, usage, model fields.

        Raises:
            RuntimeError: If the provider returns an error or connection fails.
        """
        import json

        req_bytes = json.dumps(request).encode("utf-8")
        try:
            stream = await self._host.new_stream(
                target_peer_id, [INFERENCE_REQUEST_PROTOCOL]
            )
            # Request-response on one stream: write the request WITHOUT
            # closing (a full close tears down the read half on py-libp2p,
            # so the response could never come back), read the response,
            # then close.
            await _write_frame_no_close(stream, req_bytes)

            # Read response
            resp_bytes = await _read_frame_no_close(stream)
            try:
                await stream.close()
            except Exception:
                pass
            response = json.loads(resp_bytes.decode("utf-8"))

            if "error" in response:
                raise RuntimeError(f"Provider error: {response['error']}")

            self.logger.info(
                "Inference response from %s (%d bytes)",
                str(target_peer_id)[:16], len(resp_bytes),
            )
            return response
        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.warning("Inference request to %s failed: %s", str(target_peer_id)[:16], e)
            raise RuntimeError(f"Inference request failed: {e}") from e

    async def _handle_inference_stream(self, stream):
        """Handle incoming inference request from a dependent node.

        Path A: Forward the request through this node's own provider
        (Claude, OpenAI, etc.) and return the response.
        """
        import json

        try:
            # Request-response on one stream: read the request without
            # closing, serve, write the response, then close. (Closing on
            # the read tears down the stream before the response is sent.)
            data = await _read_frame_no_close(stream)
            request = json.loads(data.decode("utf-8"))

            self.logger.info("Received inference request (%d bytes)", len(data))

            # Forward through this node's configured provider
            response = await self._serve_inference_locally(request)

            resp_bytes = json.dumps(response).encode("utf-8")
            await _write_frame_no_close(stream, resp_bytes)
            try:
                await stream.close()
            except Exception:
                pass

        except Exception as e:
            self.logger.warning("Failed to handle inference request: %s", e)
            try:
                error_resp = json.dumps({"error": str(e)}).encode("utf-8")
                await _write_frame_no_close(stream, error_resp)
                await stream.close()
            except Exception:
                pass

    def set_inference_handler(
        self,
        handler: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """Register the coroutine that serves incoming /rpb/inference/1.0.0
        requests.

        The handler takes the request dict and returns the response dict
        (or {"error": "..."}), e.g. the sponsor/dependent flow."""
        self._inference_handler = handler

    async def _serve_inference_locally(self, request: dict) -> dict:
        """Serve an inference request using this node's registered
        handler. Phase 7.3+: typically the substrate-backed handler.

        Returns an error dict when no handler is registered."""
        handler = getattr(self, "_inference_handler", None)
        if handler:
            return await handler(request)

        return {"error": "This node has no inference handler registered"}

    # =========================================================================
    # Robot poll rail (docs/robots.md, 2026-10-25)
    # =========================================================================

    def set_robot_handler(
        self,
        handler: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """Register the coroutine (run in the host's trio loop) that serves
        incoming /autonet/robot/1.0.0 messages: dict in, dict out."""
        self._robot_handler = handler

    async def request_robot(self, target_peer_id: Any, message: dict,
                            timeout: float = 10.0) -> dict:
        """Send one robot-rail message to a peer and return its response.
        Raises on a connection failure or timeout."""
        import json

        if isinstance(target_peer_id, str):
            target_peer_id = PeerID.from_base58(target_peer_id)
        with trio.fail_after(timeout):
            stream = await self._host.new_stream(target_peer_id, [ROBOT_PROTOCOL])
            # Same framing as inference: write without closing, read, close.
            await _write_frame_no_close(stream, json.dumps(message).encode("utf-8"))
            resp = await _read_frame_no_close(stream)
        try:
            await stream.close()
        except Exception:
            pass
        return json.loads(resp.decode("utf-8"))

    async def _handle_robot_stream(self, stream):
        import json

        try:
            request = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            handler = getattr(self, "_robot_handler", None)
            response = (await handler(request)) if handler else {
                "error": "this node does not run the robot rail"}
        except Exception as e:
            self.logger.warning("robot rail request failed: %s", e)
            response = {"error": str(e)}
        try:
            await _write_frame_no_close(stream, json.dumps(response).encode("utf-8"))
            await stream.close()
        except Exception:
            pass

    # =========================================================================
    # Service calls (/autonet/service/1.0.0), relayed at most one hop
    # =========================================================================

    def set_service_handler(
        self,
        handler: Optional[Callable[[dict], Awaitable[dict]]],
    ) -> None:
        """Register the coroutine (run in the host's trio loop) that serves
        service requests addressed to this peer: the ``service_request``
        dict in, the reply dict out. None unregisters. A host without a
        handler still forwards requests addressed to other peers."""
        self._service_handler = handler

    async def _service_exchange(self, peer_id: Any, envelope: dict,
                                timeout: float,
                                connect_timeout: float = 15.0) -> dict:
        """One envelope out, one reply back on a fresh stream.

        Raises ``ConnectionError`` when the stream cannot be opened or the
        request cannot be written (the request never left this host), and
        ``TimeoutError`` when it was sent but no reply came in time."""
        if self._host is None:
            raise ConnectionError("libp2p host not running")
        if isinstance(peer_id, str):
            peer_id = PeerID.from_base58(peer_id)
        try:
            with trio.fail_after(connect_timeout):
                stream = await self._host.new_stream(peer_id, [SERVICE_PROTOCOL])
                await _write_frame_no_close(
                    stream, json.dumps(envelope).encode("utf-8"))
        except Exception as e:
            raise ConnectionError(f"could not reach {str(peer_id)[:16]}: {e}") from e
        try:
            with trio.fail_after(timeout):
                raw = await _read_frame_no_close(stream)
        except trio.TooSlowError:
            raise TimeoutError(f"no reply within {int(timeout)}s") from None
        except Exception as e:
            raise TimeoutError(f"the stream broke before the reply: {e}") from e
        finally:
            try:
                await stream.close()
            except Exception:
                pass
        if not raw:
            raise TimeoutError("the stream closed without a reply")
        return json.loads(raw.decode("utf-8"))

    def _seed_peer_ids(self) -> set:
        """Peer ids named in this host's seed list (the DAO Registry's
        ``bootstrap`` key, or local config)."""
        out = set()
        for addr in self._bootstrap_peers or []:
            text = str(addr)
            if "/p2p/" in text:
                pid = text.rsplit("/p2p/", 1)[1].split("/", 1)[0]
                if pid:
                    out.add(pid)
        return out

    async def reach_peer(self, peer_id: str, timeout: float = 15.0) -> bool:
        """Connect to ``peer_id`` if we are not: at the addresses we know
        (capability gossip, the peerstore), else at the ones the DHT finds
        for it; directly or through one of its relays."""
        if peer_id in self.get_connected_peers():
            return True
        if not self._host:
            return False
        try:
            pid = PeerID.from_base58(peer_id)
        except Exception:
            return False
        cap = self._known_capabilities.get(peer_id)
        known = []
        for a in list(cap.listen_addrs if cap else []):
            try:
                known.append(Multiaddr(a.split(f"/p2p/{peer_id}")[0]))
            except Exception:
                pass
        with trio.move_on_after(timeout):
            try:
                if known:
                    self._host.get_peerstore().add_addrs(pid, known, 3600)
                try:
                    have = bool(self._host.get_peerstore().addrs(pid))
                except Exception:
                    have = False
                if not have and self._dht is not None:
                    info = await self._dht.find_peer(pid)
                    if info is not None and info.addrs:
                        self._host.get_peerstore().add_addrs(pid, list(info.addrs), 3600)
                await self._host.connect(PeerInfo(pid, []))
                await self._nudge_pubsub_peer(pid)
            except Exception as e:  # noqa: BLE001
                self.logger.debug("could not reach %s: %s", peer_id[:16], e)
        return peer_id in self.get_connected_peers()

    def _relay_candidates(self, target: str) -> List[str]:
        """Connected peers that may forward a service request to the target:
        seeds and peers that advertised the relay role, never the target.

        A request's args (a prompt, a document) are plaintext to the peer
        that forwards it, so it is not offered to every daemon in the mesh,
        only to relays. Seeds count as relays from the moment they connect:
        a relay advertises its role on an interval, and the DAO lists its
        seeds."""
        seeds = self._seed_peer_ids()

        def forwards(pid: str) -> bool:
            cap = self._known_capabilities.get(pid)
            return pid in seeds or bool(cap and "relay" in (cap.roles or []))
        return [p for p in self.get_connected_peers()
                if p != target and forwards(p)]

    async def request_service(self, target_peer_id: str, message: dict,
                              timeout: float = 300.0,
                              connect_timeout: float = 15.0,
                              accept: Optional[Callable[[dict], bool]] = None,
                              ) -> dict:
        """Send a ``service_request`` to the peer ``target_peer_id`` and
        return its reply.

        Direct when connected to the target; otherwise each connected relay
        (see ``_relay_candidates``) is asked in turn to forward it one hop.
        A relay that is not connected to the target answers with
        ``relay_error`` at once and the next is tried.

        ``accept(reply)`` says whether a relayed reply is the target's own
        (the buyer checks the provider's signature on it). A reply it
        refuses was made up or altered on the way, so the next relay is
        tried: the provider's replay guard serves one payment once, so a
        second copy costs nothing. When every relay fails that way, the last
        refused reply is returned (the request may have reached the target).
        All relay attempts share one deadline of ``timeout``.

        Raises ``ConnectionError`` if no path reached the target (nothing
        was delivered) and ``TimeoutError`` if it was delivered but no
        reply came within ``timeout``."""
        target = str(target_peer_id)
        if target not in self.get_connected_peers():
            await self.reach_peer(target, timeout=connect_timeout)
        if target in self.get_connected_peers():
            try:
                return await self._service_exchange(
                    target, {"target": target, "hops": 0, "msg": message},
                    timeout, connect_timeout)
            except ConnectionError as e:
                self.logger.debug("direct service call to %s failed: %s",
                                  target[:16], e)
        errors: List[str] = []
        refused: Optional[dict] = None
        deadline = trio.current_time() + float(timeout)
        for n, relay in enumerate(self._relay_candidates(target)):
            remaining = deadline - trio.current_time()
            if n and remaining < 1.0:
                errors.append("out of time")
                break
            remaining = max(remaining, 0.1)
            envelope = {"target": target, "hops": SERVICE_RELAY_MAX_HOPS,
                        "timeout": remaining, "msg": message}
            try:
                reply = await self._service_exchange(
                    relay, envelope, remaining + connect_timeout,
                    connect_timeout)
            except ConnectionError as e:
                errors.append(str(e))
                continue
            if isinstance(reply, dict) and reply.get("relay_error"):
                errors.append(f"{relay[:16]}: {reply['relay_error']}")
                continue
            if accept is not None and not accept(reply):
                self.logger.warning(
                    "service reply through %s is not the target's own; "
                    "trying the next relay", relay[:16])
                refused = reply if isinstance(reply, dict) else {"ok": False}
                errors.append(f"{relay[:16]}: reply not from the target")
                continue
            return reply
        if refused is not None:
            return refused
        detail = "; ".join(errors) if errors else "no connected relay"
        raise ConnectionError(f"no path to {target[:16]} ({detail})")

    async def _handle_service_request_stream(self, stream):
        """Serve a service request addressed to us, or forward it one hop
        to the target if we are connected to it. Forwarding carries the
        bytes only: payment, payer signature and provider identity are
        checked end to end."""
        try:
            envelope = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("envelope must be an object")
            target = str(envelope.get("target") or "")
            try:
                hops = min(int(envelope.get("hops") or 0), SERVICE_RELAY_MAX_HOPS)
            except (TypeError, ValueError):
                hops = 0
            msg = envelope.get("msg")
            if not isinstance(msg, dict):
                raise ValueError("missing 'msg'")
            if not target or target == self.peer_id:
                handler = self._service_handler
                response = (await handler(msg)) if handler else {
                    "msg_id": msg.get("msg_id"), "ok": False,
                    "error": "this node serves no services"}
            elif hops <= 0:
                response = {"relay_error": "not the target and no hops left"}
            elif target not in self.get_connected_peers():
                response = {"relay_error": "not connected to the target"}
            else:
                try:
                    wait = float(envelope.get("timeout") or self.service_relay_timeout)
                except (TypeError, ValueError):
                    wait = self.service_relay_timeout
                wait = max(1.0, min(wait, self.service_relay_timeout))
                try:
                    response = await self._service_exchange(
                        target, {"target": target, "hops": 0, "msg": msg}, wait)
                except ConnectionError as e:
                    response = {"relay_error": str(e)}
                except TimeoutError as e:
                    response = {"msg_id": msg.get("msg_id"), "ok": False,
                                "error": f"the provider did not reply through "
                                         f"the relay: {e}"}
        except Exception as e:
            self.logger.warning("service request failed: %s", e)
            response = {"ok": False, "error": f"bad service request: {e}"}
        try:
            await _write_frame_no_close(stream, json.dumps(response).encode("utf-8"))
            await stream.close()
        except Exception:
            pass

    # =========================================================================
    # Service sessions (service_session.py; Decision 2026-11-12 in
    # docs/services_market.md) over the mesh's connections: direct, hole
    # punched (DCUtR) or a circuit through a relay
    # =========================================================================

    def _pid(self, peer_id: Any) -> Any:
        return PeerID.from_base58(peer_id) if isinstance(peer_id, str) else peer_id

    @staticmethod
    def _conn_relayed(conn: Any) -> bool:
        """Whether a connection runs through a relay circuit."""
        try:
            addrs = list(conn.get_transport_addresses())
        except Exception:  # noqa: BLE001
            return False
        return bool(addrs) and all(p2p_mesh.is_circuit(a) for a in addrs)

    def connection_path(self, peer_id: Any) -> Optional[str]:
        """"direct" when a direct connection to ``peer_id`` is open (dialled,
        or hole punched by DCUtR), "relayed" when only circuits are, None
        when not connected."""
        if self._host is None:
            return None
        try:
            conns = [c for c in self._host.get_network().get_connections(self._pid(peer_id))
                     if not getattr(c, "is_closed", False)]
        except Exception:  # noqa: BLE001
            return None
        if not conns:
            return None
        return "relayed" if all(self._conn_relayed(c) for c in conns) else "direct"

    async def _handle_peer_info_stream(self, stream):
        """Tell the asker what we know of a peer: whether we are connected
        to it and what its own capability says (stored under the peer id
        that sent it, so it is that peer's word)."""
        try:
            req = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            peer = str(req.get("peer") or "") if isinstance(req, dict) else ""
            out: Dict[str, Any] = {"peer": peer,
                                   "connected": peer in self.get_connected_peers()}
            cap = self._known_capabilities.get(peer)
            if cap is not None:
                out.update({"reachable": bool(getattr(cap, "reachable", False)),
                            "addrs": list(cap.listen_addrs or []),
                            "roles": list(cap.roles or []),
                            "timestamp": cap.timestamp})
            await _write_frame_no_close(stream, json.dumps(out).encode("utf-8"))
            await stream.close()
        except Exception as e:
            self.logger.debug(f"peer-info request failed: {e}")

    async def peer_info(self, via: Any, peer: str, timeout: float = 5.0) -> Optional[dict]:
        """What the connected peer ``via`` knows of ``peer`` (see
        PEER_INFO_PROTOCOL), or None."""
        if self._host is None:
            return None
        try:
            with trio.fail_after(timeout):
                stream = await self._host.new_stream(self._pid(via), [PEER_INFO_PROTOCOL])
                await _write_frame_no_close(stream, json.dumps({"peer": peer}).encode("utf-8"))
                data = await _read_frame_no_close(stream)
            try:
                await stream.close()
            except Exception:
                pass
            info = json.loads(data.decode("utf-8")) if data else None
        except Exception as e:
            self.logger.debug(f"peer-info via {str(via)[:16]} failed: {e}")
            return None
        return info if isinstance(info, dict) and info.get("peer") == peer else None

    async def relay_peer_infos(self, target: str) -> Dict[str, dict]:
        """Ask every connected relay at once what it knows of ``target``.
        The newest capability any of them reports is cached for
        ``peer_reachable``."""
        out: Dict[str, dict] = {}

        async def one(relay: str) -> None:
            info = await self.peer_info(relay, target)
            if info is not None:
                out[relay] = info
        async with trio.open_nursery() as nursery:
            for relay in self._relay_candidates(target):
                nursery.start_soon(one, relay)
        known = [i for i in out.values() if "reachable" in i]
        if known:
            best = max(known, key=lambda i: float(i.get("timestamp") or 0))
            self._peer_info_cache[target] = (time.time(), best)
        return out

    def peer_reachable(self, peer_id: str) -> Optional[bool]:
        """Whether ``peer_id`` says it is directly reachable: its own
        capability when it sent us one, else the last answer a relay gave;
        None when unknown. A miss or a stale answer asks the relays in the
        background. Safe from any thread; never waits."""
        peer_id = str(peer_id or "")
        if not peer_id:
            return None
        cap = self._known_capabilities.get(peer_id)
        if cap is not None:
            return bool(getattr(cap, "reachable", False))
        hit = self._peer_info_cache.get(peer_id)
        now = time.time()
        if hit is None or now - hit[0] > PEER_INFO_TTL:
            if self._peer_info_due.get(peer_id, 0.0) <= now:
                self._peer_info_due[peer_id] = now + PEER_INFO_RETRY
                token, nursery = self._trio_token, self._nursery
                if token is not None and nursery is not None:
                    try:
                        token.run_sync_soon(nursery.start_soon,
                                            self.relay_peer_infos, peer_id)
                    except Exception:  # noqa: BLE001 (host shutting down)
                        pass
        if hit is None:
            return None
        value = hit[1].get("reachable")
        return bool(value) if isinstance(value, bool) else None

    async def route_to(self, target: str, direct_timeout: float = 15.0) -> Tuple[str, str]:
        """Have a connection to ``target`` for a service stream, in the
        order the owner decided (2026-11-10): one already open; else the
        mesh dialer (``reach_peer``: the target's direct addresses first,
        its relay circuits racing them ``CIRCUIT_DIAL_DELAY`` later, the DHT
        when we know no address). DCUtR then upgrades a circuit between two
        NAT'd daemons to a direct, hole-punched connection where both NATs
        allow it, and the stream's next use rides that.

        Returns ``(path, "")``: path "direct" or "relayed". Raises
        ``ConnectionError`` when nothing reaches the target."""
        if self._host is None:
            raise ConnectionError("libp2p host not running")
        target = str(target)
        if target == self.peer_id:
            raise ConnectionError("cannot open a stream to this node itself")
        path = self.connection_path(target)
        if path is not None:
            return path, ""
        if not await self.reach_peer(target, timeout=direct_timeout):
            raise ConnectionError(
                f"no path to {target[:16]} (not reachable directly, through a "
                "relay circuit, or found in the DHT)")
        path = self.connection_path(target)
        if path is None:
            raise ConnectionError(f"the connection to {target[:16]} closed")
        return path, ""

    def set_service_session_handler(self, handler: Optional[Any]) -> None:
        """Serve ``/autonet/service-session/1.0.0`` with ``handler`` (a
        ``service_session.SessionHandler``, run in the host's trio loop).
        None refuses every session."""
        self._service_session_handler = handler

    async def _handle_service_session_open(self, stream):
        from .service_session import serve_incoming
        try:
            peer = str(stream.muxed_conn.peer_id)
            path = "relayed" if self._conn_relayed(stream.muxed_conn) else "direct"
        except Exception:  # noqa: BLE001
            peer, path = "", ""
        await serve_incoming(stream, self._service_session_handler,
                             path=path, peer=peer)

    @asynccontextmanager
    async def service_session(self, target: str, open_msg: dict, *,
                              voucher_source: Optional[Callable[[int, int], Awaitable[Optional[dict]]]] = None,
                              on_balance: Optional[Callable[[int, int], Any]] = None,
                              timeout: float = 60.0, direct_timeout: float = 10.0):
        """Open a session to the provider ``target`` (its peer id) and yield
        the accepted ``service_session.ServiceSession``.

        The path is ``route_to``'s: an open connection, else the mesh
        dialer (direct addresses, then the target's relay circuits). The
        transport authenticates ``target`` (the peer id the provider bound
        on chain). ``voucher_source(consumed, cumulative)`` gives the next
        voucher when the provider asks; ``on_balance(consumed, cumulative)``
        hears its count (see service_session.py). Raises ``ConnectionError``
        when no path reached the target (nothing was sent),
        ``service_session.SessionRefused`` when the provider refused it, and
        ``TimeoutError`` when it was sent but not answered. Connections are
        the mesh's: nothing here opens or closes a circuit of its own."""
        from .service_session import SERVICE_SESSION_PROTOCOL, open_on
        target = str(target)
        await self.route_to(target, direct_timeout)
        try:
            with trio.fail_after(15):
                raw = await self._host.new_stream(self._pid(target),
                                                  [SERVICE_SESSION_PROTOCOL])
        except Exception as e:
            raise ConnectionError(f"could not open a session to {target[:16]}: {e}") from e
        path = "relayed" if self._conn_relayed(raw.muxed_conn) else "direct"
        async with open_on(raw, open_msg, voucher_source=voucher_source,
                           on_balance=on_balance, timeout=timeout, path=path,
                           peer=target, relay="") as session:
            yield session

    # =========================================================================
    # Batch catch-up (/autonet/batches/1.0.0), issue #41
    # =========================================================================

    def set_batch_source(self, topic: str,
                         serve: Optional[Callable[[dict], dict]]) -> None:
        """Register ``serve(request) -> {"batches": [...]}`` (sync, cheap)
        for catch-up requests on ``topic``. None unregisters."""
        if serve is None:
            self._batch_sources.pop(topic, None)
        else:
            self._batch_sources[topic] = serve

    async def _handle_batches_stream(self, stream):
        try:
            request = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            serve = self._batch_sources.get(str(request.get("topic") or ""))
            response = serve(request) if serve else {"batches": []}
        except Exception as e:
            self.logger.debug("catch-up request failed: %s", e)
            response = {"error": str(e)}
        try:
            await _write_frame_no_close(stream, json.dumps(response).encode("utf-8"))
            await stream.close()
        except Exception:
            pass

    async def request_batches(self, request: dict, max_peers: int = 4,
                              timeout: float = 10.0) -> List[str]:
        """Ask up to ``max_peers`` connected peers (relays first, then the
        rest) for catch-up batches, concurrently; returns every wire batch
        they sent (the caller verifies and dedupes)."""
        if not self._host:
            return []
        peers = self._relay_candidates("")[:max_peers]
        out: List[str] = []
        frame = json.dumps(request).encode("utf-8")

        async def ask(pid: str) -> None:
            with trio.move_on_after(timeout):
                try:
                    stream = await self._host.new_stream(
                        PeerID.from_base58(pid), [BATCHES_PROTOCOL])
                    await _write_frame_no_close(stream, frame)
                    raw = await _read_frame_no_close(stream)
                    await stream.close()
                    reply = json.loads(raw.decode("utf-8")) if raw else {}
                    out.extend(str(w) for w in (reply.get("batches") or []))
                except Exception as e:
                    self.logger.debug("catch-up from %s failed: %s", pid[:16], e)

        async with trio.open_nursery() as nursery:
            for pid in peers:
                nursery.start_soon(ask, pid)
        return out

    # =========================================================================
    # Blob offers (issue #44)
    # =========================================================================

    def set_blob_offer_store(self, has: Optional[Callable[[str], bool]],
                             put: Optional[Callable[[bytes], Any]]) -> None:
        """Accept blob offers: ``has(cid)`` skips what is already kept,
        ``put(blob)`` keeps a fetched one. None (the default) declines."""
        self._blob_offer_store = (has, put) if has and put else None

    async def _handle_blob_offer_stream(self, stream):
        kept = 0
        try:
            request = json.loads((await _read_frame_no_close(stream)).decode("utf-8"))
            store = self._blob_offer_store
            cids = [str(c) for c in (request.get("cids") or [])][:BLOB_OFFER_MAX_CIDS]
            if store is not None:
                has, put = store
                offerer = stream.muxed_conn.peer_id
                for cid in cids:
                    if len(cid) != 64 or not all(ch in "0123456789abcdef" for ch in cid) or has(cid):
                        continue
                    # Direct only: the offerer said it holds the blob.
                    blob = await self.fetch_blob(
                        offerer, cid, timeout=self._blob_relay_timeout * 4)
                    if blob is not None:
                        put(blob)
                        kept += 1
                        # A kept blob is announced like any other we hold.
                        self.provide_blobs_threadsafe([cid])
        except Exception as e:
            self.logger.debug("blob offer failed: %s", e)
        try:
            await _write_frame_no_close(stream, json.dumps({"kept": kept}).encode("utf-8"))
            await stream.close()
        except Exception:
            pass

    async def offer_blobs(self, cids: List[str], timeout: float = 60.0) -> int:
        """Offer blobs this host holds to every connected relay, so they
        stay fetchable after this host goes offline. Returns how many the
        relays kept (0 when they already had them)."""
        if not self._host or not cids:
            return 0
        self.provide_blobs_threadsafe(list(cids))
        relays = self.connected_relays()
        frame = json.dumps({"cids": list(cids)[:BLOB_OFFER_MAX_CIDS]}).encode("utf-8")
        kept = [0]

        async def offer(pid: str) -> None:
            with trio.move_on_after(timeout):
                try:
                    stream = await self._host.new_stream(
                        PeerID.from_base58(pid), [BLOB_OFFER_PROTOCOL])
                    await _write_frame_no_close(stream, frame)
                    raw = await _read_frame_no_close(stream)
                    await stream.close()
                    kept[0] += int((json.loads(raw.decode("utf-8")) if raw else {}).get("kept", 0))
                except Exception as e:
                    self.logger.debug("blob offer to %s failed: %s", pid[:16], e)

        async with trio.open_nursery() as nursery:
            for pid in relays:
                nursery.start_soon(offer, pid)
        return kept[0]

    def set_inference_advertisement(
        self,
        *,
        renderer_model: str,
        price_atn: int,
        agent_address: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a specific agent on this node as an inference provider.

        Phase 7.5b: writes the inference advertisement into the
        matching ``cap.agents[i].inference`` dict so a daemon hosting
        multiple agents can advertise per-agent offers. If no agent
        with this address exists in ``cap.agents`` yet, also writes to
        ``cap.inference`` for backward compatibility with single-agent
        daemons.

        Adds ``"inference-provider"`` to ``cap.roles`` (idempotent).
        Includes a ``schema`` key so future format changes can fan out.

        Call ``advertise_capability()`` afterwards (or rely on the
        periodic advertisement loop) to push the new capability to
        peers.
        """
        cap = self._capability
        if "inference-provider" not in cap.roles:
            cap.roles.append("inference-provider")
        adv: Dict[str, Any] = {
            "schema": 1,
            "renderer_model": str(renderer_model),
            "price_atn": int(price_atn),
            "agent_address": str(agent_address),
        }
        if extras:
            adv.update(extras)
        # Try to find the matching agent in cap.agents and update its
        # inference dict. Backward compat: also keep cap.inference
        # populated when the matching agent isn't listed (single-agent
        # daemons that don't populate cap.agents at all).
        agent_addr_lower = str(agent_address).lower()
        matched = False
        for entry in cap.agents:
            if str(entry.get("address", "")).lower() == agent_addr_lower:
                entry["inference"] = adv
                matched = True
                break
        if not matched:
            # No agent record to attach to; daemon-wide fallback so
            # peers that read only cap.inference still find it.
            cap.inference = adv

    def clear_inference_advertisement(self, agent_address: Optional[str] = None) -> None:
        """Remove inference advertisement.

        ``agent_address`` is None: clears every per-agent inference
        dict AND ``cap.inference``, drops the role.

        ``agent_address`` set: clears only that agent's inference
        dict; the role stays if any other agent still advertises.
        """
        cap = self._capability
        if agent_address is None:
            if "inference-provider" in cap.roles:
                cap.roles = [r for r in cap.roles if r != "inference-provider"]
            cap.inference = {}
            for entry in cap.agents:
                entry["inference"] = {}
            return
        target = str(agent_address).lower()
        for entry in cap.agents:
            if str(entry.get("address", "")).lower() == target:
                entry["inference"] = {}
                break
        if cap.inference and str(cap.inference.get("agent_address", "")).lower() == target:
            cap.inference = {}
        # Drop the role if nothing else still advertises.
        any_advert = bool(cap.inference) or any(
            entry.get("inference") for entry in cap.agents
        )
        if not any_advert and "inference-provider" in cap.roles:
            cap.roles = [r for r in cap.roles if r != "inference-provider"]

    async def advertise_models(self, models: List[str], agent_address: str) -> None:
        """Backward-compatible shim — Phase 7.4 reframed advertisement
        as a richer inference-capability record (see
        ``set_inference_advertisement``). This shim wires the old
        signature into the new path: the first model is treated as
        the renderer model and price defaults to 0.

        Prefer ``set_inference_advertisement`` for new code.
        """
        if not models:
            return
        self.set_inference_advertisement(
            renderer_model=models[0],
            price_atn=0,
            agent_address=agent_address,
        )
        await self.advertise_capability()

    def discover_inference_providers(
        self,
        *,
        max_price_atn: Optional[int] = None,
        renderer_model: Optional[str] = None,
        require_agent_address: bool = True,
    ) -> List[InferenceProviderMatch]:
        """Find agents advertising inference capability.

        Phase 7.5b: returns one ``InferenceProviderMatch`` per agent
        offering inference, not one per daemon. A single daemon hosting
        multiple inference-providing agents shows up as multiple
        matches — callers can route to the specific agent by
        ``agent_address`` rather than treating the whole daemon as one
        offer.

        Sources of inference advertisements (per peer cap):
          1. Each entry in ``cap.agents`` with a non-empty ``inference``
             dict — the per-agent advertisement.
          2. ``cap.inference`` itself — daemon-wide fallback for
             pre-Phase-7.5b daemons that only set the legacy field.

        Filters:
          - ``max_price_atn`` (optional): keep only matches with
            ``price_atn`` <= this cap.
          - ``renderer_model`` (optional): exact match on
            ``inference["renderer_model"]``.
          - ``require_agent_address`` (default True): keep only
            matches with a non-empty agent_address.

        Sorted: known-latency peers first, ascending by latency;
        unmeasured peers after. Within each tier, cheapest price wins.
        """
        out: List[InferenceProviderMatch] = []
        tracker = getattr(self, "_latency_tracker", None)

        def _consider(cap: NodeCapability, adv: Dict[str, Any]) -> None:
            if not adv:
                return
            agent_addr = str(adv.get("agent_address", ""))
            if require_agent_address and not agent_addr:
                return
            if (
                max_price_atn is not None
                and int(adv.get("price_atn", 0)) > int(max_price_atn)
            ):
                return
            if renderer_model is not None:
                if str(adv.get("renderer_model", "")) != str(renderer_model):
                    return
            lat_ms: Optional[float] = None
            if tracker is not None:
                lat_obj = tracker.get_latency(cap.peer_id)
                if (
                    lat_obj is not None
                    and getattr(lat_obj, "reachable", False)
                    and getattr(lat_obj, "samples", 0) > 0
                ):
                    lat_ms = float(lat_obj.ema_rtt_ms)
            out.append(InferenceProviderMatch(
                peer_id=cap.peer_id,
                agent_address=agent_addr,
                renderer_model=str(adv.get("renderer_model", "")),
                price_atn=int(adv.get("price_atn", 0)),
                inference=dict(adv),
                capability=cap,
                latency_ms=lat_ms,
            ))

        for cap in self._known_capabilities.values():
            if "inference-provider" not in cap.roles:
                continue
            # Per-agent advertisements (preferred path).
            seen_addrs: set = set()
            for entry in cap.agents:
                adv = entry.get("inference") or {}
                if adv:
                    _consider(cap, adv)
                    addr = str(adv.get("agent_address", "")).lower()
                    if addr:
                        seen_addrs.add(addr)
            # Legacy daemon-wide advertisement, only if it doesn't
            # duplicate an already-included per-agent one.
            legacy_addr = str(cap.inference.get("agent_address", "")).lower() if cap.inference else ""
            if cap.inference and (not legacy_addr or legacy_addr not in seen_addrs):
                _consider(cap, cap.inference)

        def _sort_key(m: InferenceProviderMatch) -> Tuple[int, float, int]:
            if m.latency_ms is not None:
                return (0, m.latency_ms, m.price_atn)
            return (1, float("inf"), m.price_atn)

        out.sort(key=_sort_key)
        return out

    # =========================================================================
    # Embedding Exchange Protocol (two-speed inference architecture)
    # =========================================================================

    def set_embedding_handler(self, handler: Callable):
        """Set handler for incoming embeddings: async handler(peer_id, metadata, tensor_bytes)"""
        self._embedding_handler = handler

    async def send_embedding(
        self,
        target_peer_id: PeerID,
        tensor_bytes: bytes,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Send an embedding tensor to a peer for the two-speed inference architecture.

        Frame format: [4-byte metadata_len][metadata_json][tensor_bytes]
        """
        if not self._host:
            return False
        try:
            meta_bytes = json.dumps(metadata or {}).encode("utf-8")
            meta_len = struct.pack(">I", len(meta_bytes))
            payload = meta_len + meta_bytes + tensor_bytes

            stream = await self._host.new_stream(target_peer_id, [EMBEDDING_PROTOCOL])
            await _write_framed(stream, payload)

            self.logger.debug(
                f"Sent embedding ({len(tensor_bytes)} bytes) "
                f"to {str(target_peer_id)[:16]}..."
            )
            return True
        except Exception as e:
            self.logger.warning(f"Failed to send embedding: {e}")
            return False

    async def _handle_embedding_stream(self, stream):
        """Handle incoming embedding tensor."""
        try:
            data = await _read_framed(stream)
            if len(data) < 4:
                return

            meta_len = struct.unpack(">I", data[:4])[0]
            meta_bytes = data[4:4 + meta_len]
            tensor_bytes = data[4 + meta_len:]

            metadata = json.loads(meta_bytes.decode("utf-8"))
            remote_peer = str(stream.muxed_conn.peer_id)

            self.logger.debug(
                f"Received embedding ({len(tensor_bytes)} bytes) "
                f"from {remote_peer[:16]}..."
            )

            if self._embedding_handler:
                await self._embedding_handler(remote_peer, metadata, tensor_bytes)
        except Exception as e:
            self.logger.warning(f"Error handling embedding stream: {e}")

    # =========================================================================
    # Internal setup
    # =========================================================================

    def _setup_stream_handlers(self):
        """Register protocol stream handlers on the libp2p host."""
        if not self._host:
            return

        self._host.set_stream_handler(
            WEIGHTS_PROTOCOL, self._handle_weight_stream
        )
        self._host.set_stream_handler(
            ACTIVATIONS_PROTOCOL, self._handle_activation_stream
        )
        self._host.set_stream_handler(
            CAPABILITY_PROTOCOL, self._handle_capability_stream
        )
        self._host.set_stream_handler(
            BLOB_PROTOCOL, self._handle_blob_stream
        )
        self._host.set_stream_handler(
            EMBEDDING_PROTOCOL, self._handle_embedding_stream
        )
        self._host.set_stream_handler(
            INFERENCE_REQUEST_PROTOCOL, self._handle_inference_stream
        )
        self._host.set_stream_handler(
            ROBOT_PROTOCOL, self._handle_robot_stream
        )
        self._host.set_stream_handler(
            REACH_PROTOCOL, self._handle_reach_stream
        )
        self._host.set_stream_handler(
            SERVICE_PROTOCOL, self._handle_service_request_stream
        )
        self._host.set_stream_handler(
            BATCHES_PROTOCOL, self._handle_batches_stream
        )
        self._host.set_stream_handler(
            BLOB_OFFER_PROTOCOL, self._handle_blob_offer_stream
        )
        self._host.set_stream_handler(
            PEER_INFO_PROTOCOL, self._handle_peer_info_stream
        )
        # Service sessions (service_session.py) ride the mesh's connections;
        # relayed ones are circuit relay v2 (_start_mesh_services).
        try:
            from .service_session import SERVICE_SESSION_PROTOCOL
            self._host.set_stream_handler(
                SERVICE_SESSION_PROTOCOL, self._handle_service_session_open
            )
        except Exception as e:  # noqa: BLE001 (sessions off, the rest works)
            self.logger.warning(f"service sessions unavailable: {e}")

    # =========================================================================
    # Summary / Introspection
    # =========================================================================

    def summary(self) -> Dict:
        """Return a summary dict for dashboard display."""
        return {
            "node_id": self.node_id,
            "peer_id": self.peer_id,
            "addrs": self.addrs,
            "connected_peers": len(self.get_connected_peers()),
            "known_capabilities": len(self._known_capabilities),
            "guilds": list(self._guilds.keys()),
            "running": self._running,
            "transport": self._transport,
            "connections": self.connection_kinds() if self._host else {},
            "reservations": self.reservations,
            "dht_mode": self.dht_mode,
            "dht_peers": (self._dht.get_routing_table_size() if self._dht else 0),
            "provided_blobs": len(self._provided),
            "peer_cache": len(self._peer_cache),
        }
