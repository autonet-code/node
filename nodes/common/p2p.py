"""
P2P Communication Layer for Autonet

Wraps py-libp2p to provide:
- Node discovery via Kademlia DHT
- QUIC transport (preferred) with TCP fallback
- NAT traversal via UPnP
- Latency-aware peer selection
- Weight delta transfer between solver/aggregator
- Activation relay for pipeline-parallel inference
- Guild-local gossip via GossipSub
- Cross-guild inference routing
- Node capability advertisement via DHT

Uses trio as the async runtime (required by py-libp2p).

Usage:
    from nodes.common.p2p import AutonetHost, NodeCapability
    host = AutonetHost(node_id="solver-0", config=p2p_config)
    async with host.run():
        await host.send_weight_delta(peer_id, delta_bytes)
        await host.publish_guild_message(guild_id, message)
"""

import hashlib
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

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
    from libp2p.tools.async_service.trio_service import background_trio_service

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


async def _read_framed(stream) -> bytes:
    """Read a length-prefixed frame and return the payload."""
    header = await stream.read(4)
    if len(header) < 4:
        raise ValueError("Truncated frame header")
    length = struct.unpack(">I", header)[0]
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


async def _read_frame_no_close(stream) -> bytes:
    """Read a length-prefixed frame without closing the stream (for request-response)."""
    header = await stream.read(4)
    if len(header) < 4:
        raise ValueError("Truncated frame header")
    length = struct.unpack(">I", header)[0]
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
BLOB_PROTOCOL = TProtocol("/autonet/blob/1.0.0")
# A blob request frame is "<hash>" or "<hash>|<hops>": how many times the
# serving peer may forward the request to its own peers on a miss, so a
# NAT'd daemon can reach a blob held only by another NAT'd daemon through
# a reachable peer both are connected to. A bare hash allows one hop.
BLOB_RELAY_MAX_HOPS = 1
EMBEDDING_PROTOCOL = TProtocol("/autonet/embeddings/1.0.0")
INFERENCE_REQUEST_PROTOCOL = TProtocol("/rpb/inference/1.0.0")
# Robot poll rail (docs/robots.md, 2026-10-25): polls, link pushes and
# receipts between daemons, one JSON request and one JSON response a stream.
ROBOT_PROTOCOL = TProtocol("/autonet/robot/1.0.0")
# Reachability check: we send our listen port, the peer answers with the IP
# it sees us on and whether a fresh TCP connect to that IP and port worked.
# Only the requester's own observed IP is ever dialled.
REACH_PROTOCOL = TProtocol("/autonet/reach/1.0.0")
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
    # Phase 7.4: substrate-backed inference advertisement.
    # Loose dict so the wire format can evolve. Set when this node
    # serves /rpb/inference/1.0.0 via SubstrateProvider. Empty dict
    # means "this node is not advertising inference."
    # Conventional keys (callers may add others):
    #   - "renderer_model": str  (e.g. "qwen3:4b")
    #   - "price_atn":      int  (ATN per probe, scaled to uint256
    #                              like the chain side)
    #   - "agent_address":  str  (0x address that should receive
    #                              payForService funds)
    #   - "schema":         int  (1, for forward-compat)
    inference: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "NodeCapability":
        d = json.loads(data.decode("utf-8"))
        return cls(**d)


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


class AutonetHost:
    """
    Main P2P host for Autonet nodes.

    Wraps a libp2p host and provides high-level P2P operations:
    - Discovery via Kademlia DHT and bootstrap peers
    - Latency-aware peer selection
    - Weight delta transfer (solver → aggregator)
    - Activation relay (inference pipeline)
    - Guild gossip (GossipSub)
    - Node capability advertisement

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
        transport: str = "tcp",
    ):
        self.node_id = node_id
        self._listen_port = listen_port
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
        # "tcp" or "ws". py-libp2p runs ONE transport per host, picked from
        # its listen address, so every node of a network must share it. "ws"
        # lets a relay sit behind an HTTPS proxy (wss on 443) and lets
        # HTTPS-only machines dial out (docs/p2p_networking.md).
        if transport not in ("tcp", "ws"):
            raise ValueError(f"unknown p2p transport {transport!r}")
        self._transport = transport
        if transport == "ws":
            # HTTP CONNECT proxies (HTTPS-only egress) for the ws dialer.
            from .ws_proxy import install as _install_ws_proxy
            _install_ws_proxy()
        self._bootstrap_redial_interval = bootstrap_redial_interval
        self._enable_quic = enable_quic
        self._enable_upnp = enable_upnp

        # Handlers registered by user code
        self._weight_handler: Optional[Callable] = None
        self._activation_handler: Optional[Callable] = None
        self._guild_handlers: Dict[str, Callable] = {}
        self._blob_handler: Optional[Callable] = None       # async (hash: str) -> Optional[bytes]
        self._blob_relay_sink: Optional[Callable] = None    # (hash, bytes) -> None, caches relayed blobs
        self._blob_relay_timeout = 5.0
        self._embedding_handler: Optional[Callable] = None  # async (peer_id, metadata, tensor_bytes)

        # State (initialized in run())
        self._host: Optional[IHost] = None
        self._latency_tracker: Optional[PeerLatencyTracker] = None
        self._gossipsub: Optional[GossipSub] = None
        self._pubsub: Optional[Pubsub] = None
        self._guild_subscriptions: Dict[str, Any] = {}
        self._known_capabilities: Dict[str, NodeCapability] = {}
        self._guilds: Dict[str, GuildMembership] = {}
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

        self._running = False
        self.logger = logging.getLogger(f"P2P[{node_id}]")

    @asynccontextmanager
    async def run(self):
        """
        Start the P2P host and all services.

        Usage:
            async with host.run():
                # host is now listening and discoverable
                await host.send_weight_delta(...)
        """
        suffix = "/ws" if self._transport == "ws" else ""
        listen_addrs = [
            Multiaddr(f"/ip4/{self._listen_host}/tcp/{self._listen_port}{suffix}")
        ]
        # py-libp2p's TCP transport dials only /ip4 and /ip6 (its bootstrap
        # resolves /dnsaddr only), so /dns4 tunnel hostnames resolve here.
        # The configured names are kept and resolved again on every redial.
        bootstrap = [self._resolve(a) for a in self._bootstrap_peers]

        key_pair = None
        if self._key_seed is not None:
            from libp2p.crypto.ed25519 import create_new_key_pair
            key_pair = create_new_key_pair(self._key_seed)
        self._host = new_host(
            key_pair=key_pair,
            listen_addrs=listen_addrs,
            enable_quic=self._enable_quic,
            # UPnP is handled in _auto_reach_loop: py-libp2p's own mapping
            # reads get_addrs, which we replace with the announce addrs.
            enable_upnp=False,
            bootstrap=bootstrap if bootstrap else None,
        )

        async with self._host.run(listen_addrs=listen_addrs):
            self._running = True
            peer_id = str(self._host.get_id())
            if self._announce_addrs:
                self._announce_resolved = self._resolve_announce_addrs()
            if self._announce_addrs or self._auto_reach or self._enable_upnp:
                p2p_part = Multiaddr(f"/p2p/{peer_id}")
                listen = self._host.get_addrs
                # Identify, the signed peer record and our capability all
                # read get_addrs, so replacing it on the host instance is
                # enough; the listen socket is unchanged.
                self._host.get_addrs = lambda: (
                    [a.encapsulate(p2p_part) for a in self._announce_resolved]
                    or listen()
                )
            addrs = self._host.get_addrs()

            # Update our capability with actual peer ID and addresses
            self._capability.peer_id = peer_id
            self._capability.listen_addrs = [str(a) for a in addrs]

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
            import trio
            self._trio_token = trio.lowlevel.current_trio_token()
            try:
                async with trio.open_nursery() as nursery:
                    self._nursery = nursery
                    if ((self._bootstrap_peers or self._announce_addrs)
                            and self._bootstrap_redial_interval > 0):
                        nursery.start_soon(self._maintain_addrs)
                    if not self._announce_addrs and (self._auto_reach or self._enable_upnp):
                        nursery.start_soon(self._auto_reach_loop)
                    self._ready_event.set()
                    try:
                        yield self
                    finally:
                        # Cancel background tasks (topic readers).
                        nursery.cancel_scope.cancel()
            finally:
                self._ready_event.clear()
                self._nursery = None
                self._trio_token = None
                self._running = False
                self.logger.info("P2P host shutting down")

    def _resolve(self, addr: str) -> str:
        """TCP dials only IPs, so names resolve here. The ws transport keeps
        names: TLS (SNI) and the HTTP Host header must carry them (an HTTPS
        proxy in front of a relay routes on them), and behind an HTTP proxy
        the proxy resolves them."""
        return addr if self._transport == "ws" else resolve_dns_multiaddr(addr)

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

    async def _maintain_addrs(self) -> None:
        """Track address changes, every ``bootstrap_redial_interval``.

        Announce names are resolved again; if the result changed (a
        dynamic-DNS or tunnel name now points elsewhere) we advertise the
        new address at once. Bootstrap peers we are not connected to are
        redialled at their freshly resolved address: py-libp2p dials them
        once at start, so a daemon that started first, lost its link, or
        whose bootstrap peer moved would otherwise stay cut off.
        """
        import trio
        while True:
            await trio.sleep(self._bootstrap_redial_interval)
            if self._announce_addrs:
                fresh = await trio.to_thread.run_sync(self._resolve_announce_addrs)
                if fresh and [str(a) for a in fresh] != [str(a) for a in self._announce_resolved]:
                    self.logger.info(f"Announce address changed to {[str(a) for a in fresh]}")
                    self._announce_resolved = fresh
                    self._capability.listen_addrs = self.addrs
                    await self.advertise_capability()
            connected = set(self.get_connected_peers())
            for addr in list(self._bootstrap_peers):
                pid = addr.split("/p2p/")[-1] if "/p2p/" in addr else ""
                if not pid or pid == self.peer_id or pid in connected:
                    continue
                addr = await trio.to_thread.run_sync(self._resolve, addr)
                # Drop cached addresses so a peer that moved is dialled
                # only at its current address, not the stale one first.
                try:
                    self._host.get_peerstore().clear_addrs(PeerID.from_base58(pid))
                except Exception:
                    pass
                with trio.move_on_after(10):
                    await self.connect_to_peer(addr)

    def _listen_port_actual(self) -> int:
        for addr in self._host.get_transport_addrs():
            port = addr.value_for_protocol("tcp")
            if port:
                return int(port)
        return 0

    async def _set_auto_announce(self, addr: Optional[str]) -> None:
        """Announce ``addr`` (or go back to the listen addrs on None) and
        tell peers if that changed."""
        fresh = [Multiaddr(addr)] if addr else []
        if [str(a) for a in fresh] == [str(a) for a in self._announce_resolved]:
            return
        self.logger.info(
            f"Reachable at {addr}" if addr else "No longer reachable; announcing listen addrs"
        )
        self._announce_resolved = fresh
        self._capability.listen_addrs = self.addrs
        await self.advertise_capability()

    async def _auto_reach_loop(self) -> None:
        """Find and confirm our public address, every ``reach_interval``.

        With UPnP on, the router maps our listen port and reports its
        public IP. Either way a connected peer is asked what IP it sees
        us on and whether it can connect to us there; we announce
        ``/ip4/<ip>/tcp/<port>`` only when it can. Repeating the check
        catches a changed public IP. A daemon nobody can reach keeps
        working NAT'd (dialling out).
        """
        import trio
        port = self._listen_port_actual()
        upnp = None
        if self._enable_upnp and port:
            try:
                from libp2p.discovery.upnp.upnp import UpnpManager
                upnp = UpnpManager()
                if not (await upnp.discover() and await upnp.add_port_mapping(port, "TCP")):
                    upnp = None
            except Exception as e:  # miniupnpc missing or the router said no
                self.logger.info(f"UPnP unavailable: {e}")
                upnp = None
        try:
            while True:
                if upnp is not None:
                    # Leases expire and routers reboot: renew each round.
                    await upnp.add_port_mapping(port, "TCP")
                peers = self._host.get_connected_peers()
                verdict = None
                for pid in peers[:3]:
                    with trio.move_on_after(15):
                        verdict = await self.check_reachability(pid, port)
                    if verdict is not None:
                        break
                if verdict is None:
                    await trio.sleep(2.0)  # no peer answered yet; retry soon
                    continue
                ip = verdict.get("ip") or ""
                if verdict.get("reachable") and _is_public_ip(ip):
                    proto = "ip6" if ":" in ip else "ip4"
                    await self._set_auto_announce(f"/{proto}/{ip}/tcp/{port}")
                else:
                    await self._set_auto_announce(None)
                await trio.sleep(self._reach_interval)
        finally:
            if upnp is not None:
                with trio.CancelScope(shield=True), trio.move_on_after(3):
                    await upnp.remove_port_mapping(port, "TCP")

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
        to it on the port it named. We dial only that observed IP."""
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
        import trio
        while True:
            try:
                if self._pubsub is not None and self._host is not None:
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
            from libp2p.tools.async_service.trio_service import TrioManager
            self._pubsub_manager = TrioManager(self._pubsub)
            self._gossipsub_manager = TrioManager(self._gossipsub)
            self._nursery.start_soon(self._pubsub_manager.run)
            self._nursery.start_soon(self._gossipsub_manager.run)
            await self._pubsub_manager.wait_started()
            await self._gossipsub_manager.wait_started()
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
        """Handle incoming capability advertisement."""
        try:
            data = await _read_framed(stream)
            cap = NodeCapability.from_bytes(data)
            self._known_capabilities[cap.peer_id] = cap

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

    def set_blob_relay_sink(self, sink: Callable):
        """Set where blobs fetched on a peer's behalf are cached: sink(content_hash, data)"""
        self._blob_relay_sink = sink

    async def fetch_blob(
        self,
        target_peer_id: PeerID,
        content_hash: str,
        timeout: float = 30.0,
        hops: int = BLOB_RELAY_MAX_HOPS,
    ) -> Optional[bytes]:
        """
        Request a blob from a peer by content hash.

        ``hops`` is how many times the peer may forward the request on a
        miss (see ``BLOB_RELAY_MAX_HOPS``).

        Returns the blob bytes if found and integrity-verified, None otherwise.
        """
        if not self._host:
            return None
        try:
            stream = await self._host.new_stream(target_peer_id, [BLOB_PROTOCOL])
            # Write request: hash as UTF-8 (no stream close — waiting for response)
            request = content_hash if hops == BLOB_RELAY_MAX_HOPS else f"{content_hash}|{hops}"
            await _write_frame_no_close(stream, request.encode("utf-8"))
            # Read response with timeout
            data = b""
            with trio.move_on_after(timeout):
                data = await _read_frame_no_close(stream)
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
        except Exception as e:
            self.logger.debug(
                f"Failed to fetch blob {content_hash[:16]} "
                f"from {str(target_peer_id)[:16]}: {e}"
            )
            return None

    async def _handle_blob_stream(self, stream):
        """Handle incoming blob request: read content hash, write blob bytes."""
        try:
            hash_bytes = await _read_frame_no_close(stream)
            content_hash, _, hops_str = hash_bytes.decode("utf-8").strip().partition("|")
            try:
                hops = min(int(hops_str), BLOB_RELAY_MAX_HOPS) if hops_str else BLOB_RELAY_MAX_HOPS
            except ValueError:
                hops = 0

            blob_data: Optional[bytes] = None
            if self._blob_handler:
                try:
                    blob_data = await self._blob_handler(content_hash)
                except Exception as e:
                    self.logger.debug(f"Blob handler error for {content_hash[:16]}: {e}")

            if blob_data is None and hops > 0:
                requester = stream.muxed_conn.peer_id
                others = [p for p in self._host.get_connected_peers() if p != requester]
                blob_data = await self.fetch_blob_from_peers(
                    others, content_hash,
                    timeout_per_peer=self._blob_relay_timeout, hops=hops - 1,
                )
                if blob_data is not None and self._blob_relay_sink:
                    try:
                        self._blob_relay_sink(content_hash, blob_data)
                    except Exception as e:
                        self.logger.debug(f"Blob relay sink error for {content_hash[:16]}: {e}")

            # Respond with blob data, or empty frame if not found
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
        """
        Try to fetch a blob from any connected peer.

        Tries peers concurrently and returns the first successful response.
        """
        if not self._host:
            return None
        return await self.fetch_blob_from_peers(
            self._host.get_connected_peers(), content_hash,
            timeout_per_peer=timeout_per_peer,
        )

    async def fetch_blob_from_peers(
        self,
        peer_ids: List[Any],
        content_hash: str,
        timeout_per_peer: float = 10.0,
        hops: int = BLOB_RELAY_MAX_HOPS,
    ) -> Optional[bytes]:
        """Ask the given peers (PeerIDs or base58 strings) concurrently;
        the first integrity-verified answer wins and cancels the rest."""
        if not self._host or not peer_ids:
            return None
        from libp2p.peer.id import ID as _PeerID

        result: List[Optional[bytes]] = [None]

        async def try_peer(pid, cancel_scope):
            with trio.move_on_after(timeout_per_peer):
                data = await self.fetch_blob(
                    pid, content_hash, timeout=timeout_per_peer, hops=hops,
                )
                if data is not None and result[0] is None:
                    result[0] = data
                    cancel_scope.cancel()

        async with trio.open_nursery() as nursery:
            for pid in peer_ids:
                if isinstance(pid, str):
                    try:
                        pid = _PeerID.from_base58(pid)
                    except Exception:
                        continue
                nursery.start_soon(try_peer, pid, nursery.cancel_scope)

        return result[0]

    # =========================================================================
    # RPB Inference Protocol (/rpb/inference/1.0.0)
    # =========================================================================

    async def request_inference(self, target_peer_id: str, request: dict) -> dict:
        """Send an inference request to a serving peer.

        Uses the /rpb/inference/1.0.0 protocol. The serving peer dispatches
        the request through whatever handler it registered via
        ``set_inference_handler`` — typically the substrate-backed handler
        from Phase 7.3 (substrate locate + local LLM render).

        Args:
            target_peer_id: Peer ID of the serving peer.
            request: Dict with messages, system, model, max_tokens, tools,
                temperature. The exact shape SubstrateProvider.send accepts.

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
        (or {"error": "..."}). Phase 7.3 wires the substrate-backed
        handler from ``nodes.common.substrate_inference_handler``;
        future phases may install other handlers on top of the same
        protocol (e.g. the sponsor/dependent flow in Phase 8+)."""
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
        }
