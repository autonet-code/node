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
EMBEDDING_PROTOCOL = TProtocol("/autonet/embeddings/1.0.0")
INFERENCE_REQUEST_PROTOCOL = TProtocol("/rpb/inference/1.0.0")
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
    #                              payForInference funds)
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

    Requires ``pip install autonet-computer[network]`` for P2P dependencies.
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
    ):
        self.node_id = node_id
        self._listen_port = listen_port
        self._listen_host = listen_host
        self._bootstrap_peers = bootstrap_peers or []
        self._enable_quic = enable_quic
        self._enable_upnp = enable_upnp

        # Handlers registered by user code
        self._weight_handler: Optional[Callable] = None
        self._activation_handler: Optional[Callable] = None
        self._guild_handlers: Dict[str, Callable] = {}
        self._blob_handler: Optional[Callable] = None       # async (hash: str) -> Optional[bytes]
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
        listen_addrs = [
            Multiaddr(f"/ip4/{self._listen_host}/tcp/{self._listen_port}")
        ]

        self._host = new_host(
            listen_addrs=listen_addrs,
            enable_quic=self._enable_quic,
            enable_upnp=self._enable_upnp,
            bootstrap=self._bootstrap_peers if self._bootstrap_peers else None,
        )

        async with self._host.run(listen_addrs=listen_addrs):
            self._running = True
            peer_id = str(self._host.get_id())
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

    async def connect_to_peer(self, multiaddr_str: str) -> bool:
        """Connect to a peer by multiaddr string."""
        if not self._host:
            return False
        try:
            maddr = Multiaddr(multiaddr_str)
            info = info_from_p2p_addr(maddr)
            await self._host.connect(info)
            self.logger.info(f"Connected to {str(info.peer_id)[:16]}...")
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
        # Pubsub is an async-service; it has to be run via TrioManager
        # so its peer-queue handler (which wires libp2p's notifee into
        # router.add_peer / mesh GRAFT) actually starts. Without this,
        # the mesh never forms and messages don't propagate.
        # Phase 10.6.
        if self._nursery is not None:
            from libp2p.tools.async_service.trio_service import TrioManager
            pubsub_manager = TrioManager(self._pubsub)
            self._nursery.start_soon(pubsub_manager.run)
            await pubsub_manager.wait_started()

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
        if token is None:
            self.logger.debug("publish_topic_threadsafe before host run; dropping")
            return
        import trio

        async def _do():
            await self.publish_topic(topic, data)

        try:
            trio.from_thread.run(_do, trio_token=token)
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

    async def fetch_blob(
        self,
        target_peer_id: PeerID,
        content_hash: str,
        timeout: float = 30.0,
    ) -> Optional[bytes]:
        """
        Request a blob from a peer by content hash.

        Returns the blob bytes if found and integrity-verified, None otherwise.
        """
        if not self._host:
            return None
        try:
            stream = await self._host.new_stream(target_peer_id, [BLOB_PROTOCOL])
            # Write request: hash as UTF-8 (no stream close — waiting for response)
            await _write_frame_no_close(stream, content_hash.encode("utf-8"))
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
            content_hash = hash_bytes.decode("utf-8").strip()

            blob_data: Optional[bytes] = None
            if self._blob_handler:
                try:
                    blob_data = await self._blob_handler(content_hash)
                except Exception as e:
                    self.logger.debug(f"Blob handler error for {content_hash[:16]}: {e}")

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
        peers = self._host.get_connected_peers()
        if not peers:
            return None

        result: List[Optional[bytes]] = [None]

        async def try_peer(pid):
            if result[0] is not None:
                return
            data = await self.fetch_blob(pid, content_hash, timeout=timeout_per_peer)
            if data is not None:
                result[0] = data

        async with trio.open_nursery() as nursery:
            for pid in peers:
                nursery.start_soon(try_peer, pid)

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
            await _write_framed(stream, req_bytes)

            # Read response
            resp_bytes = await _read_framed(stream)
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
            data = await _read_framed(stream)
            request = json.loads(data.decode("utf-8"))

            self.logger.info("Received inference request (%d bytes)", len(data))

            # Forward through this node's configured provider
            response = await self._serve_inference_locally(request)

            resp_bytes = json.dumps(response).encode("utf-8")
            await _write_framed(stream, resp_bytes)

        except Exception as e:
            self.logger.warning("Failed to handle inference request: %s", e)
            try:
                error_resp = json.dumps({"error": str(e)}).encode("utf-8")
                await _write_framed(stream, error_resp)
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

    def set_inference_advertisement(
        self,
        *,
        renderer_model: str,
        price_atn: int,
        agent_address: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark this node as an inference provider with the given pricing.

        Updates ``self._capability``: adds ``"inference-provider"`` to
        ``roles`` (idempotent) and sets ``inference`` to a dict carrying
        the renderer model id, the per-probe price, and the address that
        should receive ``payForInference`` payments.

        ``extras`` lets callers extend the dict with custom fields
        (e.g. allowed model tiers, regional hints). The schema key is
        always set so future format changes can fan out.

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
        cap.inference = adv

    def clear_inference_advertisement(self) -> None:
        """Remove this node's inference advertisement: drop the role
        and clear the ``inference`` dict. Useful when the agent
        served by this daemon stops offering inference (e.g. its
        local renderer goes offline, or the agent deactivates)."""
        cap = self._capability
        if "inference-provider" in cap.roles:
            cap.roles = [r for r in cap.roles if r != "inference-provider"]
        cap.inference = {}

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
    ) -> List[NodeCapability]:
        """Find peers advertising inference capability.

        Filters ``known_capabilities`` for entries that:
          - have ``"inference-provider"`` in ``roles``,
          - carry a non-empty ``inference`` advertisement dict,
          - if ``max_price_atn`` is set, ``price_atn`` <= the cap,
          - if ``renderer_model`` is set, exact match,
          - if ``require_agent_address`` is True (default), the
            advertised agent_address is non-empty (so payment can
            actually route to a real address).

        Returns a list of ``NodeCapability``, sorted by latency
        ascending (peers we have measurements for) then by price
        ascending (cheapest wins ties / unmeasured peers).
        """
        out: List[NodeCapability] = []
        for cap in self._known_capabilities.values():
            if "inference-provider" not in cap.roles:
                continue
            if not cap.inference:
                continue
            if require_agent_address and not cap.inference.get("agent_address"):
                continue
            if (
                max_price_atn is not None
                and int(cap.inference.get("price_atn", 0)) > int(max_price_atn)
            ):
                continue
            if renderer_model is not None:
                if str(cap.inference.get("renderer_model", "")) != str(renderer_model):
                    continue
            out.append(cap)

        # Sort: known-latency peers ascending by latency, then by
        # advertised price ascending. Peers without a latency reading
        # land after all measured peers, sorted just by price.
        def _sort_key(c: NodeCapability) -> Tuple[int, float, int]:
            tracker = getattr(self, "_latency_tracker", None)
            lat_obj = tracker.get_latency(c.peer_id) if tracker is not None else None
            # PeerLatencyTracker.get_latency returns a PeerLatency
            # object (or None). Use ema_rtt_ms when reachable; treat
            # unreachable / missing as "unmeasured".
            if (
                lat_obj is not None
                and getattr(lat_obj, "reachable", False)
                and getattr(lat_obj, "samples", 0) > 0
            ):
                return (0, float(lat_obj.ema_rtt_ms), int(c.inference.get("price_atn", 0)))
            return (1, float("inf"), int(c.inference.get("price_atn", 0)))

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
