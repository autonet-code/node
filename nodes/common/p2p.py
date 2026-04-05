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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    # Aggregate (computed from peer gossip)
    known_contributors: int = 0                           # unique peers with cycles > 0


@dataclass
class NodeCapability:
    """A node's advertised capabilities."""
    peer_id: str
    node_id: str
    roles: List[str] = field(default_factory=list)       # e.g. ["solver", "aggregator"]
    gpu_type: str = ""                                    # e.g. "RTX 4090"
    gpu_memory_mb: int = 0
    bandwidth_mbps: float = 0.0
    modules_hosted: List[str] = field(default_factory=list)  # module IDs
    listen_addrs: List[str] = field(default_factory=list)
    agents: List[Dict] = field(default_factory=list)      # list of AgentAdvertisement dicts
    model_state: Dict = field(default_factory=dict)       # ModelState as dict
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
            # Note: GossipSub is initialized lazily in join_guild()

            try:
                yield self
            finally:
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
        self._gossipsub = GossipSub(
            protocols=protocols,
            degree=6,
            degree_low=4,
            degree_high=12,
            heartbeat_interval=120,
        )
        self._pubsub = Pubsub(
            host=self._host,
            router=self._gossipsub,
            strict_signing=False,
        )

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
        """Send an inference request to a provider node.

        Uses the /rpb/inference/1.0.0 protocol. The provider node forwards
        the request to its own centralized API subscription (Path A) and
        returns the response.

        Args:
            target_peer_id: Peer ID of the provider node.
            request: Dict with messages, system, model, max_tokens, tools, temperature.

        Returns:
            Dict with text, tool_calls, usage, model fields from the provider.

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

    async def _serve_inference_locally(self, request: dict) -> dict:
        """Serve an inference request using this node's own provider.

        This is the sponsor side of Path A — the sponsor node has a
        centralized API subscription and uses it to serve dependent nodes.

        Override this method or set self._inference_handler to customize.
        """
        handler = getattr(self, "_inference_handler", None)
        if handler:
            return await handler(request)

        # Default: return error indicating no provider configured
        return {"error": "This node has no inference provider configured for sponsorship"}

    async def advertise_models(self, models: List[str], agent_address: str) -> None:
        """Advertise available models on the provider registry gossip topic.

        Publishes to PROVIDER_REGISTRY_TOPIC so that RPB provider nodes
        can discover available inference endpoints.
        """
        # TODO: Publish to PROVIDER_REGISTRY_TOPIC
        # 1. Ensure GossipSub is initialized
        # 2. Build advertisement message with models, peer_id, agent_address
        # 3. Publish to PROVIDER_REGISTRY_TOPIC
        pass

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
