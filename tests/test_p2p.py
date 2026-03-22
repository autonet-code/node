"""
Tests for the P2P communication layer.

Uses real localhost libp2p hosts for integration tests (no mocking of
the network layer — these test actual TCP connections between hosts).

Run: python -m pytest tests/test_p2p.py -v
"""

import sys
import json
import hashlib
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio
import pytest

from multiaddr import Multiaddr
from libp2p import new_host
from libp2p.custom_types import TProtocol
from libp2p.peer.peerinfo import PeerInfo, info_from_p2p_addr

from nodes.common.p2p import (
    AutonetHost,
    NodeCapability,
    GuildMembership,
    PeerLatencyTracker,
    PeerLatency,
    WEIGHTS_PROTOCOL,
    ACTIVATIONS_PROTOCOL,
    CAPABILITY_PROTOCOL,
)
from nodes.common.config import P2PConfig, load_config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def p2p_config():
    return P2PConfig(listen_port=0, listen_host="127.0.0.1")


# =============================================================================
# Tests: NodeCapability serialization
# =============================================================================


class TestNodeCapability:
    def test_serialize_roundtrip(self):
        """NodeCapability survives JSON serialization."""
        cap = NodeCapability(
            peer_id="12D3KooW...",
            node_id="solver-0",
            roles=["solver", "aggregator"],
            gpu_type="RTX 4090",
            gpu_memory_mb=24576,
            bandwidth_mbps=1000.0,
            modules_hosted=["visual_encoder", "text_encoder"],
            listen_addrs=["/ip4/127.0.0.1/tcp/9000"],
        )

        data = cap.to_bytes()
        restored = NodeCapability.from_bytes(data)

        assert restored.peer_id == cap.peer_id
        assert restored.node_id == cap.node_id
        assert restored.roles == cap.roles
        assert restored.gpu_type == "RTX 4090"
        assert restored.gpu_memory_mb == 24576
        assert restored.modules_hosted == cap.modules_hosted

    def test_defaults(self):
        """NodeCapability has sensible defaults."""
        cap = NodeCapability(peer_id="abc", node_id="test")
        assert cap.roles == []
        assert cap.gpu_memory_mb == 0
        assert cap.modules_hosted == []
        assert cap.timestamp > 0


# =============================================================================
# Tests: GuildMembership
# =============================================================================


class TestGuildMembership:
    def test_topic(self):
        """Guild topic is derived from guild_id."""
        guild = GuildMembership(guild_id="vision_guild")
        assert guild.topic == "/autonet/guild/vision_guild"

    def test_members(self):
        """Guild tracks members."""
        guild = GuildMembership(
            guild_id="lang_guild",
            member_peer_ids={"peer1", "peer2"},
            modules=["text_encoder"],
        )
        assert "peer1" in guild.member_peer_ids
        assert "text_encoder" in guild.modules


# =============================================================================
# Tests: P2PConfig in config system
# =============================================================================


class TestP2PConfig:
    def test_config_has_p2p(self):
        """AutonetConfig includes P2P section."""
        cfg = load_config()
        assert hasattr(cfg, "p2p")
        assert cfg.p2p.listen_port == 0
        assert cfg.p2p.enable_quic is False
        assert cfg.p2p.enable_upnp is False

    def test_p2p_defaults(self):
        """P2PConfig has correct defaults."""
        p2p = P2PConfig()
        assert p2p.listen_port == 0
        assert p2p.listen_host == "0.0.0.0"
        assert p2p.bootstrap_peers == []
        assert p2p.capability_advertise_interval == 60
        assert p2p.ping_interval == 30


# =============================================================================
# Tests: AutonetHost basic lifecycle (Story 4.1)
# =============================================================================


class TestAutonetHostLifecycle:
    @pytest.mark.trio
    async def test_host_starts_and_stops(self):
        """Host starts, gets peer ID and addresses, then stops."""
        host = AutonetHost(node_id="test-0", listen_host="127.0.0.1")
        async with host.run():
            assert host.peer_id != ""
            assert len(host.addrs) > 0
            assert host._running is True
        assert host._running is False

    @pytest.mark.trio
    async def test_host_summary(self):
        """Host summary returns correct info."""
        host = AutonetHost(node_id="test-1", listen_host="127.0.0.1")
        async with host.run():
            summary = host.summary()
            assert summary["node_id"] == "test-1"
            assert summary["peer_id"] != ""
            assert summary["running"] is True
            assert isinstance(summary["addrs"], list)

    @pytest.mark.trio
    async def test_two_hosts_connect(self):
        """Two hosts can connect to each other."""
        host1 = AutonetHost(node_id="node-1", listen_host="127.0.0.1")
        host2 = AutonetHost(node_id="node-2", listen_host="127.0.0.1")

        async with host1.run():
            async with host2.run():
                # Connect host1 to host2 via multiaddr
                h2_addr = host2.addrs[0]
                success = await host1.connect_to_peer(h2_addr)
                assert success is True

                peers = host1.get_connected_peers()
                assert len(peers) >= 1

    @pytest.mark.trio
    async def test_connect_failure(self):
        """Connecting to a non-existent peer returns False."""
        host = AutonetHost(node_id="test-fail", listen_host="127.0.0.1")
        async with host.run():
            success = await host.connect_to_peer(
                "/ip4/127.0.0.1/tcp/1/p2p/12D3KooWDeadBeefDeadBeefDeadBeefDeadBeefDeadBeefDeadBe"
            )
            assert success is False


# =============================================================================
# Tests: Weight Delta Transfer (Story 4.3)
# =============================================================================


class TestWeightDeltaTransfer:
    @pytest.mark.trio
    async def test_send_receive_weight_delta(self):
        """Weight deltas are sent and received with hash verification."""
        host1 = AutonetHost(node_id="solver-0", listen_host="127.0.0.1")
        host2 = AutonetHost(node_id="aggregator-0", listen_host="127.0.0.1")

        received_data = []

        async def weight_handler(peer_id, data_hash, data):
            received_data.append((peer_id, data_hash, data))

        host2.set_weight_handler(weight_handler)

        async with host1.run():
            async with host2.run():
                # Connect
                h2_info = PeerInfo(
                    host2._host.get_id(), host2._host.get_addrs()
                )
                await host1.connect_to_info(h2_info)

                # Send weight delta
                delta = b"fake-weight-delta-payload-12345"
                h2_pid = host2._host.get_id()
                success = await host1.send_weight_delta(h2_pid, delta)
                assert success is True

                # Allow handler to process
                await trio.sleep(0.2)

                assert len(received_data) == 1
                _, received_hash, received_payload = received_data[0]
                assert received_payload == delta
                expected_hash = hashlib.sha256(delta).hexdigest()
                assert received_hash == expected_hash


# =============================================================================
# Tests: Activation Relay (Story 4.4)
# =============================================================================


class TestActivationRelay:
    @pytest.mark.trio
    async def test_send_receive_activation(self):
        """Activation tensors are relayed between hosts with metadata."""
        host1 = AutonetHost(node_id="module-a", listen_host="127.0.0.1")
        host2 = AutonetHost(node_id="module-b", listen_host="127.0.0.1")

        received_data = []

        async def activation_handler(peer_id, metadata, tensor_bytes):
            received_data.append((peer_id, metadata, tensor_bytes))

        host2.set_activation_handler(activation_handler)

        async with host1.run():
            async with host2.run():
                h2_info = PeerInfo(
                    host2._host.get_id(), host2._host.get_addrs()
                )
                await host1.connect_to_info(h2_info)

                # Simulate a 196x768 fp16 activation (300KB)
                tensor_data = b"\x00\x01" * (196 * 768)  # ~300KB
                metadata = {
                    "module_id": "visual_encoder",
                    "batch_size": 1,
                    "shape": [196, 768],
                }

                h2_pid = host2._host.get_id()
                success = await host1.send_activation(
                    h2_pid, tensor_data, metadata
                )
                assert success is True

                await trio.sleep(0.3)

                assert len(received_data) == 1
                _, recv_meta, recv_tensor = received_data[0]
                assert recv_meta["module_id"] == "visual_encoder"
                assert recv_meta["shape"] == [196, 768]
                assert recv_tensor == tensor_data


# =============================================================================
# Tests: Node Capability Advertisement (Story 4.7)
# =============================================================================


class TestCapabilityAdvertisement:
    @pytest.mark.trio
    async def test_advertise_and_discover(self):
        """Node advertises capability, peer discovers it."""
        cap1 = NodeCapability(
            peer_id="", node_id="solver-0",
            roles=["solver"],
            gpu_type="RTX 4090",
            gpu_memory_mb=24576,
            modules_hosted=["visual_encoder"],
        )
        host1 = AutonetHost(
            node_id="solver-0", listen_host="127.0.0.1", capability=cap1
        )
        host2 = AutonetHost(node_id="watcher-0", listen_host="127.0.0.1")

        async with host1.run():
            async with host2.run():
                # Connect
                h2_info = PeerInfo(
                    host2._host.get_id(), host2._host.get_addrs()
                )
                await host1.connect_to_info(h2_info)

                # Advertise capability from host1
                await host1.advertise_capability()
                await trio.sleep(0.2)

                # host2 should now know about host1's capability
                known = host2.known_capabilities
                assert len(known) >= 1

                cap = list(known.values())[0]
                assert cap.node_id == "solver-0"
                assert "solver" in cap.roles
                assert cap.gpu_type == "RTX 4090"
                assert "visual_encoder" in cap.modules_hosted

    def test_find_capable_peers(self):
        """find_capable_peers filters by role, GPU, and module."""
        host = AutonetHost(node_id="test", listen_host="127.0.0.1")

        host._known_capabilities = {
            "peer1": NodeCapability(
                peer_id="peer1", node_id="solver-0",
                roles=["solver"], gpu_memory_mb=24576,
                modules_hosted=["visual_encoder"],
            ),
            "peer2": NodeCapability(
                peer_id="peer2", node_id="aggregator-0",
                roles=["aggregator"], gpu_memory_mb=8192,
                modules_hosted=["text_encoder"],
            ),
            "peer3": NodeCapability(
                peer_id="peer3", node_id="solver-1",
                roles=["solver"], gpu_memory_mb=4096,
                modules_hosted=["visual_encoder"],
            ),
        }

        # Filter by role
        solvers = host.find_capable_peers(role="solver")
        assert len(solvers) == 2

        # Filter by GPU memory
        big_gpu = host.find_capable_peers(min_gpu_memory=10000)
        assert len(big_gpu) == 1
        assert big_gpu[0].node_id == "solver-0"

        # Filter by module
        visual = host.find_capable_peers(module_id="visual_encoder")
        assert len(visual) == 2

        # Combined filter
        specific = host.find_capable_peers(
            role="solver", min_gpu_memory=10000, module_id="visual_encoder"
        )
        assert len(specific) == 1

    def test_update_capability(self):
        """update_capability modifies local capability."""
        host = AutonetHost(node_id="test", listen_host="127.0.0.1")
        host.update_capability(
            roles=["solver", "aggregator"],
            gpu_type="A100",
            gpu_memory_mb=40000,
        )
        assert host._capability.roles == ["solver", "aggregator"]
        assert host._capability.gpu_type == "A100"
        assert host._capability.gpu_memory_mb == 40000


# =============================================================================
# Tests: PeerLatencyTracker (Story 4.2)
# =============================================================================


class TestPeerLatencyTracker:
    @pytest.mark.trio
    async def test_ping_peer(self):
        """Ping measures RTT to a connected peer."""
        host1 = AutonetHost(node_id="pinger", listen_host="127.0.0.1")
        host2 = AutonetHost(node_id="target", listen_host="127.0.0.1")

        async with host1.run():
            async with host2.run():
                h2_info = PeerInfo(
                    host2._host.get_id(), host2._host.get_addrs()
                )
                await host1.connect_to_info(h2_info)

                tracker = host1.latency_tracker
                rtt = await tracker.ping_peer(host2._host.get_id())

                assert rtt is not None
                assert rtt > 0  # Should be measurable
                assert rtt < 1000  # < 1 second for localhost

                # Check stored data
                pid_str = str(host2._host.get_id())
                entry = tracker.get_latency(pid_str)
                assert entry is not None
                assert entry.samples == 1
                assert entry.reachable is True

    def test_select_fastest_peers(self):
        """select_fastest_peers returns sorted by EMA RTT."""
        host = AutonetHost(node_id="test", listen_host="127.0.0.1")
        tracker = PeerLatencyTracker.__new__(PeerLatencyTracker)
        tracker._alpha = 0.3
        tracker._peers = {
            "peer1": PeerLatency(
                peer_id="peer1", rtt_ms=50, ema_rtt_ms=50, samples=5, reachable=True
            ),
            "peer2": PeerLatency(
                peer_id="peer2", rtt_ms=10, ema_rtt_ms=10, samples=5, reachable=True
            ),
            "peer3": PeerLatency(
                peer_id="peer3", rtt_ms=30, ema_rtt_ms=30, samples=5, reachable=True
            ),
            "peer4": PeerLatency(
                peer_id="peer4", rtt_ms=5, ema_rtt_ms=5, samples=0, reachable=True
            ),  # No samples
            "peer5": PeerLatency(
                peer_id="peer5", rtt_ms=1, ema_rtt_ms=1, samples=5, reachable=False
            ),  # Unreachable
        }

        fastest = tracker.select_fastest_peers(3)
        assert len(fastest) == 3
        assert fastest[0].peer_id == "peer2"  # 10ms
        assert fastest[1].peer_id == "peer3"  # 30ms
        assert fastest[2].peer_id == "peer1"  # 50ms

    def test_select_peers_for_route(self):
        """Route selection picks fastest peer per module."""
        tracker = PeerLatencyTracker.__new__(PeerLatencyTracker)
        tracker._alpha = 0.3
        tracker._peers = {
            "peer_a": PeerLatency(
                peer_id="peer_a", ema_rtt_ms=20, samples=5, reachable=True
            ),
            "peer_b": PeerLatency(
                peer_id="peer_b", ema_rtt_ms=10, samples=5, reachable=True
            ),
            "peer_c": PeerLatency(
                peer_id="peer_c", ema_rtt_ms=30, samples=5, reachable=True
            ),
        }

        capabilities = {
            "peer_a": NodeCapability(
                peer_id="peer_a", node_id="n1",
                modules_hosted=["visual_encoder", "predictor"]
            ),
            "peer_b": NodeCapability(
                peer_id="peer_b", node_id="n2",
                modules_hosted=["visual_encoder", "text_encoder"]
            ),
            "peer_c": NodeCapability(
                peer_id="peer_c", node_id="n3",
                modules_hosted=["predictor"]
            ),
        }

        route = tracker.select_peers_for_route(
            ["visual_encoder", "text_encoder", "predictor"],
            capabilities,
        )

        assert route[0] == ("visual_encoder", "peer_b")  # peer_b is fastest (10ms)
        assert route[1] == ("text_encoder", "peer_b")    # Only peer_b has it
        assert route[2] == ("predictor", "peer_a")        # peer_a (20ms) < peer_c (30ms)


# =============================================================================
# Tests: Guild Gossip (Story 4.5)
# =============================================================================


class TestGuildGossip:
    def test_guild_membership(self):
        """Guild membership abstraction works."""
        guild = GuildMembership(
            guild_id="vision_guild",
            member_peer_ids={"peer1", "peer2"},
            modules=["visual_encoder"],
        )
        assert guild.topic == "/autonet/guild/vision_guild"
        assert "peer1" in guild.member_peer_ids

    def test_get_guild_for_module(self):
        """get_guild_for_module looks up guild by module ID."""
        host = AutonetHost(node_id="test", listen_host="127.0.0.1")
        host._guilds = {
            "vision": GuildMembership(
                guild_id="vision", modules=["visual_encoder"]
            ),
            "lang": GuildMembership(
                guild_id="lang", modules=["text_encoder"]
            ),
        }

        assert host.get_guild_for_module("visual_encoder").guild_id == "vision"
        assert host.get_guild_for_module("text_encoder").guild_id == "lang"
        assert host.get_guild_for_module("predictor") is None


# =============================================================================
# Tests: Cross-guild routing (Story 4.6)
# =============================================================================


class TestCrossGuildRouting:
    def test_select_peers_for_route_no_tracker(self):
        """Route selection returns empty peer IDs when no tracker."""
        host = AutonetHost(node_id="test", listen_host="127.0.0.1")
        # Not running, so no tracker
        route = host.select_peers_for_route(["mod_a", "mod_b"])
        assert route == [("mod_a", ""), ("mod_b", "")]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
