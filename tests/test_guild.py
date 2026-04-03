"""
Tests for guild formation (Epic 8).

Exercises:
1. GuildManager: discovery, recommendation, join/leave, health monitoring
2. GuildConfig: config loading and defaults
3. Aggregator guild-level and network-level aggregation logic
4. Weighted guild aggregation math

Run: python -m pytest tests/test_guild.py -v
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.common.blockchain import TransactionResult
from nodes.common.config import (
    AutonetConfig,
    GuildConfig,
    NodeConfig,
    load_config,
)
from nodes.common.guild_manager import (
    GuildManager,
    GuildInfo,
    GuildReputationInfo,
    GuildRecommendation,
)


# =============================================================================
# Helpers
# =============================================================================


def _ok() -> TransactionResult:
    return TransactionResult(success=True, tx_hash="0xabc")


def _fail(msg: str = "reverted") -> TransactionResult:
    return TransactionResult(success=False, error=msg)


class FakeContractHandle:
    """Minimal stand-in for a contract object."""
    def __init__(self, name: str, address: str = "0xGuildRegistry"):
        self.name = name
        self.address = address
        self.functions = MagicMock()


def make_guild_registry(
    has_guild_registry: bool = True,
    guild_count: int = 0,
) -> MagicMock:
    """Build a mock ContractRegistry for guild tests."""
    registry = MagicMock()

    guild_contract = FakeContractHandle("GuildRegistry") if has_guild_registry else None

    def _get(name: str):
        if name == "GuildRegistry" and has_guild_registry:
            return guild_contract
        if name == "AutonetEconomy":
            return None
        if name == "AutonetDAO":
            return None
        return None

    registry.get = MagicMock(side_effect=_get)
    registry.call_contract = MagicMock(return_value=_ok())
    registry.approve_atn = MagicMock(return_value=_ok())
    registry.blockchain = MagicMock()
    registry.blockchain.account = MagicMock()
    registry.blockchain.account.address = "0xAggregator1"

    # Wire up guild contract functions
    if guild_contract:
        guild_contract.functions.getGuildCount = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=guild_count))
        )
        guild_contract.functions.minMemberStake = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=50 * 10**18))
        )

    return registry


def make_config(**guild_overrides) -> AutonetConfig:
    """Build a config with guild settings."""
    cfg = AutonetConfig()
    for key, value in guild_overrides.items():
        if hasattr(cfg.guild, key):
            setattr(cfg.guild, key, value)
    return cfg


# =============================================================================
# Tests: GuildConfig
# =============================================================================


class TestGuildConfig:
    def test_defaults(self):
        """GuildConfig has sensible defaults."""
        cfg = GuildConfig()
        assert cfg.guild_id is None
        assert cfg.auto_join is False
        assert cfg.module_ids == []
        assert cfg.aggregation_level == "flat"
        assert cfg.min_guild_updates == 2
        assert cfg.reputation_weight == 0.3
        assert cfg.member_count_weight == 0.7

    def test_config_in_autonet(self):
        """GuildConfig is accessible from AutonetConfig."""
        cfg = AutonetConfig()
        assert isinstance(cfg.guild, GuildConfig)
        assert cfg.guild.aggregation_level == "flat"

    def test_config_override(self):
        """GuildConfig values can be overridden."""
        cfg = make_config(guild_id=5, aggregation_level="guild", auto_join=True)
        assert cfg.guild.guild_id == 5
        assert cfg.guild.aggregation_level == "guild"
        assert cfg.guild.auto_join is True


# =============================================================================
# Tests: GuildManager — Discovery
# =============================================================================


class TestGuildManagerDiscovery:
    def test_init_no_registry(self):
        """GuildManager gracefully handles missing GuildRegistry."""
        registry = make_guild_registry(has_guild_registry=False)
        gm = GuildManager(registry, "test-node", make_config())
        assert gm.guild_id is None
        assert gm.is_in_guild is False
        assert gm.aggregation_level == "flat"

    def test_init_with_registry(self):
        """GuildManager detects GuildRegistry contract."""
        registry = make_guild_registry(has_guild_registry=True)
        gm = GuildManager(registry, "test-node", make_config())
        assert not gm.is_in_guild

    def test_refresh_guilds_empty(self):
        """refresh_guilds returns 0 when no guilds exist."""
        registry = make_guild_registry(guild_count=0)
        gm = GuildManager(registry, "test-node", make_config())
        assert gm.refresh_guilds() == 0
        assert gm.get_all_guilds() == []

    def test_refresh_guilds_with_data(self):
        """refresh_guilds fetches guilds from chain."""
        registry = make_guild_registry(guild_count=2)
        guild_contract = registry.get("GuildRegistry")

        # Mock getGuild responses
        def mock_get_guild(guild_id):
            guilds = {
                1: ("Vision Guild", [b'\x01' * 32], 5, "0xAgg1", "0xCreator1", 500, True),
                2: ("Language Guild", [b'\x02' * 32], 3, "0xAgg2", "0xCreator2", 300, True),
            }
            mock = MagicMock()
            mock.call = MagicMock(return_value=guilds.get(guild_id))
            return mock

        guild_contract.functions.getGuild = MagicMock(side_effect=mock_get_guild)

        gm = GuildManager(registry, "test-node", make_config())
        count = gm.refresh_guilds()
        assert count == 2
        assert len(gm.get_all_guilds()) == 2

        g1 = gm.get_guild(1)
        assert g1.name == "Vision Guild"
        assert g1.member_count == 5

    def test_refresh_filters_inactive(self):
        """refresh_guilds only returns active guilds."""
        registry = make_guild_registry(guild_count=2)
        guild_contract = registry.get("GuildRegistry")

        def mock_get_guild(guild_id):
            guilds = {
                1: ("Active Guild", [b'\x01' * 32], 5, "0xAgg1", "0xCreator1", 500, True),
                2: ("Inactive Guild", [b'\x02' * 32], 0, "0xAgg2", "0xCreator2", 0, False),
            }
            mock = MagicMock()
            mock.call = MagicMock(return_value=guilds.get(guild_id))
            return mock

        guild_contract.functions.getGuild = MagicMock(side_effect=mock_get_guild)

        gm = GuildManager(registry, "test-node", make_config())
        count = gm.refresh_guilds()
        assert count == 1  # Only active guild


# =============================================================================
# Tests: GuildManager — Recommendation
# =============================================================================


class TestGuildManagerRecommendation:
    def _setup_guilds(self, gm):
        """Populate the guild manager with test guilds."""
        gm._guilds = {
            1: GuildInfo(
                guild_id=1, name="Vision Guild", creator="0xC1",
                aggregator="0xA1", module_ids=[b'\x01' * 32, b'\x03' * 32],
                member_count=5, total_staked=500, active=True,
            ),
            2: GuildInfo(
                guild_id=2, name="Language Guild", creator="0xC2",
                aggregator="0xA2", module_ids=[b'\x02' * 32],
                member_count=2, total_staked=200, active=True,
            ),
            3: GuildInfo(
                guild_id=3, name="Audio Guild", creator="0xC3",
                aggregator="0xA3", module_ids=[b'\x04' * 32],
                member_count=10, total_staked=1000, active=True,
            ),
        }

    def test_recommend_no_guilds(self):
        """recommend_guild returns None when no guilds are available."""
        registry = make_guild_registry(guild_count=0)
        gm = GuildManager(registry, "test-node", make_config())
        assert gm.recommend_guild() is None

    def test_recommend_no_capabilities(self):
        """Without capabilities, recommends least-populated guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())
        self._setup_guilds(gm)

        rec = gm.recommend_guild(capabilities=[])
        assert rec is not None
        assert rec.guild_id == 2  # Language Guild has fewest members (2)
        assert rec.module_overlap == 0

    def test_recommend_with_matching_modules(self):
        """Recommends guild with best module overlap."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())
        self._setup_guilds(gm)

        # Node can train modules 0x01 and 0x03 — matches Vision Guild
        rec = gm.recommend_guild(capabilities=[b'\x01' * 32, b'\x03' * 32])
        assert rec is not None
        assert rec.guild_id == 1  # Vision Guild
        assert rec.module_overlap == 2

    def test_recommend_single_module_match(self):
        """Single module match still works."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())
        self._setup_guilds(gm)

        rec = gm.recommend_guild(capabilities=[b'\x02' * 32])
        assert rec is not None
        assert rec.guild_id == 2  # Language Guild
        assert rec.module_overlap == 1

    def test_recommend_no_matching_modules(self):
        """Returns None when no guild matches the node's capabilities."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())
        self._setup_guilds(gm)

        rec = gm.recommend_guild(capabilities=[b'\xFF' * 32])
        assert rec is None


# =============================================================================
# Tests: GuildManager — Join / Leave
# =============================================================================


class TestGuildManagerMembership:
    def test_join_guild_success(self):
        """Successfully join a guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())

        result = gm.join_guild(1)
        assert result == 1
        assert gm.guild_id == 1
        assert gm.is_in_guild is True

    def test_join_guild_approval_failure(self):
        """Join fails if ATN approval fails."""
        registry = make_guild_registry()
        registry.approve_atn = MagicMock(return_value=_fail("insufficient balance"))
        gm = GuildManager(registry, "test-node", make_config())

        result = gm.join_guild(1)
        assert result is None
        assert gm.guild_id is None

    def test_join_guild_contract_failure(self):
        """Join fails if contract call fails."""
        registry = make_guild_registry()
        registry.call_contract = MagicMock(return_value=_fail("AlreadyMember"))
        gm = GuildManager(registry, "test-node", make_config())

        result = gm.join_guild(1)
        assert result is None

    def test_leave_guild(self):
        """Successfully leave a guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(guild_id=1))

        assert gm.guild_id == 1
        result = gm.leave_guild()
        assert result is True
        assert gm.guild_id is None

    def test_leave_no_guild(self):
        """leave_guild when not in a guild is a no-op."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())

        assert gm.leave_guild() is True

    def test_auto_join_disabled(self):
        """auto_join returns None when not configured."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(auto_join=False))
        assert gm.auto_join() is None

    def test_auto_join_enabled(self):
        """auto_join joins the recommended guild when configured."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(auto_join=True))

        # Setup guilds for recommendation
        gm._guilds = {
            1: GuildInfo(
                guild_id=1, name="Test Guild", creator="0xC1",
                aggregator="0xA1", module_ids=[b'\x01' * 32],
                member_count=3, total_staked=300, active=True,
            ),
        }

        result = gm.auto_join()
        assert result == 1
        assert gm.guild_id == 1


# =============================================================================
# Tests: GuildManager — Health Monitoring
# =============================================================================


class TestGuildManagerHealth:
    def test_health_not_in_guild(self):
        """monitor_guild_health returns None when not in a guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())
        assert gm.monitor_guild_health() is None

    def test_health_healthy_guild(self):
        """monitor_guild_health reports healthy for a good guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(guild_id=1))

        gm._guilds = {
            1: GuildInfo(
                guild_id=1, name="Healthy Guild", creator="0xC1",
                aggregator="0xA1", module_ids=[b'\x01' * 32],
                member_count=5, total_staked=500, active=True,
            ),
        }

        # Mock reputation fetch
        guild_contract = registry.get("GuildRegistry")
        guild_contract.functions.getReputation = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=(7000, 10, 1000)))
        )

        health = gm.monitor_guild_health()
        assert health is not None
        assert health["status"] == "healthy"
        assert health["member_count"] == 5
        assert health["guild_id"] == 1

    def test_health_no_members(self):
        """monitor_guild_health flags guilds with no members."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(guild_id=1))

        gm._guilds = {
            1: GuildInfo(
                guild_id=1, name="Empty Guild", creator="0xC1",
                aggregator="0xA1", module_ids=[b'\x01' * 32],
                member_count=0, total_staked=0, active=True,
            ),
        }

        health = gm.monitor_guild_health()
        assert health["status"] == "critical"


# =============================================================================
# Tests: GuildManager — Metrics Reporting
# =============================================================================


class TestGuildManagerMetrics:
    def test_report_metrics_success(self):
        """report_guild_metrics calls the contract."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config(guild_id=1))

        result = gm.report_guild_metrics(b'\x01' * 32, 8000)
        assert result is True
        registry.call_contract.assert_called_once_with(
            "GuildRegistry",
            "recordGuildMetrics",
            1,
            b'\x01' * 32,
            8000,
        )

    def test_report_metrics_no_guild(self):
        """report_guild_metrics is a no-op when not in a guild."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())

        result = gm.report_guild_metrics(b'\x01' * 32, 8000)
        assert result is True
        registry.call_contract.assert_not_called()

    def test_reward_multiplier(self):
        """get_guild_reward_multiplier calls the contract."""
        registry = make_guild_registry()
        guild_contract = registry.get("GuildRegistry")
        guild_contract.functions.getGuildRewardMultiplier = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=12000))
        )

        gm = GuildManager(registry, "test-node", make_config())
        mult = gm.get_guild_reward_multiplier(1, b'\x01' * 32)
        assert mult == 12000


# =============================================================================
# Tests: GuildManager — Aggregation Weights
# =============================================================================


class TestGuildAggregationWeights:
    def test_weights_empty(self):
        """get_guild_weights returns empty when no guilds."""
        registry = make_guild_registry(guild_count=0)
        gm = GuildManager(registry, "test-node", make_config())
        assert gm.get_guild_weights_for_aggregation() == {}

    def test_weights_normalized(self):
        """Weights sum to approximately 1.0."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())

        gm._guilds = {
            1: GuildInfo(1, "G1", "0xC1", "0xA1", [b'\x01' * 32], 5, 500, True),
            2: GuildInfo(2, "G2", "0xC2", "0xA2", [b'\x02' * 32], 3, 300, True),
        }

        # Mock reputation — returns (score, count, timestamp)
        guild_contract = registry.get("GuildRegistry")
        guild_contract.functions.getReputation = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=(5000, 5, 1000)))
        )

        weights = gm.get_guild_weights_for_aggregation()
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_weights_favor_larger_guilds(self):
        """With equal reputation, larger guilds get more weight."""
        registry = make_guild_registry()
        gm = GuildManager(registry, "test-node", make_config())

        gm._guilds = {
            1: GuildInfo(1, "Big", "0xC1", "0xA1", [b'\x01' * 32], 10, 1000, True),
            2: GuildInfo(2, "Small", "0xC2", "0xA2", [b'\x02' * 32], 1, 100, True),
        }

        guild_contract = registry.get("GuildRegistry")
        guild_contract.functions.getReputation = MagicMock(
            return_value=MagicMock(call=MagicMock(return_value=(5000, 5, 1000)))
        )

        weights = gm.get_guild_weights_for_aggregation()
        assert weights[1] > weights[2]


# =============================================================================
# Tests: Aggregator — Guild & Network Aggregation
# =============================================================================


class TestAggregatorGuildLevel:
    """Test the aggregator's guild-level aggregation path."""

    def _make_aggregator(self, aggregation_level="flat", guild_id=None):
        """Create an AggregatorNode with mocked dependencies."""
        from nodes.aggregator.main import AggregatorNode, ProjectAggregationState

        registry = make_guild_registry()
        registry.get_new_events = MagicMock(return_value=[])
        registry.stake = MagicMock(return_value=_ok())
        registry.set_mature_model = MagicMock(return_value=_ok())
        registry.get_revealed_solution = MagicMock(return_value=None)

        store = MagicMock()
        store.get_json = MagicMock(return_value={"accuracy": 0.85, "loss": 0.3})
        store.add_json = MagicMock(return_value="QmAggregated123")

        config = make_config(
            aggregation_level=aggregation_level,
            guild_id=guild_id,
        )

        node = AggregatorNode(
            registry=registry,
            store=store,
            node_id="test-agg",
            rpb_address="",
            config=config,
        )
        node.is_staked = True

        return node

    def test_flat_aggregation_level(self):
        """Flat level uses the legacy path."""
        node = self._make_aggregator(aggregation_level="flat")
        assert node.aggregation_level == "flat"

        # Add some updates
        node.project_state.collected_updates = ["cid1", "cid2"]

        # Should trigger flat aggregation
        node._cycle()
        assert node.metrics.aggregations_done >= 1

    def test_guild_aggregation_no_guild(self):
        """Guild aggregation falls back to flat when no guild_id."""
        node = self._make_aggregator(aggregation_level="guild", guild_id=None)

        node.project_state.collected_updates = ["cid1", "cid2"]
        node._cycle()
        # Falls back to flat aggregation
        assert node.metrics.aggregations_done >= 1

    def test_guild_aggregation_filters_by_membership(self):
        """Guild aggregation only includes guild member updates."""
        node = self._make_aggregator(aggregation_level="guild", guild_id=1)

        # Add updates with solver tracking
        node.project_state.collected_updates = ["cid_member", "cid_outsider"]
        node.project_state.update_solvers = {
            "cid_member": "0xMember1",
            "cid_outsider": "0xOutsider",
        }

        # Mock: 0xMember1 is in guild 1, 0xOutsider is not
        def is_member(addr, gid):
            return addr == "0xMember1" and gid == 1

        node.guild_manager.is_member_of_guild = MagicMock(side_effect=is_member)

        # Only 1 member update < min_guild_updates (2), so no aggregation
        node._cycle()
        assert node.metrics.guild_aggregations_done == 0

    def test_network_aggregation_with_guild_aggregates(self):
        """Network aggregation combines guild-level updates."""
        node = self._make_aggregator(aggregation_level="network")

        # Simulate guild aggregates already available
        node.project_state.guild_aggregated_cids = {
            1: "QmGuild1Agg",
            2: "QmGuild2Agg",
        }

        # Mock guild weights
        node.guild_manager.get_guild_weights_for_aggregation = MagicMock(
            return_value={1: 0.6, 2: 0.4}
        )

        node._cycle()
        assert node.metrics.network_aggregations_done >= 1
        # Guild aggregated CIDs should be cleared after network aggregation
        assert len(node.project_state.guild_aggregated_cids) == 0

    def test_network_aggregation_fallback_to_flat(self):
        """Network aggregation falls back to flat when no guild aggregates."""
        node = self._make_aggregator(aggregation_level="network")

        # No guild aggregates, but flat updates available
        node.project_state.collected_updates = ["cid1", "cid2"]

        node._cycle()
        # Should fall back to flat aggregation
        assert node.metrics.aggregations_done >= 1


# =============================================================================
# Tests: Weighted Guild Aggregation Math
# =============================================================================


class TestWeightedAggregation:
    """Test the weighted aggregation math."""

    def _make_aggregator(self):
        """Create a minimal aggregator for testing aggregation methods."""
        from nodes.aggregator.main import AggregatorNode

        registry = make_guild_registry()
        registry.get_new_events = MagicMock(return_value=[])
        registry.stake = MagicMock(return_value=_ok())

        store = MagicMock()
        config = make_config()

        node = AggregatorNode(
            registry=registry,
            store=store,
            node_id="test-agg",
            rpb_address="",
            config=config,
        )
        return node

    def test_weighted_mock_aggregation(self):
        """Mock weighted aggregation applies weights correctly."""
        node = self._make_aggregator()

        updates = [
            {"accuracy": 0.9, "loss": 0.1},
            {"accuracy": 0.7, "loss": 0.3},
        ]
        weights = [0.75, 0.25]

        result = node._weighted_guild_aggregate_mock(updates, weights)

        # accuracy = 0.9 * 0.75 + 0.7 * 0.25 = 0.675 + 0.175 = 0.85
        assert abs(result["accuracy"] - 0.85) < 0.001
        # loss = 0.1 * 0.75 + 0.3 * 0.25 = 0.075 + 0.075 = 0.15
        assert abs(result["loss"] - 0.15) < 0.001

    def test_weighted_real_aggregation(self):
        """Real weighted aggregation applies weights to weight deltas."""
        node = self._make_aggregator()

        updates = [
            {"aggregated_weight_delta": {"param1": [1.0, 2.0], "param2": [3.0]}},
            {"aggregated_weight_delta": {"param1": [3.0, 4.0], "param2": [5.0]}},
        ]
        weights = [0.6, 0.4]

        result = node._weighted_guild_aggregate_real(updates, weights)

        delta = result["aggregated_weight_delta"]
        # param1 = [1.0, 2.0] * 0.6 + [3.0, 4.0] * 0.4 = [0.6+1.2, 1.2+1.6] = [1.8, 2.8]
        assert abs(delta["param1"][0] - 1.8) < 0.001
        assert abs(delta["param1"][1] - 2.8) < 0.001
        # param2 = [3.0] * 0.6 + [5.0] * 0.4 = [1.8 + 2.0] = [3.8]
        assert abs(delta["param2"][0] - 3.8) < 0.001

    def test_equal_weights(self):
        """Equal weights give simple average."""
        node = self._make_aggregator()

        updates = [
            {"accuracy": 0.8, "loss": 0.2},
            {"accuracy": 0.6, "loss": 0.4},
        ]
        weights = [0.5, 0.5]

        result = node._weighted_guild_aggregate_mock(updates, weights)
        assert abs(result["accuracy"] - 0.7) < 0.001
        assert abs(result["loss"] - 0.3) < 0.001


# =============================================================================
# Tests: ProjectAggregationState guild fields
# =============================================================================


class TestRPBAggregationState:
    def test_guild_tracking_fields(self):
        """RPBAggregationState has guild tracking fields."""
        from nodes.aggregator.main import RPBAggregationState

        state = RPBAggregationState(rpb_address="")
        assert state.guild_updates == {}
        assert state.guild_aggregated_cids == {}
        assert state.update_solvers == {}

    def test_solver_tracking(self):
        """update_solvers maps CIDs to solver addresses."""
        from nodes.aggregator.main import RPBAggregationState

        state = RPBAggregationState(rpb_address="")
        state.update_solvers["cid1"] = "0xSolver1"
        state.update_solvers["cid2"] = "0xSolver2"

        assert state.update_solvers["cid1"] == "0xSolver1"
        assert len(state.update_solvers) == 2
