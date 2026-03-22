"""
Guild Manager for Autonet nodes.

Reads guild registry from chain, recommends guilds based on node capabilities,
handles auto-join, and monitors guild health metrics.

Wires into the P2P layer's GuildMembership for gossip coordination.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .config import AutonetConfig, GuildConfig, load_config
from .contracts import ContractRegistry

logger = logging.getLogger(__name__)


@dataclass
class GuildInfo:
    """Cached guild information from on-chain registry."""
    guild_id: int
    name: str
    creator: str
    aggregator: str
    module_ids: List[bytes]
    member_count: int
    total_staked: int
    active: bool


@dataclass
class GuildReputationInfo:
    """Cached reputation for a guild on a specific module."""
    guild_id: int
    module_id: bytes
    score: int            # 0-10000 bps
    evaluation_count: int
    last_updated_at: int


@dataclass
class GuildRecommendation:
    """Result of recommend_guild()."""
    guild_id: int
    name: str
    module_overlap: int       # Number of matching modules
    reputation_score: int     # Average reputation across matched modules
    member_count: int
    reason: str


class GuildManager:
    """
    Manages guild interactions for a node.

    Responsibilities:
    - Read guild registry from chain (cached locally)
    - Recommend the best guild for this node's capabilities
    - Auto-join a guild if configured
    - Monitor guild health (member count, reputation trends)
    - Report guild metrics after aggregation
    """

    def __init__(
        self,
        registry: ContractRegistry,
        node_id: str,
        config: Optional[AutonetConfig] = None,
    ):
        self.registry = registry
        self.node_id = node_id
        self.config = config or load_config()
        self.guild_config: GuildConfig = self.config.guild

        self._guilds: Dict[int, GuildInfo] = {}
        self._my_guild_id: Optional[int] = self.guild_config.guild_id
        self._guild_registry_available: bool = False

        self.logger = logging.getLogger(f"GuildManager[{node_id}]")

        # Check if GuildRegistry contract is deployed
        contract = self.registry.get("GuildRegistry")
        if contract is not None:
            self._guild_registry_available = True
            self.logger.info("GuildRegistry contract available")
        else:
            self.logger.info(
                "GuildRegistry contract not deployed, guild features disabled"
            )

    @property
    def guild_id(self) -> Optional[int]:
        """The guild this node currently belongs to."""
        return self._my_guild_id

    @property
    def aggregation_level(self) -> str:
        """Current aggregation level: 'flat', 'guild', or 'network'."""
        return self.guild_config.aggregation_level

    @property
    def is_in_guild(self) -> bool:
        """Whether this node is currently in a guild."""
        return self._my_guild_id is not None

    # =========================================================================
    # Guild Discovery
    # =========================================================================

    def refresh_guilds(self) -> int:
        """
        Fetch all guilds from the on-chain registry.

        Returns:
            Number of active guilds found.
        """
        if not self._guild_registry_available:
            return 0

        try:
            count = self._get_guild_count()
            self._guilds.clear()

            for gid in range(1, count + 1):
                info = self._fetch_guild(gid)
                if info and info.active:
                    self._guilds[gid] = info

            self.logger.info(
                f"Refreshed guild list: {len(self._guilds)} active guilds "
                f"out of {count} total"
            )
            return len(self._guilds)

        except Exception as e:
            self.logger.warning(f"Failed to refresh guilds: {e}")
            return 0

    def get_guild(self, guild_id: int) -> Optional[GuildInfo]:
        """Get cached guild info. Call refresh_guilds() first."""
        return self._guilds.get(guild_id)

    def get_all_guilds(self) -> List[GuildInfo]:
        """Get all cached active guilds."""
        return list(self._guilds.values())

    # =========================================================================
    # Guild Recommendation
    # =========================================================================

    def recommend_guild(
        self,
        capabilities: Optional[List[bytes]] = None,
    ) -> Optional[GuildRecommendation]:
        """
        Recommend the best guild for this node based on capabilities.

        Scoring:
        1. Module overlap (how many of the node's modules match the guild's)
        2. Guild reputation (higher is better)
        3. Member count (lower is better — less competition for rewards)

        Args:
            capabilities: Module IDs this node can train. If None, uses config.

        Returns:
            GuildRecommendation or None if no guilds available.
        """
        if not self._guilds:
            self.refresh_guilds()

        if not self._guilds:
            return None

        # Use config module_ids if not specified
        if capabilities is None:
            capabilities = [
                bytes.fromhex(mid) if isinstance(mid, str) else mid
                for mid in self.guild_config.module_ids
            ]

        if not capabilities:
            # No capabilities specified — recommend the guild with fewest members
            # (best chance of getting rewards)
            best = min(self._guilds.values(), key=lambda g: g.member_count)
            return GuildRecommendation(
                guild_id=best.guild_id,
                name=best.name,
                module_overlap=0,
                reputation_score=0,
                member_count=best.member_count,
                reason="No capabilities specified, recommending least-populated guild",
            )

        # Score each guild
        best_score = -1
        best_guild = None
        best_overlap = 0
        best_rep = 0

        for guild in self._guilds.values():
            guild_modules = set(
                m if isinstance(m, bytes) else bytes.fromhex(m)
                for m in guild.module_ids
            )
            node_modules = set(capabilities)

            overlap = len(guild_modules & node_modules)
            if overlap == 0:
                continue

            # Fetch average reputation for matched modules
            rep_scores = []
            for mod_id in guild_modules & node_modules:
                rep = self._fetch_reputation(guild.guild_id, mod_id)
                if rep:
                    rep_scores.append(rep.score)

            avg_rep = sum(rep_scores) / len(rep_scores) if rep_scores else 5000

            # Composite score: overlap * 1000 + reputation - member_count * 10
            # Prioritizes module match, then reputation, with mild preference
            # for smaller guilds.
            score = overlap * 1000 + avg_rep - guild.member_count * 10

            if score > best_score:
                best_score = score
                best_guild = guild
                best_overlap = overlap
                best_rep = int(avg_rep)

        if best_guild is None:
            return None

        return GuildRecommendation(
            guild_id=best_guild.guild_id,
            name=best_guild.name,
            module_overlap=best_overlap,
            reputation_score=best_rep,
            member_count=best_guild.member_count,
            reason=f"Best module overlap ({best_overlap}), reputation {best_rep}/10000",
        )

    # =========================================================================
    # Guild Joining
    # =========================================================================

    def auto_join(self) -> Optional[int]:
        """
        Auto-join the recommended guild if configured.

        Returns:
            Guild ID joined, or None if not configured or failed.
        """
        if not self.guild_config.auto_join:
            return None

        if self._my_guild_id is not None:
            self.logger.debug(f"Already in guild {self._my_guild_id}")
            return self._my_guild_id

        recommendation = self.recommend_guild()
        if recommendation is None:
            self.logger.info("No guilds available for auto-join")
            return None

        return self.join_guild(recommendation.guild_id)

    def join_guild(self, guild_id: int) -> Optional[int]:
        """
        Join a guild by staking ATN on-chain.

        Args:
            guild_id: The guild to join.

        Returns:
            Guild ID if successful, None on failure.
        """
        if not self._guild_registry_available:
            self.logger.warning("GuildRegistry not available")
            return None

        try:
            # Approve ATN spending for the guild registry
            guild_contract = self.registry.get("GuildRegistry")
            if guild_contract is None:
                return None

            # Get min member stake
            min_stake = guild_contract.functions.minMemberStake().call()

            if min_stake > 0:
                approve_result = self.registry.approve_atn(
                    guild_contract.address, min_stake
                )
                if not approve_result.success:
                    self.logger.error(
                        f"ATN approval for guild join failed: {approve_result.error}"
                    )
                    return None

            # Join
            result = self.registry.call_contract(
                "GuildRegistry", "joinGuild", guild_id
            )
            if result.success:
                self._my_guild_id = guild_id
                self.logger.info(f"Joined guild {guild_id}")
                return guild_id
            else:
                self.logger.error(f"Failed to join guild {guild_id}: {result.error}")
                return None

        except Exception as e:
            self.logger.error(f"Error joining guild {guild_id}: {e}")
            return None

    def leave_guild(self) -> bool:
        """
        Leave the current guild and reclaim staked ATN.

        Returns:
            True if successful.
        """
        if self._my_guild_id is None:
            return True

        if not self._guild_registry_available:
            return False

        try:
            result = self.registry.call_contract(
                "GuildRegistry", "leaveGuild", self._my_guild_id
            )
            if result.success:
                self.logger.info(f"Left guild {self._my_guild_id}")
                self._my_guild_id = None
                return True
            else:
                self.logger.error(f"Failed to leave guild: {result.error}")
                return False
        except Exception as e:
            self.logger.error(f"Error leaving guild: {e}")
            return False

    # =========================================================================
    # Guild Health Monitoring
    # =========================================================================

    def monitor_guild_health(self) -> Optional[Dict[str, Any]]:
        """
        Check health metrics for the current guild.

        Returns:
            Dict with health indicators, or None if not in a guild.
        """
        if self._my_guild_id is None:
            return None

        guild = self._guilds.get(self._my_guild_id)
        if guild is None:
            # Refresh and retry
            self.refresh_guilds()
            guild = self._guilds.get(self._my_guild_id)

        if guild is None:
            return {"status": "unknown", "reason": "Guild not found in registry"}

        # Gather reputation for each module
        module_reps = {}
        for mod_id in guild.module_ids:
            rep = self._fetch_reputation(guild.guild_id, mod_id)
            if rep:
                module_reps[mod_id.hex() if isinstance(mod_id, bytes) else str(mod_id)] = {
                    "score": rep.score,
                    "evaluation_count": rep.evaluation_count,
                }

        avg_rep = (
            sum(r["score"] for r in module_reps.values()) / len(module_reps)
            if module_reps
            else 5000
        )

        # Health assessment
        if guild.member_count == 0:
            status = "critical"
            reason = "No members in guild"
        elif avg_rep < 3000:
            status = "poor"
            reason = f"Low average reputation ({avg_rep:.0f}/10000)"
        elif avg_rep < 5000:
            status = "fair"
            reason = f"Below-average reputation ({avg_rep:.0f}/10000)"
        elif not guild.active:
            status = "inactive"
            reason = "Guild is deactivated"
        else:
            status = "healthy"
            reason = f"Active with {guild.member_count} members, reputation {avg_rep:.0f}/10000"

        return {
            "guild_id": guild.guild_id,
            "name": guild.name,
            "status": status,
            "reason": reason,
            "member_count": guild.member_count,
            "total_staked": guild.total_staked,
            "aggregator": guild.aggregator,
            "active": guild.active,
            "avg_reputation": int(avg_rep),
            "modules": module_reps,
        }

    # =========================================================================
    # Guild Metrics Reporting
    # =========================================================================

    def report_guild_metrics(
        self, module_id: bytes, improvement_score: int
    ) -> bool:
        """
        Report improvement metrics for a guild's module training.

        Called by aggregator after evaluating training results. The on-chain
        GuildRegistry uses EMA to smooth scores over time.

        Args:
            module_id: The module that was trained.
            improvement_score: How much the guild improved this module (0-10000 bps).

        Returns:
            True if report succeeded.
        """
        if not self._guild_registry_available:
            return True  # Graceful skip

        if self._my_guild_id is None:
            return True

        try:
            result = self.registry.call_contract(
                "GuildRegistry",
                "recordGuildMetrics",
                self._my_guild_id,
                module_id,
                improvement_score,
            )
            if result.success:
                self.logger.info(
                    f"Reported guild metrics: guild={self._my_guild_id}, "
                    f"module={module_id[:8].hex()}, score={improvement_score}"
                )
                return True
            else:
                self.logger.warning(f"Guild metrics report failed: {result.error}")
                return False
        except Exception as e:
            self.logger.warning(f"Guild metrics report error: {e}")
            return False

    def get_guild_reward_multiplier(
        self, guild_id: int, module_id: bytes
    ) -> int:
        """
        Get the reward multiplier for a guild on a specific module.

        Args:
            guild_id: The guild.
            module_id: The module.

        Returns:
            Multiplier in basis points (10000 = 1x). Returns 10000 if unavailable.
        """
        if not self._guild_registry_available:
            return 10000

        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return 10000
            return contract.functions.getGuildRewardMultiplier(
                guild_id, module_id
            ).call()
        except Exception:
            return 10000

    # =========================================================================
    # Network Aggregation Helpers
    # =========================================================================

    def get_guild_weights_for_aggregation(self) -> Dict[int, float]:
        """
        Compute weights for each guild to use in network-level aggregation.

        Weights are based on a combination of member count and guild reputation.
        Used by Story 8.3 (network-level aggregation).

        Returns:
            Dict mapping guild_id to aggregation weight (normalized to sum to 1).
        """
        if not self._guilds:
            self.refresh_guilds()

        if not self._guilds:
            return {}

        rep_weight = self.guild_config.reputation_weight
        count_weight = self.guild_config.member_count_weight

        raw_weights: Dict[int, float] = {}

        for guild in self._guilds.values():
            # Average reputation across all modules
            reps = []
            for mod_id in guild.module_ids:
                rep = self._fetch_reputation(guild.guild_id, mod_id)
                if rep:
                    reps.append(rep.score)
            avg_rep = sum(reps) / len(reps) if reps else 5000

            # Composite: weighted combination of member count and reputation
            # Normalize reputation to [0,1] range (it's 0-10000 bps)
            norm_rep = avg_rep / 10000.0
            norm_count = guild.member_count  # Raw count

            weight = count_weight * norm_count + rep_weight * norm_rep * 10
            raw_weights[guild.guild_id] = max(weight, 0.01)  # Floor at 0.01

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            return {gid: w / total for gid, w in raw_weights.items()}
        return raw_weights

    def get_guild_members(self, guild_id: int) -> List[str]:
        """
        Get member addresses for a guild.

        Returns:
            List of member addresses, or empty list if unavailable.
        """
        if not self._guild_registry_available:
            return []

        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return []
            return contract.functions.getGuildMembers(guild_id).call()
        except Exception:
            return []

    def is_member_of_guild(self, address: str, guild_id: int) -> bool:
        """Check if an address is a member of a guild."""
        if not self._guild_registry_available:
            return False

        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return False
            return contract.functions.isMember(address, guild_id).call()
        except Exception:
            return False

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_guild_count(self) -> int:
        """Get total guild count from chain."""
        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return 0
            return contract.functions.getGuildCount().call()
        except Exception:
            return 0

    def _fetch_guild(self, guild_id: int) -> Optional[GuildInfo]:
        """Fetch guild info from chain."""
        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return None

            result = contract.functions.getGuild(guild_id).call()
            name, module_ids, member_count, aggregator, creator, total_staked, active = result

            return GuildInfo(
                guild_id=guild_id,
                name=name,
                creator=creator,
                aggregator=aggregator,
                module_ids=list(module_ids),
                member_count=member_count,
                total_staked=total_staked,
                active=active,
            )
        except Exception as e:
            self.logger.debug(f"Failed to fetch guild {guild_id}: {e}")
            return None

    def _fetch_reputation(
        self, guild_id: int, module_id: bytes
    ) -> Optional[GuildReputationInfo]:
        """Fetch guild reputation for a module from chain."""
        try:
            contract = self.registry.get("GuildRegistry")
            if contract is None:
                return None

            score, eval_count, last_updated = contract.functions.getReputation(
                guild_id, module_id
            ).call()

            return GuildReputationInfo(
                guild_id=guild_id,
                module_id=module_id,
                score=score,
                evaluation_count=eval_count,
                last_updated_at=last_updated,
            )
        except Exception:
            return None
