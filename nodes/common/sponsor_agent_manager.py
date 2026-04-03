"""
Sponsor Agent Manager for RPB (Recursive Principial Body) nodes.

Replaces guild-based coordination with sponsor agent relationships.
A sponsor agent funds and oversees a set of child agents, defining their
training charter, alignment requirements, and budget.

Reads sponsor registry from chain, manages sponsored agent relationships,
checks alignment between sponsor and child charters, and monitors health.

Wires into the P2P layer for gossip coordination (replacing guild gossip).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .config import AutonetConfig, load_config
from .contracts import ContractRegistry

logger = logging.getLogger(__name__)


@dataclass
class SponsorAgentInfo:
    """Cached sponsor agent information from on-chain registry."""
    agent_id: str
    address: str
    name: str
    training_charter: str
    alignment_embedding: list[float] = field(default_factory=list)
    module_ids: list[bytes] = field(default_factory=list)
    sponsored_agents: list[str] = field(default_factory=list)  # agent addresses
    budget: int = 0                  # Total budget in ATN (wei)
    active: bool = True


@dataclass
class SponsorRecommendation:
    """Result of recommend_sponsor()."""
    agent_id: str
    name: str
    module_overlap: int              # Number of matching modules
    alignment_score: float           # 0.0 - 1.0, how well charters align
    sponsored_count: int             # Current number of sponsored agents
    budget_remaining: int
    reason: str


class SponsorAgentManager:
    """
    Manages sponsor agent interactions for a node.

    Responsibilities:
    - Read sponsor registry from chain (cached locally)
    - Register this node as a sponsor agent
    - Add/remove sponsored agents
    - Check alignment between sponsor and child charters
    - Monitor sponsor health (budget, sponsored agent count)
    - Report metrics after training rounds
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

        self._sponsors: Dict[str, SponsorAgentInfo] = {}
        self._my_sponsor_id: Optional[str] = None
        self._is_sponsor: bool = False
        self._sponsor_registry_available: bool = False

        self.logger = logging.getLogger(f"SponsorAgentManager[{node_id}]")

        # Check if SponsorRegistry contract is deployed
        # TODO: Deploy SponsorRegistry contract and register its ABI
        contract = self.registry.get("SponsorRegistry")
        if contract is not None:
            self._sponsor_registry_available = True
            self.logger.info("SponsorRegistry contract available")
        else:
            self.logger.info(
                "SponsorRegistry contract not deployed, sponsor features disabled"
            )

    @property
    def sponsor_id(self) -> Optional[str]:
        """The sponsor agent this node is registered under."""
        return self._my_sponsor_id

    @property
    def is_sponsor(self) -> bool:
        """Whether this node is registered as a sponsor agent."""
        return self._is_sponsor

    @property
    def is_sponsored(self) -> bool:
        """Whether this node is sponsored by another agent."""
        return self._my_sponsor_id is not None

    # =========================================================================
    # Sponsor Registration
    # =========================================================================

    def register_as_sponsor(
        self,
        name: str,
        training_charter: str,
        module_ids: Optional[List[bytes]] = None,
        initial_budget: int = 0,
    ) -> Optional[str]:
        """
        Register this node as a sponsor agent on-chain.

        Args:
            name: Human-readable sponsor name.
            training_charter: Text defining the scope of allowed work.
            module_ids: Module IDs this sponsor covers.
            initial_budget: Initial ATN budget to deposit.

        Returns:
            Sponsor agent ID if successful, None on failure.
        """
        if not self._sponsor_registry_available:
            self.logger.warning("SponsorRegistry not available")
            return None

        try:
            sponsor_contract = self.registry.get("SponsorRegistry")
            if sponsor_contract is None:
                return None

            # TODO: Call the actual on-chain registration
            # result = self.registry.call_contract(
            #     "SponsorRegistry", "registerSponsor",
            #     name, training_charter, module_ids or [], initial_budget,
            # )
            # if result.success:
            #     self._is_sponsor = True
            #     self._my_sponsor_id = result.return_value
            #     return self._my_sponsor_id

            self.logger.info(
                f"Sponsor registration pending: name={name}, "
                f"modules={len(module_ids or [])}, budget={initial_budget}"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error registering as sponsor: {e}")
            return None

    # =========================================================================
    # Sponsored Agent Management
    # =========================================================================

    def add_sponsored_agent(
        self,
        agent_address: str,
        allocation: int = 0,
    ) -> bool:
        """
        Add an agent to this sponsor's list of sponsored agents.

        Args:
            agent_address: The 0x address of the agent to sponsor.
            allocation: ATN budget allocation for this agent.

        Returns:
            True if successful.
        """
        if not self._is_sponsor:
            self.logger.warning("Cannot add sponsored agent: not a sponsor")
            return False

        if not self._sponsor_registry_available:
            return False

        try:
            # TODO: Call on-chain addSponsoredAgent
            # result = self.registry.call_contract(
            #     "SponsorRegistry", "addSponsoredAgent",
            #     self._my_sponsor_id, agent_address, allocation,
            # )
            # return result.success

            self.logger.info(
                f"Add sponsored agent pending: address={agent_address}, "
                f"allocation={allocation}"
            )
            return False

        except Exception as e:
            self.logger.error(f"Error adding sponsored agent: {e}")
            return False

    def remove_sponsored_agent(self, agent_address: str) -> bool:
        """
        Remove an agent from this sponsor's list.

        Args:
            agent_address: The 0x address of the agent to remove.

        Returns:
            True if successful.
        """
        if not self._is_sponsor:
            return False

        if not self._sponsor_registry_available:
            return False

        try:
            # TODO: Call on-chain removeSponsoredAgent
            # result = self.registry.call_contract(
            #     "SponsorRegistry", "removeSponsoredAgent",
            #     self._my_sponsor_id, agent_address,
            # )
            # return result.success

            self.logger.info(
                f"Remove sponsored agent pending: address={agent_address}"
            )
            return False

        except Exception as e:
            self.logger.error(f"Error removing sponsored agent: {e}")
            return False

    # =========================================================================
    # Alignment Checking
    # =========================================================================

    def check_alignment(
        self,
        sponsor_charter: str,
        child_charter: str,
        threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """
        Check alignment between a sponsor's charter and a child agent's charter.

        Uses cosine similarity of charter embeddings when available,
        falls back to keyword overlap heuristic.

        Args:
            sponsor_charter: The sponsor's training charter text.
            child_charter: The child agent's charter/system prompt text.
            threshold: Minimum similarity score (0.0 - 1.0).

        Returns:
            (aligned, similarity_score) tuple.
        """
        if not sponsor_charter or not child_charter:
            return True, 1.0  # No charter = no restriction

        # TODO: Use actual embedding model for semantic similarity
        # For now, use simple keyword overlap heuristic
        sponsor_words = set(sponsor_charter.lower().split())
        child_words = set(child_charter.lower().split())

        if not sponsor_words or not child_words:
            return True, 1.0

        overlap = len(sponsor_words & child_words)
        total = len(sponsor_words | child_words)
        similarity = overlap / total if total > 0 else 0.0

        aligned = similarity >= threshold
        return aligned, similarity

    # =========================================================================
    # Sponsor Discovery
    # =========================================================================

    def refresh_sponsors(self) -> int:
        """
        Fetch all sponsor agents from the on-chain registry.

        Returns:
            Number of active sponsors found.
        """
        if not self._sponsor_registry_available:
            return 0

        try:
            # TODO: Implement actual on-chain query
            # count = self._get_sponsor_count()
            # self._sponsors.clear()
            # for i in range(count):
            #     info = self._fetch_sponsor(i)
            #     if info and info.active:
            #         self._sponsors[info.agent_id] = info

            self.logger.info(
                f"Refreshed sponsor list: {len(self._sponsors)} active sponsors"
            )
            return len(self._sponsors)

        except Exception as e:
            self.logger.warning(f"Failed to refresh sponsors: {e}")
            return 0

    def get_sponsor_info(self, sponsor_id: str) -> Optional[SponsorAgentInfo]:
        """Get cached sponsor info. Call refresh_sponsors() first."""
        return self._sponsors.get(sponsor_id)

    def list_sponsors(self) -> List[SponsorAgentInfo]:
        """Get all cached active sponsors."""
        return [s for s in self._sponsors.values() if s.active]

    def get_sponsored_agents(self, sponsor_id: str) -> List[str]:
        """
        Get addresses of agents sponsored by a given sponsor.

        Args:
            sponsor_id: The sponsor agent ID.

        Returns:
            List of sponsored agent addresses.
        """
        sponsor = self._sponsors.get(sponsor_id)
        if sponsor:
            return list(sponsor.sponsored_agents)

        if not self._sponsor_registry_available:
            return []

        try:
            # TODO: Query on-chain for sponsored agents
            # contract = self.registry.get("SponsorRegistry")
            # if contract is None:
            #     return []
            # return contract.functions.getSponsoredAgents(sponsor_id).call()
            return []

        except Exception:
            return []

    # =========================================================================
    # Sponsor Recommendation
    # =========================================================================

    def recommend_sponsor(
        self,
        capabilities: Optional[List[bytes]] = None,
        charter: str = "",
    ) -> Optional[SponsorRecommendation]:
        """
        Recommend the best sponsor agent for this node based on capabilities
        and charter alignment.

        Scoring:
        1. Module overlap (how many of the node's modules match the sponsor's)
        2. Charter alignment (semantic similarity)
        3. Budget remaining (higher is better)
        4. Sponsored agent count (lower is better for individual attention)

        Args:
            capabilities: Module IDs this node can train. If None, uses config.
            charter: This node's charter text for alignment checking.

        Returns:
            SponsorRecommendation or None if no sponsors available.
        """
        if not self._sponsors:
            self.refresh_sponsors()

        if not self._sponsors:
            return None

        best_score = -1.0
        best_sponsor = None
        best_overlap = 0
        best_alignment = 0.0

        for sponsor in self._sponsors.values():
            if not sponsor.active:
                continue

            # Module overlap
            if capabilities:
                sponsor_modules = set(
                    m if isinstance(m, bytes) else bytes.fromhex(m)
                    for m in sponsor.module_ids
                )
                node_modules = set(capabilities)
                overlap = len(sponsor_modules & node_modules)
            else:
                overlap = 0

            # Charter alignment
            _, alignment = self.check_alignment(
                sponsor.training_charter, charter
            )

            # Composite score: overlap * 1000 + alignment * 500
            #   + budget_remaining / 1000 - sponsored_count * 50
            score = (
                overlap * 1000
                + alignment * 500
                + sponsor.budget / 1000
                - len(sponsor.sponsored_agents) * 50
            )

            if score > best_score:
                best_score = score
                best_sponsor = sponsor
                best_overlap = overlap
                best_alignment = alignment

        if best_sponsor is None:
            return None

        return SponsorRecommendation(
            agent_id=best_sponsor.agent_id,
            name=best_sponsor.name,
            module_overlap=best_overlap,
            alignment_score=best_alignment,
            sponsored_count=len(best_sponsor.sponsored_agents),
            budget_remaining=best_sponsor.budget,
            reason=f"Best module overlap ({best_overlap}), alignment {best_alignment:.2f}",
        )

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def monitor_sponsor_health(self) -> Optional[Dict[str, Any]]:
        """
        Check health metrics for the current sponsor relationship.

        Returns:
            Dict with health indicators, or None if not sponsored.
        """
        if self._my_sponsor_id is None:
            return None

        sponsor = self._sponsors.get(self._my_sponsor_id)
        if sponsor is None:
            self.refresh_sponsors()
            sponsor = self._sponsors.get(self._my_sponsor_id)

        if sponsor is None:
            return {"status": "unknown", "reason": "Sponsor not found in registry"}

        if not sponsor.active:
            status = "inactive"
            reason = "Sponsor agent is deactivated"
        elif sponsor.budget <= 0:
            status = "critical"
            reason = "Sponsor has no remaining budget"
        elif len(sponsor.sponsored_agents) == 0:
            status = "warning"
            reason = "Sponsor has no sponsored agents"
        else:
            status = "healthy"
            reason = (
                f"Active with {len(sponsor.sponsored_agents)} sponsored agents, "
                f"budget {sponsor.budget} ATN"
            )

        return {
            "sponsor_id": sponsor.agent_id,
            "name": sponsor.name,
            "status": status,
            "reason": reason,
            "sponsored_count": len(sponsor.sponsored_agents),
            "budget": sponsor.budget,
            "active": sponsor.active,
            "module_count": len(sponsor.module_ids),
        }
