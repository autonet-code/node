"""
Governance integration for Autonet nodes.

Provides attestation, heartbeat listening, service registration,
reward claiming, reputation accrual, evolution proposal evaluation,
and RPB consensus that all node types share. This bridges the
training loop to the economic layer (Autonet.sol) and the evolution
mechanism (EvolutionProposal.sol) on the jurisdiction chain.

3-Tier Constitutional Evaluation (Phase C Step 6)
--------------------------------------------------
Actions and proposals are evaluated through a tiered pipeline:

  Tier 1 — Geometric pre-filter (O(1), ~80-90% of cases)
    ConstitutionalGeometry.evaluate() checks cosine similarity against
    7 RPB principle direction vectors in JEPA embedding space.

  Tier 2 — Concept bottleneck classifier (lightweight MLP)
    Handles nuanced cases where Tier 1 is uncertain. Each bottleneck
    neuron corresponds to one RPB principle (interpretable).

  Tier 3 — Full LLM evaluation via AIProvider
    A direct AIProvider call, used for high-stakes and adversarial-seeming
    inputs (Verdict.UNCERTAIN from Tiers 1+2). Invoked by
    ThreeTierConstitutionalEvaluator.

Evolution proposals require 95% quorum (CONSTITUTIONAL_QUORUM_BPS)
for constitutional amendments (parameter changes).
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from .contracts import ContractRegistry

logger = logging.getLogger(__name__)

# Quorum requirement for constitutional amendments (EvolutionProposal.sol parameter).
# 95% per the RPB specification ("requiring 95% quorum").
CONSTITUTIONAL_QUORUM_BPS: int = 9500  # 95% in basis points


def compute_service_id(rpb_address: str = "") -> bytes:
    """Deterministic service ID for an RPB."""
    key = rpb_address.lower() if rpb_address else "autonet-rpb"
    return hashlib.sha256(f"autonet-rpb-{key}".encode()).digest()


class GovernanceBridge:
    """
    Bridge between a node and the governance/economic layer.

    Each node owns one GovernanceBridge. It handles:
    - Usage attestation after task completion
    - Heartbeat listening (consensus liveness check)
    - Service registration at startup
    """

    def __init__(
        self,
        registry: ContractRegistry,
        node_id: str,
        rpb_address: str = "",
    ):
        self.registry = registry
        self.node_id = node_id
        self.rpb_address = rpb_address
        self.service_id = compute_service_id(rpb_address)

        self._last_heartbeat: Optional[float] = None
        self._heartbeat_interval: float = 60.0
        self._service_registered: bool = False
        self._economy_available: bool = self.registry.get("AutonetEconomy") is not None
        self._dao_available: bool = self.registry.get("AutonetDAO") is not None
        self._evolution_available: bool = self.registry.get("EvolutionProposal") is not None

        self.logger = logging.getLogger(f"GovernanceBridge[{node_id}]")

        if self._economy_available:
            self.logger.info("Economic layer (AutonetEconomy) available")
        else:
            self.logger.info("Economic layer not deployed, attestation will be skipped")

    # =========================================================================
    # Usage Attestation
    # =========================================================================

    def attest_task_completion(self, units: int = 1) -> bool:
        """
        Attest that this node completed training work.

        Called by solver after training, coordinator after verification,
        aggregator after model publication.

        Args:
            units: Number of work units to attest (default 1 per task).

        Returns:
            True if attestation succeeded (or was skipped because no economy).
        """
        if not self._economy_available:
            return True  # Graceful skip

        try:
            result = self.registry.attest_usage(self.service_id, units)
            if result.success:
                self.logger.info(
                    f"Attested {units} unit(s) for service "
                    f"{self.service_id[:8].hex()}... epoch={self._get_epoch()}"
                )
                return True
            else:
                self.logger.warning(f"Attestation failed: {result.error}")
                return False
        except Exception as e:
            self.logger.warning(f"Attestation error: {e}")
            return False

    def _get_epoch(self) -> int:
        try:
            return self.registry.get_current_epoch()
        except Exception:
            return 0

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def check_heartbeat(self) -> bool:
        """
        Check for governance heartbeat events from the DAO.

        If the DAO contract emits HeartbeatEmitted events, this resets the
        local heartbeat timer. If no heartbeat is detected within the
        interval, returns False (work should halt).

        Returns:
            True if heartbeat is alive, False if missed.
        """
        if not self._dao_available:
            # No DAO deployed — treat heartbeat as always alive
            # (development/simulation mode)
            return True

        # Check for new HeartbeatEmitted events
        try:
            events = self.registry.get_new_events("AutonetDAO", "HeartbeatEmitted")
            if events:
                self._last_heartbeat = time.time()
                self.logger.debug(f"Heartbeat received ({len(events)} events)")
        except Exception:
            pass  # DAO may not have heartbeat function yet

        # Evaluate liveness
        if self._last_heartbeat is None:
            # First check — give benefit of the doubt
            self._last_heartbeat = time.time()
            return True

        elapsed = time.time() - self._last_heartbeat
        if elapsed > self._heartbeat_interval:
            self.logger.warning(
                f"Governance heartbeat missed ({elapsed:.0f}s > {self._heartbeat_interval:.0f}s)"
            )
            return False

        return True

    def receive_heartbeat(self):
        """Manually record a heartbeat (e.g., from orchestrator in sim mode)."""
        self._last_heartbeat = time.time()

    # =========================================================================
    # Service Registration
    # =========================================================================

    def register_if_needed(self, rpb_contract_address: str) -> bool:
        """
        Register this RPB as a service in the economic layer.

        Called once at node startup. Idempotent — will skip if already
        registered or if the economy contract isn't deployed.

        Args:
            rpb_contract_address: Address of the RPB contract.

        Returns:
            True if registered (or already was), False on failure.
        """
        if self._service_registered or not self._economy_available:
            return True

        # Check if already registered
        try:
            info = self.registry.get_service(self.service_id)
            if info and info.get("rpbContract", info.get("projectContract")) not in (None, "0x" + "0" * 40):
                self._service_registered = True
                self.logger.info(
                    f"Service already registered: {self.service_id[:8].hex()}..."
                )
                return True
        except Exception:
            pass

        # Register
        try:
            import subprocess
            codebase_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(__import__("pathlib").Path(__file__).parent.parent.parent),
                timeout=5,
            ).decode().strip()
        except Exception:
            codebase_hash = "unknown"

        try:
            result = self.registry.register_service(
                self.service_id,
                rpb_contract_address,
                codebase_hash,
            )
            if result.success:
                self._service_registered = True
                self.logger.info(
                    f"Service registered: {self.service_id[:8].hex()}... "
                    f"codebase={codebase_hash[:12]}"
                )
                return True
            else:
                self.logger.warning(f"Service registration failed: {result.error}")
                return False
        except Exception as e:
            self.logger.warning(f"Service registration error: {e}")
            return False

    # =========================================================================
    # Reward Claiming (train → attest → earn ATN)
    # =========================================================================

    def claim_epoch_rewards(self, epoch_id: Optional[int] = None) -> bool:
        """
        Claim participant rewards for a finalized epoch.

        Calls Autonet.sol's claimParticipantReward(serviceId, epochId).
        The reward is proportional to this node's attested usage vs total
        service usage in that epoch.

        Args:
            epoch_id: Specific epoch to claim. If None, claims the last
                      finalized epoch (current - 1).

        Returns:
            True if claim succeeded (or was skipped because no economy).
        """
        if not self._economy_available:
            return True

        if epoch_id is None:
            current = self._get_epoch()
            epoch_id = current - 1 if current > 1 else 0

        if epoch_id <= 0:
            self.logger.debug("No finalized epoch to claim rewards for")
            return True

        try:
            result = self.registry.claim_participant_reward(
                self.service_id, epoch_id
            )
            if result.success:
                self.logger.info(
                    f"Claimed rewards for epoch {epoch_id}, "
                    f"service={self.service_id[:8].hex()}..."
                )
                return True
            else:
                # "AlreadyClaimed" or "NothingToClaim" are expected
                if "AlreadyClaimed" in str(result.error):
                    self.logger.debug(f"Epoch {epoch_id} already claimed")
                    return True
                if "NothingToClaim" in str(result.error):
                    self.logger.debug(f"Nothing to claim in epoch {epoch_id}")
                    return True
                self.logger.warning(f"Reward claim failed: {result.error}")
                return False
        except Exception as e:
            self.logger.warning(f"Reward claim error: {e}")
            return False

    def claim_reputation(self) -> bool:
        """
        Claim reputation (RepToken) based on economic activity.

        Calls RepToken.claimReputationFromEconomy(). The RepToken contract
        reads the user's earnings from the Economy contract and mints
        governance tokens proportional to contribution.

        Returns:
            True if claim succeeded (or was skipped).
        """
        if not self._economy_available:
            return True

        try:
            result = self.registry.claim_reputation()
            if result.success:
                self.logger.info("Reputation claimed from economic activity")
                return True
            else:
                self.logger.debug(f"Reputation claim skipped: {result.error}")
                return True  # Not fatal — may have nothing to claim
        except Exception as e:
            self.logger.warning(f"Reputation claim error: {e}")
            return False

    # =========================================================================
    # Capability Reporting
    # =========================================================================

    def report_capability_score(
        self, module_id: bytes, score: int
    ) -> bool:
        """
        Report a capability evaluation score for a training module.

        Called by coordinator/aggregator after evaluating training results.
        The CapabilityScorecard uses EMA to smooth multiple evaluations.

        Args:
            module_id: 32-byte module identifier (e.g., keccak of "visual_encoder")
            score: Capability score in basis points (0-10000)

        Returns:
            True if report succeeded (or was skipped).
        """
        if not self._economy_available:
            return True

        try:
            result = self.registry.update_capability_score(module_id, score)
            if result.success:
                self.logger.info(
                    f"Reported capability score {score}/10000 for module "
                    f"{module_id[:8].hex()}..."
                )
                return True
            else:
                self.logger.debug(f"Score report skipped: {result.error}")
                return True
        except Exception as e:
            self.logger.warning(f"Capability score report error: {e}")
            return False

    def get_training_scorecard(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the current capability scorecard.

        Returns a dict with module IDs, scores, targets, and multipliers.
        Proposers use this to create tasks targeting underserved modules.

        Returns:
            Dict with keys {modules: [{id, score, target, multiplier}, ...]},
            or None if scorecard isn't available.
        """
        if not self._economy_available:
            return None

        try:
            return self.registry.get_capability_scorecard()
        except Exception as e:
            self.logger.debug(f"Could not fetch scorecard: {e}")
            return None

    # =========================================================================
    # Evolution Proposals
    # =========================================================================

    def get_pending_proposals(self) -> List[Dict[str, Any]]:
        """
        Fetch proposals in Evaluating status that need RPB evaluation.

        Returns:
            List of proposal dicts, or empty list if unavailable.
        """
        if not self._evolution_available:
            return []

        try:
            count = self.registry.get_evolution_proposal_count()
            pending = []
            for i in range(1, count + 1):
                prop = self.registry.get_evolution_proposal(i)
                if prop and prop.get("status") == 1:  # Evaluating
                    prop["id"] = i
                    pending.append(prop)
            return pending
        except Exception as e:
            self.logger.debug(f"Could not fetch pending proposals: {e}")
            return []

    def submit_rpb_evaluation(
        self,
        proposal_id: int,
        approve: bool,
        confidence: int,
        reason_cid: str,
    ) -> bool:
        """
        Submit an RPB evaluation for a proposal.

        Called after the RPB evaluator produces a structured recommendation.
        The node evaluates the proposal using the constitutional prompt
        and an AI provider, then submits the result on-chain.

        Args:
            proposal_id: The proposal to evaluate
            approve: Whether to recommend adoption
            confidence: Confidence level (0-10000 bps)
            reason_cid: CID pointing to structured recommendation

        Returns:
            True if submission succeeded (or was skipped).
        """
        if not self._evolution_available:
            return True

        try:
            result = self.registry.submit_rpb_evaluation(
                proposal_id, approve, confidence, reason_cid
            )
            if result.success:
                self.logger.info(
                    f"RPB evaluation submitted for proposal {proposal_id}: "
                    f"{'approve' if approve else 'reject'} (confidence={confidence})"
                )
                return True
            else:
                if "AlreadyEvaluated" in str(result.error):
                    self.logger.debug(f"Already evaluated proposal {proposal_id}")
                    return True
                self.logger.warning(f"RPB evaluation failed: {result.error}")
                return False
        except Exception as e:
            self.logger.warning(f"RPB evaluation error: {e}")
            return False

    def submit_evolution_proposal(
        self,
        content_cid: str,
        stake_amount: int,
        is_constitutional_amendment: bool = False,
    ) -> Optional[int]:
        """
        Submit an evolution proposal to EvolutionProposal.sol.

        Constitutional amendments (parameter changes that affect the RPB
        constitution) require a 95% quorum enforced at the contract level.
        This method records the intent; the on-chain admin must call
        EvolutionProposal.setApprovalThreshold(9500) before evaluation
        to enforce the higher bar.

        Args:
            content_cid:                CID pointing to proposal description + spec
            stake_amount:               ATN to stake (must meet contract's minStake)
            is_constitutional_amendment: If True, flags this proposal as requiring
                                         the 95% constitutional quorum (9500 bps)

        Returns:
            Proposal ID on success, None on failure.
        """
        if not self._evolution_available:
            self.logger.debug("EvolutionProposal not deployed, skipping submission")
            return None

        try:
            result = self.registry.submit_evolution_proposal(content_cid, stake_amount)
            if result.success:
                proposal_id = result.data.get("proposal_id")
                quorum_note = (
                    f" [CONSTITUTIONAL AMENDMENT — 95% quorum required "
                    f"({CONSTITUTIONAL_QUORUM_BPS} bps)]"
                    if is_constitutional_amendment
                    else ""
                )
                self.logger.info(
                    f"Evolution proposal submitted: id={proposal_id}, "
                    f"stake={stake_amount}{quorum_note}"
                )
                return proposal_id
            else:
                self.logger.warning(
                    f"Evolution proposal submission failed: {result.error}"
                )
                return None
        except Exception as e:
            self.logger.warning(f"Evolution proposal submission error: {e}")
            return None

    def get_rpb_prompt(self) -> Optional[str]:
        """
        Get the current RPB constitutional prompt CID.

        Returns:
            The prompt CID string, or None if unavailable.
        """
        if not self._evolution_available:
            return None

        try:
            return self.registry.get_current_rpb_prompt()
        except Exception as e:
            self.logger.debug(f"Could not fetch RPB prompt: {e}")
            return None


# =============================================================================
# AI Provider Abstraction (Tier 3 constitutional evaluation)
# =============================================================================


class AIProvider(ABC):
    """
    Abstract interface for AI providers used in Tier 3 constitutional
    evaluation (ThreeTierConstitutionalEvaluator).

    Callers invoke evaluate() with the constitutional prompt and the action
    text. The provider returns a structured recommendation. Different providers
    (Claude, GPT, local LLM) implement this interface.
    """

    @abstractmethod
    def evaluate(
        self, system_prompt: str, proposal_content: str
    ) -> "RPBRecommendation":
        """
        Evaluate a proposal against the constitutional prompt.

        Args:
            system_prompt: The RPB constitutional prompt text
            proposal_content: The proposal description + spec

        Returns:
            Structured recommendation with approve/reject, confidence, reasoning.
        """
        ...


@dataclass
class RPBRecommendation:
    """Structured output from an RPB evaluation."""
    approve: bool
    confidence: int          # 0-10000 bps
    reasoning: str           # Human-readable reasoning
    constitutional_alignment: float  # 0.0-1.0 score against principles
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON for CID storage."""
        return json.dumps({
            "approve": self.approve,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "constitutional_alignment": self.constitutional_alignment,
            "risks": self.risks,
            "benefits": self.benefits,
        })


class PlaceholderAIProvider(AIProvider):
    """
    Placeholder AI provider for development/testing.

    Always returns a neutral recommendation. Replace with real
    provider (Claude, GPT, local LLM) in production.
    """

    def evaluate(
        self, system_prompt: str, proposal_content: str
    ) -> RPBRecommendation:
        return RPBRecommendation(
            approve=True,
            confidence=5000,
            reasoning="Placeholder evaluation — real AI provider not configured",
            constitutional_alignment=0.5,
            risks=["No real evaluation performed"],
            benefits=["Proposal submitted for review"],
        )


# =============================================================================
# Three-Tier Constitutional Evaluator
# =============================================================================


class ThreeTierConstitutionalEvaluator:
    """
    Convenience wrapper that combines ConstitutionalGeometry (Tiers 1+2) with
    a direct LLM AIProvider call (Tier 3).

    Used by GovernanceEngine to evaluate individual node instructions before
    they are queued for execution.

    Quick usage (per-instruction compliance check):
        evaluator = ThreeTierConstitutionalEvaluator(geometry=geometry)
        verdict, conf = evaluator.check_action(action_text, encode_fn)
        if verdict == Verdict.VIOLATION:
            reject instruction

    The LLM path (Tier 3) is only invoked when Tiers 1+2 return UNCERTAIN,
    preserving the O(1) fast path for the overwhelming majority of decisions.
    """

    def __init__(
        self,
        geometry: Optional["ConstitutionalGeometry"] = None,  # type: ignore[name-defined]
        provider: Optional[AIProvider] = None,
    ):
        self._geometry = geometry
        self._provider = provider or PlaceholderAIProvider()
        self.logger = logging.getLogger("ThreeTierConstitutionalEvaluator")

    def check_action(
        self,
        action_text: str,
        encode_fn: Optional[object],
        justification: str = "",
    ) -> Tuple[str, float]:
        """
        Evaluate a proposed action for constitutional compliance.

        Returns:
            (verdict_str, confidence) where verdict_str is one of:
            "compliant", "uncertain", "violation"
        """
        from .constitutional_geometry import Verdict

        # Tier 1+2 via geometry
        if self._geometry is not None and encode_fn is not None:
            try:
                embedding = encode_fn(action_text)
                result = self._geometry.evaluate(embedding)
                if not result.drift_warning and result.verdict != Verdict.UNCERTAIN:
                    return result.verdict.value, result.overall_confidence
            except Exception as e:
                self.logger.debug(f"Geometric evaluation failed: {e}")

        # Tier 3: LLM evaluation
        try:
            prompt = (
                "You are evaluating whether a proposed action complies with the "
                "RPB constitutional principles (Human Dignity, Freedom of Thought, "
                "Democratic Governance, Transparency, Privacy, Non-Discrimination, "
                "Cultural Respect). Respond with COMPLIANT or VIOLATION and a "
                "brief reason."
            )
            recommendation = self._provider.evaluate(prompt, action_text)
            verdict = "compliant" if recommendation.approve else "violation"
            confidence = recommendation.confidence / 10_000.0
            return verdict, confidence
        except Exception as e:
            self.logger.warning(f"Tier 3 LLM evaluation failed: {e}")
            # Fail open (uncertain) — never block on evaluator failure
            return "uncertain", 0.0
