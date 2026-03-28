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

  Tier 3 — Full LLM evaluation with multi-node Yuma consensus
    The existing RPBEvaluator/AIProvider chain. Used for high-stakes
    and adversarial-seeming inputs (Verdict.UNCERTAIN from Tiers 1+2).

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


def compute_service_id(project_id: int) -> bytes:
    """Deterministic service ID for a training project."""
    return hashlib.sha256(f"autonet-training-project-{project_id}".encode()).digest()


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
        project_id: int,
    ):
        self.registry = registry
        self.node_id = node_id
        self.project_id = project_id
        self.service_id = compute_service_id(project_id)

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

    def register_if_needed(self, project_contract_address: str) -> bool:
        """
        Register this project as a service in the economic layer.

        Called once at node startup. Idempotent — will skip if already
        registered or if the economy contract isn't deployed.

        Args:
            project_contract_address: Address of the Project.sol contract.

        Returns:
            True if registered (or already was), False on failure.
        """
        if self._service_registered or not self._economy_available:
            return True

        # Check if already registered
        try:
            info = self.registry.get_service(self.service_id)
            if info and info.get("projectContract") not in (None, "0x" + "0" * 40):
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
                project_contract_address,
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
# AI Provider Abstraction (for RPB evaluation)
# =============================================================================


class AIProvider(ABC):
    """
    Abstract interface for AI providers used in RPB evaluation.

    Nodes call evaluate() with the constitutional prompt and proposal data.
    The provider returns a structured recommendation. Different providers
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
# RPB Evaluator (node-side)
# =============================================================================


class RPBEvaluator:
    """
    Node-side RPB evaluator.

    Listens for new proposals, loads the constitutional prompt from Registry,
    evaluates proposals through the 3-tier pipeline, and submits structured
    recommendations on-chain.

    Tier routing:
      - Tier 1+2 (geometric/bottleneck): used when an embedding of the proposal
        content is available (e.g., encoded via the node's TextEncoder)
      - Tier 3 (LLM): always used when Tier 1+2 returns UNCERTAIN, and always
        used when no geometry evaluator is configured

    Usage:
        evaluator = RPBEvaluator(governance_bridge, provider)
        evaluator.evaluate_pending_proposals()  # Call periodically
    """

    def __init__(
        self,
        governance: GovernanceBridge,
        provider: Optional[AIProvider] = None,
        blob_store: Optional[object] = None,
        geometry: Optional["ConstitutionalGeometry"] = None,  # type: ignore[name-defined]
        encode_fn: Optional[object] = None,
    ):
        self.governance = governance
        self.provider = provider or PlaceholderAIProvider()
        self.blob_store = blob_store
        self._geometry = geometry    # ConstitutionalGeometry instance (optional)
        self._encode_fn = encode_fn  # Callable[[str], Tensor]: text → embedding
        self._evaluated_proposals: set = set()
        self.logger = logging.getLogger(
            f"RPBEvaluator[{governance.node_id}]"
        )

    def evaluate_pending_proposals(self) -> int:
        """
        Evaluate all pending proposals that this node hasn't evaluated yet.

        Routes each proposal through the 3-tier constitutional pipeline:
          - Tier 1+2: fast geometric/bottleneck evaluation (if geometry configured)
          - Tier 3:   full LLM evaluation via AIProvider (always available fallback)

        Returns:
            Number of proposals evaluated.
        """
        pending = self.governance.get_pending_proposals()
        if not pending:
            return 0

        prompt_cid = self.governance.get_rpb_prompt()
        prompt_text: Optional[str] = None
        if prompt_cid:
            prompt_text = self._resolve_prompt(prompt_cid)

        evaluated = 0
        for proposal in pending:
            pid = proposal["id"]
            if pid in self._evaluated_proposals:
                continue

            try:
                recommendation = self._evaluate_proposal_tiered(
                    proposal, prompt_text
                )
                if recommendation is None:
                    continue

                reason_cid = f"rpb-eval-{pid}-{self.governance.node_id}"
                success = self.governance.submit_rpb_evaluation(
                    pid,
                    recommendation.approve,
                    recommendation.confidence,
                    reason_cid,
                )

                if success:
                    self._evaluated_proposals.add(pid)
                    evaluated += 1
                    self.logger.info(
                        f"Evaluated proposal {pid}: "
                        f"{'approve' if recommendation.approve else 'reject'} "
                        f"(confidence={recommendation.confidence})"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to evaluate proposal {pid}: {e}")

        return evaluated

    def _evaluate_proposal_tiered(
        self,
        proposal: Dict[str, Any],
        prompt_text: Optional[str],
    ) -> Optional[RPBRecommendation]:
        """
        Route a proposal through the 3-tier constitutional pipeline.

        Tries Tier 1+2 (geometric) first when available. Falls through to
        Tier 3 (LLM) when:
          - Geometry evaluator is not configured
          - Proposal embedding is unavailable
          - Tier 1+2 returns Verdict.UNCERTAIN
          - drift_warning is set (geometry may be unreliable)
        """
        content_cid = proposal.get("contentCid", "")

        # Attempt Tier 1+2 if geometry + encoder are available
        if self._geometry is not None and self._encode_fn is not None and content_cid:
            tier_result = self._run_geometric_tiers(content_cid)
            if tier_result is not None:
                return tier_result

        # Tier 3: full LLM evaluation
        if not prompt_text:
            self.logger.debug(
                f"Proposal {proposal.get('id')}: no RPB prompt resolved — "
                f"using PlaceholderAIProvider"
            )
        llm_prompt = prompt_text or "Evaluate the following proposal."
        return self.provider.evaluate(llm_prompt, content_cid)

    def _run_geometric_tiers(
        self, content_cid: str
    ) -> Optional[RPBRecommendation]:
        """
        Try Tier 1+2 for a proposal. Returns None if LLM fallback is needed.
        """
        from .constitutional_geometry import Verdict

        try:
            embedding = self._encode_fn(content_cid)
        except Exception as e:
            self.logger.debug(f"Could not encode proposal content: {e}")
            return None

        result = self._geometry.evaluate(embedding)

        # Always use LLM if drift detected (geometric results may be wrong)
        if result.drift_warning:
            self.logger.info(
                "Embedding drift detected — skipping Tier 1+2, using LLM (Tier 3)"
            )
            return None

        if result.verdict == Verdict.UNCERTAIN:
            return None  # Tier 1+2 inconclusive → LLM

        # Convert geometric result to RPBRecommendation
        approve = result.verdict == Verdict.COMPLIANT
        # Scale 0-1 confidence to 0-10000 bps
        confidence_bps = int(result.overall_confidence * 10_000)

        weakest_note = ""
        if result.weakest_principle and not approve:
            weakest_note = (
                f" Weakest principle: {result.weakest_principle.value}."
            )

        self.logger.info(
            f"Geometric Tier {result.tier_used} decision: "
            f"{'approve' if approve else 'reject'} "
            f"(conf={confidence_bps} bps){weakest_note}"
        )

        return RPBRecommendation(
            approve=approve,
            confidence=confidence_bps,
            reasoning=result.explanation + weakest_note,
            constitutional_alignment=result.overall_confidence,
            risks=(
                []
                if approve
                else [
                    f"Potential violation of {result.weakest_principle.value}"
                    if result.weakest_principle
                    else "Constitutional concern detected"
                ]
            ),
            benefits=(
                ["Geometric constitutional evaluation: compliant"] if approve else []
            ),
        )

    def _resolve_prompt(self, prompt_cid: str) -> Optional[str]:
        """
        Resolve a prompt CID to the full evaluation prompt text.

        Uses rpb_prompt.load_prompt_text which tries blob store first,
        then falls back to the local constitution/v1_udhr.txt file.
        """
        try:
            from .rpb_prompt import load_prompt_text
            return load_prompt_text(self.blob_store, prompt_cid)
        except Exception as e:
            self.logger.warning(f"Failed to resolve prompt {prompt_cid}: {e}")
            return None


# =============================================================================
# RPB Consensus
# =============================================================================


class RPBConsensus:
    """
    Consensus mechanism for RPB evaluations.

    Multiple nodes evaluate proposals independently using (potentially
    different) AI providers. Non-deterministic LLM outputs are resolved
    through weighted confidence voting:

    1. Each evaluator submits (approve/reject, confidence, reasoning)
    2. Approval is computed as: sum(approve_confidence) / sum(all_confidence)
    3. If approval >= threshold, proposal is approved for trial
    4. Resolution is permissionless (anyone calls resolveEvaluation)

    This extends the Yuma consensus pattern used for training verification
    to handle the non-deterministic nature of LLM evaluations.
    """

    def __init__(self, governance: GovernanceBridge):
        self.governance = governance
        self.logger = logging.getLogger(
            f"RPBConsensus[{governance.node_id}]"
        )

    def try_resolve_proposals(self) -> int:
        """
        Attempt to resolve any proposals that have reached quorum.

        Calls the permissionless resolveEvaluation on-chain function.
        The contract enforces quorum and time requirements.

        Returns:
            Number of proposals resolved.
        """
        if not self.governance._evolution_available:
            return 0

        pending = self.governance.get_pending_proposals()
        resolved = 0

        for proposal in pending:
            pid = proposal["id"]
            try:
                result = self.governance.registry.resolve_proposal_evaluation(pid)
                if result.success:
                    self.logger.info(f"Resolved proposal {pid}")
                    resolved += 1
                else:
                    # QuorumNotReached or EvaluationPeriodActive are expected
                    if "QuorumNotReached" not in str(result.error) and \
                       "EvaluationPeriodActive" not in str(result.error):
                        self.logger.debug(
                            f"Could not resolve proposal {pid}: {result.error}"
                        )
            except Exception as e:
                self.logger.debug(f"Resolution attempt for {pid} failed: {e}")

        return resolved


# =============================================================================
# Three-Tier Constitutional Evaluator
# =============================================================================


class ThreeTierConstitutionalEvaluator:
    """
    Convenience wrapper that combines ConstitutionalGeometry (Tiers 1+2) with
    the existing LLM-based RPBEvaluator (Tier 3).

    Used by GovernanceEngine to evaluate individual node instructions before
    they are queued for execution, and by RPBEvaluator for on-chain proposals.

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
