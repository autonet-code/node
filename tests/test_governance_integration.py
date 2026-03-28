"""
Integration test: governance bridge, attestation, heartbeat, constitution.

Exercises the economic and governance integration layer offline (mocked
blockchain) to prove that:
1. GovernanceBridge correctly attests task completions
2. Heartbeat checks work (alive / missed / no-DAO fallback)
3. Service registration is idempotent
4. Constitution can be populated from a ContractRegistry
5. compute_service_id is deterministic

Run: python -m pytest tests/test_governance_integration.py -v
"""

import sys
import time
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.common.blockchain import TransactionResult
from nodes.common.governance import (
    GovernanceBridge,
    compute_service_id,
    RPBEvaluator,
    RPBConsensus,
    RPBRecommendation,
    PlaceholderAIProvider,
    AIProvider,
)
from nodes.core.constitution import (
    Constitution,
    AUTONET_PRINCIPLES,
    AUTONET_BLUEPRINT,
    constitution_from_registry,
)


# =============================================================================
# Helpers: mock registry
# =============================================================================


def _ok() -> TransactionResult:
    return TransactionResult(success=True, tx_hash="0xabc")


def _fail(msg: str = "reverted") -> TransactionResult:
    return TransactionResult(success=False, error=msg)


class FakeContractHandle:
    """Minimal stand-in for ContractHandle."""
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address


def make_registry(
    has_economy: bool = True,
    has_dao: bool = True,
    has_evolution: bool = False,
) -> MagicMock:
    """Build a mock ContractRegistry for governance tests."""
    registry = MagicMock()

    def _get(name: str):
        if name == "AutonetEconomy" and has_economy:
            return FakeContractHandle("AutonetEconomy", "0x1111111111111111111111111111111111111111")
        if name == "AutonetDAO" and has_dao:
            return FakeContractHandle("AutonetDAO", "0x2222222222222222222222222222222222222222")
        if name == "ParticipantStaking":
            return FakeContractHandle("ParticipantStaking", "0x3333333333333333333333333333333333333333")
        if name == "TaskContract":
            return FakeContractHandle("TaskContract", "0x4444444444444444444444444444444444444444")
        if name == "ResultsRewards":
            return FakeContractHandle("ResultsRewards", "0x5555555555555555555555555555555555555555")
        if name == "Project":
            return FakeContractHandle("Project", "0x6666666666666666666666666666666666666666")
        if name == "EvolutionProposal" and has_evolution:
            return FakeContractHandle("EvolutionProposal", "0x7777777777777777777777777777777777777777")
        return None

    registry.get = MagicMock(side_effect=_get)
    registry.attest_usage = MagicMock(return_value=_ok())
    registry.register_service = MagicMock(return_value=_ok())
    registry.activate_service = MagicMock(return_value=_ok())
    registry.get_service = MagicMock(return_value=None)  # not yet registered
    registry.get_current_epoch = MagicMock(return_value=1)
    registry.get_new_events = MagicMock(return_value=[])
    registry.blockchain = MagicMock()
    registry.blockchain.rpc_url = "http://127.0.0.1:8545"

    return registry


# =============================================================================
# Tests: compute_service_id
# =============================================================================


class TestComputeServiceId:
    def test_deterministic(self):
        """Same project_id always produces the same service_id."""
        a = compute_service_id(1)
        b = compute_service_id(1)
        assert a == b
        assert len(a) == 32  # SHA-256 → 32 bytes

    def test_different_projects(self):
        """Different project IDs produce different service IDs."""
        a = compute_service_id(1)
        b = compute_service_id(2)
        assert a != b

    def test_matches_manual_sha256(self):
        """The function uses the expected hashing scheme."""
        expected = hashlib.sha256(b"autonet-training-project-42").digest()
        assert compute_service_id(42) == expected


# =============================================================================
# Tests: GovernanceBridge — Attestation
# =============================================================================


class TestAttestation:
    def test_attest_success(self):
        """Attestation succeeds when economy is deployed."""
        registry = make_registry(has_economy=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.attest_task_completion(units=3)

        assert result is True
        registry.attest_usage.assert_called_once_with(bridge.service_id, 3)

    def test_attest_skipped_no_economy(self):
        """Attestation is gracefully skipped when economy is not deployed."""
        registry = make_registry(has_economy=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.attest_task_completion(units=1)

        assert result is True  # Skipped = success (no error)
        registry.attest_usage.assert_not_called()

    def test_attest_failure_returns_false(self):
        """Attestation returns False when the TX fails."""
        registry = make_registry(has_economy=True)
        registry.attest_usage.return_value = _fail("epoch not started")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.attest_task_completion(units=1)

        assert result is False

    def test_attest_exception_returns_false(self):
        """Attestation returns False on unexpected exceptions."""
        registry = make_registry(has_economy=True)
        registry.attest_usage.side_effect = RuntimeError("connection lost")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.attest_task_completion(units=1)

        assert result is False

    def test_default_units_is_one(self):
        """Default attestation unit count is 1."""
        registry = make_registry(has_economy=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        bridge.attest_task_completion()

        registry.attest_usage.assert_called_once_with(bridge.service_id, 1)


# =============================================================================
# Tests: GovernanceBridge — Heartbeat
# =============================================================================


class TestHeartbeat:
    def test_heartbeat_alive_on_first_check(self):
        """First heartbeat check returns True (benefit of the doubt)."""
        registry = make_registry(has_dao=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.check_heartbeat() is True

    def test_heartbeat_alive_with_events(self):
        """Heartbeat returns True when HeartbeatEmitted events arrive."""
        registry = make_registry(has_dao=True)
        registry.get_new_events.return_value = [{"event": "HeartbeatEmitted"}]
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.check_heartbeat() is True

    def test_heartbeat_missed_after_timeout(self):
        """Heartbeat returns False when no event for > interval."""
        registry = make_registry(has_dao=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        bridge._heartbeat_interval = 0.1  # 100ms for fast test

        # First check sets _last_heartbeat
        bridge.check_heartbeat()

        # Wait for timeout
        time.sleep(0.15)

        assert bridge.check_heartbeat() is False

    def test_heartbeat_always_alive_no_dao(self):
        """When no DAO deployed, heartbeat is always alive (dev mode)."""
        registry = make_registry(has_dao=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        # Even after arbitrary time, always returns True
        assert bridge.check_heartbeat() is True
        assert bridge.check_heartbeat() is True

    def test_receive_heartbeat_manual(self):
        """Manual heartbeat resets the timer."""
        registry = make_registry(has_dao=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        bridge._heartbeat_interval = 0.1

        # First check
        bridge.check_heartbeat()
        time.sleep(0.08)

        # Manual heartbeat before timeout
        bridge.receive_heartbeat()
        time.sleep(0.05)

        # Still alive because we received manual heartbeat
        assert bridge.check_heartbeat() is True

    def test_heartbeat_event_exception_ignored(self):
        """Exceptions from event fetching are silently ignored."""
        registry = make_registry(has_dao=True)
        registry.get_new_events.side_effect = RuntimeError("RPC error")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        # Should not raise, and should still give benefit of the doubt
        assert bridge.check_heartbeat() is True


# =============================================================================
# Tests: GovernanceBridge — Service Registration
# =============================================================================


class TestServiceRegistration:
    def test_register_new_service(self):
        """First registration succeeds and marks the bridge as registered."""
        registry = make_registry(has_economy=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.register_if_needed("0x6666666666666666666666666666666666666666")

        assert result is True
        assert bridge._service_registered is True
        registry.register_service.assert_called_once()

    def test_register_skipped_when_already_registered(self):
        """Second registration call is a no-op."""
        registry = make_registry(has_economy=True)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        bridge.register_if_needed("0x6666666666666666666666666666666666666666")
        bridge.register_if_needed("0x6666666666666666666666666666666666666666")

        # Only called once despite two calls
        registry.register_service.assert_called_once()

    def test_register_skipped_no_economy(self):
        """Registration is skipped when economy is not deployed."""
        registry = make_registry(has_economy=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.register_if_needed("0x6666666666666666666666666666666666666666")

        assert result is True
        registry.register_service.assert_not_called()

    def test_register_detects_existing_on_chain(self):
        """If the service already exists on-chain, skip re-registration."""
        registry = make_registry(has_economy=True)
        registry.get_service.return_value = {
            "projectContract": "0x6666666666666666666666666666666666666666",
            "isActive": True,
        }
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.register_if_needed("0x6666666666666666666666666666666666666666")

        assert result is True
        assert bridge._service_registered is True
        registry.register_service.assert_not_called()  # Didn't need to register

    def test_register_failure_returns_false(self):
        """Registration failure returns False."""
        registry = make_registry(has_economy=True)
        registry.register_service.return_value = _fail("already registered")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.register_if_needed("0x6666666666666666666666666666666666666666")

        assert result is False
        assert bridge._service_registered is False


# =============================================================================
# Tests: Constitution from registry
# =============================================================================


class TestConstitutionFromRegistry:
    def test_populates_addresses_from_registry(self):
        """constitution_from_registry fills blueprint with real addresses."""
        registry = make_registry(has_economy=True, has_dao=True)

        constitution = constitution_from_registry(registry)

        assert isinstance(constitution, Constitution)
        bp = constitution.operational_blueprint

        # Staking address should be populated from the mock registry
        assert bp["staking_contract_address"] == "0x3333333333333333333333333333333333333333"
        assert bp["task_contract_address"] == "0x4444444444444444444444444444444444444444"
        assert bp["results_contract_address"] == "0x5555555555555555555555555555555555555555"
        assert bp["project_contract_address"] == "0x6666666666666666666666666666666666666666"
        assert bp["consensus_contract_address"] == "0x2222222222222222222222222222222222222222"

    def test_uses_rpc_url_from_blockchain(self):
        """RPC URL comes from the registry's blockchain interface."""
        registry = make_registry()
        constitution = constitution_from_registry(registry)
        assert constitution.operational_blueprint["chain_rpc_url"] == "http://127.0.0.1:8545"

    def test_principles_are_standard(self):
        """Constitution from registry uses the standard AUTONET_PRINCIPLES."""
        registry = make_registry()
        constitution = constitution_from_registry(registry)
        assert constitution.principles == AUTONET_PRINCIPLES

    def test_missing_contract_keeps_default(self):
        """If a contract isn't deployed, the blueprint keeps its default address."""
        registry = make_registry(has_economy=False, has_dao=False)
        # AutonetDAO not deployed → consensus_contract_address stays default
        constitution = constitution_from_registry(registry)
        bp = constitution.operational_blueprint
        # Still has the zero address from AUTONET_BLUEPRINT
        assert bp["consensus_contract_address"] == "0x0000000000000000000000000000000000000000"

    def test_constitution_is_frozen(self):
        """Constitution from registry is immutable (frozen dataclass)."""
        registry = make_registry()
        constitution = constitution_from_registry(registry)

        with pytest.raises(AttributeError):
            constitution.principles = frozenset(["new principle"])


# =============================================================================
# Tests: GovernanceBridge — epoch helper
# =============================================================================


class TestEpochHelper:
    def test_get_epoch_returns_current(self):
        """_get_epoch returns the current epoch number."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 5
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge._get_epoch() == 5

    def test_get_epoch_returns_zero_on_error(self):
        """_get_epoch returns 0 when the call fails."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.side_effect = RuntimeError("RPC down")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge._get_epoch() == 0


# =============================================================================
# Tests: Bridge initialization state
# =============================================================================


class TestBridgeInit:
    def test_economy_available_flag(self):
        """Bridge detects economy availability at init."""
        bridge_with = GovernanceBridge(make_registry(has_economy=True), "n", 1)
        bridge_without = GovernanceBridge(make_registry(has_economy=False), "n", 1)

        assert bridge_with._economy_available is True
        assert bridge_without._economy_available is False

    def test_dao_available_flag(self):
        """Bridge detects DAO availability at init."""
        bridge_with = GovernanceBridge(make_registry(has_dao=True), "n", 1)
        bridge_without = GovernanceBridge(make_registry(has_dao=False), "n", 1)

        assert bridge_with._dao_available is True
        assert bridge_without._dao_available is False

    def test_service_id_set_from_project_id(self):
        """Service ID is derived from project_id at init."""
        bridge = GovernanceBridge(make_registry(), "solver-0", project_id=7)
        assert bridge.service_id == compute_service_id(7)


# =============================================================================
# Tests: GovernanceBridge — Reward Claiming
# =============================================================================


class TestRewardClaiming:
    def test_claim_rewards_success(self):
        """Successful reward claim for previous epoch."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 3
        registry.claim_participant_reward = MagicMock(return_value=_ok())
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is True
        # Should claim epoch 2 (current - 1)
        registry.claim_participant_reward.assert_called_once_with(
            bridge.service_id, 2
        )

    def test_claim_rewards_specific_epoch(self):
        """Claim rewards for a specific epoch."""
        registry = make_registry(has_economy=True)
        registry.claim_participant_reward = MagicMock(return_value=_ok())
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards(epoch_id=5)

        assert result is True
        registry.claim_participant_reward.assert_called_once_with(
            bridge.service_id, 5
        )

    def test_claim_rewards_skipped_no_economy(self):
        """Reward claiming is skipped when economy is not deployed."""
        registry = make_registry(has_economy=False)
        registry.claim_participant_reward = MagicMock()
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is True
        registry.claim_participant_reward.assert_not_called()

    def test_claim_already_claimed_is_ok(self):
        """AlreadyClaimed error is treated as success (idempotent)."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 3
        registry.claim_participant_reward = MagicMock(
            return_value=_fail("AlreadyClaimed")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is True  # Not an error

    def test_claim_nothing_to_claim_is_ok(self):
        """NothingToClaim error is treated as success."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 3
        registry.claim_participant_reward = MagicMock(
            return_value=_fail("NothingToClaim")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is True

    def test_claim_failure_returns_false(self):
        """Real claim failure returns False."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 3
        registry.claim_participant_reward = MagicMock(
            return_value=_fail("InsufficientFunds")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is False

    def test_claim_exception_returns_false(self):
        """Exception during claim returns False."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 3
        registry.claim_participant_reward = MagicMock(
            side_effect=RuntimeError("RPC error")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is False

    def test_claim_skipped_epoch_zero(self):
        """No claim when there's no finalized epoch."""
        registry = make_registry(has_economy=True)
        registry.get_current_epoch.return_value = 1
        registry.claim_participant_reward = MagicMock()
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_epoch_rewards()

        assert result is True
        registry.claim_participant_reward.assert_not_called()


# =============================================================================
# Tests: GovernanceBridge — Reputation
# =============================================================================


class TestReputation:
    def test_claim_reputation_success(self):
        """Successful reputation claim."""
        registry = make_registry(has_economy=True)
        registry.claim_reputation = MagicMock(return_value=_ok())
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_reputation()

        assert result is True
        registry.claim_reputation.assert_called_once()

    def test_claim_reputation_skipped_no_economy(self):
        """Reputation claim skipped when economy is not deployed."""
        registry = make_registry(has_economy=False)
        registry.claim_reputation = MagicMock()
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_reputation()

        assert result is True
        registry.claim_reputation.assert_not_called()

    def test_claim_reputation_failure_is_not_fatal(self):
        """Reputation claim failure returns True (not fatal)."""
        registry = make_registry(has_economy=True)
        registry.claim_reputation = MagicMock(
            return_value=_fail("Nothing to claim")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_reputation()

        assert result is True  # Failed claim is not fatal

    def test_claim_reputation_exception_returns_false(self):
        """Exception during reputation claim returns False."""
        registry = make_registry(has_economy=True)
        registry.claim_reputation = MagicMock(
            side_effect=RuntimeError("RPC error")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.claim_reputation()

        assert result is False


# =============================================================================
# Tests: GovernanceBridge — Capability Reporting
# =============================================================================


class TestCapabilityReporting:
    def test_report_score_success(self):
        """Successful capability score report."""
        registry = make_registry(has_economy=True)
        registry.update_capability_score = MagicMock(return_value=_ok())
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        module_id = hashlib.sha256(b"visual_encoder").digest()
        result = bridge.report_capability_score(module_id, 7500)

        assert result is True
        registry.update_capability_score.assert_called_once_with(module_id, 7500)

    def test_report_score_skipped_no_economy(self):
        """Score report skipped when economy is not deployed."""
        registry = make_registry(has_economy=False)
        registry.update_capability_score = MagicMock()
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        module_id = hashlib.sha256(b"visual_encoder").digest()
        result = bridge.report_capability_score(module_id, 7500)

        assert result is True
        registry.update_capability_score.assert_not_called()

    def test_get_scorecard_success(self):
        """Successful scorecard fetch."""
        registry = make_registry(has_economy=True)
        registry.get_capability_scorecard = MagicMock(return_value={
            "modules": [
                {"id": b"\x01" * 32, "score": 4000, "target": 8000, "multiplier": 17500},
                {"id": b"\x02" * 32, "score": 0, "target": 6000, "multiplier": 30000},
            ]
        })
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        scorecard = bridge.get_training_scorecard()

        assert scorecard is not None
        assert len(scorecard["modules"]) == 2
        assert scorecard["modules"][0]["multiplier"] == 17500

    def test_get_scorecard_returns_none_no_economy(self):
        """Scorecard returns None when economy is not deployed."""
        registry = make_registry(has_economy=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_training_scorecard() is None

    def test_get_scorecard_exception_returns_none(self):
        """Scorecard returns None on exception."""
        registry = make_registry(has_economy=True)
        registry.get_capability_scorecard = MagicMock(
            side_effect=RuntimeError("not deployed")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_training_scorecard() is None


# =============================================================================
# Tests: GovernanceBridge — Evolution Proposals
# =============================================================================


class TestEvolutionProposals:
    def test_get_pending_proposals_success(self):
        """Fetch proposals in Evaluating status."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=3)
        registry.get_evolution_proposal = MagicMock(side_effect=[
            {"status": 0, "contentCid": "cid1"},   # Proposed
            {"status": 1, "contentCid": "cid2"},   # Evaluating
            {"status": 2, "contentCid": "cid3"},   # Trial
        ])
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        pending = bridge.get_pending_proposals()

        assert len(pending) == 1
        assert pending[0]["contentCid"] == "cid2"
        assert pending[0]["id"] == 2

    def test_get_pending_proposals_no_evolution(self):
        """Returns empty list when evolution contract not deployed."""
        registry = make_registry(has_evolution=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_pending_proposals() == []

    def test_get_pending_proposals_exception(self):
        """Returns empty list on exception."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(
            side_effect=RuntimeError("RPC error")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_pending_proposals() == []

    def test_submit_rpb_evaluation_success(self):
        """Successful RPB evaluation submission."""
        registry = make_registry(has_evolution=True)
        registry.submit_rpb_evaluation = MagicMock(return_value=_ok())
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.submit_rpb_evaluation(1, True, 8000, "reason-cid")

        assert result is True
        registry.submit_rpb_evaluation.assert_called_once_with(
            1, True, 8000, "reason-cid"
        )

    def test_submit_rpb_evaluation_no_evolution(self):
        """Evaluation skipped when evolution contract not deployed."""
        registry = make_registry(has_evolution=False)
        registry.submit_rpb_evaluation = MagicMock()
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.submit_rpb_evaluation(1, True, 8000, "reason-cid")

        assert result is True
        registry.submit_rpb_evaluation.assert_not_called()

    def test_submit_rpb_evaluation_already_evaluated(self):
        """AlreadyEvaluated treated as success."""
        registry = make_registry(has_evolution=True)
        registry.submit_rpb_evaluation = MagicMock(
            return_value=_fail("AlreadyEvaluated")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.submit_rpb_evaluation(1, True, 8000, "reason-cid")

        assert result is True

    def test_submit_rpb_evaluation_failure(self):
        """Real failure returns False."""
        registry = make_registry(has_evolution=True)
        registry.submit_rpb_evaluation = MagicMock(
            return_value=_fail("InsufficientStake")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.submit_rpb_evaluation(1, True, 8000, "reason-cid")

        assert result is False

    def test_submit_rpb_evaluation_exception(self):
        """Exception returns False."""
        registry = make_registry(has_evolution=True)
        registry.submit_rpb_evaluation = MagicMock(
            side_effect=RuntimeError("connection lost")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        result = bridge.submit_rpb_evaluation(1, True, 8000, "reason-cid")

        assert result is False

    def test_get_rpb_prompt_success(self):
        """Successful RPB prompt fetch."""
        registry = make_registry(has_evolution=True)
        registry.get_current_rpb_prompt = MagicMock(return_value="bafyrpb_prompt_v1")
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_rpb_prompt() == "bafyrpb_prompt_v1"

    def test_get_rpb_prompt_no_evolution(self):
        """Returns None when evolution contract not deployed."""
        registry = make_registry(has_evolution=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_rpb_prompt() is None

    def test_get_rpb_prompt_exception(self):
        """Returns None on exception."""
        registry = make_registry(has_evolution=True)
        registry.get_current_rpb_prompt = MagicMock(
            side_effect=RuntimeError("RPC error")
        )
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)

        assert bridge.get_rpb_prompt() is None

    def test_evolution_available_flag(self):
        """Bridge detects evolution contract availability."""
        bridge_with = GovernanceBridge(
            make_registry(has_evolution=True), "n", 1
        )
        bridge_without = GovernanceBridge(
            make_registry(has_evolution=False), "n", 1
        )

        assert bridge_with._evolution_available is True
        assert bridge_without._evolution_available is False


# =============================================================================
# Tests: RPBRecommendation
# =============================================================================


class TestRPBRecommendation:
    def test_to_json(self):
        """Recommendation serializes to JSON."""
        rec = RPBRecommendation(
            approve=True,
            confidence=8000,
            reasoning="Aligns with constitutional principles",
            constitutional_alignment=0.85,
            risks=["Requires significant compute"],
            benefits=["Improves visual encoder"],
        )
        import json
        data = json.loads(rec.to_json())

        assert data["approve"] is True
        assert data["confidence"] == 8000
        assert len(data["risks"]) == 1
        assert data["constitutional_alignment"] == 0.85

    def test_defaults(self):
        """Recommendation has sensible defaults for optional fields."""
        rec = RPBRecommendation(
            approve=False,
            confidence=3000,
            reasoning="Too risky",
            constitutional_alignment=0.3,
        )
        assert rec.risks == []
        assert rec.benefits == []


# =============================================================================
# Tests: PlaceholderAIProvider
# =============================================================================


class TestPlaceholderAIProvider:
    def test_returns_recommendation(self):
        """Placeholder provider returns a valid recommendation."""
        provider = PlaceholderAIProvider()
        rec = provider.evaluate("system prompt", "proposal content")

        assert isinstance(rec, RPBRecommendation)
        assert rec.approve is True
        assert rec.confidence == 5000
        assert "Placeholder" in rec.reasoning


# =============================================================================
# Tests: RPBEvaluator
# =============================================================================


class TestRPBEvaluator:
    def test_evaluate_pending_proposals(self):
        """Evaluator processes pending proposals."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=2)
        registry.get_evolution_proposal = MagicMock(side_effect=[
            {"status": 1, "contentCid": "cid1"},   # Evaluating
            {"status": 0, "contentCid": "cid2"},   # Proposed (not pending)
        ])
        registry.get_current_rpb_prompt = MagicMock(return_value="prompt-cid")
        registry.submit_rpb_evaluation = MagicMock(return_value=_ok())

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge)

        count = evaluator.evaluate_pending_proposals()

        assert count == 1
        registry.submit_rpb_evaluation.assert_called_once()
        args = registry.submit_rpb_evaluation.call_args
        assert args[0][0] == 1  # proposal_id
        assert args[0][1] is True  # approve (placeholder always approves)
        assert args[0][2] == 5000  # confidence (placeholder default)

    def test_skips_already_evaluated(self):
        """Evaluator skips proposals it already evaluated."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.get_current_rpb_prompt = MagicMock(return_value="prompt-cid")
        registry.submit_rpb_evaluation = MagicMock(return_value=_ok())

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge)

        # First call evaluates
        assert evaluator.evaluate_pending_proposals() == 1
        # Second call skips (already evaluated)
        assert evaluator.evaluate_pending_proposals() == 0

    def test_no_prompt_skips(self):
        """Evaluator skips when no RPB prompt is configured."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.get_current_rpb_prompt = MagicMock(return_value=None)

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge)

        assert evaluator.evaluate_pending_proposals() == 0

    def test_no_pending_returns_zero(self):
        """Evaluator returns 0 when no proposals pending."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=0)

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge)

        assert evaluator.evaluate_pending_proposals() == 0

    def test_custom_provider(self):
        """Evaluator uses a custom AI provider."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.get_current_rpb_prompt = MagicMock(return_value="prompt-cid")
        registry.submit_rpb_evaluation = MagicMock(return_value=_ok())

        # Custom provider that always rejects
        class RejectProvider(AIProvider):
            def evaluate(self, system_prompt, proposal_content):
                return RPBRecommendation(
                    approve=False, confidence=9000,
                    reasoning="Against principles",
                    constitutional_alignment=0.1,
                )

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge, provider=RejectProvider())

        count = evaluator.evaluate_pending_proposals()

        assert count == 1
        args = registry.submit_rpb_evaluation.call_args
        assert args[0][1] is False  # reject
        assert args[0][2] == 9000   # confidence

    def test_evaluation_exception_handled(self):
        """Evaluator handles provider exceptions gracefully."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.get_current_rpb_prompt = MagicMock(return_value="prompt-cid")

        # Provider that throws
        class ErrorProvider(AIProvider):
            def evaluate(self, system_prompt, proposal_content):
                raise RuntimeError("model unavailable")

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        evaluator = RPBEvaluator(bridge, provider=ErrorProvider())

        # Should not raise
        count = evaluator.evaluate_pending_proposals()
        assert count == 0


# =============================================================================
# Tests: RPBConsensus
# =============================================================================


class TestRPBConsensus:
    def test_try_resolve_success(self):
        """Consensus resolves a proposal that reached quorum."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.resolve_proposal_evaluation = MagicMock(return_value=_ok())

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        consensus = RPBConsensus(bridge)

        resolved = consensus.try_resolve_proposals()

        assert resolved == 1
        registry.resolve_proposal_evaluation.assert_called_once_with(1)

    def test_try_resolve_quorum_not_reached(self):
        """Expected failure (quorum not reached) is handled silently."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.resolve_proposal_evaluation = MagicMock(
            return_value=_fail("QuorumNotReached")
        )

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        consensus = RPBConsensus(bridge)

        resolved = consensus.try_resolve_proposals()

        assert resolved == 0

    def test_try_resolve_period_active(self):
        """Expected failure (evaluation period active) is handled."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.resolve_proposal_evaluation = MagicMock(
            return_value=_fail("EvaluationPeriodActive")
        )

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        consensus = RPBConsensus(bridge)

        resolved = consensus.try_resolve_proposals()

        assert resolved == 0

    def test_try_resolve_no_evolution(self):
        """Returns 0 when evolution contract not deployed."""
        registry = make_registry(has_evolution=False)
        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        consensus = RPBConsensus(bridge)

        assert consensus.try_resolve_proposals() == 0

    def test_try_resolve_exception(self):
        """Exception during resolution is handled."""
        registry = make_registry(has_evolution=True)
        registry.get_evolution_proposal_count = MagicMock(return_value=1)
        registry.get_evolution_proposal = MagicMock(return_value={
            "status": 1, "contentCid": "cid1"
        })
        registry.resolve_proposal_evaluation = MagicMock(
            side_effect=RuntimeError("RPC error")
        )

        bridge = GovernanceBridge(registry, "solver-0", project_id=1)
        consensus = RPBConsensus(bridge)

        # Should not raise
        resolved = consensus.try_resolve_proposals()
        assert resolved == 0


# =============================================================================
# Tests: AutonetService governance wiring (Phase 2.2)
# =============================================================================


class TestAutonetServiceGovernanceWiring:
    """Verify that AutonetService._init_governance() creates a GovernanceBridge
    when blockchain config has a private_key, and passes it to TrainingDataFeed."""

    def test_init_governance_returns_bridge_with_private_key(self):
        """With private_key configured, _init_governance returns a GovernanceBridge."""
        from nodes.common.config import AutonetConfig, BlockchainConfig
        from nodes.service import AutonetService

        config = AutonetConfig()
        config.blockchain = BlockchainConfig(
            rpc_url="http://127.0.0.1:8545",
            private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            chain_id=31337,
        )

        service = AutonetService(config=config, data_dir="/tmp/test")

        with patch("nodes.common.blockchain.BlockchainInterface") as MockBI, \
             patch("nodes.common.contracts.ContractRegistry") as MockCR, \
             patch("nodes.common.governance.GovernanceBridge") as MockGB:

            # Mock blockchain is connected
            mock_bi = MockBI.return_value
            mock_bi.web3 = MagicMock()
            mock_bi.web3.is_connected.return_value = True
            mock_bi.account = MagicMock()
            mock_bi.account.address = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

            result = service._init_governance()

            assert result is not None
            MockBI.assert_called_once_with(
                rpc_url="http://127.0.0.1:8545",
                private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                chain_id=31337,
            )
            MockCR.assert_called_once_with(mock_bi)
            MockGB.assert_called_once()

    def test_init_governance_returns_none_without_private_key(self):
        """Without private_key, _init_governance returns None (offline mode)."""
        from nodes.common.config import AutonetConfig, BlockchainConfig
        from nodes.service import AutonetService

        config = AutonetConfig()
        config.blockchain = BlockchainConfig(
            rpc_url="http://127.0.0.1:8545",
            private_key=None,
            chain_id=31337,
        )

        service = AutonetService(config=config)
        result = service._init_governance()
        assert result is None

    def test_init_governance_returns_none_on_connection_failure(self):
        """If blockchain unreachable, returns None (offline mode)."""
        from nodes.common.config import AutonetConfig, BlockchainConfig
        from nodes.service import AutonetService

        config = AutonetConfig()
        config.blockchain = BlockchainConfig(
            rpc_url="http://127.0.0.1:9999",
            private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            chain_id=31337,
        )

        service = AutonetService(config=config, data_dir="/tmp/test")

        with patch("nodes.common.blockchain.BlockchainInterface") as MockBI:
            mock_bi = MockBI.return_value
            mock_bi.web3 = MagicMock()
            mock_bi.web3.is_connected.return_value = False

            result = service._init_governance()
            assert result is None

    def test_training_feed_receives_governance(self):
        """_init_training_feed passes governance bridge to TrainingDataFeed."""
        from nodes.common.config import AutonetConfig, BlockchainConfig
        from nodes.service import AutonetService

        config = AutonetConfig()
        config.blockchain = BlockchainConfig(
            rpc_url="http://127.0.0.1:8545",
            private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            chain_id=31337,
        )

        service = AutonetService(config=config, data_dir="/tmp/test")

        mock_gov = MagicMock()
        with patch.object(service, "_init_governance", return_value=mock_gov):
            service._init_training_feed()

        assert service._training_feed is not None
        # The attestor inside the feed should have governance set
        assert service._training_feed._attestor.governance is mock_gov

    def test_training_feed_works_offline(self):
        """Training feed works without governance (offline mode)."""
        from nodes.common.config import AutonetConfig, BlockchainConfig
        from nodes.service import AutonetService

        config = AutonetConfig()
        config.blockchain = BlockchainConfig(private_key=None)

        service = AutonetService(config=config, data_dir="/tmp/test")

        with patch.object(service, "_init_governance", return_value=None):
            service._init_training_feed()

        assert service._training_feed is not None
        assert service._training_feed._attestor.governance is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
