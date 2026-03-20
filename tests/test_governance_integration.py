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
from nodes.common.governance import GovernanceBridge, compute_service_id
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
