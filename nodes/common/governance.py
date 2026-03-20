"""
Governance integration for Autonet nodes.

Provides attestation, heartbeat listening, and service registration
that all node types share. This bridges the training loop to the
economic layer (Autonet.sol on the jurisdiction chain).
"""

import hashlib
import logging
import time
from typing import Optional

from .contracts import ContractRegistry

logger = logging.getLogger(__name__)


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
