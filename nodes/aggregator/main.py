"""
Autonet Aggregator Node

Autonomous node that combines verified model updates into improved global models.
Implements federated averaging (FedAvg) and publishes mature models on-chain.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..common.contracts import ContractRegistry
from ..common.ipfs import IPFSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AggregatorMetrics:
    """Metrics tracked by the aggregator node."""
    tasks_proposed: int = 0
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0
    aggregations_done: int = 0
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0


@dataclass
class ProjectAggregationState:
    """State tracking for a single project."""
    project_id: int
    collected_updates: List[str] = field(default_factory=list)
    aggregation_rounds: int = 0
    last_model_cid: Optional[str] = None


class AggregatorNode:
    """
    Autonomous Aggregator Node.

    Responsibilities:
    1. Stake as AGGREGATOR role
    2. Monitor RewardsDistributed events to collect verified update CIDs
    3. Aggregate multiple updates via FedAvg
    4. Publish mature models on-chain via setMatureModel
    """

    AGGREGATOR_ROLE = 4
    STAKE_AMOUNT = 1000 * 10**18  # 1000 ATN
    MIN_UPDATES_FOR_AGGREGATION = 2

    def __init__(
        self,
        registry: ContractRegistry,
        ipfs: IPFSClient,
        node_id: str,
        project_id: int = 1,
    ):
        """
        Initialize the aggregator node.

        Args:
            registry: ContractRegistry for blockchain calls
            ipfs: IPFSClient for IPFS operations
            node_id: Unique identifier for this node (e.g., "aggregator-0")
            project_id: Project ID to aggregate updates for
        """
        self.registry = registry
        self.ipfs = ipfs
        self.node_id = node_id
        self.project_id = project_id

        self.metrics = AggregatorMetrics()
        self.project_state = ProjectAggregationState(project_id=project_id)
        self.is_staked = False
        self.should_stop = False

        self.my_address = self.registry.blockchain.account.address
        logger.info(f"[{self.node_id}] Initialized with address {self.my_address[:10]}...")

    def stop(self):
        """Signal the node to stop running."""
        self.should_stop = True
        logger.info(f"[{self.node_id}] Stop signal received")

    def run(self, max_cycles: int = 100, cycle_delay: float = 5.0):
        """
        Main event loop for the aggregator node.

        Args:
            max_cycles: Maximum number of cycles to run (0 = unlimited)
            cycle_delay: Seconds to wait between cycles
        """
        logger.info(f"[{self.node_id}] Starting aggregator node for project {self.project_id}")
        logger.info(f"[{self.node_id}] max_cycles={max_cycles}, cycle_delay={cycle_delay}s")

        cycle = 0
        while not self.should_stop:
            if max_cycles > 0 and cycle >= max_cycles:
                logger.info(f"[{self.node_id}] Reached max_cycles={max_cycles}, stopping")
                break

            try:
                self._cycle()
                self.metrics.cycles += 1
                cycle += 1
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"[{self.node_id}] Cycle error: {e}", exc_info=True)

            time.sleep(cycle_delay)

        logger.info(f"[{self.node_id}] Stopped after {cycle} cycles")
        self._log_final_metrics()

    def _cycle(self):
        """Execute one cycle of the aggregator loop."""
        # Step 1: Stake on first cycle
        if not self.is_staked:
            self._stake()
            return

        # Step 2: Poll for new RewardsDistributed events
        self._poll_rewards_distributed()

        # Step 3: Check if we have enough updates to aggregate
        if len(self.project_state.collected_updates) >= self.MIN_UPDATES_FOR_AGGREGATION:
            self._aggregate_and_publish()

    def _stake(self):
        """Stake as AGGREGATOR role."""
        logger.info(f"[{self.node_id}] Staking {self.STAKE_AMOUNT / 10**18} ATN as AGGREGATOR")

        # First approve ATN spending
        staking_contract = self.registry.get("ParticipantStaking")
        if not staking_contract:
            logger.error(f"[{self.node_id}] ParticipantStaking contract not found")
            self.metrics.errors += 1
            return

        approve_result = self.registry.approve_atn(
            staking_contract.address,
            self.STAKE_AMOUNT
        )
        if not approve_result.success:
            logger.error(f"[{self.node_id}] ATN approval failed: {approve_result.error}")
            self.metrics.errors += 1
            return

        logger.info(f"[{self.node_id}] ATN approved for staking")

        # Stake
        stake_result = self.registry.stake(self.AGGREGATOR_ROLE, self.STAKE_AMOUNT)
        if stake_result.success:
            self.is_staked = True
            logger.info(f"[{self.node_id}] Successfully staked as AGGREGATOR")
        else:
            logger.error(f"[{self.node_id}] Staking failed: {stake_result.error}")
            self.metrics.errors += 1

    def _poll_rewards_distributed(self):
        """
        Poll for RewardsDistributed events from ResultsRewards contract.

        Event signature (from ResultsRewards.sol):
        event RewardsDistributed(
            uint256 indexed taskId,
            address indexed solver,
            uint256 solverReward,
            uint256 proposerReward
        );

        We extract the task, fetch its details, and collect the update CID.
        """
        try:
            events = self.registry.get_new_events("ResultsRewards", "RewardsDistributed")

            for event in events:
                task_id = event["args"]["taskId"]
                solver = event["args"]["solver"]

                logger.info(
                    f"[{self.node_id}] RewardsDistributed: task={task_id}, solver={solver[:10]}..."
                )

                # Get task details to extract the solution CID
                task = self.registry.get_task(task_id)
                if not task:
                    logger.warning(f"[{self.node_id}] Could not fetch task {task_id}")
                    continue

                # Get solution CID from ResultsRewards
                # In the actual contract, we'd call getSolution(taskId, solver) or similar
                # For now, we'll get it from the revealed solution
                solution_cid = self._get_solution_cid(task_id, solver)
                if solution_cid:
                    self._collect_update(task_id, solution_cid)

        except Exception as e:
            logger.error(f"[{self.node_id}] Error polling events: {e}", exc_info=True)
            self.metrics.errors += 1

    def _get_solution_cid(self, task_id: int, solver: str) -> Optional[str]:
        """
        Retrieve the solution CID for a task/solver pair.

        This would call a contract method like ResultsRewards.getRevealedSolution(taskId, solver).
        For now, we'll attempt to read from contract state directly.
        """
        try:
            # Try calling a method to get the revealed solution CID
            # The actual method name depends on the contract implementation
            # Assuming there's a public mapping: taskSolutions[taskId][solver] -> (cid, ...)
            result = self.registry.call("ResultsRewards", "getRevealedSolution", task_id, solver)
            if result and len(result) > 0:
                return result[0]  # First element is typically the CID
        except Exception as e:
            logger.debug(f"[{self.node_id}] Could not get solution CID for task {task_id}: {e}")

        return None

    def _collect_update(self, task_id: int, update_cid: str):
        """Collect a verified update CID."""
        if update_cid in self.project_state.collected_updates:
            logger.debug(f"[{self.node_id}] Update {update_cid[:20]}... already collected")
            return

        self.project_state.collected_updates.append(update_cid)
        logger.info(
            f"[{self.node_id}] Collected update {update_cid[:20]}... "
            f"(total: {len(self.project_state.collected_updates)})"
        )

    def _aggregate_and_publish(self):
        """
        Aggregate collected updates and publish the new model.

        Steps:
        1. Download all update CIDs from IPFS
        2. Perform FedAvg aggregation
        3. Upload aggregated model to IPFS
        4. Call setMatureModel on-chain
        5. Clear collected updates
        """
        logger.info(
            f"[{self.node_id}] Starting aggregation of "
            f"{len(self.project_state.collected_updates)} updates"
        )

        # Download all updates
        updates = []
        for cid in self.project_state.collected_updates:
            try:
                update_data = self.ipfs.get_json(cid)
                if update_data:
                    updates.append(update_data)
                    logger.debug(f"[{self.node_id}] Downloaded update {cid[:20]}...")
                else:
                    logger.warning(f"[{self.node_id}] Failed to download {cid[:20]}...")
            except Exception as e:
                logger.error(f"[{self.node_id}] Error downloading {cid}: {e}")

        if not updates:
            logger.error(f"[{self.node_id}] No updates downloaded, aborting aggregation")
            self.metrics.errors += 1
            self.project_state.collected_updates.clear()
            return

        # Perform FedAvg
        aggregated_model = self._fedavg(updates)

        # Add metadata
        aggregated_model["metadata"] = {
            "project_id": self.project_id,
            "aggregation_round": self.project_state.aggregation_rounds + 1,
            "updates_count": len(updates),
            "aggregator": self.my_address,
            "timestamp": int(time.time()),
        }

        # Upload to IPFS
        try:
            new_model_cid = self.ipfs.add_json(aggregated_model)
            if not new_model_cid:
                logger.error(f"[{self.node_id}] Failed to upload aggregated model to IPFS")
                self.metrics.errors += 1
                return

            logger.info(f"[{self.node_id}] Aggregated model uploaded: {new_model_cid[:20]}...")
        except Exception as e:
            logger.error(f"[{self.node_id}] Error uploading to IPFS: {e}")
            self.metrics.errors += 1
            return

        # Publish on-chain
        result = self.registry.set_mature_model(
            self.project_id,
            new_model_cid,
            price=0  # Free model for now
        )

        if result.success:
            logger.info(
                f"[{self.node_id}] Published mature model for project {self.project_id}: "
                f"{new_model_cid[:20]}..."
            )
            self.metrics.aggregations_done += 1
            self.project_state.aggregation_rounds += 1
            self.project_state.last_model_cid = new_model_cid
            self.project_state.collected_updates.clear()
        else:
            logger.error(f"[{self.node_id}] Failed to publish model: {result.error}")
            self.metrics.errors += 1

    def _fedavg(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform Federated Averaging on model updates.

        In production, this would:
        - Parse model weights from each update
        - Compute weighted average based on training samples
        - Return the averaged weights

        For demo purposes, we average all numeric values across updates.
        """
        logger.info(f"[{self.node_id}] Performing FedAvg on {len(updates)} updates")

        if not updates:
            return {}

        # Initialize aggregated model with structure from first update
        aggregated = {}

        # Collect all keys across all updates
        all_keys = set()
        for update in updates:
            all_keys.update(update.keys())

        # For each key, aggregate values
        for key in all_keys:
            if key == "metadata":
                continue  # Skip metadata, we'll add our own

            values = [update.get(key) for update in updates if key in update]

            # Try to average numeric values
            try:
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    aggregated[key] = sum(numeric_values) / len(numeric_values)
                else:
                    # For non-numeric, take the first value
                    aggregated[key] = values[0] if values else None
            except Exception:
                # Fallback: take first value
                aggregated[key] = values[0] if values else None

        # Add aggregation-specific fields
        aggregated["aggregation_method"] = "fedavg"
        aggregated["num_updates"] = len(updates)

        logger.debug(f"[{self.node_id}] Aggregated model keys: {list(aggregated.keys())}")

        return aggregated

    def _log_final_metrics(self):
        """Log final metrics at shutdown."""
        logger.info(f"[{self.node_id}] Final metrics:")
        logger.info(f"  Cycles: {self.metrics.cycles}")
        logger.info(f"  Aggregations: {self.metrics.aggregations_done}")
        logger.info(f"  Errors: {self.metrics.errors}")
        logger.info(f"  Aggregation rounds: {self.project_state.aggregation_rounds}")
        if self.project_state.last_model_cid:
            logger.info(f"  Last model CID: {self.project_state.last_model_cid[:20]}...")


def main():
    """Demo/test entry point."""
    from ..common.blockchain import BlockchainInterface

    # Create blockchain and registry (assuming local Hardhat node)
    blockchain = BlockchainInterface()
    registry = ContractRegistry(blockchain)
    ipfs = IPFSClient()

    # Create aggregator node
    node = AggregatorNode(
        registry=registry,
        ipfs=ipfs,
        node_id="aggregator-demo",
        project_id=1,
    )

    # Run for a few cycles
    try:
        node.run(max_cycles=10, cycle_delay=5.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        node.stop()


if __name__ == "__main__":
    main()
