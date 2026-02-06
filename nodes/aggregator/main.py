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
        aggregation_method: str = "fedavg",
        trim_ratio: float = 0.2,
    ):
        """
        Initialize the aggregator node.

        Args:
            registry: ContractRegistry for blockchain calls
            ipfs: IPFSClient for IPFS operations
            node_id: Unique identifier for this node (e.g., "aggregator-0")
            project_id: Project ID to aggregate updates for
            aggregation_method: Aggregation method to use ("fedavg" or "trimmed_mean")
            trim_ratio: Ratio to trim from top/bottom for trimmed_mean (default: 0.2 = 20%)
        """
        self.registry = registry
        self.ipfs = ipfs
        self.node_id = node_id
        self.project_id = project_id
        self.aggregation_method = aggregation_method
        self.trim_ratio = trim_ratio

        self.metrics = AggregatorMetrics()
        self.project_state = ProjectAggregationState(project_id=project_id)
        self.is_staked = False
        self.should_stop = False

        self.my_address = self.registry.blockchain.account.address
        logger.info(f"[{self.node_id}] Initialized with address {self.my_address[:10]}...")
        logger.info(f"[{self.node_id}] Aggregation method: {self.aggregation_method}")

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
            address indexed recipient,
            uint256 amount,
            string rewardType
        );

        We filter for "SolverReward" events, then extract the solution CID from the revealed solutions.
        """
        try:
            events = self.registry.get_new_events("ResultsRewards", "RewardsDistributed")
            if events:
                logger.info(f"[{self.node_id}] Found {len(events)} RewardsDistributed events")

            for event in events:
                task_id = event["args"]["taskId"]
                recipient = event["args"]["recipient"]
                reward_type = event["args"]["rewardType"]

                logger.info(f"[{self.node_id}] RewardsDistributed: type={reward_type}, task={task_id}")

                # Only process solver rewards (these contain the model updates we want to aggregate)
                if reward_type != "SolverReward":
                    continue

                logger.info(
                    f"[{self.node_id}] SolverReward distributed: task={task_id}, solver={recipient[:10]}..."
                )

                # Get solution CID from ResultsRewards.revealedSolutions mapping
                solution_cid = self._get_solution_cid(task_id, recipient)
                if solution_cid:
                    self._collect_update(task_id, solution_cid)
                else:
                    logger.warning(
                        f"[{self.node_id}] No solution CID found for task {task_id}, solver {recipient[:10]}..."
                    )

        except Exception as e:
            logger.error(f"[{self.node_id}] Error polling events: {e}", exc_info=True)
            self.metrics.errors += 1

    def _get_solution_cid(self, task_id: int, solver: str) -> Optional[str]:
        """
        Retrieve the solution CID for a task/solver pair.

        Calls ResultsRewards.revealedSolutions(taskId, solver) which is a public mapping.
        """
        try:
            cid = self.registry.get_revealed_solution(task_id, solver)
            if cid and len(cid) > 0:
                return cid
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

        # Perform aggregation based on configured method
        if self.aggregation_method == "trimmed_mean":
            aggregated_model = self._trimmed_mean_aggregate(updates)
        else:
            aggregated_model = self._fedavg(updates)

        # Add metadata
        aggregated_model["metadata"] = {
            "project_id": self.project_id,
            "aggregation_round": self.project_state.aggregation_rounds + 1,
            "updates_count": len(updates),
            "aggregator": self.my_address,
            "timestamp": int(time.time()),
        }

        # Convert numpy arrays/tensors to lists for JSON serialization
        aggregated_model = self._numpy_to_python(aggregated_model)

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

        This handles both:
        - Real PyTorch weight deltas (from real ML training)
        - Mock training results (fallback)

        For real training, aggregates weight deltas using sample-weighted averaging.
        """
        logger.info(f"[{self.node_id}] Performing FedAvg on {len(updates)} updates")

        if not updates:
            return {}

        # Check if we have real weight deltas
        has_real_deltas = all("weight_delta" in u for u in updates)

        if has_real_deltas:
            logger.info(f"[{self.node_id}] Aggregating real PyTorch weight deltas")
            return self._fedavg_real_weights(updates)
        else:
            logger.info(f"[{self.node_id}] Aggregating mock training results")
            return self._fedavg_mock(updates)

    def _fedavg_real_weights(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate real PyTorch weight deltas using FedAvg.

        Uses sample-weighted averaging:
        - Extract weight_delta from each update
        - Weight by number of training samples
        - Average across all updates
        """
        try:
            from ..common.ml import aggregate_weight_deltas

            # Extract weight deltas and sample counts
            deltas = []
            weights = []
            for update in updates:
                deltas.append(update["weight_delta"])
                num_samples = update.get("metrics", {}).get("num_samples", 1)
                weights.append(num_samples)

            # Aggregate using FedAvg
            aggregated_delta = aggregate_weight_deltas(deltas, weights)

            # Build result
            aggregated = {
                "aggregated_weight_delta": aggregated_delta,
                "aggregation_method": "fedavg",
                "num_updates": len(updates),
                "total_samples": sum(weights),
                "real_training": True,
            }

            # Aggregate metrics
            avg_loss = sum(u.get("metrics", {}).get("loss", 0) for u in updates) / len(updates)
            avg_accuracy = sum(u.get("metrics", {}).get("accuracy", 0) for u in updates) / len(updates)

            aggregated["aggregated_metrics"] = {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "total_samples": sum(weights),
            }

            logger.info(
                f"[{self.node_id}] FedAvg complete: "
                f"avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.4f}, "
                f"total_samples={sum(weights)}"
            )

            return aggregated

        except Exception as e:
            logger.error(f"[{self.node_id}] Error in real weight aggregation: {e}", exc_info=True)
            # Fallback to mock aggregation
            return self._fedavg_mock(updates)

    def _fedavg_mock(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback aggregation for mock training results.

        Averages all numeric values across updates.
        """
        # Initialize aggregated model with structure from first update
        aggregated = {}

        # Collect all keys across all updates
        all_keys = set()
        for update in updates:
            all_keys.update(update.keys())

        # For each key, aggregate values
        for key in all_keys:
            if key in ["metadata", "weight_delta"]:
                continue  # Skip metadata and weight_delta

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
        aggregated["aggregation_method"] = "fedavg_mock"
        aggregated["num_updates"] = len(updates)
        aggregated["real_training"] = False

        logger.debug(f"[{self.node_id}] Aggregated model keys: {list(aggregated.keys())}")

        return aggregated

    def _trimmed_mean_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform Trimmed Mean aggregation on model updates.

        This is a Byzantine-resistant aggregation method that protects against
        malicious nodes by trimming extreme values before averaging.

        For each parameter:
        1. Collect values from all updates
        2. Sort values
        3. Trim top and bottom trim_ratio (default 20%)
        4. Average the remaining values

        This ensures that up to trim_ratio of malicious nodes cannot influence
        the aggregated result.

        Args:
            updates: List of model update dictionaries

        Returns:
            Aggregated model dictionary
        """
        logger.info(f"[{self.node_id}] Performing Trimmed Mean aggregation on {len(updates)} updates")
        logger.info(f"[{self.node_id}] Trim ratio: {self.trim_ratio} (trimming top/bottom {self.trim_ratio*100}%)")

        if not updates:
            return {}

        # Check if we have real weight deltas
        has_real_deltas = all("weight_delta" in u for u in updates)

        if has_real_deltas:
            logger.info(f"[{self.node_id}] Aggregating real PyTorch weight deltas with trimmed mean")
            return self._trimmed_mean_real_weights(updates)
        else:
            logger.info(f"[{self.node_id}] Aggregating mock training results with trimmed mean")
            return self._trimmed_mean_mock(updates)

    def _trimmed_mean_real_weights(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate real PyTorch weight deltas using Trimmed Mean.

        Args:
            updates: List of update dictionaries with weight_delta fields

        Returns:
            Aggregated model dictionary
        """
        try:
            import numpy as np
            import torch

            # Extract weight deltas
            deltas = [update["weight_delta"] for update in updates]
            num_updates = len(deltas)

            # Calculate how many to trim from each end
            trim_count = int(num_updates * self.trim_ratio)
            logger.info(f"[{self.node_id}] Trimming {trim_count} updates from each end ({num_updates} total)")

            # If we don't have enough updates to trim, fall back to regular mean
            if trim_count * 2 >= num_updates:
                logger.warning(
                    f"[{self.node_id}] Not enough updates for trimming "
                    f"({num_updates} updates, need >{trim_count*2}). Using regular mean."
                )
                trim_count = 0

            # Aggregate each parameter separately
            aggregated_delta = {}

            # Get all parameter keys from first delta
            param_keys = list(deltas[0].keys())

            for key in param_keys:
                # Collect all values for this parameter across all updates
                # Convert to numpy arrays for easier manipulation
                param_values = []
                for delta in deltas:
                    value = delta[key]
                    if isinstance(value, list):
                        param_values.append(np.array(value))
                    elif isinstance(value, np.ndarray):
                        param_values.append(value)
                    else:
                        param_values.append(np.array(value))

                # Stack along new axis (axis 0 = update dimension)
                # Shape: (num_updates, *param_shape)
                stacked = np.stack(param_values, axis=0)

                if trim_count > 0:
                    # Sort along update dimension (axis 0)
                    sorted_values = np.sort(stacked, axis=0)

                    # Trim top and bottom
                    trimmed_values = sorted_values[trim_count:-trim_count]

                    # Compute mean of trimmed values
                    aggregated_value = np.mean(trimmed_values, axis=0)
                else:
                    # No trimming, just regular mean
                    aggregated_value = np.mean(stacked, axis=0)

                # Store as list for JSON serialization
                aggregated_delta[key] = aggregated_value.tolist()

            # Build result
            aggregated = {
                "aggregated_weight_delta": aggregated_delta,
                "aggregation_method": "trimmed_mean",
                "trim_ratio": self.trim_ratio,
                "num_updates": num_updates,
                "num_trimmed_per_end": trim_count,
                "num_used_for_mean": num_updates - (2 * trim_count),
                "real_training": True,
            }

            # Aggregate metrics (using trimmed mean on metrics too)
            losses = [u.get("metrics", {}).get("loss", 0) for u in updates]
            accuracies = [u.get("metrics", {}).get("accuracy", 0) for u in updates]
            sample_counts = [u.get("metrics", {}).get("num_samples", 0) for u in updates]

            # Trimmed mean for metrics
            if trim_count > 0 and len(losses) > trim_count * 2:
                losses_sorted = sorted(losses)
                accuracies_sorted = sorted(accuracies)
                avg_loss = np.mean(losses_sorted[trim_count:-trim_count])
                avg_accuracy = np.mean(accuracies_sorted[trim_count:-trim_count])
            else:
                avg_loss = np.mean(losses)
                avg_accuracy = np.mean(accuracies)

            aggregated["aggregated_metrics"] = {
                "avg_loss": float(avg_loss),
                "avg_accuracy": float(avg_accuracy),
                "total_samples": sum(sample_counts),
            }

            logger.info(
                f"[{self.node_id}] Trimmed Mean complete: "
                f"avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.4f}, "
                f"used {num_updates - (2 * trim_count)}/{num_updates} updates"
            )

            return aggregated

        except Exception as e:
            logger.error(f"[{self.node_id}] Error in trimmed mean aggregation: {e}", exc_info=True)
            # Fallback to mock aggregation
            return self._trimmed_mean_mock(updates)

    def _trimmed_mean_mock(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback trimmed mean aggregation for mock training results.

        Args:
            updates: List of update dictionaries

        Returns:
            Aggregated model dictionary
        """
        import numpy as np

        aggregated = {}
        num_updates = len(updates)
        trim_count = int(num_updates * self.trim_ratio)

        # If we don't have enough updates to trim, fall back to regular mean
        if trim_count * 2 >= num_updates:
            logger.warning(
                f"[{self.node_id}] Not enough updates for trimming "
                f"({num_updates} updates). Using regular mean."
            )
            trim_count = 0

        # Collect all keys across all updates
        all_keys = set()
        for update in updates:
            all_keys.update(update.keys())

        # For each key, aggregate values using trimmed mean
        for key in all_keys:
            if key in ["metadata", "weight_delta"]:
                continue  # Skip metadata and weight_delta

            values = [update.get(key) for update in updates if key in update]

            # Try to apply trimmed mean to numeric values
            try:
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values and len(numeric_values) > trim_count * 2:
                    # Sort and trim
                    sorted_values = sorted(numeric_values)
                    if trim_count > 0:
                        trimmed_values = sorted_values[trim_count:-trim_count]
                    else:
                        trimmed_values = sorted_values
                    aggregated[key] = float(np.mean(trimmed_values))
                elif numeric_values:
                    # Not enough to trim, use regular mean
                    aggregated[key] = float(np.mean(numeric_values))
                else:
                    # For non-numeric, take the first value
                    aggregated[key] = values[0] if values else None
            except Exception:
                # Fallback: take first value
                aggregated[key] = values[0] if values else None

        # Add aggregation-specific fields
        aggregated["aggregation_method"] = "trimmed_mean_mock"
        aggregated["trim_ratio"] = self.trim_ratio
        aggregated["num_updates"] = num_updates
        aggregated["num_trimmed_per_end"] = trim_count
        aggregated["num_used_for_mean"] = num_updates - (2 * trim_count)
        aggregated["real_training"] = False

        logger.debug(f"[{self.node_id}] Aggregated model keys: {list(aggregated.keys())}")

        return aggregated

    def _numpy_to_python(self, obj: Any) -> Any:
        """
        Convert numpy arrays and torch tensors to Python lists for JSON serialization.

        Args:
            obj: Object to convert (can be dict, list, numpy array, tensor, or primitive)

        Returns:
            Object with all numpy arrays/tensors converted to lists
        """
        import numpy as np
        import torch

        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._numpy_to_python(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_python(v) for v in obj)
        else:
            return obj

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
