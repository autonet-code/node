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
from ..common.blob_store import BlobStore
from ..common.config import AutonetConfig, load_config
from ..common.governance import GovernanceBridge
from ..common.guild_manager import GuildManager, GuildInfo

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
    guild_aggregations_done: int = 0
    network_aggregations_done: int = 0
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0


@dataclass
class RPBAggregationState:
    """State tracking for an RPB."""
    rpb_address: str
    collected_updates: List[str] = field(default_factory=list)
    aggregation_rounds: int = 0
    last_model_cid: Optional[str] = None
    # Guild-level tracking: guild_id -> list of update CIDs from that guild
    guild_updates: Dict[int, List[str]] = field(default_factory=dict)
    # Network-level tracking: guild_id -> guild-level aggregated CID
    guild_aggregated_cids: Dict[int, str] = field(default_factory=dict)
    # Mapping: update CID -> solver address (for guild membership filtering)
    update_solvers: Dict[str, str] = field(default_factory=dict)


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

    def __init__(
        self,
        registry: ContractRegistry,
        store: BlobStore,
        node_id: str,
        rpb_address: str = "",
        aggregation_method: str = "fedavg",
        trim_ratio: float = 0.2,
        task_mode: str = "ground_truth",
        config: Optional[AutonetConfig] = None,
    ):
        """
        Initialize the aggregator node.

        Args:
            registry: ContractRegistry for blockchain calls
            store: BlobStore for content-addressed storage
            node_id: Unique identifier for this node (e.g., "aggregator-0")
            rpb_address: RPB contract address for this jurisdiction
            aggregation_method: Aggregation method to use ("fedavg" or "trimmed_mean")
            trim_ratio: Ratio to trim from top/bottom for trimmed_mean (default: 0.2 = 20%)
            task_mode: "ground_truth" (legacy) or "consensus_truth" (MM-Zero)
            config: AutonetConfig (loaded from yaml/env if not provided)
        """
        self.registry = registry
        self.store = store
        self.node_id = node_id
        self.rpb_address = rpb_address
        self.config = config or load_config()
        self.aggregation_method = self.config.node.aggregation_method
        self.trim_ratio = self.config.node.trim_ratio
        self.task_mode = task_mode

        self.stake_amount = self.config.staking.aggregator * 10**18
        self.min_updates = self.config.node.min_updates_for_aggregation

        self.metrics = AggregatorMetrics()
        self.project_state = RPBAggregationState(rpb_address=rpb_address)
        self.is_staked = False
        self.should_stop = False

        self.my_address = self.registry.blockchain.account.address

        # Governance bridge for attestation and heartbeat
        self.governance = GovernanceBridge(registry, node_id)

        # Guild manager for guild-aware aggregation
        self.guild_manager = GuildManager(registry, node_id, self.config)
        self.aggregation_level = self.config.guild.aggregation_level

        logger.info(f"[{self.node_id}] Initialized with address {self.my_address[:10]}...")
        logger.info(f"[{self.node_id}] Aggregation method: {self.aggregation_method}")
        logger.info(f"[{self.node_id}] Aggregation level: {self.aggregation_level}")

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
        logger.info(f"[{self.node_id}] Starting aggregator node for project {self.rpb_address}")
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
        # Poll for autonomous training deltas
        self._poll_autonomous_training()

        # Step 3: Aggregate based on configured level
        if self.aggregation_level == "guild":
            self._cycle_guild_aggregation()
        elif self.aggregation_level == "network":
            self._cycle_network_aggregation()
        else:
            # Flat (legacy) aggregation
            if len(self.project_state.collected_updates) >= self.min_updates:
                self._aggregate_and_publish()

    def _stake(self):
        """Stake as AGGREGATOR role."""
        logger.info(f"[{self.node_id}] Staking {self.stake_amount / 10**18} ATN as AGGREGATOR")

        # First approve ATN spending
        staking_contract = self.registry.get("ParticipantStaking")
        if not staking_contract:
            logger.error(f"[{self.node_id}] ParticipantStaking contract not found")
            self.metrics.errors += 1
            return

        approve_result = self.registry.approve_atn(
            staking_contract.address,
            self.stake_amount
        )
        if not approve_result.success:
            logger.error(f"[{self.node_id}] ATN approval failed: {approve_result.error}")
            self.metrics.errors += 1
            return

        logger.info(f"[{self.node_id}] ATN approved for staking")

        # Stake
        stake_result = self.registry.stake(self.AGGREGATOR_ROLE, self.stake_amount)
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
                    self._collect_update(task_id, solution_cid, solver=recipient)
                else:
                    logger.warning(
                        f"[{self.node_id}] No solution CID found for task {task_id}, solver {recipient[:10]}..."
                    )

        except Exception as e:
            logger.error(f"[{self.node_id}] Error polling events: {e}", exc_info=True)
            self.metrics.errors += 1

    def _poll_autonomous_training(self):
        """Poll for TrainingRecorded events from the RPB contract.

        Autonomous training nodes upload deltas to blob store and record
        contributions on-chain. The delta CID is discovered via P2P gossip
        (each node advertises its latest_delta_cid in ModelState).

        For each TrainingRecorded event, we look up the agent's delta CID
        from peer gossip and collect it for aggregation.
        """
        try:
            events = self.registry.get_new_events("RPB", "TrainingRecorded")
            if not events:
                return

            logger.info(f"[{self.node_id}] Found {len(events)} TrainingRecorded events")

            for event in events:
                agent = event["args"]["agent"]
                contribution = event["args"]["contribution"]

                logger.info(
                    f"[{self.node_id}] TrainingRecorded: agent={agent[:10]}..., "
                    f"contribution={contribution}"
                )

                # Look up this agent's latest delta CID from peer gossip
                delta_cid = self._find_delta_cid_for_agent(agent)
                if delta_cid:
                    self._collect_update(
                        task_id=0,  # autonomous (no task)
                        update_cid=delta_cid,
                        solver=agent,
                    )
                else:
                    logger.debug(
                        f"[{self.node_id}] No delta CID found for agent {agent[:10]}... "
                        "(peer may not be connected or gossip not yet received)"
                    )

        except Exception as e:
            logger.debug(f"[{self.node_id}] Error polling TrainingRecorded: {e}")

    def _find_delta_cid_for_agent(self, agent_address: str) -> Optional[str]:
        """Find the latest delta CID for an agent from P2P peer gossip.

        Each training node advertises its latest_delta_cid in its ModelState
        via the capability gossip protocol. We scan known peers for a match.
        """
        p2p_host = getattr(self, "_p2p_host", None)
        if not p2p_host:
            return None

        try:
            for peer_id, capability in p2p_host.get_peer_capabilities().items():
                model_state = capability.get("model_state", {})
                delta_cid = model_state.get("latest_delta_cid", "")
                if delta_cid:
                    # In the current model, we collect any advertised delta.
                    # Future: match peer's wallet address to agent_address.
                    return delta_cid
        except Exception as e:
            logger.debug(f"[{self.node_id}] Error scanning peer capabilities: {e}")

        return None

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

    def _collect_update(self, task_id: int, update_cid: str, solver: Optional[str] = None):
        """Collect a verified update CID."""
        if update_cid in self.project_state.collected_updates:
            logger.debug(f"[{self.node_id}] Update {update_cid[:20]}... already collected")
            return

        self.project_state.collected_updates.append(update_cid)

        # Track solver address for guild-level filtering
        if solver:
            self.project_state.update_solvers[update_cid] = solver

        logger.info(
            f"[{self.node_id}] Collected update {update_cid[:20]}... "
            f"(total: {len(self.project_state.collected_updates)})"
        )

    def _aggregate_and_publish(self):
        """
        Aggregate collected updates and publish the new model.

        Steps:
        1. Download all update CIDs from blob store
        2. Perform FedAvg aggregation
        3. Upload aggregated model to blob store
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
                update_data = self.store.get_json(cid)
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
            "rpb_address": self.rpb_address,
            "aggregation_round": self.project_state.aggregation_rounds + 1,
            "updates_count": len(updates),
            "aggregator": self.my_address,
            "timestamp": int(time.time()),
        }

        # If we have real aggregated deltas, apply them to the current global model
        # to produce a new full model checkpoint (not just the delta)
        if aggregated_model.get("real_training") and "aggregated_weight_delta" in aggregated_model:
            try:
                aggregated_model = self._apply_delta_to_global(aggregated_model)
            except Exception as e:
                logger.warning(f"[{self.node_id}] Could not apply delta to global model: {e}")

        # Convert numpy arrays/tensors to lists for JSON serialization
        aggregated_model = self._numpy_to_python(aggregated_model)

        # Upload to blob store
        try:
            new_model_cid = self.store.add_json(aggregated_model)
            if not new_model_cid:
                logger.error(f"[{self.node_id}] Failed to upload aggregated model")
                self.metrics.errors += 1
                return

            logger.info(f"[{self.node_id}] Aggregated model uploaded: {new_model_cid[:20]}...")
        except Exception as e:
            logger.error(f"[{self.node_id}] Error uploading model: {e}")
            self.metrics.errors += 1
            return

        # Publish on-chain
        result = self.registry.set_mature_model(
            new_model_cid,
            price=0  # Free model for now
        )

        if result.success:
            logger.info(
                f"[{self.node_id}] Published mature model for project {self.rpb_address}: "
                f"{new_model_cid[:20]}..."
            )
            self.metrics.aggregations_done += 1
            self.project_state.aggregation_rounds += 1
            self.project_state.last_model_cid = new_model_cid
            self.project_state.collected_updates.clear()

            # Attest model aggregation work to the economic layer
            self.governance.attest_task_completion(units=len(updates))

            # Claim rewards for previous epoch + accrue reputation
            self.governance.claim_epoch_rewards()
            self.governance.claim_reputation()
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

            # Extract weight deltas, sample counts, and alignment scores.
            # Weight = num_samples * alignment_score so higher-alignment
            # nodes have proportionally more influence on the merged model.
            deltas = []
            weights = []
            for update in updates:
                deltas.append(update["weight_delta"])
                num_samples = update.get("metrics", {}).get("num_samples", 1)
                alignment = update.get("metrics", {}).get("alignment_score", 1.0)
                # Clamp alignment to [0.01, 1.0] — never zero (would discard entirely)
                alignment = max(0.01, min(1.0, alignment))
                weights.append(num_samples * alignment)

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

    def _apply_delta_to_global(self, aggregated_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the aggregated weight delta to the current global model weights.

        Loads the current mature model from blob store, adds the delta,
        and stores the resulting full weights in the aggregated_model dict.
        This way the published model CID contains a complete, loadable model.
        """
        from ..common.ml import apply_weight_delta, load_weights, save_weights
        import torch

        delta = aggregated_model["aggregated_weight_delta"]

        # Load current global model
        global_model_cid = None
        try:
            global_model_cid = self.registry.get_mature_model()
        except Exception:
            pass

        if global_model_cid:
            base_data = self.store.get_json(global_model_cid)
            if base_data and "weights" in base_data:
                base_weights = {k: torch.tensor(v) for k, v in base_data["weights"].items()}
            else:
                base_weights = None
        else:
            base_weights = None

        if base_weights:
            # Convert delta values to tensors
            delta_tensors = {}
            for k, v in delta.items():
                delta_tensors[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v

            # Apply delta: new_weights = base_weights + delta
            new_weights = apply_weight_delta(base_weights, delta_tensors)
            logger.info(f"[{self.node_id}] Applied delta to global model ({len(new_weights)} params)")
        else:
            # No base model — the delta IS the model weights (first round)
            new_weights = {}
            for k, v in delta.items():
                new_weights[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            logger.info(f"[{self.node_id}] No base model, using delta as initial weights")

        # Store the full weights in the model for publication
        aggregated_model["weights"] = {k: v.tolist() for k, v in new_weights.items()}
        aggregated_model["format"] = "pytorch_state_dict"

        return aggregated_model

    # =========================================================================
    # Story 8.2: Guild-Level Aggregation
    # =========================================================================

    def _cycle_guild_aggregation(self):
        """
        Guild-level aggregation cycle.

        Only aggregates updates from solvers that are members of this node's guild.
        Produces a guild-level model update that can be consumed by a network
        aggregator (Story 8.3).
        """
        guild_id = self.guild_manager.guild_id
        if guild_id is None:
            logger.warning(f"[{self.node_id}] Guild aggregation configured but no guild_id set")
            # Fall back to flat aggregation
            if len(self.project_state.collected_updates) >= self.min_updates:
                self._aggregate_and_publish()
            return

        # Filter updates to only those from guild members
        guild_updates = []
        for cid in self.project_state.collected_updates:
            solver = self.project_state.update_solvers.get(cid)
            if solver and self.guild_manager.is_member_of_guild(solver, guild_id):
                guild_updates.append(cid)

        min_guild_updates = self.config.guild.min_guild_updates
        if len(guild_updates) < min_guild_updates:
            return

        logger.info(
            f"[{self.node_id}] Guild aggregation: {len(guild_updates)} updates "
            f"from guild {guild_id} (of {len(self.project_state.collected_updates)} total)"
        )

        # Download guild member updates
        updates = []
        for cid in guild_updates:
            try:
                update_data = self.store.get_json(cid)
                if update_data:
                    updates.append(update_data)
            except Exception as e:
                logger.error(f"[{self.node_id}] Error downloading {cid}: {e}")

        if not updates:
            return

        # Aggregate using configured method
        if self.aggregation_method == "trimmed_mean":
            aggregated = self._trimmed_mean_aggregate(updates)
        else:
            aggregated = self._fedavg(updates)

        # Add guild-level metadata
        aggregated["metadata"] = {
            "rpb_address": self.rpb_address,
            "guild_id": guild_id,
            "aggregation_level": "guild",
            "aggregation_round": self.project_state.aggregation_rounds + 1,
            "updates_count": len(updates),
            "aggregator": self.my_address,
            "timestamp": int(time.time()),
        }

        # Convert for serialization
        aggregated = self._numpy_to_python(aggregated)

        # Upload guild-level aggregate
        try:
            guild_cid = self.store.add_json(aggregated)
            if not guild_cid:
                logger.error(f"[{self.node_id}] Failed to upload guild aggregate")
                self.metrics.errors += 1
                return

            logger.info(
                f"[{self.node_id}] Guild {guild_id} aggregate uploaded: {guild_cid[:20]}..."
            )

            # Track this guild's aggregate for network-level consumption
            self.project_state.guild_aggregated_cids[guild_id] = guild_cid

            # Remove processed updates from collected pool
            for cid in guild_updates:
                if cid in self.project_state.collected_updates:
                    self.project_state.collected_updates.remove(cid)
                self.project_state.update_solvers.pop(cid, None)

            self.metrics.guild_aggregations_done += 1
            self.metrics.aggregations_done += 1
            self.project_state.aggregation_rounds += 1

            # Attest work
            self.governance.attest_task_completion(units=len(updates))

            # Report guild metrics if we have improvement data
            if aggregated.get("aggregated_metrics"):
                avg_accuracy = aggregated["aggregated_metrics"].get("avg_accuracy", 0)
                # Convert accuracy to basis points (0-10000)
                improvement_bps = int(min(avg_accuracy * 10000, 10000))
                for mod_id in (self.guild_manager.get_guild(guild_id) or GuildInfo(0, "", "", "", [], 0, 0, True)).module_ids:
                    self.guild_manager.report_guild_metrics(mod_id, improvement_bps)

        except Exception as e:
            logger.error(f"[{self.node_id}] Guild aggregation error: {e}", exc_info=True)
            self.metrics.errors += 1

    # =========================================================================
    # Story 8.3: Network-Level Aggregation Across Guilds
    # =========================================================================

    def _cycle_network_aggregation(self):
        """
        Network-level aggregation cycle.

        Collects guild-level aggregated updates and combines them using
        reputation-weighted averaging. The network aggregator does NOT see
        individual solver updates — only guild-level aggregates.
        """
        guild_cids = self.project_state.guild_aggregated_cids

        # Also check flat collected_updates — in case guilds aren't fully
        # set up yet, we can still aggregate what we have
        if not guild_cids and len(self.project_state.collected_updates) >= self.min_updates:
            # Fallback: flat aggregation when no guild aggregates available
            self._aggregate_and_publish()
            return

        if len(guild_cids) < 1:
            return

        logger.info(
            f"[{self.node_id}] Network aggregation: {len(guild_cids)} guild aggregates"
        )

        # Get guild weights from GuildManager
        guild_weights = self.guild_manager.get_guild_weights_for_aggregation()

        # Download guild aggregates
        guild_updates = []
        weights = []
        guild_ids_processed = []

        for gid, cid in guild_cids.items():
            try:
                data = self.store.get_json(cid)
                if data:
                    guild_updates.append(data)
                    weights.append(guild_weights.get(gid, 1.0 / len(guild_cids)))
                    guild_ids_processed.append(gid)
            except Exception as e:
                logger.error(f"[{self.node_id}] Error downloading guild {gid} aggregate: {e}")

        if not guild_updates:
            return

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Perform weighted aggregation across guild updates
        aggregated = self._weighted_guild_aggregate(guild_updates, weights)

        aggregated["metadata"] = {
            "rpb_address": self.rpb_address,
            "aggregation_level": "network",
            "aggregation_round": self.project_state.aggregation_rounds + 1,
            "guild_count": len(guild_updates),
            "guild_ids": guild_ids_processed,
            "guild_weights": {str(gid): w for gid, w in zip(guild_ids_processed, weights)},
            "aggregator": self.my_address,
            "timestamp": int(time.time()),
        }

        # Apply to global model if real training
        if aggregated.get("real_training") and "aggregated_weight_delta" in aggregated:
            try:
                aggregated = self._apply_delta_to_global(aggregated)
            except Exception as e:
                logger.warning(f"[{self.node_id}] Could not apply delta to global model: {e}")

        aggregated = self._numpy_to_python(aggregated)

        # Upload and publish
        try:
            new_model_cid = self.store.add_json(aggregated)
            if not new_model_cid:
                logger.error(f"[{self.node_id}] Failed to upload network aggregate")
                self.metrics.errors += 1
                return

            logger.info(f"[{self.node_id}] Network aggregate uploaded: {new_model_cid[:20]}...")

            # Publish on-chain
            result = self.registry.set_mature_model(
                new_model_cid,
                price=0,
            )

            if result.success:
                logger.info(
                    f"[{self.node_id}] Published network-aggregated model: {new_model_cid[:20]}..."
                )
                self.metrics.network_aggregations_done += 1
                self.metrics.aggregations_done += 1
                self.project_state.aggregation_rounds += 1
                self.project_state.last_model_cid = new_model_cid
                self.project_state.guild_aggregated_cids.clear()

                self.governance.attest_task_completion(units=len(guild_updates))
                self.governance.claim_epoch_rewards()
                self.governance.claim_reputation()
            else:
                logger.error(f"[{self.node_id}] Failed to publish model: {result.error}")
                self.metrics.errors += 1

        except Exception as e:
            logger.error(f"[{self.node_id}] Network aggregation error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _weighted_guild_aggregate(
        self, guild_updates: List[Dict[str, Any]], weights: List[float]
    ) -> Dict[str, Any]:
        """
        Weighted aggregation of guild-level updates.

        Each guild's contribution is weighted by its reputation and member count.
        For real weight deltas, this performs a weighted average of parameters.
        For mock updates, averages numeric values with weights.
        """
        has_real_deltas = all(
            "aggregated_weight_delta" in u or "weight_delta" in u
            for u in guild_updates
        )

        if has_real_deltas:
            return self._weighted_guild_aggregate_real(guild_updates, weights)
        else:
            return self._weighted_guild_aggregate_mock(guild_updates, weights)

    def _weighted_guild_aggregate_real(
        self, guild_updates: List[Dict[str, Any]], weights: List[float]
    ) -> Dict[str, Any]:
        """Weighted aggregation of real weight deltas from guilds."""
        try:
            import numpy as np

            # Extract deltas (guild aggregates use "aggregated_weight_delta")
            deltas = []
            for update in guild_updates:
                delta = update.get("aggregated_weight_delta") or update.get("weight_delta")
                deltas.append(delta)

            # Get parameter keys from first delta
            param_keys = list(deltas[0].keys())
            aggregated_delta = {}

            for key in param_keys:
                param_values = []
                for delta in deltas:
                    value = delta[key]
                    if isinstance(value, list):
                        param_values.append(np.array(value))
                    elif isinstance(value, np.ndarray):
                        param_values.append(value)
                    else:
                        param_values.append(np.array(value))

                # Weighted average
                result = np.zeros_like(param_values[0], dtype=np.float64)
                for val, w in zip(param_values, weights):
                    result += val * w
                aggregated_delta[key] = result.tolist()

            return {
                "aggregated_weight_delta": aggregated_delta,
                "aggregation_method": f"{self.aggregation_method}_guild_weighted",
                "num_updates": len(guild_updates),
                "real_training": True,
            }

        except Exception as e:
            logger.error(f"[{self.node_id}] Weighted real aggregation error: {e}", exc_info=True)
            return self._weighted_guild_aggregate_mock(guild_updates, weights)

    def _weighted_guild_aggregate_mock(
        self, guild_updates: List[Dict[str, Any]], weights: List[float]
    ) -> Dict[str, Any]:
        """Weighted aggregation of mock guild updates."""
        aggregated = {}
        all_keys = set()
        for update in guild_updates:
            all_keys.update(update.keys())

        for key in all_keys:
            if key in ("metadata", "weight_delta", "aggregated_weight_delta"):
                continue

            values = []
            value_weights = []
            for update, w in zip(guild_updates, weights):
                if key in update and isinstance(update[key], (int, float)):
                    values.append(update[key])
                    value_weights.append(w)

            if values:
                total_w = sum(value_weights)
                if total_w > 0:
                    aggregated[key] = sum(v * w for v, w in zip(values, value_weights)) / total_w
                else:
                    aggregated[key] = sum(values) / len(values)
            else:
                # Non-numeric: take first value
                for update in guild_updates:
                    if key in update:
                        aggregated[key] = update[key]
                        break

        aggregated["aggregation_method"] = f"{self.aggregation_method}_guild_weighted_mock"
        aggregated["num_updates"] = len(guild_updates)
        aggregated["real_training"] = False
        return aggregated

    def _log_final_metrics(self):
        """Log final metrics at shutdown."""
        logger.info(f"[{self.node_id}] Final metrics:")
        logger.info(f"  Cycles: {self.metrics.cycles}")
        logger.info(f"  Aggregations: {self.metrics.aggregations_done}")
        logger.info(f"  Guild aggregations: {self.metrics.guild_aggregations_done}")
        logger.info(f"  Network aggregations: {self.metrics.network_aggregations_done}")
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
    store = BlobStore()

    # Create aggregator node
    node = AggregatorNode(
        registry=registry,
        store=store,
        node_id="aggregator-demo",
        rpb_address="",
    )

    # Run for a few cycles
    try:
        node.run(max_cycles=10, cycle_delay=5.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        node.stop()


if __name__ == "__main__":
    main()
