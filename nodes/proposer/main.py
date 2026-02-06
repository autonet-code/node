"""
Autonomous Proposer Node

Proposes training tasks with ground truth, reveals ground truth after solutions are committed.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from web3 import Web3

from ..common.contracts import ContractRegistry
from ..common.ipfs import IPFSClient


@dataclass
class ProposerMetrics:
    """Metrics tracked by the proposer node."""
    tasks_proposed: int = 0
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0
    aggregations_done: int = 0
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0


class ProposerNode:
    """
    Autonomous proposer node that:
    1. Stakes as PROPOSER role
    2. Creates and proposes training tasks
    3. Reveals ground truth after solutions are committed
    """

    def __init__(
        self,
        registry: ContractRegistry,
        ipfs: IPFSClient,
        node_id: str,
        project_id: int,
    ):
        """
        Initialize the proposer node.

        Args:
            registry: ContractRegistry instance for blockchain interactions
            ipfs: IPFSClient instance for IPFS operations
            node_id: Unique identifier for this node (e.g., "proposer-0")
            project_id: Project ID to propose tasks for
        """
        self.registry = registry
        self.ipfs = ipfs
        self.node_id = node_id
        self.project_id = project_id
        self.metrics = ProposerMetrics()
        self.running = False
        self.staked = False
        self.logger = logging.getLogger(f"ProposerNode[{node_id}]")

        # Track proposed tasks (task_id -> ground_truth_cid)
        self.proposed_tasks = {}

        # Track tasks where we've revealed ground truth
        self.revealed_tasks = set()

    def stop(self):
        """Stop the node."""
        self.running = False
        self.logger.info(f"Node {self.node_id} stopped")

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 5.0):
        """
        Main execution loop.

        Args:
            max_cycles: Maximum number of cycles to run (None = infinite)
            cycle_delay: Delay between cycles in seconds
        """
        self.running = True
        self.logger.info(f"Starting ProposerNode {self.node_id} for project {self.project_id}")

        cycle_count = 0
        while self.running:
            try:
                # Check if we should stop
                if max_cycles is not None and cycle_count >= max_cycles:
                    self.logger.info(f"Reached max cycles ({max_cycles}), stopping")
                    break

                # Perform stake on first cycle
                if not self.staked:
                    self._perform_stake()

                # Main work: propose task and reveal ground truth
                self._propose_task_cycle()
                self._reveal_ground_truth_cycle()

                # Increment cycle counter
                cycle_count += 1
                self.metrics.cycles += 1

                # Sleep between cycles
                if self.running:
                    time.sleep(cycle_delay)

            except KeyboardInterrupt:
                self.logger.info("Received interrupt, stopping...")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in cycle {cycle_count}: {e}", exc_info=True)
                self.metrics.errors += 1
                # Continue running despite errors
                time.sleep(cycle_delay)

        self.logger.info(f"ProposerNode {self.node_id} finished. Metrics: {self.metrics}")

    def _perform_stake(self):
        """Approve ATN and stake as PROPOSER role."""
        try:
            # PROPOSER role = 1, stake amount = 100 ATN
            role = 1
            stake_amount = 100 * 10**18  # 100 ATN in wei

            # Get staking contract address
            staking_contract = self.registry.get("ParticipantStaking")
            staking_address = staking_contract.address

            self.logger.info(f"Approving {stake_amount} ATN for staking contract {staking_address}")

            # Approve ATN spending
            approve_result = self.registry.approve_atn(staking_address, stake_amount)
            if not approve_result.success:
                self.logger.error(f"Failed to approve ATN: {approve_result.error}")
                self.metrics.errors += 1
                return

            self.logger.info(f"Approval successful: {approve_result.tx_hash}")

            # Stake
            self.logger.info(f"Staking {stake_amount} wei as PROPOSER (role={role})")
            stake_result = self.registry.stake(role, stake_amount)

            if stake_result.success:
                self.logger.info(f"Staking successful: {stake_result.tx_hash}")
                self.staked = True
            else:
                self.logger.error(f"Staking failed: {stake_result.error}")
                self.metrics.errors += 1

        except Exception as e:
            self.logger.error(f"Error during staking: {e}", exc_info=True)
            self.metrics.errors += 1

    def _propose_task_cycle(self):
        """Generate and propose a new training task."""
        try:
            # Generate task data
            task_spec = self._generate_task_spec()
            ground_truth = self._generate_ground_truth()

            # Upload to IPFS
            self.logger.info("Uploading task spec to IPFS...")
            spec_cid = self.ipfs.add_json(task_spec)
            self.logger.info(f"Task spec CID: {spec_cid}")

            self.logger.info("Uploading ground truth to IPFS...")
            ground_truth_cid = self.ipfs.add_json(ground_truth)
            self.logger.info(f"Ground truth CID: {ground_truth_cid}")

            # Hash both CIDs to bytes32 (contracts expect bytes32)
            spec_hash = Web3.keccak(text=spec_cid)
            ground_truth_hash = Web3.keccak(text=ground_truth_cid)
            self.logger.info(f"Spec hash: {spec_hash.hex()[:16]}..., Ground truth hash: {ground_truth_hash.hex()[:16]}...")

            # Propose task on-chain
            learnability_reward = 10 * 10**18  # 10 ATN
            solver_reward = 5 * 10**18         # 5 ATN

            self.logger.info(
                f"Proposing task for project {self.project_id} "
                f"(spec={spec_cid[:16]}..., rewards={learnability_reward}/{solver_reward})"
            )

            result = self.registry.propose_task(
                self.project_id,
                spec_hash,
                ground_truth_hash,
                learnability_reward,
                solver_reward
            )

            if result.success:
                # Get the task ID from the next_task_id
                task_id = self.registry.get_next_task_id() - 1
                self.logger.info(f"Task proposed successfully! Task ID: {task_id}, TX: {result.tx_hash}")

                # Store task info
                self.proposed_tasks[task_id] = ground_truth_cid
                self.metrics.tasks_proposed += 1
            else:
                self.logger.error(f"Failed to propose task: {result.error}")
                self.metrics.errors += 1

        except Exception as e:
            self.logger.error(f"Error proposing task: {e}", exc_info=True)
            self.metrics.errors += 1

    def _reveal_ground_truth_cycle(self):
        """Check for committed solutions and reveal ground truth."""
        try:
            # Get new SolutionCommitted events (event is on TaskContract)
            events = self.registry.get_new_events("TaskContract", "SolutionCommitted")

            for event in events:
                task_id = event['args']['taskId']
                solver = event['args']['solver']

                self.logger.info(f"Solution committed for task {task_id} by {solver}")
                self.metrics.solutions_committed += 1

                # Reveal ground truth if we proposed this task and haven't revealed yet
                if task_id in self.proposed_tasks and task_id not in self.revealed_tasks:
                    ground_truth_cid = self.proposed_tasks[task_id]

                    self.logger.info(f"Revealing ground truth for task {task_id}: {ground_truth_cid}")
                    result = self.registry.reveal_ground_truth(task_id, ground_truth_cid)

                    if result.success:
                        self.logger.info(f"Ground truth revealed! TX: {result.tx_hash}")
                        self.revealed_tasks.add(task_id)
                        self.metrics.tasks_completed += 1
                    else:
                        self.logger.error(f"Failed to reveal ground truth: {result.error}")
                        self.metrics.errors += 1

        except Exception as e:
            self.logger.error(f"Error revealing ground truth: {e}", exc_info=True)
            self.metrics.errors += 1

    def _generate_task_spec(self) -> dict:
        """Generate a task specification."""
        # Simple MNIST-like task
        return {
            "type": "image_classification",
            "dataset": "mnist_subset",
            "batch_size": 32,
            "num_samples": 1000,
            "model_architecture": "simple_cnn",
            "timestamp": int(time.time()),
            "proposer": self.node_id,
        }

    def _generate_ground_truth(self) -> dict:
        """Generate ground truth data for the task."""
        # In a real implementation, this would be actual training data
        # For now, we generate a placeholder
        return {
            "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "num_samples": 1000,
            "accuracy_threshold": 0.85,
            "timestamp": int(time.time()),
            "proposer": self.node_id,
        }


def main():
    """Standalone entry point for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # This would need to be configured with actual registry and IPFS instances
    # For testing with orchestrator, this won't be called
    print("ProposerNode initialized. Use orchestrator to run.")


if __name__ == "__main__":
    main()
