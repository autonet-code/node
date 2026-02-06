"""
Autonomous Solver Node for Autonet

Discovers tasks from the blockchain, performs mock training with checkpoints,
commits solutions, and reveals them after ground truth is revealed.

Uses ContractRegistry for real blockchain interactions.
"""

import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State of a task from solver's perspective."""
    DISCOVERED = "discovered"
    TRAINING = "training"
    COMMITTED = "committed"
    REVEALED = "revealed"


@dataclass
class TrainingCheckpoint:
    """A checkpoint during training for Gensyn-style verification."""
    step_number: int
    weights_hash: str
    data_indices_hash: str
    random_seed: str
    timestamp: float


@dataclass
class TaskInfo:
    """Track task lifecycle from solver perspective."""
    task_id: int
    state: TaskState
    solution_cid: Optional[str] = None
    solution_hash: Optional[bytes] = None
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)
    discovered_at: float = field(default_factory=time.time)


@dataclass
class SolverMetrics:
    """Metrics for the solver node."""
    tasks_proposed: int = 0  # Not used by solver, but required by orchestrator
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0  # Not used by solver
    aggregations_done: int = 0  # Not used by solver
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0


class SolverNode:
    """
    Autonomous solver node that:
    1. Stakes as SOLVER on first cycle
    2. Discovers tasks from TaskProposed events
    3. Performs mock training with checkpoints
    4. Commits solutions to blockchain
    5. Reveals solutions after ground truth is revealed
    """

    SOLVER_ROLE = 2
    SOLVER_STAKE_AMOUNT = 50 * 10**18  # 50 ATN
    CHECKPOINT_FREQUENCY = 10
    TOTAL_TRAINING_STEPS = 20

    def __init__(
        self,
        registry,
        ipfs,
        node_id: str,
        project_id: int,
        deterministic_seed: Optional[int] = None,
    ):
        """
        Initialize SolverNode.

        Args:
            registry: ContractRegistry instance
            ipfs: IPFSClient instance
            node_id: Unique identifier for this node (e.g., "solver-0")
            project_id: Project ID to work on
            deterministic_seed: Seed for reproducible training
        """
        self.registry = registry
        self.ipfs = ipfs
        self.node_id = node_id
        self.project_id = project_id
        self.deterministic_seed = deterministic_seed or int(time.time())

        self.metrics = SolverMetrics()
        self.tasks: Dict[int, TaskInfo] = {}
        self.processed_task_ids: Set[int] = set()
        self.staked = False
        self.running = False

        logger.info(f"[{self.node_id}] Initialized solver node for project {project_id}")

    def run(self, max_cycles: int = 10, cycle_delay: float = 2.0):
        """
        Main event loop.

        Args:
            max_cycles: Maximum number of cycles to run
            cycle_delay: Delay between cycles in seconds
        """
        self.running = True
        logger.info(f"[{self.node_id}] Starting main loop: max_cycles={max_cycles}, delay={cycle_delay}s")

        for cycle in range(max_cycles):
            if not self.running:
                break

            try:
                logger.info(f"[{self.node_id}] === Cycle {cycle + 1}/{max_cycles} ===")

                # First cycle: stake as solver
                if cycle == 0 and not self.staked:
                    self._stake_as_solver()

                # Discover new tasks
                self._discover_tasks()

                # Process tasks: train, commit, reveal
                self._process_tasks()

                # Check for ground truth reveals and reveal our solutions
                self._check_and_reveal_solutions()

                self.metrics.cycles += 1

            except Exception as e:
                logger.error(f"[{self.node_id}] Error in cycle {cycle + 1}: {e}", exc_info=True)
                self.metrics.errors += 1

            # Sleep before next cycle
            if cycle < max_cycles - 1:
                time.sleep(cycle_delay)

        logger.info(f"[{self.node_id}] Main loop completed. Cycles: {self.metrics.cycles}")

    def stop(self):
        """Stop the node."""
        logger.info(f"[{self.node_id}] Stopping...")
        self.running = False

    def _stake_as_solver(self):
        """Approve and stake ATN as SOLVER."""
        logger.info(f"[{self.node_id}] Staking as SOLVER...")

        try:
            # Approve ParticipantStaking to spend ATN
            staking_contract = self.registry.get("ParticipantStaking")
            if not staking_contract:
                logger.error(f"[{self.node_id}] ParticipantStaking contract not found")
                return

            approve_result = self.registry.approve_atn(
                staking_contract.address,
                self.SOLVER_STAKE_AMOUNT
            )

            if not approve_result.success:
                logger.error(f"[{self.node_id}] Failed to approve ATN: {approve_result.error}")
                return

            logger.info(f"[{self.node_id}] ATN approved: {approve_result.tx_hash}")

            # Stake
            stake_result = self.registry.stake(self.SOLVER_ROLE, self.SOLVER_STAKE_AMOUNT)

            if stake_result.success:
                logger.info(f"[{self.node_id}] Staked {self.SOLVER_STAKE_AMOUNT // 10**18} ATN as SOLVER: {stake_result.tx_hash}")
                self.staked = True
            else:
                logger.error(f"[{self.node_id}] Failed to stake: {stake_result.error}")

        except Exception as e:
            logger.error(f"[{self.node_id}] Staking error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _discover_tasks(self):
        """Poll for TaskProposed events and discover new tasks."""
        try:
            events = self.registry.get_new_events("TaskContract", "TaskProposed")

            for event in events:
                task_id = event["args"]["taskId"]

                # Skip if already processed
                if task_id in self.processed_task_ids:
                    continue

                # Filter by project
                project_id = event["args"]["projectId"]
                if project_id != self.project_id:
                    continue

                logger.info(f"[{self.node_id}] Discovered task {task_id} for project {project_id}")

                # Create task info
                task_info = TaskInfo(
                    task_id=task_id,
                    state=TaskState.DISCOVERED,
                )
                self.tasks[task_id] = task_info
                self.processed_task_ids.add(task_id)

        except Exception as e:
            logger.error(f"[{self.node_id}] Error discovering tasks: {e}", exc_info=True)
            self.metrics.errors += 1

    def _process_tasks(self):
        """Process discovered tasks: train and commit solutions."""
        for task_id, task_info in list(self.tasks.items()):
            try:
                if task_info.state == TaskState.DISCOVERED:
                    self._train_and_commit(task_id, task_info)

            except Exception as e:
                logger.error(f"[{self.node_id}] Error processing task {task_id}: {e}", exc_info=True)
                self.metrics.errors += 1

    def _train_and_commit(self, task_id: int, task_info: TaskInfo):
        """Perform mock training and commit solution."""
        logger.info(f"[{self.node_id}] Training on task {task_id}...")
        task_info.state = TaskState.TRAINING

        # Perform mock training with checkpoints
        start_time = time.time()
        model_update, checkpoints = self._mock_train(task_id)
        training_time = time.time() - start_time

        logger.info(f"[{self.node_id}] Training completed in {training_time:.2f}s with {len(checkpoints)} checkpoints")

        # Upload solution to IPFS
        solution_cid = self.ipfs.add_json(model_update)
        if not solution_cid:
            logger.error(f"[{self.node_id}] Failed to upload solution to IPFS")
            return

        task_info.solution_cid = solution_cid
        task_info.checkpoints = checkpoints

        # Compute solution hash (keccak256)
        from web3 import Web3
        solution_hash = Web3.keccak(text=solution_cid)
        task_info.solution_hash = solution_hash

        logger.info(f"[{self.node_id}] Solution uploaded: {solution_cid}, hash: {solution_hash.hex()[:16]}...")

        # Commit solution hash to blockchain
        commit_result = self.registry.commit_solution(task_id, solution_hash)

        if commit_result.success:
            logger.info(f"[{self.node_id}] Solution committed for task {task_id}: {commit_result.tx_hash}")
            task_info.state = TaskState.COMMITTED
            self.metrics.solutions_committed += 1
            self.metrics.tasks_completed += 1

            # Submit a few checkpoints
            self._submit_checkpoints(task_id, checkpoints)
        else:
            logger.error(f"[{self.node_id}] Failed to commit solution: {commit_result.error}")

    def _mock_train(self, task_id: int) -> tuple:
        """
        Perform real ML training with checkpoint generation.

        Returns:
            (model_update_dict, checkpoints_list)
        """
        try:
            # Import ML module
            from ..common.ml import train_on_task

            # Fetch current global model (if any) for federated learning
            global_model_cid = self.registry.get_mature_model(self.project_id)
            if global_model_cid:
                logger.info(f"[{self.node_id}] Using global model: {global_model_cid[:20]}...")
            else:
                logger.info(f"[{self.node_id}] No global model yet, training from scratch")

            # Prepare task spec
            task_spec = {
                "task_id": task_id,
                "epochs": 1,  # Single epoch for fast training
            }

            # Perform real training (starting from global model if available)
            logger.info(f"[{self.node_id}] Starting real ML training for task {task_id}...")
            weight_delta, metrics = train_on_task(
                task_spec=task_spec,
                ipfs_client=self.ipfs,
                epochs=1,
                batch_size=32,
                learning_rate=0.01,
                num_samples=500,  # Small subset for fast training (~5 seconds)
                global_model_cid=global_model_cid,  # Continue from global model
            )

            # Generate checkpoints (mock for now, but based on real training)
            checkpoints = []
            training_steps = metrics.get("num_samples", 500) // 32  # batches
            current_seed = self._generate_deterministic_seed(task_id, 0)

            for step in range(0, training_steps, max(1, training_steps // 3)):
                checkpoint = self._create_checkpoint(step, current_seed)
                checkpoints.append(checkpoint)
                current_seed = self._generate_deterministic_seed(task_id, step + 1)

            # Final checkpoint
            final_checkpoint = self._create_checkpoint(training_steps, current_seed)
            checkpoints.append(final_checkpoint)

            # Build model update with real training results
            model_update = {
                "task_id": task_id,
                "weight_delta": weight_delta,  # Real weight updates for FedAvg
                "metrics": metrics,
                "training_steps": training_steps,
                "checkpoint_frequency": max(1, training_steps // 3),
                "final_seed": current_seed,
                "solver": self.node_id,
                "real_training": True,
            }

            logger.info(f"[{self.node_id}] Real training completed: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
            return model_update, checkpoints

        except Exception as e:
            logger.error(f"[{self.node_id}] Real training failed, falling back to mock: {e}")
            # Fallback to mock training
            return self._mock_train_fallback(task_id)

    def _mock_train_fallback(self, task_id: int) -> tuple:
        """
        Fallback mock training if real training fails.

        Returns:
            (model_update_dict, checkpoints_list)
        """
        checkpoints = []
        current_seed = self._generate_deterministic_seed(task_id, 0)

        for step in range(self.TOTAL_TRAINING_STEPS):
            # Fast mock training
            time.sleep(0.01)

            # Generate checkpoint at frequency
            if step > 0 and step % self.CHECKPOINT_FREQUENCY == 0:
                checkpoint = self._create_checkpoint(step, current_seed)
                checkpoints.append(checkpoint)
                logger.debug(f"[{self.node_id}] Checkpoint at step {step}")

            # Update seed deterministically
            current_seed = self._generate_deterministic_seed(task_id, step + 1)

        # Final checkpoint
        final_checkpoint = self._create_checkpoint(self.TOTAL_TRAINING_STEPS, current_seed)
        checkpoints.append(final_checkpoint)

        # Mock training result
        model_update = {
            "task_id": task_id,
            "model_weights": f"mock_weights_task_{task_id}",
            "metrics": {
                "loss": 0.15,
                "accuracy": 0.92,
            },
            "training_steps": self.TOTAL_TRAINING_STEPS,
            "checkpoint_frequency": self.CHECKPOINT_FREQUENCY,
            "final_seed": current_seed,
            "solver": self.node_id,
            "real_training": False,
        }

        return model_update, checkpoints

    def _create_checkpoint(self, step: int, seed: str) -> TrainingCheckpoint:
        """Create a training checkpoint with deterministic hashes."""
        weights_hash = hashlib.sha256(f"weights_step_{step}_{seed}".encode()).hexdigest()
        data_indices_hash = hashlib.sha256(f"data_indices_step_{step}_{seed}".encode()).hexdigest()

        return TrainingCheckpoint(
            step_number=step,
            weights_hash=weights_hash,
            data_indices_hash=data_indices_hash,
            random_seed=seed,
            timestamp=time.time(),
        )

    def _generate_deterministic_seed(self, task_id: int, step: int) -> str:
        """Generate a deterministic seed for reproducibility (full 32 bytes)."""
        seed_input = f"{self.deterministic_seed}:{task_id}:{step}"
        # Return full SHA256 hash (64 hex chars = 32 bytes when decoded)
        return hashlib.sha256(seed_input.encode()).hexdigest()

    def _submit_checkpoints(self, task_id: int, checkpoints: List[TrainingCheckpoint]):
        """Submit a few checkpoints to the blockchain."""
        # Submit first 3 checkpoints (or all if fewer)
        checkpoints_to_submit = checkpoints[:3]

        for checkpoint in checkpoints_to_submit:
            try:
                from web3 import Web3

                # Convert hashes to bytes32
                weights_hash = bytes.fromhex(checkpoint.weights_hash)
                data_indices_hash = bytes.fromhex(checkpoint.data_indices_hash)
                # Convert hex string to bytes (64 hex chars -> 32 bytes)
                random_seed = bytes.fromhex(checkpoint.random_seed)

                result = self.registry.submit_checkpoint(
                    task_id,
                    checkpoint.step_number,
                    weights_hash,
                    data_indices_hash,
                    random_seed,
                )

                if result.success:
                    logger.debug(f"[{self.node_id}] Submitted checkpoint step {checkpoint.step_number}: {result.tx_hash}")
                else:
                    logger.warning(f"[{self.node_id}] Failed to submit checkpoint: {result.error}")

            except Exception as e:
                logger.error(f"[{self.node_id}] Error submitting checkpoint: {e}", exc_info=True)

    def _check_and_reveal_solutions(self):
        """Check for GroundTruthRevealed events and reveal our solutions."""
        try:
            events = self.registry.get_new_events("ResultsRewards", "GroundTruthRevealed")

            for event in events:
                task_id = event["args"]["taskId"]

                # Skip if not our task or already revealed
                if task_id not in self.tasks:
                    continue

                task_info = self.tasks[task_id]
                if task_info.state == TaskState.REVEALED:
                    continue

                if task_info.state != TaskState.COMMITTED:
                    logger.warning(f"[{self.node_id}] Ground truth revealed for task {task_id} but we haven't committed yet")
                    continue

                logger.info(f"[{self.node_id}] Ground truth revealed for task {task_id}, revealing our solution...")

                # Reveal our solution
                reveal_result = self.registry.reveal_solution(task_id, task_info.solution_cid)

                if reveal_result.success:
                    logger.info(f"[{self.node_id}] Solution revealed for task {task_id}: {reveal_result.tx_hash}")
                    task_info.state = TaskState.REVEALED
                else:
                    logger.error(f"[{self.node_id}] Failed to reveal solution: {reveal_result.error}")

        except Exception as e:
            logger.error(f"[{self.node_id}] Error checking ground truth reveals: {e}", exc_info=True)
            self.metrics.errors += 1


def main():
    """Standalone demo of SolverNode."""
    from ..common.contracts import ContractRegistry
    from ..common.ipfs import IPFSClient
    from ..common.blockchain import BlockchainInterface

    # Use Hardhat account #2
    blockchain = BlockchainInterface(
        rpc_url="http://127.0.0.1:8545",
        private_key="0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
        chain_id=31337,
    )

    registry = ContractRegistry(blockchain=blockchain)
    ipfs = IPFSClient()

    node = SolverNode(
        registry=registry,
        ipfs=ipfs,
        node_id="solver-demo",
        project_id=1,
        deterministic_seed=42,
    )

    try:
        node.run(max_cycles=5, cycle_delay=3.0)
    except KeyboardInterrupt:
        node.stop()

    print(f"\nFinal metrics:")
    print(f"  Cycles: {node.metrics.cycles}")
    print(f"  Tasks completed: {node.metrics.tasks_completed}")
    print(f"  Solutions committed: {node.metrics.solutions_committed}")
    print(f"  Errors: {node.metrics.errors}")


if __name__ == "__main__":
    main()
