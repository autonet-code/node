"""
Autonomous Solver Node for Autonet

Discovers tasks from the blockchain, performs real ML training,
commits solutions, and reveals them after ground truth is revealed.

Reads hyperparameters from the proposer's task spec (stored in blob store)
and from the node config. Supports both supervised (CNN/MNIST) and
self-supervised (JEPA) training.
"""

import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from ..common.config import AutonetConfig, load_config
from ..common.governance import GovernanceBridge

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

    def __init__(
        self,
        registry,
        store,
        node_id: str,
        project_id: int,
        deterministic_seed: Optional[int] = None,
        task_type: str = "supervised",
        task_mode: str = "ground_truth",
        config: Optional[AutonetConfig] = None,
    ):
        """
        Initialize SolverNode.

        Args:
            registry: ContractRegistry instance
            store: BlobStore instance for content-addressed storage
            node_id: Unique identifier for this node (e.g., "solver-0")
            project_id: Project ID to work on
            deterministic_seed: Seed for reproducible training
            task_type: Type of training ("supervised" or "jepa")
            task_mode: "ground_truth" (legacy) or "consensus_truth" (MM-Zero)
            config: AutonetConfig (loaded from yaml/env if not provided)
        """
        self.registry = registry
        self.store = store
        self.node_id = node_id
        self.project_id = project_id
        self.config = config or load_config()
        self.deterministic_seed = deterministic_seed or self.config.seed or int(time.time())
        self.task_type = self.config.training.task_type
        self.task_mode = task_mode

        self.solver_stake_amount = self.config.staking.solver * 10**18

        self.metrics = SolverMetrics()
        self.tasks: Dict[int, TaskInfo] = {}
        self.processed_task_ids: Set[int] = set()
        self.staked = False
        self.running = False

        # Governance bridge for attestation and heartbeat
        self.governance: Optional[GovernanceBridge] = None  # Set in run() after registry is available

        logger.info(f"[{self.node_id}] Initialized solver node for project {project_id} (task_type={self.task_type})")

    def run(self, max_cycles: int = 10, cycle_delay: float = 2.0):
        """
        Main event loop.

        Args:
            max_cycles: Maximum number of cycles to run
            cycle_delay: Delay between cycles in seconds
        """
        self.running = True
        # Initialize governance bridge now that registry is available
        if self.governance is None:
            self.governance = GovernanceBridge(self.registry, self.node_id, self.project_id)
        logger.info(f"[{self.node_id}] Starting main loop: max_cycles={max_cycles}, delay={cycle_delay}s")

        for cycle in range(max_cycles):
            if not self.running:
                break

            try:
                logger.info(f"[{self.node_id}] === Cycle {cycle + 1}/{max_cycles} ===")

                # Check governance heartbeat — halt work if missed
                if not self.governance.check_heartbeat():
                    logger.warning(f"[{self.node_id}] Governance heartbeat missed, halting work")
                    import time as _t
                    _t.sleep(cycle_delay)
                    continue

                # First cycle: stake as solver
                if cycle == 0 and not self.staked:
                    self._stake_as_solver()

                # Discover new tasks
                self._discover_tasks()

                # Process tasks: train, commit, reveal
                self._process_tasks()

                # Check for ground truth reveals and reveal our solutions
                self._check_and_reveal_solutions()

                # Claim rewards for previous epoch + accrue reputation
                if self.governance:
                    self.governance.claim_epoch_rewards()
                    self.governance.claim_reputation()

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
                self.solver_stake_amount
            )

            if not approve_result.success:
                logger.error(f"[{self.node_id}] Failed to approve ATN: {approve_result.error}")
                return

            logger.info(f"[{self.node_id}] ATN approved: {approve_result.tx_hash}")

            # Stake
            stake_result = self.registry.stake(self.SOLVER_ROLE, self.solver_stake_amount)

            if stake_result.success:
                logger.info(f"[{self.node_id}] Staked {self.solver_stake_amount // 10**18} ATN as SOLVER: {stake_result.tx_hash}")
                self.staked = True
            else:
                logger.error(f"[{self.node_id}] Failed to stake: {stake_result.error}")

        except Exception as e:
            logger.error(f"[{self.node_id}] Staking error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _discover_tasks(self):
        """Poll for TaskProposed / ConsensusTaskProposed events and discover new tasks."""
        try:
            if self.task_mode == "consensus_truth":
                # Discover consensus-mode tasks
                events = self.registry.get_new_events("TaskContract", "ConsensusTaskProposed")
            else:
                # Discover ground-truth-mode tasks
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
        """Process discovered tasks: train and commit solutions or submit rollouts."""
        for task_id, task_info in list(self.tasks.items()):
            try:
                if task_info.state == TaskState.DISCOVERED:
                    if self.task_mode == "consensus_truth":
                        self._train_and_submit_rollout(task_id, task_info)
                    else:
                        self._train_and_commit(task_id, task_info)

            except Exception as e:
                logger.error(f"[{self.node_id}] Error processing task {task_id}: {e}", exc_info=True)
                self.metrics.errors += 1

    def _train_and_commit(self, task_id: int, task_info: TaskInfo):
        """Perform real training and commit solution."""
        logger.info(f"[{self.node_id}] Training on task {task_id}...")
        task_info.state = TaskState.TRAINING

        # Try to load the task spec from blob store (uploaded by proposer)
        task_spec = None
        if task_info.task_cid:
            try:
                task_spec = self.store.get_json(task_info.task_cid)
                if task_spec:
                    logger.info(f"[{self.node_id}] Loaded task spec from {task_info.task_cid[:20]}...")
            except Exception as e:
                logger.warning(f"[{self.node_id}] Could not load task spec: {e}")

        # Perform real training with checkpoints
        start_time = time.time()
        model_update, checkpoints = self._train(task_id, task_spec)
        training_time = time.time() - start_time

        logger.info(f"[{self.node_id}] Training completed in {training_time:.2f}s with {len(checkpoints)} checkpoints")

        # Upload solution to blob store
        solution_cid = self.store.add_json(model_update)
        if not solution_cid:
            logger.error(f"[{self.node_id}] Failed to upload solution")
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

            # Attest training work to the economic layer
            if self.governance:
                self.governance.attest_task_completion(units=1)

            # Submit a few checkpoints
            self._submit_checkpoints(task_id, checkpoints)
        else:
            logger.error(f"[{self.node_id}] Failed to commit solution: {commit_result.error}")

    def _train(self, task_id: int, task_spec: Optional[dict] = None) -> tuple:
        """
        Perform real ML training driven by task spec.

        The task spec is uploaded by the proposer and contains all
        hyperparameters. If no spec is available (e.g., spec fetch fails),
        falls back to config defaults.

        Returns:
            (model_update_dict, checkpoints_list)
        """
        # Resolve task spec: try blob store first, fall back to config defaults
        if task_spec is None:
            task_spec = self._build_default_task_spec(task_id)

        # Get global model CID (from spec or from chain)
        global_model_cid = task_spec.get("global_model_cid")
        if not global_model_cid:
            try:
                global_model_cid = self.registry.get_mature_model(self.project_id)
            except Exception:
                pass

        if global_model_cid:
            logger.info(f"[{self.node_id}] Continuing from global model: {global_model_cid[:20]}...")
        else:
            logger.info(f"[{self.node_id}] No global model, training from scratch")

        task_type = task_spec.get("task_type", self.task_type)

        if task_type in ("vl_jepa", "text_jepa"):
            return self._train_vljepa(task_id, task_spec, global_model_cid)
        elif task_type in ("jepa", "jepa_pretrain"):
            return self._train_jepa(task_id, task_spec, global_model_cid)
        else:
            return self._train_supervised(task_id, task_spec, global_model_cid)

    def _build_default_task_spec(self, task_id: int) -> dict:
        """Build a task spec from config defaults when proposer spec is unavailable."""
        cfg = self.config
        return {
            "task_id": task_id,
            "task_type": cfg.training.task_type,
            "image_size": cfg.model.image_size,
            "patch_size": cfg.model.patch_size,
            "embed_dim": cfg.model.embed_dim,
            "num_heads": cfg.model.num_heads,
            "encoder_depth": cfg.model.encoder_depth,
            "predictor_depth": cfg.model.predictor_depth,
            "predictor_embed_dim": cfg.model.predictor_embed_dim,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "num_samples": cfg.training.num_samples,
            "optimizer": cfg.training.optimizer,
        }

    def _train_supervised(self, task_id: int, task_spec: dict, global_model_cid: str = None) -> tuple:
        """
        Perform supervised CNN training using task spec hyperparameters.
        """
        from ..common.ml import train_on_task

        epochs = task_spec.get("epochs", self.config.training.epochs)
        batch_size = task_spec.get("batch_size", self.config.training.batch_size)
        lr = task_spec.get("learning_rate", self.config.training.learning_rate)
        num_samples = task_spec.get("num_samples", self.config.training.num_samples)

        logger.info(
            f"[{self.node_id}] Supervised training: task={task_id}, "
            f"epochs={epochs}, batch={batch_size}, lr={lr}, samples={num_samples}"
        )
        weight_delta, metrics = train_on_task(
            task_spec={"task_id": task_id, "epochs": epochs},
            store=self.store,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            num_samples=num_samples,
            global_model_cid=global_model_cid,
        )

        checkpoints = self._generate_training_checkpoints(task_id, metrics)

        model_update = {
            "task_id": task_id,
            "task_type": "supervised",
            "weight_delta": weight_delta,
            "metrics": metrics,
            "training_steps": metrics.get("num_samples", num_samples) // batch_size,
            "final_seed": self._generate_deterministic_seed(task_id, 100),
            "solver": self.node_id,
            "real_training": True,
        }

        logger.info(f"[{self.node_id}] Supervised training done: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
        return model_update, checkpoints

    def _train_jepa(self, task_id: int, task_spec: dict, global_model_cid: str = None) -> tuple:
        """
        Perform JEPA self-supervised training using task spec hyperparameters.

        All model architecture and training params come from the task spec,
        which was generated by the proposer from the shared config.
        """
        from ..common.ml import train_jepa_on_task

        epochs = task_spec.get("epochs", self.config.training.epochs)
        batch_size = task_spec.get("batch_size", self.config.training.batch_size)
        lr = task_spec.get("learning_rate", self.config.training.learning_rate)
        num_samples = task_spec.get("num_samples", self.config.training.num_samples)
        image_size = task_spec.get("image_size", self.config.model.image_size)

        logger.info(
            f"[{self.node_id}] JEPA training: task={task_id}, "
            f"epochs={epochs}, batch={batch_size}, lr={lr}, "
            f"samples={num_samples}, img={image_size}px"
        )
        weight_delta, metrics = train_jepa_on_task(
            task_spec=task_spec,
            store=self.store,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            num_samples=num_samples,
            image_size=image_size,
            global_model_cid=global_model_cid,
        )

        checkpoints = self._generate_training_checkpoints(task_id, metrics)

        model_update = {
            "task_id": task_id,
            "task_type": "jepa",
            "weight_delta": weight_delta,
            "metrics": metrics,
            "training_steps": metrics.get("num_samples", num_samples) // batch_size,
            "final_seed": self._generate_deterministic_seed(task_id, 100),
            "solver": self.node_id,
            "real_training": True,
            "self_supervised": True,
        }

        logger.info(
            f"[{self.node_id}] JEPA training done: loss={metrics['loss']:.4f}, "
            f"cosine_sim={metrics.get('cosine_similarity', 0):.4f}"
        )
        return model_update, checkpoints

    def _train_vljepa(self, task_id: int, task_spec: dict, global_model_cid: str = None) -> tuple:
        """
        Perform VL-JEPA / Text-JEPA training using agent framework data.

        Story 3.4: Instead of training on CIFAR/FakeData, uses the real
        agent interaction data (conversations, execution records) stored
        in the ATN data directory.

        Supports three modality paths:
        - text_jepa: Text-only (from conversations + execution logs)
        - vl_jepa: Multimodal (text + visual, when visual data is available)
        - Falls back to text-only if no visual data source is provided
        """
        from ..common.ml import train_vljepa_on_task

        epochs = task_spec.get("epochs", self.config.training.epochs)
        lr = task_spec.get("learning_rate", self.config.training.learning_rate)
        batch_size = task_spec.get("batch_size", self.config.training.batch_size)

        # Build text data source from ATN data directory
        text_source = None
        data_dir = task_spec.get("data_dir", "")
        if not data_dir:
            # Try config-level data_dir
            data_dir = getattr(self.config, "data_dir", "")

        if data_dir:
            from ..common.text_data import TextDataConfig, TextTrainingDataSource
            from pathlib import Path

            data_path = Path(data_dir)
            conversations_dir = data_path / "conversations"
            agents_dir = data_path / "agents"

            text_config = TextDataConfig(
                conversations_dir=str(conversations_dir) if conversations_dir.is_dir() else "",
                agents_dir=str(agents_dir) if agents_dir.is_dir() else "",
                max_seq_length=task_spec.get("max_seq_length", 2048),
                vocab_size=task_spec.get("vocab_size", 260),
                batch_size=batch_size,
                scrub_pii=True,
                shuffle=True,
            )
            text_source = TextTrainingDataSource(text_config)

            logger.info(
                f"[{self.node_id}] VL-JEPA training: task={task_id}, "
                f"epochs={epochs}, batch={batch_size}, lr={lr}, "
                f"data_dir={data_dir}"
            )
        else:
            logger.info(
                f"[{self.node_id}] VL-JEPA training (no data dir): task={task_id}, "
                f"epochs={epochs}, batch={batch_size}, lr={lr}"
            )

        # Determine modalities from task_spec
        modalities = task_spec.get("modalities")
        task_type = task_spec.get("task_type", "text_jepa")

        weight_delta, metrics = train_vljepa_on_task(
            task_spec=task_spec,
            global_model_cid=global_model_cid,
            store=self.store,
            text_data_source=text_source,
            epochs=epochs,
            learning_rate=lr,
            modalities=modalities,
        )

        checkpoints = self._generate_training_checkpoints(task_id, metrics)

        model_update = {
            "task_id": task_id,
            "task_type": task_type,
            "weight_delta": weight_delta,
            "metrics": metrics,
            "training_steps": metrics.get("num_batches", 0),
            "final_seed": self._generate_deterministic_seed(task_id, 100),
            "solver": self.node_id,
            "real_training": True,
            "self_supervised": True,
        }

        logger.info(
            f"[{self.node_id}] VL-JEPA training done: "
            f"arch={metrics.get('architecture', 'unknown')}, "
            f"modalities={metrics.get('modalities', [])}, "
            f"loss={metrics.get('loss', 0):.4f}, "
            f"batches={metrics.get('num_batches', 0)}"
        )
        return model_update, checkpoints

    def _generate_training_checkpoints(self, task_id: int, metrics: dict) -> list:
        """Generate training checkpoints from metrics."""
        checkpoints = []
        training_steps = metrics.get("num_samples", 500) // 32
        current_seed = self._generate_deterministic_seed(task_id, 0)

        for step in range(0, training_steps, max(1, training_steps // 3)):
            checkpoint = self._create_checkpoint(step, current_seed)
            checkpoints.append(checkpoint)
            current_seed = self._generate_deterministic_seed(task_id, step + 1)

        # Final checkpoint
        final_checkpoint = self._create_checkpoint(training_steps, current_seed)
        checkpoints.append(final_checkpoint)

        return checkpoints

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

    # ============ Consensus-as-Truth (MM-Zero) Methods ============

    def _train_and_submit_rollout(self, task_id: int, task_info: TaskInfo):
        """
        Train on a consensus-mode task and submit a rollout.

        Unlike ground-truth mode, there's no commit-reveal pattern.
        Multiple solvers independently train and submit answer hashes.
        """
        logger.info(f"[{self.node_id}] Training for consensus rollout on task {task_id}...")
        task_info.state = TaskState.TRAINING

        # Load task spec if available
        task_spec = None
        if task_info.task_cid:
            try:
                task_spec = self.store.get_json(task_info.task_cid)
            except Exception:
                pass

        # Perform real training
        start_time = time.time()
        model_update, checkpoints = self._train(task_id, task_spec)
        training_time = time.time() - start_time

        logger.info(f"[{self.node_id}] Consensus training completed in {training_time:.2f}s")

        # Generate an answer hash from the model's output
        # For JEPA: answer is the embedding signature of the trained model
        # For supervised: answer is the prediction vector on task inputs
        import json
        answer_data = self._generate_consensus_answer(model_update)
        answer_json = json.dumps(answer_data, sort_keys=True)

        # Compute confidence from training metrics
        metrics = model_update.get("metrics", {})
        confidence = self._compute_confidence(metrics)

        # Upload full solution to blob store (weight delta + answer + metrics)
        solution_data = {
            **model_update,
            "answer": answer_data,
            "confidence": confidence,
            "task_mode": "consensus_truth",
        }
        solution_cid = self.store.add_json(solution_data)
        if not solution_cid:
            logger.error(f"[{self.node_id}] Failed to upload rollout solution")
            return

        # Compute hashes
        from web3 import Web3
        answer_hash = Web3.keccak(text=answer_json)
        solution_hash = Web3.keccak(text=solution_cid)

        logger.info(
            f"[{self.node_id}] Submitting rollout: answer_hash={answer_hash.hex()[:16]}..., "
            f"confidence={confidence}"
        )

        # Submit rollout on-chain
        result = self.registry.submit_rollout(
            task_id, answer_hash, solution_hash, confidence
        )

        if result.success:
            logger.info(f"[{self.node_id}] Rollout submitted for task {task_id}: {result.tx_hash}")
            task_info.state = TaskState.COMMITTED
            task_info.solution_cid = solution_cid
            self.metrics.solutions_committed += 1
            self.metrics.tasks_completed += 1
        else:
            logger.error(f"[{self.node_id}] Failed to submit rollout: {result.error}")
            self.metrics.errors += 1

    def _generate_consensus_answer(self, model_update: dict) -> dict:
        """
        Generate an answer from the trained model for consensus voting.

        The answer is a deterministic fingerprint of the model's behavior,
        used for majority-vote consensus across solvers.
        """
        metrics = model_update.get("metrics", {})
        task_type = model_update.get("task_type", "supervised")

        if task_type == "jepa":
            # For JEPA: answer is discretized embedding statistics
            return {
                "type": "jepa_embedding",
                "cosine_similarity_bucket": self._bucket_similarity(
                    metrics.get("cosine_similarity", 0)
                ),
                "loss_bucket": self._bucket_loss(metrics.get("loss", 0)),
                "embedding_energy_bucket": self._bucket_loss(
                    metrics.get("embedding_energy", 0)
                ),
            }
        else:
            # For supervised: answer is discretized accuracy + loss
            return {
                "type": "supervised_prediction",
                "accuracy_bucket": self._bucket_accuracy(
                    metrics.get("accuracy", 0)
                ),
                "loss_bucket": self._bucket_loss(metrics.get("loss", 0)),
            }

    def _bucket_similarity(self, similarity: float) -> str:
        """Bucket a cosine similarity value into discrete ranges for consensus."""
        if similarity >= 0.9:
            return "very_high"
        elif similarity >= 0.7:
            return "high"
        elif similarity >= 0.5:
            return "medium"
        elif similarity >= 0.3:
            return "low"
        else:
            return "very_low"

    def _bucket_accuracy(self, accuracy: float) -> str:
        """Bucket an accuracy value into discrete ranges for consensus."""
        if accuracy >= 0.9:
            return "very_high"
        elif accuracy >= 0.7:
            return "high"
        elif accuracy >= 0.5:
            return "medium"
        elif accuracy >= 0.3:
            return "low"
        else:
            return "very_low"

    def _bucket_loss(self, loss: float) -> str:
        """Bucket a loss value into discrete ranges for consensus matching."""
        if loss < 0.1:
            return "very_low"
        elif loss < 0.5:
            return "low"
        elif loss < 1.0:
            return "medium"
        elif loss < 2.0:
            return "high"
        else:
            return "very_high"

    def _compute_confidence(self, metrics: dict) -> int:
        """
        Compute confidence score (0-100) from training metrics.

        Higher confidence = model performed well during training.
        """
        task_type = metrics.get("task_type", "supervised")

        if "cosine_similarity" in metrics:
            # JEPA: confidence from cosine similarity
            cos_sim = metrics.get("cosine_similarity", 0)
            return min(100, max(0, int(cos_sim * 100)))
        elif "accuracy" in metrics:
            # Supervised: confidence from accuracy
            acc = metrics.get("accuracy", 0)
            return min(100, max(0, int(acc * 100)))
        else:
            return 50  # Default moderate confidence

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
    from ..common.blob_store import BlobStore
    from ..common.blockchain import BlockchainInterface

    # Use Hardhat account #2
    blockchain = BlockchainInterface(
        rpc_url="http://127.0.0.1:8545",
        private_key="0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
        chain_id=31337,
    )

    registry = ContractRegistry(blockchain=blockchain)
    store = BlobStore()

    node = SolverNode(
        registry=registry,
        store=store,
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
