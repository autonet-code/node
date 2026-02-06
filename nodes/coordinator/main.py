"""
CoordinatorNode - Autonomous verification and voting node for Autonet.

Responsibilities:
- Stake as COORDINATOR (500 ATN)
- Monitor SolutionRevealed events
- Download and verify solutions against ground truth
- Submit verification votes
- Finalize voting when threshold reached
- Track forced errors and bond strength
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..common.contracts import ContractRegistry
from ..common.ipfs import IPFSClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoordinatorMetrics:
    """Metrics tracked by the coordinator node."""
    tasks_proposed: int = 0
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0
    aggregations_done: int = 0
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0

    # Coordinator-specific metrics
    solutions_verified: int = 0
    voting_finalized: int = 0
    ema_bond_strength: float = 1.0


class CoordinatorNode:
    """
    Autonomous coordinator node that verifies solver solutions.

    Interface Requirements (for orchestrator.py):
    - __init__(registry, ipfs, node_id, project_id)
    - run(max_cycles, cycle_delay)
    - stop()
    - metrics attribute
    """

    COORDINATOR_ROLE = 3
    STAKE_AMOUNT = 500 * 10**18  # 500 ATN
    EMA_ALPHA = 0.1  # Exponential moving average decay
    VOTE_THRESHOLD = 2  # Minimum votes to finalize

    def __init__(
        self,
        registry: ContractRegistry,
        ipfs: IPFSClient,
        node_id: str,
        project_id: int,
    ):
        """
        Initialize the coordinator node.

        Args:
            registry: ContractRegistry for blockchain interactions
            ipfs: IPFSClient for content storage/retrieval
            node_id: Unique identifier for this node instance
            project_id: Project ID to coordinate for
        """
        self.registry = registry
        self.ipfs = ipfs
        self.node_id = node_id
        self.project_id = project_id

        self.metrics = CoordinatorMetrics()
        self._running = False
        self._staked = False

        # Track processed events to avoid duplicates
        self._processed_solutions = set()

        # Track finalized tasks to avoid duplicate finalization attempts
        self._finalized_tasks = set()

        # Cache ground truths: task_id -> groundTruthCid
        self._ground_truth_cache: dict[int, str] = {}

        logger.info(
            f"CoordinatorNode initialized: {node_id} for project {project_id}"
        )

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 2.0):
        """
        Main execution loop.

        Args:
            max_cycles: Maximum number of cycles to run (None = infinite)
            cycle_delay: Delay between cycles in seconds
        """
        self._running = True
        logger.info(f"CoordinatorNode {self.node_id} starting...")

        try:
            cycle = 0
            while self._running and (max_cycles is None or cycle < max_cycles):
                try:
                    self._run_cycle()
                    self.metrics.cycles += 1
                    cycle += 1

                    if self._running:
                        time.sleep(cycle_delay)

                except Exception as e:
                    logger.error(f"Error in cycle {cycle}: {e}", exc_info=True)
                    self.metrics.errors += 1
                    time.sleep(cycle_delay)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self._running = False
            logger.info(
                f"CoordinatorNode {self.node_id} stopped after {cycle} cycles"
            )

    def stop(self):
        """Stop the node gracefully."""
        logger.info(f"Stopping CoordinatorNode {self.node_id}...")
        self._running = False

    def _run_cycle(self):
        """Execute one cycle of coordinator operations."""
        # First cycle: stake if not already staked
        if not self._staked:
            self._stake()
            return

        # Fetch and cache ground truths first (must happen before solution reveals)
        self._fetch_ground_truths()

        # Poll for new SolutionRevealed events
        self._process_solution_reveals()

        # Check for forced errors
        self._check_forced_errors()

        # Update bond strength (EMA decay)
        self._update_bond_strength()

    def _stake(self):
        """Approve ATN and stake as coordinator."""
        logger.info(f"Staking {self.STAKE_AMOUNT / 10**18} ATN as COORDINATOR...")

        try:
            # Get staking contract address
            staking_contract = self.registry.get("ParticipantStaking")

            # Approve ATN transfer
            logger.info("Approving ATN transfer...")
            result = self.registry.approve_atn(
                staking_contract.address,
                self.STAKE_AMOUNT
            )

            if not result.success:
                logger.error(f"ATN approval failed: {result.error}")
                self.metrics.errors += 1
                return

            logger.info(f"ATN approved: {result.tx_hash}")

            # Stake
            logger.info(f"Staking as coordinator (role={self.COORDINATOR_ROLE})...")
            result = self.registry.stake(self.COORDINATOR_ROLE, self.STAKE_AMOUNT)

            if result.success:
                logger.info(f"Staked successfully: {result.tx_hash}")
                self._staked = True
            else:
                logger.error(f"Staking failed: {result.error}")
                self.metrics.errors += 1

        except Exception as e:
            logger.error(f"Staking error: {e}", exc_info=True)
            self.metrics.errors += 1

    def _process_solution_reveals(self):
        """Poll for and process SolutionRevealed events."""
        try:
            events = self.registry.get_new_events("ResultsRewards", "SolutionRevealed")

            for event in events:
                self._process_solution_reveal(event)

        except Exception as e:
            logger.error(f"Error processing solution reveals: {e}", exc_info=True)
            self.metrics.errors += 1

    def _process_solution_reveal(self, event):
        """
        Process a single SolutionRevealed event.

        Event args should contain:
        - taskId
        - solver
        - cid (the solution CID)
        - groundTruthCid (from separate GroundTruthRevealed event)
        """
        try:
            args = event.get("args", {})
            task_id = args.get("taskId")
            solver = args.get("solver")
            solution_cid = args.get("cid")  # Event emits 'cid', not 'solutionCid'

            # Create unique key to avoid duplicate processing
            event_key = (task_id, solver)
            if event_key in self._processed_solutions:
                return

            logger.info(
                f"Processing solution reveal: task={task_id}, solver={solver}"
            )

            # Check if voting is still open
            if not self.registry.is_voting_open(task_id, solver):
                logger.info(f"Voting closed for task {task_id}, solver {solver}")
                self._processed_solutions.add(event_key)
                return

            # Get ground truth CID - check GroundTruthRevealed events
            ground_truth_cid = self._get_ground_truth_cid(task_id)
            if not ground_truth_cid:
                logger.warning(f"Ground truth not found for task {task_id}")
                return

            # Download and verify
            is_correct, score = self._verify_solution(
                solution_cid,
                ground_truth_cid
            )

            # Create verification report
            report = {
                "task_id": task_id,
                "solver": solver,
                "solution_cid": solution_cid,
                "ground_truth_cid": ground_truth_cid,
                "is_correct": is_correct,
                "score": score,
                "coordinator": self.registry.blockchain.account.address,
                "timestamp": int(time.time()),
                "node_id": self.node_id,
            }

            # Upload report to IPFS
            report_cid = self.ipfs.add_json(report)
            logger.info(f"Verification report uploaded: {report_cid}")

            # Submit vote
            result = self.registry.submit_vote(
                task_id,
                solver,
                is_correct,
                score,
                report_cid
            )

            if result.success:
                logger.info(
                    f"Vote submitted for task {task_id}: "
                    f"correct={is_correct}, score={score}"
                )
                self.metrics.votes_submitted += 1
                self.metrics.solutions_verified += 1
                self._processed_solutions.add(event_key)

                # Update bond strength based on verification
                self._update_bond_strength(success=True)

                # Try to finalize voting if threshold reached
                self._try_finalize_voting(task_id, solver)
            else:
                logger.error(f"Vote submission failed: {result.error}")
                self.metrics.errors += 1
                self._update_bond_strength(success=False)

        except Exception as e:
            logger.error(f"Error processing solution reveal: {e}", exc_info=True)
            self.metrics.errors += 1

    def _fetch_ground_truths(self):
        """
        Fetch new GroundTruthRevealed events and cache them.
        Must be called each cycle BEFORE processing solution reveals.
        """
        try:
            events = self.registry.get_new_events(
                "ResultsRewards",
                "GroundTruthRevealed"
            )

            for event in events:
                args = event.get("args", {})
                task_id = args.get("taskId")
                ground_truth_cid = args.get("cid")  # Event uses 'cid', not 'groundTruthCid'

                if task_id is not None and ground_truth_cid:
                    self._ground_truth_cache[task_id] = ground_truth_cid
                    logger.info(f"Cached ground truth for task {task_id}: {ground_truth_cid[:20]}...")

        except Exception as e:
            logger.error(f"Error fetching ground truths: {e}", exc_info=True)

    def _get_ground_truth_cid(self, task_id: int) -> Optional[str]:
        """
        Get ground truth CID for a task from the cache.

        Args:
            task_id: Task ID to get ground truth for

        Returns:
            Ground truth CID or None if not found
        """
        return self._ground_truth_cache.get(task_id)

    def _verify_solution(
        self,
        solution_cid: str,
        ground_truth_cid: str
    ) -> tuple[bool, int]:
        """
        Verify a solution against ground truth.

        Supports both supervised (accuracy-based) and self-supervised (embedding-based) tasks.

        Args:
            solution_cid: IPFS CID of solution
            ground_truth_cid: IPFS CID of ground truth

        Returns:
            Tuple of (is_correct, score) where score is 0-100
        """
        try:
            # Download both from IPFS
            solution_data = self.ipfs.get_json(solution_cid)
            ground_truth_data = self.ipfs.get_json(ground_truth_cid)

            # Extract metrics
            solution_metrics = solution_data.get("metrics", {})
            task_type = solution_data.get("task_type", "supervised")

            # Determine verification method based on task type
            if task_type == "jepa" or solution_metrics.get("self_supervised"):
                # JEPA/Self-supervised verification: use embedding similarity
                return self._verify_jepa_solution(solution_metrics, ground_truth_data)
            else:
                # Supervised verification: use accuracy
                return self._verify_supervised_solution(solution_metrics, ground_truth_data)

        except Exception as e:
            logger.error(f"Error verifying solution: {e}", exc_info=True)
            # Conservative: assume incorrect on error
            return False, 0

    def _verify_supervised_solution(
        self,
        solution_metrics: dict,
        ground_truth_data: dict
    ) -> tuple[bool, int]:
        """Verify supervised (accuracy-based) solution."""
        solution_accuracy = solution_metrics.get("accuracy", 0.0)
        accuracy_threshold = ground_truth_data.get("accuracy_threshold", 0.0)

        if accuracy_threshold > 0 and solution_accuracy > 0:
            if solution_accuracy >= accuracy_threshold:
                score = 100
            else:
                score = int((solution_accuracy / accuracy_threshold) * 100)
        else:
            # Fallback: give passing score
            score = 80

        is_correct = score >= 70

        logger.info(
            f"Supervised verification: score={score}, correct={is_correct}, "
            f"solution_acc={solution_accuracy:.3f}, threshold={accuracy_threshold:.3f}"
        )

        return is_correct, score

    def _verify_jepa_solution(
        self,
        solution_metrics: dict,
        ground_truth_data: dict
    ) -> tuple[bool, int]:
        """
        Verify JEPA (embedding-based) solution.

        For JEPA, we verify by checking:
        - cosine_similarity: how well predictions match target embeddings
        - embedding_energy (L2 distance): lower is better

        Ground truth for JEPA is the target encoder's behavior.
        """
        cosine_sim = solution_metrics.get("cosine_similarity", 0.0)
        embedding_energy = solution_metrics.get("embedding_energy", float('inf'))
        loss = solution_metrics.get("loss", float('inf'))

        # Get thresholds from ground truth (or use defaults)
        min_cosine_sim = ground_truth_data.get("min_cosine_similarity", 0.5)
        max_energy = ground_truth_data.get("max_embedding_energy", 1.0)

        # Score based on cosine similarity (main metric for JEPA)
        # Cosine similarity ranges from -1 to 1, but we expect > 0 for trained models
        if cosine_sim >= min_cosine_sim:
            # Above threshold: full credit + bonus for exceeding
            score = min(100, int(50 + cosine_sim * 50))
        else:
            # Below threshold: partial credit
            score = int(cosine_sim * 100)

        # Adjust based on embedding energy if available
        if embedding_energy < max_energy:
            score = min(100, score + 10)

        is_correct = cosine_sim >= min_cosine_sim

        logger.info(
            f"JEPA verification: score={score}, correct={is_correct}, "
            f"cosine_sim={cosine_sim:.3f} (min={min_cosine_sim}), "
            f"energy={embedding_energy:.3f} (max={max_energy}), loss={loss:.3f}"
        )

        return is_correct, score

    def _compute_structural_similarity(
        self,
        data1: dict,
        data2: dict
    ) -> int:
        """
        Compute structural similarity between two JSON objects.

        Args:
            data1: First JSON object
            data2: Second JSON object

        Returns:
            Similarity score 0-100
        """
        try:
            # Count matching keys at top level
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())

            if not keys1 and not keys2:
                return 100

            common_keys = keys1 & keys2
            all_keys = keys1 | keys2

            if not all_keys:
                return 0

            # Base similarity on key overlap
            key_similarity = len(common_keys) / len(all_keys)

            # Check value similarity for common keys
            value_matches = 0
            for key in common_keys:
                if data1[key] == data2[key]:
                    value_matches += 1

            if common_keys:
                value_similarity = value_matches / len(common_keys)
            else:
                value_similarity = 0

            # Weighted average: 40% key similarity, 60% value similarity
            overall_similarity = (
                0.4 * key_similarity + 0.6 * value_similarity
            )

            return int(overall_similarity * 100)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0

    def _try_finalize_voting(self, task_id: int, solver: str):
        """
        Try to finalize voting if threshold is reached.

        Args:
            task_id: Task ID to finalize
            solver: Solver address
        """
        try:
            # Check if we've already tried to finalize this task
            finalize_key = (task_id, solver)
            if finalize_key in self._finalized_tasks:
                logger.debug(f"Task {task_id} already finalized by this node")
                return

            vote_count = self.registry.get_vote_count(task_id, solver)

            if vote_count >= self.VOTE_THRESHOLD:
                logger.info(
                    f"Vote threshold reached ({vote_count} >= {self.VOTE_THRESHOLD}), "
                    f"finalizing task {task_id}..."
                )

                # Mark as finalized before attempting to prevent race conditions
                self._finalized_tasks.add(finalize_key)

                result = self.registry.finalize_voting(task_id, solver)

                if result.success:
                    logger.info(f"Voting finalized for task {task_id}: {result.tx_hash}")
                    self.metrics.voting_finalized += 1
                    self.metrics.tasks_completed += 1
                else:
                    # If it failed, it might be because another coordinator finalized first
                    if "Already finalized" in str(result.error):
                        logger.info(f"Task {task_id} already finalized by another coordinator")
                    else:
                        logger.warning(f"Finalization failed: {result.error}")

        except Exception as e:
            logger.error(f"Error finalizing voting: {e}", exc_info=True)

    def _check_forced_errors(self):
        """Check for and report forced errors in solutions."""
        try:
            # This would typically scan for known attack patterns
            # or anomalies in recent solutions
            # Placeholder implementation
            pass

        except Exception as e:
            logger.error(f"Error checking forced errors: {e}", exc_info=True)

    def _update_bond_strength(self, success: Optional[bool] = None):
        """
        Update EMA bond strength metric.

        Args:
            success: Optional success indicator (True/False/None for decay only)
        """
        if success is True:
            # Increase bond strength
            target = 1.0
        elif success is False:
            # Decrease bond strength
            target = 0.0
        else:
            # Natural decay towards neutral
            return

        # Update EMA
        self.metrics.ema_bond_strength = (
            self.EMA_ALPHA * target +
            (1 - self.EMA_ALPHA) * self.metrics.ema_bond_strength
        )
