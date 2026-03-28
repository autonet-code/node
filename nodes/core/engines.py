"""
Autonet Node Engines

The four specialized engines that power each autonomous node:
- AwarenessEngine: Environmental perception
- GovernanceEngine: Constitutional validation and collective decision-making
- WorkEngine: Task execution (training, inference, verification)
- SurvivalEngine: Self-preservation and replication
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# TYPE_CHECKING import avoids circular dependencies at runtime;
# ThreeTierConstitutionalEvaluator and ConstitutionalGeometry are optional.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nodes.common.governance import ThreeTierConstitutionalEvaluator
    from nodes.common.constitutional_geometry import ConstitutionalGeometry

logger = logging.getLogger(__name__)


class InstructionStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class Instruction:
    id: str
    action: str
    details: Dict[str, Any]
    proof_of_adherence: str
    signature: str
    status: InstructionStatus = InstructionStatus.PENDING


class BaseEngine(ABC):
    """Base class for all node engines."""

    def __init__(self, node: "Node"):
        self.node = node
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{node.node_id[:8]}]")

    @abstractmethod
    def tick(self) -> None:
        """Execute one cycle of the engine."""
        pass


class AwarenessEngine(BaseEngine):
    """
    The node's duty to perceive its environment.
    Monitors network status, local resources, and external signals.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.last_observation: Dict[str, Any] = {}

    def tick(self) -> None:
        self.last_observation = self.perceive()

    def perceive(self) -> Dict[str, Any]:
        """Gather environmental data."""
        import psutil

        observation = {
            "network_status": "OK",
            "cpu_percent": psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0.0,
            "memory_percent": psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0.0,
            "timestamp": self._current_time(),
            "consensus_alive": self.node.is_consensus_alive(),
        }

        self.logger.debug(f"Perception: {observation}")
        return observation

    def _current_time(self) -> float:
        import time
        return time.time()


class GovernanceEngine(BaseEngine):
    """
    The node's duty to participate in collective decision-making.
    Validates instructions against constitutional principles via the 3-tier
    constitutional evaluation pipeline (geometric → bottleneck → LLM).
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.pending_instructions: List[Instruction] = []
        self.validated_instructions: List[Instruction] = []

        # Optional 3-tier evaluator (wired at node startup when JEPA model available)
        self._constitutional_evaluator: Optional["ThreeTierConstitutionalEvaluator"] = None
        self._encode_fn: Optional[Any] = None  # text → embedding callable

        # Calibration: recent embeddings buffer for drift monitoring
        # Holds at most _drift_buffer_size embeddings
        self._drift_buffer: List[Any] = []
        self._drift_buffer_size: int = 64
        self._drift_check_interval: int = 16  # check every N new embeddings

    def wire_constitutional_evaluator(
        self,
        evaluator: "ThreeTierConstitutionalEvaluator",
        encode_fn: Any,
    ) -> None:
        """
        Wire the 3-tier constitutional evaluator into the engine.

        Called by the node startup sequence after the JEPA model is loaded.

        Args:
            evaluator:  ThreeTierConstitutionalEvaluator instance
            encode_fn:  Callable[[str], Tensor] — text → embedding (from TextEncoder)
        """
        self._constitutional_evaluator = evaluator
        self._encode_fn = encode_fn
        self.logger.info("3-tier constitutional evaluator wired")

    def tick(self) -> None:
        self.check_for_proposals()
        self.run_drift_calibration()
        self.process_pending_instructions()

    def check_for_proposals(self) -> None:
        """Check for new proposals from the consensus network."""
        # In production: poll blockchain or P2P for new evolution proposals
        pass

    def run_drift_calibration(self) -> None:
        """
        Run embedding drift detection using buffered recent embeddings.

        Called every tick. When the buffer is full, updates DriftMonitor
        statistics. If drift is detected, logs a warning so operators can
        trigger recalibration (by calling geometry.calibrate() again with
        the updated model's encode_fn).
        """
        if (
            self._constitutional_evaluator is None
            or self._encode_fn is None
            or not self._drift_buffer
        ):
            return

        evaluator = self._constitutional_evaluator
        geometry = getattr(evaluator, "_geometry", None)
        if geometry is None or not geometry.drift_monitor.is_calibrated:
            return

        if len(self._drift_buffer) < self._drift_check_interval:
            return

        try:
            import torch
            stacked = torch.stack(self._drift_buffer[-self._drift_check_interval:])
            drift = geometry.update_drift_stats(stacked)
            if drift:
                self.logger.warning(
                    "Embedding drift detected during governance tick — "
                    "constitutional geometry recalibration required. "
                    "Call geometry.calibrate(encode_fn, reference_embeddings) "
                    "after model weight update."
                )
            # Keep buffer bounded
            self._drift_buffer = self._drift_buffer[-self._drift_buffer_size:]
        except Exception as e:
            self.logger.debug(f"Drift calibration step failed: {e}")

    def process_pending_instructions(self) -> None:
        """Validate pending instructions against constitutional principles."""
        for instruction in list(self.pending_instructions):
            if self.validate_instruction(instruction):
                instruction.status = InstructionStatus.VALIDATED
                self.validated_instructions.append(instruction)
                self.node.work.queue_instruction(instruction)
            else:
                instruction.status = InstructionStatus.REJECTED
                self.logger.warning(
                    f"Rejected instruction {instruction.id}: violates constitutional principles"
                )

            self.pending_instructions.remove(instruction)

    def validate_instruction(self, instruction: Instruction) -> bool:
        """
        Validate an instruction against the RPB constitution (3-tier pipeline).

        Fast path: geometric evaluation (Tier 1) handles ~80-90% of cases in O(1).
        Medium path: concept bottleneck (Tier 2) for nuanced cases.
        Slow path: LLM evaluation (Tier 3) for uncertain/adversarial cases.

        Falls back to the existing constitution.validate_action() check if no
        constitutional evaluator is wired.
        """
        # Try 3-tier evaluator first
        if self._constitutional_evaluator is not None:
            action_text = f"{instruction.action}: {instruction.proof_of_adherence}"
            verdict, confidence = self._constitutional_evaluator.check_action(
                action_text, self._encode_fn, instruction.proof_of_adherence
            )

            if verdict == "violation":
                self.logger.warning(
                    f"Instruction {instruction.id} rejected by constitutional evaluator: "
                    f"verdict=violation, confidence={confidence:.3f}"
                )
                return False

            if verdict == "compliant":
                self.logger.debug(
                    f"Instruction {instruction.id} approved: "
                    f"verdict=compliant, confidence={confidence:.3f}"
                )
                # Buffer embedding for drift monitoring
                self._buffer_embedding(action_text)
                return True

            # verdict == "uncertain": fall through to constitution check
            self.logger.debug(
                f"Constitutional evaluator uncertain for {instruction.id} "
                f"(confidence={confidence:.3f}) — falling back to constitution"
            )

        # Fallback: existing constitution.validate_action()
        return self.node.constitution.validate_action(
            instruction.action,
            instruction.proof_of_adherence,
        )

    def _buffer_embedding(self, text: str) -> None:
        """Buffer an embedding for drift monitoring (best-effort)."""
        if self._encode_fn is None:
            return
        try:
            emb = self._encode_fn(text)
            if emb is not None:
                import torch
                if emb.dim() > 1:
                    emb = emb.mean(dim=0)
                self._drift_buffer.append(emb.detach())
        except Exception:
            pass  # Non-critical

    def submit_instruction(self, instruction: Instruction) -> None:
        """Add an instruction to the pending queue."""
        self.pending_instructions.append(instruction)


class WorkEngine(BaseEngine):
    """
    The node's duty to execute validated instructions.
    Handles training, inference, and verification tasks.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.instruction_queue: List[Instruction] = []
        self.current_task: Optional[Instruction] = None

    def tick(self) -> None:
        if not self.node.is_consensus_alive():
            self.logger.warning("Consensus heartbeat missed. Halting work.")
            return

        self.execute_next()

    def queue_instruction(self, instruction: Instruction) -> None:
        """Add a validated instruction to the work queue."""
        self.instruction_queue.append(instruction)
        self.logger.info(f"Queued instruction: {instruction.id}")

    def execute_next(self) -> None:
        """Execute the next instruction in the queue."""
        if not self.instruction_queue:
            return

        instruction = self.instruction_queue.pop(0)
        self.current_task = instruction

        try:
            # Defense-in-depth: re-check compliance at execution time.
            # GovernanceEngine validates before queueing; this catches any
            # instructions that bypass the governance queue (e.g., injected
            # directly during testing or via buggy code paths).
            if not self._compliance_check(instruction):
                instruction.status = InstructionStatus.REJECTED
                self.logger.warning(
                    f"WorkEngine compliance check rejected {instruction.id} "
                    f"at execution time — should have been caught by GovernanceEngine"
                )
                return

            self._execute_instruction(instruction)
            instruction.status = InstructionStatus.EXECUTED
            self.logger.info(f"Executed: {instruction.action}")
        except Exception as e:
            instruction.status = InstructionStatus.FAILED
            self.logger.error(f"Failed to execute {instruction.id}: {e}")
        finally:
            self.current_task = None

    def _compliance_check(self, instruction: Instruction) -> bool:
        """
        Lightweight compliance check before instruction execution.

        Delegates to GovernanceEngine's constitutional evaluator when available.
        Fails open (returns True) if no evaluator is configured — execution
        is never blocked by evaluator unavailability alone, since GovernanceEngine
        already validated the instruction on entry.
        """
        gov = self.node.governance
        evaluator = getattr(gov, "_constitutional_evaluator", None)
        encode_fn = getattr(gov, "_encode_fn", None)

        if evaluator is None:
            return True  # No evaluator: trust GovernanceEngine's prior check

        action_text = f"{instruction.action}: {instruction.proof_of_adherence}"
        verdict, confidence = evaluator.check_action(action_text, encode_fn)

        if verdict == "violation" and confidence >= 0.85:
            # Only hard-reject on high-confidence violations to avoid false positives
            return False

        return True

    def _execute_instruction(self, instruction: Instruction) -> None:
        """Execute an instruction based on its action type."""
        action = instruction.action
        details = instruction.details

        if action == "TRAIN_MODEL":
            self._handle_training(details)
        elif action == "VERIFY_SOLUTION":
            self._handle_verification(details)
        elif action == "AGGREGATE_UPDATES":
            self._handle_aggregation(details)
        elif action == "SERVE_INFERENCE":
            self._handle_inference(details)
        else:
            self.logger.warning(f"Unknown action: {action}")

    def _handle_training(self, details: Dict[str, Any]) -> None:
        """Handle a training task."""
        self.logger.info(f"Training on task: {details.get('task_id')}")
        # In production: download model, train, upload update

    def _handle_verification(self, details: Dict[str, Any]) -> None:
        """Handle a verification task."""
        self.logger.info(f"Verifying solution: {details.get('solution_cid')}")
        # In production: download solution, verify against ground truth

    def _handle_aggregation(self, details: Dict[str, Any]) -> None:
        """Handle model aggregation."""
        self.logger.info(f"Aggregating updates for project: {details.get('project_id')}")
        # In production: download verified updates, perform FedAvg

    def _handle_inference(self, details: Dict[str, Any]) -> None:
        """Handle an inference request."""
        self.logger.info(f"Serving inference: {details.get('request_id')}")
        # In production: load model, run inference, return result


class SurvivalEngine(BaseEngine):
    """
    The node's duty to ensure its own persistence and spread.
    Handles replication and network maintenance.
    """

    def __init__(self, node: "Node"):
        super().__init__(node)
        self.replication_threshold = 0.8  # Replicate when network coverage < 80%

    def tick(self) -> None:
        self.maintain_presence()
        self.consider_replication()

    def maintain_presence(self) -> None:
        """Maintain node's presence in the network."""
        # In production: send heartbeat, update DHT, maintain connections
        pass

    def consider_replication(self) -> None:
        """Consider whether to spawn new nodes."""
        # In production: check network coverage, spawn spores if needed
        pass
