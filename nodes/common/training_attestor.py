"""
Training Attestor — bridges training cycles to on-chain usage attestation.

After each training cycle completes, the attestor converts training metrics
(batches processed, loss improvement, etc.) into abstract "usage units"
and attests them on-chain via GovernanceBridge.attest_task_completion().

This is the training-side analog of InferenceAttestor (which attests
centralized inference usage). Together, they close the economic loop:

    Agent activity → Training → attestUsage() → epoch rewards
    Agent activity → Inference → attestUsage() → epoch rewards

Usage:
    attestor = TrainingAttestor(governance_bridge)
    # After each training cycle:
    attestor.record_cycle(metrics)
    attestor.flush()  # or auto-flushes when threshold is met

Story 4.2 (BACKLOG_TRAINING_DATA.md)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingCycleRecord:
    """A single training cycle record."""
    cycle_number: int
    num_batches: int
    loss: float
    architecture: str = ""
    modalities: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    weight_delta_keys: int = 0  # Number of weight parameters updated


@dataclass
class TrainingAttestationBatch:
    """Accumulated training cycles ready for attestation."""
    records: List[TrainingCycleRecord] = field(default_factory=list)
    total_batches: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def add(self, record: TrainingCycleRecord) -> None:
        self.records.append(record)
        self.total_batches += record.num_batches


class TrainingAttestor:
    """Tracks training cycles and attests usage on-chain.

    Converts training work into abstract "units" for on-chain attestation:
    - Default: 1 unit per batch trained (configurable)
    - Accumulated into batches to reduce transaction frequency
    - Flushes to GovernanceBridge when threshold is met

    Works without a chain connection — gracefully skips attestation
    while still tracking local metrics.
    """

    def __init__(
        self,
        governance=None,
        batches_per_unit: int = 10,
        flush_threshold_units: int = 1,
        max_batch_age_seconds: float = 300.0,
    ):
        """
        Args:
            governance: GovernanceBridge instance (None = offline mode).
            batches_per_unit: Number of training batches per attestation unit.
            flush_threshold_units: Auto-flush when accumulated units >= this.
            max_batch_age_seconds: Auto-flush when batch age exceeds this.
        """
        self.governance = governance
        self.batches_per_unit = batches_per_unit
        self.flush_threshold_units = flush_threshold_units
        self.max_batch_age_seconds = max_batch_age_seconds

        # Current accumulation batch
        self._current_batch = TrainingAttestationBatch()

        # Lifetime stats
        self._total_attested_units: int = 0
        self._total_batches_trained: int = 0
        self._total_cycles: int = 0
        self._attestation_count: int = 0
        self._last_flush_time: float = 0.0

        # History (last N attestations for status reporting)
        self._attestation_history: List[Dict[str, Any]] = []
        self._max_history: int = 50

    def record_cycle(self, metrics: Dict[str, Any], cycle_number: int = 0) -> None:
        """Record a completed training cycle.

        Args:
            metrics: Training metrics from train_vljepa_on_task().
            cycle_number: Sequential cycle number.
        """
        record = TrainingCycleRecord(
            cycle_number=cycle_number or (self._total_cycles + 1),
            num_batches=metrics.get("num_batches", 0),
            loss=metrics.get("loss", 0.0),
            architecture=metrics.get("architecture", ""),
            modalities=metrics.get("modalities", []),
            weight_delta_keys=metrics.get("weight_delta_keys", 0),
        )

        self._current_batch.add(record)
        self._total_batches_trained += record.num_batches
        self._total_cycles += 1

        logger.debug(
            "Training attestor: recorded cycle %d (%d batches, loss=%.4f)",
            record.cycle_number, record.num_batches, record.loss,
        )

        # Check auto-flush conditions
        pending_units = self._compute_pending_units()
        batch_age = time.time() - self._current_batch.start_time

        if pending_units >= self.flush_threshold_units:
            logger.debug("Auto-flushing: %d units >= threshold %d",
                        pending_units, self.flush_threshold_units)
            self.flush()
        elif batch_age > self.max_batch_age_seconds and pending_units > 0:
            logger.debug("Auto-flushing: batch age %.0fs > max %.0fs",
                        batch_age, self.max_batch_age_seconds)
            self.flush()

    def flush(self) -> bool:
        """Flush accumulated training usage to on-chain attestation.

        Returns:
            True if attestation succeeded (or nothing to flush / offline).
        """
        batch = self._current_batch

        if batch.total_batches == 0:
            return True  # Nothing to flush

        # Convert batches to units (round up)
        units = max(1, (batch.total_batches + self.batches_per_unit - 1) // self.batches_per_unit)

        logger.info(
            "Training attestor: flushing %d batches (%d cycles) → %d unit(s)",
            batch.total_batches, batch.record_count, units,
        )

        # Attempt on-chain attestation
        success = self._attest_on_chain(units)

        if success:
            self._total_attested_units += units
            self._attestation_count += 1
            self._last_flush_time = time.time()

            # Record in history
            self._attestation_history.append({
                "units": units,
                "batches": batch.total_batches,
                "cycles": batch.record_count,
                "timestamp": time.time(),
                "on_chain": self.governance is not None,
            })
            if len(self._attestation_history) > self._max_history:
                self._attestation_history = self._attestation_history[-self._max_history:]

            # Reset batch
            self._current_batch = TrainingAttestationBatch()

            logger.info(
                "Training attestor: attested %d unit(s) "
                "(lifetime: %d units, %d batches, %d cycles)",
                units, self._total_attested_units,
                self._total_batches_trained, self._total_cycles,
            )
        else:
            logger.warning(
                "Training attestor: on-chain attestation failed for %d units. "
                "Batch retained for retry.",
                units,
            )

        return success

    def _attest_on_chain(self, units: int) -> bool:
        """Attempt to attest usage on-chain via GovernanceBridge.

        Returns True if:
        - No governance bridge (offline mode, always succeeds)
        - GovernanceBridge.attest_task_completion() succeeds
        """
        if self.governance is None:
            # Offline mode — record locally, skip on-chain
            logger.debug("Training attestor: offline mode, skipping on-chain attestation")
            return True

        try:
            return self.governance.attest_task_completion(units=units)
        except Exception as e:
            logger.warning("Training attestor: attestation error: %s", e)
            return False

    def _compute_pending_units(self) -> int:
        """Compute how many units the current batch represents."""
        if self._current_batch.total_batches == 0:
            return 0
        return max(1, self._current_batch.total_batches // self.batches_per_unit)

    def get_status(self) -> Dict[str, Any]:
        """Get current attestor status for monitoring."""
        return {
            "total_attested_units": self._total_attested_units,
            "total_batches_trained": self._total_batches_trained,
            "total_cycles": self._total_cycles,
            "attestation_count": self._attestation_count,
            "pending_batches": self._current_batch.total_batches,
            "pending_units": self._compute_pending_units(),
            "last_flush_time": self._last_flush_time,
            "governance_available": self.governance is not None,
            "recent_attestations": self._attestation_history[-5:],
        }
