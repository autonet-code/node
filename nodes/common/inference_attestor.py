"""
Inference Attestor — bridges centralized inference usage to attestUsage().

When the daemon routes requests to centralized providers (Claude, GPT, etc.),
the InferenceAttestor tracks token consumption and attests it on-chain so
that Scenario B inference still participates in the economic loop.

Usage:
    attestor = InferenceAttestor(governance_bridge, blob_store)
    attestor.record_inference(provider="claude", input_tokens=1200, output_tokens=400)
    attestor.flush()  # Attest accumulated usage on-chain
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """A single inference call record."""
    provider: str
    input_tokens: int
    output_tokens: int
    model: str = ""
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AttestationBatch:
    """Accumulated records ready for on-chain attestation."""
    records: List[InferenceRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def record_count(self) -> int:
        return len(self.records)

    def add(self, record: InferenceRecord):
        self.records.append(record)
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens


class InferenceAttestor:
    """
    Tracks centralized inference usage and attests it on-chain.

    In Scenario B (centralized inference), the daemon routes requests to
    Claude, GPT, or other providers. The attestor:

    1. Records each inference call (provider, tokens, model)
    2. Accumulates usage into batches
    3. Flushes batches to on-chain attestation via GovernanceBridge
    4. Stores attestation receipts in blob store for auditability

    Token-to-unit conversion:
        On-chain attestation uses abstract "units" (not raw tokens).
        Default: 1 unit = 1000 tokens. Configurable via tokens_per_unit.

    Flush triggers:
        - Manual: call flush()
        - Threshold: auto-flush when accumulated tokens exceed threshold
        - Time: auto-flush when batch age exceeds max_batch_age_seconds
    """

    def __init__(
        self,
        governance,
        store=None,
        tokens_per_unit: int = 1000,
        flush_threshold: int = 10000,
        max_batch_age_seconds: float = 300.0,
    ):
        """
        Args:
            governance: GovernanceBridge instance for on-chain attestation
            store: BlobStore for storing attestation receipts (optional)
            tokens_per_unit: Number of tokens per on-chain usage unit
            flush_threshold: Auto-flush when tokens exceed this threshold
            max_batch_age_seconds: Auto-flush when batch is older than this
        """
        self.governance = governance
        self.store = store
        self.tokens_per_unit = tokens_per_unit
        self.flush_threshold = flush_threshold
        self.max_batch_age_seconds = max_batch_age_seconds

        self._current_batch = AttestationBatch()
        self._provider_totals: Dict[str, int] = {}
        self._total_attested_units: int = 0
        self._total_attested_tokens: int = 0
        self._attestation_count: int = 0

        self.logger = logging.getLogger(
            f"InferenceAttestor[{governance.node_id}]"
        )

    def record_inference(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        request_id: Optional[str] = None,
    ) -> InferenceRecord:
        """
        Record an inference call.

        Args:
            provider: Provider name (e.g., "claude", "gpt", "gemini")
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
            model: Model identifier (e.g., "claude-sonnet-4")
            request_id: Optional request ID for tracing

        Returns:
            The recorded InferenceRecord.
        """
        record = InferenceRecord(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            request_id=request_id,
        )

        self._current_batch.add(record)
        self._provider_totals[provider] = (
            self._provider_totals.get(provider, 0) + record.total_tokens
        )

        self.logger.debug(
            f"Recorded {provider}/{model}: {input_tokens}+{output_tokens} tokens "
            f"(batch total: {self._current_batch.total_tokens})"
        )

        # Check auto-flush triggers
        self._check_auto_flush()

        return record

    def flush(self) -> bool:
        """
        Flush accumulated usage to on-chain attestation.

        Converts accumulated tokens to usage units and calls
        GovernanceBridge.attest_task_completion().

        Returns:
            True if attestation succeeded (or nothing to flush).
        """
        batch = self._current_batch

        if batch.total_tokens == 0:
            return True  # Nothing to flush

        # Convert tokens to units (round up to not lose fractional units)
        units = max(1, (batch.total_tokens + self.tokens_per_unit - 1) // self.tokens_per_unit)

        self.logger.info(
            f"Flushing batch: {batch.total_tokens} tokens "
            f"({batch.record_count} records) -> {units} unit(s)"
        )

        # Store receipt in blob store (if available)
        receipt_cid = None
        if self.store:
            receipt = {
                "type": "inference_attestation",
                "total_input_tokens": batch.total_input_tokens,
                "total_output_tokens": batch.total_output_tokens,
                "total_tokens": batch.total_tokens,
                "units_attested": units,
                "record_count": batch.record_count,
                "provider_breakdown": self._batch_provider_breakdown(batch),
                "batch_start": batch.start_time,
                "batch_end": time.time(),
                "node_id": self.governance.node_id,
            }
            try:
                receipt_cid = self.store.add_json(receipt)
                self.logger.debug(f"Attestation receipt stored: {receipt_cid}")
            except Exception as e:
                self.logger.warning(f"Failed to store receipt: {e}")

        # Attest on-chain
        success = self.governance.attest_task_completion(units=units)

        if success:
            self._total_attested_units += units
            self._total_attested_tokens += batch.total_tokens
            self._attestation_count += 1
            self._current_batch = AttestationBatch()
            self.logger.info(
                f"Attested {units} unit(s) on-chain "
                f"(lifetime: {self._total_attested_units} units, "
                f"{self._total_attested_tokens} tokens)"
            )
        else:
            self.logger.warning(
                f"On-chain attestation failed for {units} units. "
                f"Batch retained for retry."
            )

        return success

    def _check_auto_flush(self):
        """Check if auto-flush should be triggered."""
        batch = self._current_batch

        # Threshold trigger
        if batch.total_tokens >= self.flush_threshold:
            self.logger.info(
                f"Auto-flush: threshold reached "
                f"({batch.total_tokens} >= {self.flush_threshold})"
            )
            self.flush()
            return

        # Time trigger
        age = time.time() - batch.start_time
        if age >= self.max_batch_age_seconds and batch.total_tokens > 0:
            self.logger.info(
                f"Auto-flush: batch age {age:.0f}s >= {self.max_batch_age_seconds}s"
            )
            self.flush()

    def _batch_provider_breakdown(self, batch: AttestationBatch) -> Dict[str, int]:
        """Compute per-provider token totals for a batch."""
        breakdown: Dict[str, int] = {}
        for record in batch.records:
            breakdown[record.provider] = (
                breakdown.get(record.provider, 0) + record.total_tokens
            )
        return breakdown

    # =========================================================================
    # Stats / Introspection
    # =========================================================================

    @property
    def pending_tokens(self) -> int:
        """Tokens accumulated but not yet attested."""
        return self._current_batch.total_tokens

    @property
    def pending_records(self) -> int:
        """Number of records not yet attested."""
        return self._current_batch.record_count

    @property
    def lifetime_attested_units(self) -> int:
        """Total units attested on-chain since creation."""
        return self._total_attested_units

    @property
    def lifetime_attested_tokens(self) -> int:
        """Total tokens attested since creation."""
        return self._total_attested_tokens

    @property
    def provider_totals(self) -> Dict[str, int]:
        """Lifetime token totals per provider."""
        return dict(self._provider_totals)

    def summary(self) -> Dict:
        """Return a summary dict for dashboard/API display."""
        return {
            "pending_tokens": self.pending_tokens,
            "pending_records": self.pending_records,
            "lifetime_attested_units": self._total_attested_units,
            "lifetime_attested_tokens": self._total_attested_tokens,
            "attestation_count": self._attestation_count,
            "provider_totals": self.provider_totals,
            "tokens_per_unit": self.tokens_per_unit,
        }
