"""
Tests for InferenceAttestor — bridges centralized inference to attestUsage().

Run: python -m pytest tests/test_inference_attestor.py -v
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.common.blockchain import TransactionResult
from nodes.common.blob_store import BlobStore
from nodes.common.inference_attestor import (
    InferenceAttestor,
    InferenceRecord,
    AttestationBatch,
)


# =============================================================================
# Helpers
# =============================================================================


def _ok():
    return TransactionResult(success=True, tx_hash="0xabc")


def _fail(msg="reverted"):
    return TransactionResult(success=False, error=msg)


class FakeContractHandle:
    def __init__(self, name, address):
        self.name = name
        self.address = address


def make_governance(attest_ok=True):
    """Create a mock GovernanceBridge for attestor tests."""
    gov = MagicMock()
    gov.node_id = "inference-0"
    gov.attest_task_completion = MagicMock(return_value=attest_ok)
    return gov


@pytest.fixture
def blob_store():
    tmpdir = tempfile.mkdtemp(prefix="autonet-test-attestor-")
    return BlobStore(data_dir=tmpdir)


@pytest.fixture
def governance():
    return make_governance(attest_ok=True)


# =============================================================================
# Tests: InferenceRecord
# =============================================================================


class TestInferenceRecord:
    def test_total_tokens(self):
        """total_tokens is sum of input + output."""
        record = InferenceRecord(
            provider="claude", input_tokens=1200, output_tokens=400
        )
        assert record.total_tokens == 1600

    def test_defaults(self):
        """Record has sensible defaults."""
        record = InferenceRecord(
            provider="gpt", input_tokens=100, output_tokens=50
        )
        assert record.model == ""
        assert record.request_id is None
        assert record.timestamp > 0


# =============================================================================
# Tests: AttestationBatch
# =============================================================================


class TestAttestationBatch:
    def test_empty_batch(self):
        batch = AttestationBatch()
        assert batch.total_tokens == 0
        assert batch.record_count == 0

    def test_add_records(self):
        batch = AttestationBatch()
        batch.add(InferenceRecord("claude", 1000, 200))
        batch.add(InferenceRecord("gpt", 500, 100))
        assert batch.total_tokens == 1800
        assert batch.total_input_tokens == 1500
        assert batch.total_output_tokens == 300
        assert batch.record_count == 2


# =============================================================================
# Tests: InferenceAttestor — Recording
# =============================================================================


class TestAttestorRecording:
    def test_record_inference(self, governance):
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        record = attestor.record_inference(
            provider="claude", input_tokens=1200, output_tokens=400,
            model="sonnet-4", request_id="req-1"
        )

        assert record.total_tokens == 1600
        assert attestor.pending_tokens == 1600
        assert attestor.pending_records == 1
        assert attestor.provider_totals["claude"] == 1600

    def test_multiple_records_accumulate(self, governance):
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 200)
        attestor.record_inference("gpt", 800, 300)
        attestor.record_inference("claude", 500, 100)

        assert attestor.pending_tokens == 2900
        assert attestor.pending_records == 3
        assert attestor.provider_totals["claude"] == 1800
        assert attestor.provider_totals["gpt"] == 1100

    def test_record_returns_record_object(self, governance):
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        record = attestor.record_inference("claude", 100, 50, model="haiku")
        assert isinstance(record, InferenceRecord)
        assert record.model == "haiku"


# =============================================================================
# Tests: InferenceAttestor — Flushing
# =============================================================================


class TestAttestorFlushing:
    def test_flush_converts_tokens_to_units(self, governance):
        """Tokens are converted to units at the configured rate."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 2000, 500)  # 2500 tokens
        success = attestor.flush()

        assert success is True
        governance.attest_task_completion.assert_called_once_with(units=3)  # ceil(2500/1000)
        assert attestor.pending_tokens == 0
        assert attestor.lifetime_attested_units == 3
        assert attestor.lifetime_attested_tokens == 2500

    def test_flush_rounds_up(self, governance):
        """Fractional units round up (don't lose work)."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 100, 50)  # 150 tokens -> 1 unit
        attestor.flush()

        governance.attest_task_completion.assert_called_once_with(units=1)

    def test_flush_empty_batch_noop(self, governance):
        """Flushing an empty batch is a no-op."""
        attestor = InferenceAttestor(governance, tokens_per_unit=1000)
        success = attestor.flush()

        assert success is True
        governance.attest_task_completion.assert_not_called()

    def test_flush_failure_retains_batch(self, governance):
        """Failed attestation keeps the batch for retry."""
        governance.attest_task_completion.return_value = False
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 2000, 500)
        success = attestor.flush()

        assert success is False
        assert attestor.pending_tokens == 2500  # Batch retained
        assert attestor.lifetime_attested_units == 0

    def test_flush_resets_batch(self, governance):
        """Successful flush resets the batch."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 500)
        attestor.flush()

        assert attestor.pending_tokens == 0
        assert attestor.pending_records == 0

        # Second flush is a no-op
        attestor.flush()
        assert governance.attest_task_completion.call_count == 1

    def test_multiple_flushes_accumulate_lifetime(self, governance):
        """Lifetime counters accumulate across flushes."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 500)
        attestor.flush()

        attestor.record_inference("gpt", 2000, 1000)
        attestor.flush()

        assert attestor.lifetime_attested_units == 2 + 3  # ceil(1500/1000) + ceil(3000/1000)
        assert attestor.lifetime_attested_tokens == 1500 + 3000


# =============================================================================
# Tests: InferenceAttestor — Auto-flush
# =============================================================================


class TestAttestorAutoFlush:
    def test_auto_flush_on_threshold(self, governance):
        """Auto-flush triggers when token threshold is reached."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=3000
        )

        # First record: under threshold
        attestor.record_inference("claude", 1000, 500)
        assert attestor.pending_tokens == 1500
        governance.attest_task_completion.assert_not_called()

        # Second record: exceeds threshold -> auto-flush
        attestor.record_inference("claude", 1500, 500)
        governance.attest_task_completion.assert_called_once()
        assert attestor.pending_tokens == 0

    def test_auto_flush_on_time(self, governance):
        """Auto-flush triggers when batch age exceeds max."""
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000,
            flush_threshold=100000,
            max_batch_age_seconds=0.1,  # 100ms for fast test
        )

        attestor.record_inference("claude", 500, 100)
        governance.attest_task_completion.assert_not_called()

        # Wait for batch to age
        time.sleep(0.15)

        # Next record should trigger time-based auto-flush
        attestor.record_inference("claude", 500, 100)
        governance.attest_task_completion.assert_called()


# =============================================================================
# Tests: InferenceAttestor — Blob Store Receipt
# =============================================================================


class TestAttestorReceipt:
    def test_stores_receipt_in_blob_store(self, governance, blob_store):
        """Attestation receipts are stored in blob store."""
        attestor = InferenceAttestor(
            governance, blob_store,
            tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 500, model="sonnet-4")
        attestor.record_inference("gpt", 800, 200, model="gpt-4o")
        attestor.flush()

        # Check that at least one blob was stored
        hashes = blob_store.list_hashes()
        assert len(hashes) >= 1

        # Check receipt content
        receipt = blob_store.get_json(hashes[0])
        assert receipt["type"] == "inference_attestation"
        assert receipt["total_tokens"] == 2500
        assert receipt["units_attested"] == 3
        assert receipt["record_count"] == 2
        assert "claude" in receipt["provider_breakdown"]
        assert "gpt" in receipt["provider_breakdown"]

    def test_works_without_blob_store(self, governance):
        """Attestor works without blob store (receipt storage is optional)."""
        attestor = InferenceAttestor(
            governance, store=None,
            tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 500)
        success = attestor.flush()

        assert success is True
        governance.attest_task_completion.assert_called_once()


# =============================================================================
# Tests: InferenceAttestor — Summary
# =============================================================================


class TestAttestorSummary:
    def test_summary(self, governance):
        attestor = InferenceAttestor(
            governance, tokens_per_unit=1000, flush_threshold=100000
        )

        attestor.record_inference("claude", 1000, 500)
        attestor.flush()
        attestor.record_inference("gpt", 200, 100)

        summary = attestor.summary()
        assert summary["pending_tokens"] == 300
        assert summary["lifetime_attested_tokens"] == 1500
        assert summary["lifetime_attested_units"] == 2  # ceil(1500/1000)
        assert summary["attestation_count"] == 1
        assert "claude" in summary["provider_totals"]
        assert "gpt" in summary["provider_totals"]
        assert summary["tokens_per_unit"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
