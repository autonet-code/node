"""Tests for Story 4.2 — Training Attestor.

Verifies that:
- TrainingAttestor initializes in offline and online modes
- record_cycle() accumulates batches and triggers auto-flush
- flush() converts batches to units and calls GovernanceBridge
- Offline mode records locally without on-chain calls
- Status reporting includes attestation history
- Integration with TrainingDataFeed works end-to-end
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nodes.common.training_attestor import (
    TrainingAttestor,
    TrainingAttestationBatch,
    TrainingCycleRecord,
)


class TestTrainingAttestorInit(unittest.TestCase):
    """Test attestor initialization."""

    def test_offline_mode(self):
        attestor = TrainingAttestor()
        self.assertIsNone(attestor.governance)
        self.assertEqual(attestor._total_attested_units, 0)

    def test_online_mode(self):
        gov = MagicMock()
        attestor = TrainingAttestor(governance=gov)
        self.assertEqual(attestor.governance, gov)

    def test_custom_params(self):
        attestor = TrainingAttestor(
            batches_per_unit=5,
            flush_threshold_units=10,
            max_batch_age_seconds=600,
        )
        self.assertEqual(attestor.batches_per_unit, 5)
        self.assertEqual(attestor.flush_threshold_units, 10)
        self.assertEqual(attestor.max_batch_age_seconds, 600)


class TestTrainingAttestorRecording(unittest.TestCase):
    """Test cycle recording."""

    def test_record_increments_counters(self):
        attestor = TrainingAttestor(flush_threshold_units=100)  # High threshold
        metrics = {"num_batches": 5, "loss": 0.5, "architecture": "text_jepa"}

        attestor.record_cycle(metrics, cycle_number=1)

        self.assertEqual(attestor._total_cycles, 1)
        self.assertEqual(attestor._total_batches_trained, 5)
        self.assertEqual(attestor._current_batch.total_batches, 5)

    def test_multiple_records_accumulate(self):
        attestor = TrainingAttestor(flush_threshold_units=100)

        for i in range(3):
            attestor.record_cycle({"num_batches": 10, "loss": 0.3}, cycle_number=i+1)

        self.assertEqual(attestor._total_cycles, 3)
        self.assertEqual(attestor._total_batches_trained, 30)
        self.assertEqual(attestor._current_batch.total_batches, 30)


class TestTrainingAttestorFlush(unittest.TestCase):
    """Test flush mechanics."""

    def test_flush_empty_succeeds(self):
        attestor = TrainingAttestor()
        self.assertTrue(attestor.flush())

    def test_flush_offline_records_locally(self):
        """Offline mode: flush succeeds and records in history."""
        attestor = TrainingAttestor(batches_per_unit=5, flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 15, "loss": 0.4})
        result = attestor.flush()

        self.assertTrue(result)
        self.assertEqual(attestor._total_attested_units, 3)  # 15 / 5 = 3 units
        self.assertEqual(attestor._attestation_count, 1)
        self.assertEqual(len(attestor._attestation_history), 1)
        self.assertFalse(attestor._attestation_history[0]["on_chain"])

    def test_flush_online_calls_governance(self):
        """Online mode: flush calls GovernanceBridge.attest_task_completion()."""
        gov = MagicMock()
        gov.attest_task_completion.return_value = True

        attestor = TrainingAttestor(governance=gov, batches_per_unit=10,
                                    flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 25, "loss": 0.3})
        result = attestor.flush()

        self.assertTrue(result)
        gov.attest_task_completion.assert_called_once_with(units=3)  # ceil(25/10) = 3
        self.assertEqual(attestor._total_attested_units, 3)

    def test_flush_on_chain_failure_retains_batch(self):
        """Failed attestation retains batch for retry."""
        gov = MagicMock()
        gov.attest_task_completion.return_value = False

        attestor = TrainingAttestor(governance=gov, batches_per_unit=10,
                                    flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 20, "loss": 0.3})
        result = attestor.flush()

        self.assertFalse(result)
        self.assertEqual(attestor._total_attested_units, 0)
        self.assertEqual(attestor._current_batch.total_batches, 20)  # Retained

    def test_flush_resets_batch_on_success(self):
        """Successful flush resets the current batch."""
        attestor = TrainingAttestor(batches_per_unit=5, flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 10, "loss": 0.4})
        attestor.flush()

        self.assertEqual(attestor._current_batch.total_batches, 0)
        self.assertEqual(attestor._current_batch.record_count, 0)


class TestTrainingAttestorAutoFlush(unittest.TestCase):
    """Test auto-flush triggers."""

    def test_auto_flush_on_threshold(self):
        """Auto-flushes when pending units >= threshold."""
        gov = MagicMock()
        gov.attest_task_completion.return_value = True

        attestor = TrainingAttestor(
            governance=gov,
            batches_per_unit=5,
            flush_threshold_units=2,  # Flush at 2 units (10 batches)
        )

        # Record 10 batches → 2 units → should auto-flush
        attestor.record_cycle({"num_batches": 10, "loss": 0.3})

        gov.attest_task_completion.assert_called_once_with(units=2)
        self.assertEqual(attestor._total_attested_units, 2)

    def test_no_auto_flush_below_threshold(self):
        """No auto-flush when pending units < threshold."""
        gov = MagicMock()
        gov.attest_task_completion.return_value = True

        attestor = TrainingAttestor(
            governance=gov,
            batches_per_unit=10,
            flush_threshold_units=5,  # Threshold: 50 batches
        )

        attestor.record_cycle({"num_batches": 8, "loss": 0.3})

        gov.attest_task_completion.assert_not_called()


class TestTrainingAttestorUnitConversion(unittest.TestCase):
    """Test batch-to-unit conversion."""

    def test_exact_conversion(self):
        attestor = TrainingAttestor(batches_per_unit=10, flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 20, "loss": 0.3})
        attestor.flush()

        self.assertEqual(attestor._total_attested_units, 2)

    def test_rounds_up(self):
        """Partial units round up (don't lose fractional work)."""
        attestor = TrainingAttestor(batches_per_unit=10, flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 15, "loss": 0.3})
        attestor.flush()

        self.assertEqual(attestor._total_attested_units, 2)  # ceil(15/10) = 2

    def test_minimum_one_unit(self):
        """Even 1 batch produces at least 1 unit."""
        attestor = TrainingAttestor(batches_per_unit=100, flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 1, "loss": 0.3})
        attestor.flush()

        self.assertEqual(attestor._total_attested_units, 1)


class TestTrainingAttestorStatus(unittest.TestCase):
    """Test status reporting."""

    def test_initial_status(self):
        attestor = TrainingAttestor()
        status = attestor.get_status()

        self.assertEqual(status["total_attested_units"], 0)
        self.assertEqual(status["total_batches_trained"], 0)
        self.assertEqual(status["total_cycles"], 0)
        self.assertEqual(status["attestation_count"], 0)
        self.assertFalse(status["governance_available"])

    def test_status_after_training(self):
        gov = MagicMock()
        gov.attest_task_completion.return_value = True

        attestor = TrainingAttestor(governance=gov, batches_per_unit=5,
                                    flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 10, "loss": 0.3})
        attestor.flush()

        status = attestor.get_status()
        self.assertEqual(status["total_attested_units"], 2)
        self.assertEqual(status["total_batches_trained"], 10)
        self.assertEqual(status["total_cycles"], 1)
        self.assertTrue(status["governance_available"])
        self.assertGreater(len(status["recent_attestations"]), 0)


class TestTrainingAttestorWithGovernanceBridge(unittest.TestCase):
    """Test with a mock GovernanceBridge that simulates real behavior."""

    def test_governance_exception_handled(self):
        """GovernanceBridge throwing an exception is handled gracefully."""
        gov = MagicMock()
        gov.attest_task_completion.side_effect = RuntimeError("chain down")

        attestor = TrainingAttestor(governance=gov, batches_per_unit=5,
                                    flush_threshold_units=100)

        attestor.record_cycle({"num_batches": 10, "loss": 0.3})
        result = attestor.flush()

        self.assertFalse(result)
        self.assertEqual(attestor._total_attested_units, 0)
        # Batch retained for retry
        self.assertEqual(attestor._current_batch.total_batches, 10)


class TestTrainingFeedAttestorIntegration(unittest.TestCase):
    """Test that TrainingDataFeed calls the attestor after training."""

    def _create_data(self, tmpdir: str) -> None:
        conv_dir = Path(tmpdir) / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        turns = [
            {"role": "user", "content": "Hello world",
             "timestamp": "2026-03-26T10:00:00+00:00"},
            {"role": "assistant", "content": "Hi there!",
             "timestamp": "2026-03-26T10:00:01+00:00"},
        ]
        for i in range(5):
            with open(conv_dir / f"s{i}.jsonl", "w") as f:
                for t in turns:
                    f.write(json.dumps(t) + "\n")

    def test_training_feed_attests_after_cycle(self):
        """TrainingDataFeed records attestation after each training cycle."""
        from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_data(tmpdir)

            gov = MagicMock()
            gov.attest_task_completion.return_value = True

            config = TrainingFeedConfig(
                data_dir=tmpdir,
                min_events_for_cycle=1,
                min_cycle_interval=0,
                vocab_size=260,
                max_seq_length=64,
                embed_dim=64,
                num_heads=4,
                encoder_depth=2,
                predictor_depth=1,
                predictor_embed_dim=32,
                batch_size=2,
                epochs=1,
                scrub_pii=False,
            )
            feed = TrainingDataFeed(config, governance=gov)
            feed._pending_events = 5

            result = feed.run_cycle()

            self.assertIsNotNone(result)
            # Attestor should have recorded the cycle
            self.assertEqual(feed._attestor._total_cycles, 1)
            self.assertGreater(feed._attestor._total_batches_trained, 0)

    def test_training_feed_attestor_status_in_feed_status(self):
        """TrainingDataFeed status includes attestation info."""
        from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig

        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        status = feed.get_status()
        self.assertIn("attestation", status)
        self.assertEqual(status["attestation"]["total_attested_units"], 0)

    def test_training_feed_offline_attestation(self):
        """Training works without governance (offline attestation)."""
        from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_data(tmpdir)

            config = TrainingFeedConfig(
                data_dir=tmpdir,
                min_events_for_cycle=1,
                min_cycle_interval=0,
                vocab_size=260,
                max_seq_length=64,
                embed_dim=64,
                num_heads=4,
                encoder_depth=2,
                predictor_depth=1,
                predictor_embed_dim=32,
                batch_size=2,
                epochs=1,
                scrub_pii=False,
            )
            # No governance → offline mode
            feed = TrainingDataFeed(config)
            feed._pending_events = 5

            result = feed.run_cycle()
            self.assertIsNotNone(result)
            # Attestor still records locally
            self.assertEqual(feed._attestor._total_cycles, 1)


if __name__ == "__main__":
    unittest.main()
