"""Tests for Story 2.2 — train_vljepa_on_task() unified training function.

Verifies that:
- Text-only training works end-to-end with TextTrainingDataSource
- Visual-only training works with image batches
- Multimodal training works with paired text+visual data
- Weight deltas are non-zero and exclude target encoder
- Metrics are returned in the expected format
- Modality auto-detection works correctly
- FedAvg-compatible output format
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from nodes.common.ml import train_vljepa_on_task
from nodes.common.text_data import TextDataConfig, TextTrainingDataSource


def _make_conversation_dir(tmpdir: str, num_conversations: int = 3) -> str:
    """Create synthetic conversation JSONL files for testing."""
    conv_dir = Path(tmpdir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    turns = [
        {"role": "user", "content": "Schedule a meeting with the design team",
         "timestamp": "2026-03-26T10:00:00+00:00"},
        {"role": "assistant", "content": "I've scheduled the meeting for Thursday at 2pm.",
         "timestamp": "2026-03-26T10:00:02+00:00"},
        {"role": "user", "content": "Also remind me to review the Q1 report",
         "timestamp": "2026-03-26T10:00:05+00:00"},
        {"role": "assistant", "content": "Reminder set for Thursday evening.",
         "timestamp": "2026-03-26T10:00:06+00:00"},
    ]
    for i in range(num_conversations):
        with open(conv_dir / f"session_{i}.jsonl", "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")

    return str(conv_dir)


def _make_text_source(tmpdir: str, batch_size: int = 2) -> TextTrainingDataSource:
    """Create a TextTrainingDataSource with synthetic data."""
    conv_dir = _make_conversation_dir(tmpdir, num_conversations=5)
    config = TextDataConfig(
        conversations_dir=conv_dir,
        max_seq_length=128,
        batch_size=batch_size,
        scrub_pii=False,
        shuffle=False,
    )
    return TextTrainingDataSource(config)


def _make_visual_source(num_batches: int = 3, batch_size: int = 2, image_size: int = 32):
    """Create a simple visual data source (list of image tensors)."""
    return [torch.randn(batch_size, 3, image_size, image_size) for _ in range(num_batches)]


class TestTrainVLJEPATextOnly(unittest.TestCase):
    """Test text-only training path."""

    def test_text_only_training(self):
        """Text-only training produces weight deltas and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir)
            task_spec = {
                "vocab_size": 260,
                "max_seq_length": 128,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
            }

            weight_delta, metrics = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                epochs=1,
                learning_rate=1e-3,
            )

            # Weight delta should be non-empty
            self.assertGreater(len(weight_delta), 0)

            # No target encoder weights in delta
            for key in weight_delta:
                self.assertNotIn("target_encoder", key)

            # Metrics should have expected fields
            self.assertEqual(metrics["architecture"], "text_jepa")
            self.assertEqual(metrics["modalities"], ["text"])
            self.assertTrue(metrics["self_supervised"])
            self.assertGreater(metrics["num_batches"], 0)
            self.assertIn("loss", metrics)
            self.assertIn("cosine_similarity", metrics)

    def test_auto_detects_text_modality(self):
        """Modality auto-detection picks text when only text source given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir)
            task_spec = {
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
            }

            _, metrics = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                epochs=1,
            )

            self.assertEqual(metrics["modalities"], ["text"])
            self.assertEqual(metrics["architecture"], "text_jepa")

    def test_weight_delta_nonzero(self):
        """Weight deltas are actually non-zero (real learning happened)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir)
            task_spec = {
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
            }

            weight_delta, _ = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                epochs=1,
            )

            total_delta = 0.0
            for v in weight_delta.values():
                total_delta += sum(abs(x) for x in _flatten(v))
            self.assertGreater(total_delta, 0, "Weight deltas are all zero")


class TestTrainVLJEPAVisualOnly(unittest.TestCase):
    """Test visual-only training path."""

    def test_visual_only_training(self):
        """Visual-only training produces weight deltas and metrics."""
        visual_source = _make_visual_source(num_batches=3, image_size=32)
        task_spec = {
            "image_size": 32,
            "patch_size": 4,
            "embed_dim": 64,
            "num_heads": 2,
            "encoder_depth": 2,
            "predictor_depth": 1,
            "predictor_embed_dim": 32,
        }

        weight_delta, metrics = train_vljepa_on_task(
            task_spec=task_spec,
            visual_data_source=visual_source,
            epochs=1,
            learning_rate=1e-3,
        )

        self.assertGreater(len(weight_delta), 0)
        for key in weight_delta:
            self.assertNotIn("target_encoder", key)

        self.assertEqual(metrics["architecture"], "jepa")
        self.assertEqual(metrics["modalities"], ["visual"])
        self.assertGreater(metrics["num_batches"], 0)


class TestTrainVLJEPAMultimodal(unittest.TestCase):
    """Test multimodal training path."""

    def test_multimodal_training(self):
        """Multimodal training runs VL-JEPA with both text and visual data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir, batch_size=2)
            visual_source = _make_visual_source(num_batches=3, batch_size=2, image_size=32)
            task_spec = {
                "image_size": 32,
                "patch_size": 4,
                "embed_dim": 64,
                "num_heads": 2,
                "encoder_depth": 2,
                "vocab_size": 260,
                "max_seq_length": 128,
                "text_encoder_depth": 2,
                "fusion_depth": 2,
            }

            weight_delta, metrics = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                visual_data_source=visual_source,
                epochs=1,
                learning_rate=1e-3,
            )

            self.assertGreater(len(weight_delta), 0)

            # Multimodal should not include target_encoder or text_decoder
            for key in weight_delta:
                self.assertNotIn("target_encoder", key)
                self.assertNotIn("text_decoder", key)

            self.assertEqual(metrics["architecture"], "vl_jepa")
            self.assertIn("text", metrics["modalities"])
            self.assertIn("visual", metrics["modalities"])
            self.assertGreater(metrics["num_batches"], 0)

    def test_explicit_modality_override(self):
        """Explicit modalities parameter overrides auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir)
            visual_source = _make_visual_source(num_batches=2, batch_size=2, image_size=32)
            task_spec = {
                "image_size": 32,
                "patch_size": 4,
                "embed_dim": 64,
                "num_heads": 2,
                "encoder_depth": 2,
                "vocab_size": 260,
                "max_seq_length": 128,
                "text_encoder_depth": 2,
                "fusion_depth": 2,
            }

            _, metrics = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                visual_data_source=visual_source,
                modalities=["text", "visual"],
                epochs=1,
            )

            self.assertEqual(metrics["architecture"], "vl_jepa")


class TestTrainVLJEPAFedAvgCompat(unittest.TestCase):
    """Test FedAvg compatibility of output format."""

    def test_weight_delta_format(self):
        """Weight deltas are serializable (lists of numbers, not tensors)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_source = _make_text_source(tmpdir)
            task_spec = {
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
            }

            weight_delta, _ = train_vljepa_on_task(
                task_spec=task_spec,
                text_data_source=text_source,
                epochs=1,
            )

            # All values should be lists (JSON-serializable), not tensors
            for key, value in weight_delta.items():
                self.assertIsInstance(value, list, f"Weight delta {key} is not a list")

    def test_fallback_on_no_data(self):
        """Falls back to mock result when no data sources provided."""
        task_spec = {"task_id": 42}

        weight_delta, metrics = train_vljepa_on_task(
            task_spec=task_spec,
            modalities=["text"],
        )

        # Should get mock result since no data source
        self.assertTrue(metrics.get("mock", False) or metrics.get("num_batches", -1) == 0)


# Helper for nested list flattening
def _flatten(lst):
    """Flatten a nested list of numbers."""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


if __name__ == "__main__":
    unittest.main()
