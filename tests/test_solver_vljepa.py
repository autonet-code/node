"""Tests for Story 3.4 — Solver VL-JEPA integration.

Verifies that:
- SolverNode dispatches to _train_vljepa for task_type "text_jepa" and "vl_jepa"
- _train_vljepa produces valid weight deltas and metrics
- _train_vljepa reads from ATN data directory when provided
- _train_vljepa falls back gracefully with no data
- Training output is FedAvg-compatible (dict of lists)
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _create_conversation_data(data_dir: str, num_sessions: int = 3) -> None:
    """Create synthetic JSONL conversation files."""
    conv_dir = Path(data_dir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    turns = [
        {"role": "user", "content": "What's the weather like?",
         "timestamp": "2026-03-26T10:00:00+00:00"},
        {"role": "assistant", "content": "It looks sunny today with a high of 72F.",
         "timestamp": "2026-03-26T10:00:02+00:00"},
        {"role": "user", "content": "Schedule a run for this afternoon",
         "timestamp": "2026-03-26T10:00:05+00:00"},
        {"role": "assistant", "content": "I've added a run at 4pm to your calendar.",
         "timestamp": "2026-03-26T10:00:06+00:00"},
    ]

    for i in range(num_sessions):
        with open(conv_dir / f"session_{i}.jsonl", "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")


def _create_execution_data(data_dir: str) -> None:
    """Create synthetic execution records."""
    agents_dir = Path(data_dir) / "agents" / "daily_planner"
    agents_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "agent_id": "daily_planner",
            "execution_id": "exec_1",
            "trigger_source": "scheduler",
            "status": "completed",
            "step_results": [
                {
                    "step_name": "plan",
                    "step_type": "cognitive",
                    "output": {
                        "text": "Reviewing today's tasks and priorities.",
                        "tool_calls": [],
                    },
                }
            ],
        }
    ]

    with open(agents_dir / "execution.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_solver(data_dir: str = ""):
    """Create a SolverNode with mocked chain dependencies."""
    from nodes.solver.main import SolverNode

    registry = MagicMock()
    store = MagicMock()

    # Mock store.get to return None (no cached model)
    store.get.return_value = None

    solver = SolverNode(
        registry=registry,
        store=store,
        node_id="test-solver-0",
        rpb_address="",
        task_type="text_jepa",
    )

    # Inject data_dir via config if provided
    if data_dir:
        solver.config.data_dir = data_dir

    return solver


class TestSolverVLJEPADispatch(unittest.TestCase):
    """Test that task_type dispatches correctly."""

    def test_text_jepa_dispatches(self):
        """task_type='text_jepa' routes to _train_vljepa."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)
            solver = _make_solver(data_dir=tmpdir)

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
            }

            model_update, checkpoints = solver._train(1, task_spec)

            self.assertEqual(model_update["task_type"], "text_jepa")
            self.assertTrue(model_update["self_supervised"])
            self.assertTrue(model_update["real_training"])
            self.assertGreater(len(model_update["weight_delta"]), 0)

    def test_vl_jepa_dispatches(self):
        """task_type='vl_jepa' also routes to _train_vljepa."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=3)
            solver = _make_solver(data_dir=tmpdir)

            task_spec = {
                "task_id": 2,
                "task_type": "vl_jepa",
                "data_dir": tmpdir,
                "modalities": ["text"],
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, checkpoints = solver._train(2, task_spec)

            self.assertEqual(model_update["task_type"], "vl_jepa")
            self.assertIn("text", model_update["metrics"]["modalities"])


class TestSolverVLJEPAOutput(unittest.TestCase):
    """Test training output format and content."""

    def test_weight_deltas_are_lists(self):
        """Weight deltas should be JSON-serializable (lists, not tensors)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, _ = solver._train(1, task_spec)

            for key, value in model_update["weight_delta"].items():
                self.assertIsInstance(value, list,
                                     f"Weight delta {key} should be a list")

    def test_metrics_have_required_fields(self):
        """Metrics should contain architecture, modalities, loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, _ = solver._train(1, task_spec)
            metrics = model_update["metrics"]

            self.assertEqual(metrics["architecture"], "text_jepa")
            self.assertEqual(metrics["modalities"], ["text"])
            self.assertTrue(metrics["self_supervised"])
            self.assertIn("loss", metrics)
            self.assertIn("num_batches", metrics)

    def test_no_target_encoder_in_deltas(self):
        """Target encoder weights should NOT be in weight deltas (FedAvg)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, _ = solver._train(1, task_spec)

            for key in model_update["weight_delta"]:
                self.assertNotIn("target_encoder", key,
                                 "Target encoder weights should not be in weight deltas")

    def test_checkpoints_generated(self):
        """Training should produce verification checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            _, checkpoints = solver._train(1, task_spec)
            self.assertGreater(len(checkpoints), 0)


class TestSolverVLJEPAWithExecutionData(unittest.TestCase):
    """Test with both conversation and execution data."""

    def test_trains_on_combined_data(self):
        """Solver should train on both conversations and execution records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=3)
            _create_execution_data(tmpdir)
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, _ = solver._train(1, task_spec)
            self.assertGreater(model_update["metrics"]["num_batches"], 0)


class TestSolverVLJEPAFallback(unittest.TestCase):
    """Test fallback behavior."""

    def test_no_data_dir_returns_mock(self):
        """Without data_dir, falls back to mock result."""
        solver = _make_solver()

        task_spec = {
            "task_id": 1,
            "task_type": "text_jepa",
            # No data_dir
            "vocab_size": 260,
            "max_seq_length": 64,
            "embed_dim": 64,
            "num_heads": 4,
            "encoder_depth": 2,
            "predictor_depth": 1,
            "predictor_embed_dim": 32,
            "epochs": 1,
            "batch_size": 2,
        }

        model_update, _ = solver._train(1, task_spec)
        # Should succeed — train_vljepa_on_task handles no data gracefully
        self.assertIsNotNone(model_update)

    def test_empty_data_dir_returns_mock(self):
        """With empty data dir, falls back gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            solver = _make_solver()

            task_spec = {
                "task_id": 1,
                "task_type": "text_jepa",
                "data_dir": tmpdir,  # exists but empty
                "vocab_size": 260,
                "max_seq_length": 64,
                "embed_dim": 64,
                "num_heads": 4,
                "encoder_depth": 2,
                "predictor_depth": 1,
                "predictor_embed_dim": 32,
                "epochs": 1,
                "batch_size": 2,
            }

            model_update, _ = solver._train(1, task_spec)
            self.assertIsNotNone(model_update)


if __name__ == "__main__":
    unittest.main()
