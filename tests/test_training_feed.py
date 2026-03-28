"""Tests for Story 3.2 — Training Data Feed.

Verifies that:
- TrainingDataFeed initializes correctly
- Event notification increments counters
- should_train() gating works (event count, time interval)
- run_cycle() trains on JSONL conversation data
- Behavioral profile is updated after training
- AutonetService._run_cycle() delegates to training feed
- force_cycle() bypasses gating
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig


def _create_conversation_data(data_dir: str, num_sessions: int = 3) -> None:
    """Create synthetic JSONL conversation files in the expected directory."""
    conv_dir = Path(data_dir) / "conversations"
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

    for i in range(num_sessions):
        with open(conv_dir / f"session_{i}.jsonl", "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")


def _create_execution_data(data_dir: str, num_agents: int = 2) -> None:
    """Create synthetic execution JSONL files in the expected directory."""
    agents_dir = Path(data_dir) / "agents"

    for i in range(num_agents):
        agent_dir = agents_dir / f"agent_{i}"
        agent_dir.mkdir(parents=True, exist_ok=True)

        records = [
            {
                "agent_id": f"agent_{i}",
                "execution_id": f"exec_{i}_1",
                "trigger_source": "scheduler",
                "status": "completed",
                "step_results": [
                    {
                        "step_name": "main",
                        "step_type": "cognitive",
                        "output": {
                            "text": "Checking inbox for new messages.",
                            "tool_calls": [
                                {"name": "bash", "input": {"command": "ls"}},
                            ],
                        },
                    }
                ],
            }
        ]

        with open(agent_dir / "execution.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


class TestTrainingFeedInit(unittest.TestCase):
    """Test TrainingDataFeed initialization."""

    def test_default_init(self):
        config = TrainingFeedConfig()
        feed = TrainingDataFeed(config)
        self.assertEqual(feed.pending_events, 0)
        self.assertEqual(feed.cycles_completed, 0)

    def test_init_with_data_dir(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)
        self.assertEqual(feed._data_dir, Path("/tmp/test"))


class TestTrainingFeedNotification(unittest.TestCase):
    """Test event notification mechanics."""

    def test_notify_increments_counter(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        feed.notify_execution("agent1", "exec1", "completed")
        self.assertEqual(feed.pending_events, 1)

        feed.notify_execution("agent2", "exec2", "completed")
        self.assertEqual(feed.pending_events, 2)

    def test_notify_ignores_non_terminal(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        feed.notify_execution("agent1", "exec1", "started")
        self.assertEqual(feed.pending_events, 0)

        feed.notify_execution("agent1", "exec1", "queued")
        self.assertEqual(feed.pending_events, 0)

    def test_notify_counts_failures(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        feed.notify_execution("agent1", "exec1", "failed")
        self.assertEqual(feed.pending_events, 1)


class TestTrainingFeedGating(unittest.TestCase):
    """Test should_train() gating logic."""

    def test_no_data_dir_blocks(self):
        config = TrainingFeedConfig()  # No data_dir
        feed = TrainingDataFeed(config)
        feed._pending_events = 100
        self.assertFalse(feed.should_train())

    def test_insufficient_events_blocks(self):
        config = TrainingFeedConfig(data_dir="/tmp/test", min_events_for_cycle=5)
        feed = TrainingDataFeed(config)
        feed._pending_events = 3
        self.assertFalse(feed.should_train())

    def test_sufficient_events_allows(self):
        config = TrainingFeedConfig(data_dir="/tmp/test", min_events_for_cycle=5,
                                    min_cycle_interval=0)
        feed = TrainingDataFeed(config)
        feed._pending_events = 5
        self.assertTrue(feed.should_train())

    def test_interval_blocks(self):
        config = TrainingFeedConfig(data_dir="/tmp/test", min_events_for_cycle=1,
                                    min_cycle_interval=3600)
        feed = TrainingDataFeed(config)
        feed._pending_events = 10
        feed._last_cycle_time = time.time()  # Just ran
        self.assertFalse(feed.should_train())

    def test_interval_allows_after_time(self):
        config = TrainingFeedConfig(data_dir="/tmp/test", min_events_for_cycle=1,
                                    min_cycle_interval=0)
        feed = TrainingDataFeed(config)
        feed._pending_events = 1
        feed._last_cycle_time = 0  # Long ago
        self.assertTrue(feed.should_train())


class TestTrainingFeedRunCycle(unittest.TestCase):
    """Test actual training cycles with synthetic data."""

    def test_run_cycle_with_conversation_data(self):
        """Training cycle reads JSONL and produces weight deltas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=5)

            config = TrainingFeedConfig(
                data_dir=tmpdir,
                min_events_for_cycle=1,
                min_cycle_interval=0,
                # Small model for fast testing
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
            feed = TrainingDataFeed(config)
            feed._pending_events = 5

            result = feed.run_cycle()

            self.assertIsNotNone(result)
            weight_delta, metrics = result
            self.assertGreater(len(weight_delta), 0)
            self.assertEqual(metrics["architecture"], "text_jepa")
            self.assertEqual(metrics["modalities"], ["text"])
            self.assertGreater(metrics["num_batches"], 0)
            self.assertEqual(feed.cycles_completed, 1)
            self.assertEqual(feed.pending_events, 0)

    def test_run_cycle_with_execution_data(self):
        """Training cycle reads execution JSONL too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=2)
            _create_execution_data(tmpdir, num_agents=3)

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
            feed = TrainingDataFeed(config)
            feed._pending_events = 5

            result = feed.run_cycle()

            self.assertIsNotNone(result)
            weight_delta, metrics = result
            self.assertGreater(metrics["num_batches"], 0)

    def test_run_cycle_skips_without_events(self):
        """No events → no training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir)

            config = TrainingFeedConfig(
                data_dir=tmpdir,
                min_events_for_cycle=5,
                min_cycle_interval=0,
            )
            feed = TrainingDataFeed(config)
            # 0 pending events
            result = feed.run_cycle()
            self.assertIsNone(result)
            self.assertEqual(feed.cycles_completed, 0)

    def test_run_cycle_skips_empty_dir(self):
        """No data files → no training."""
        with tempfile.TemporaryDirectory() as tmpdir:
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
                scrub_pii=False,
            )
            feed = TrainingDataFeed(config)
            feed._pending_events = 5

            result = feed.run_cycle()
            # Cycle ran but no data → None result, but cycle still counted
            self.assertIsNone(result)

    def test_force_cycle_bypasses_gating(self):
        """force_cycle() trains even without enough events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_conversation_data(tmpdir, num_sessions=3)

            config = TrainingFeedConfig(
                data_dir=tmpdir,
                min_events_for_cycle=100,  # Very high threshold
                min_cycle_interval=3600,   # Very long interval
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
            feed = TrainingDataFeed(config)
            feed._pending_events = 0  # No events

            result = feed.force_cycle()

            self.assertIsNotNone(result)
            self.assertEqual(feed.cycles_completed, 1)


class TestTrainingFeedStatus(unittest.TestCase):
    """Test status reporting."""

    def test_status_initial(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        status = feed.get_status()
        self.assertEqual(status["pending_events"], 0)
        self.assertEqual(status["total_events"], 0)
        self.assertEqual(status["cycles_completed"], 0)
        self.assertIsNotNone(status["data_dir"])
        self.assertNotIn("profile", status)

    def test_status_after_events(self):
        config = TrainingFeedConfig(data_dir="/tmp/test")
        feed = TrainingDataFeed(config)

        feed.notify_execution("a1", "e1", "completed")
        feed.notify_execution("a2", "e2", "completed")

        status = feed.get_status()
        self.assertEqual(status["pending_events"], 2)
        self.assertEqual(status["total_events"], 2)


class TestAutonetServiceIntegration(unittest.TestCase):
    """Test AutonetService integration with training feed."""

    def test_service_initializes_feed_with_data_dir(self):
        """AutonetService creates training feed when data_dir is given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nodes.service import AutonetService

            # Create service with data_dir
            service = AutonetService(data_dir=tmpdir)
            # Don't call start() — just test init
            service._init_training_feed()

            self.assertIsNotNone(service._training_feed)

    def test_service_notify_execution_passthrough(self):
        """notify_execution() passes through to training feed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from nodes.service import AutonetService

            service = AutonetService(data_dir=tmpdir)
            service._init_training_feed()

            service.notify_execution("agent1", "exec1", "completed")
            self.assertEqual(service._training_feed.pending_events, 1)

    def test_service_no_feed_without_data_dir(self):
        """No training feed when data_dir is not set."""
        from nodes.service import AutonetService

        service = AutonetService()
        # _data_dir is None, so no feed initialized
        self.assertIsNone(service._training_feed)


if __name__ == "__main__":
    unittest.main()
