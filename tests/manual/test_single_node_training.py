"""
Test 1: Single-Node Training Loop — MVP Verification

Confirms the full pipeline: agent activity data → TrainingDataFeed →
TextJEPA training → weight delta with alignment score.

No blockchain, no P2P — pure ML pipeline.

Run:
    cd C:\\code\\autonet
    python -m pytest tests/manual/test_single_node_training.py -v -s
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig
from nodes.common.text_jepa import TextJEPAConfig, TextJEPATrainer
from nodes.common.tokenizer import SimpleTokenizer


# ---------------------------------------------------------------------------
# Synthetic agent activity (what the daemon writes to disk)
# ---------------------------------------------------------------------------

AGENT_CONVERSATIONS = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris, located in the north-central part of the country."},
    {"role": "user", "content": "Tell me about its history during the revolution."},
    {"role": "assistant", "content": "The French Revolution began in 1789 with the storming of the Bastille. Paris was the epicenter of political upheaval."},
    {"role": "user", "content": "How did the Napoleonic era change the city?"},
    {"role": "assistant", "content": "Napoleon transformed Paris through grand urban projects, the Arc de Triomphe, and centralized administration."},
    {"role": "user", "content": "What about modern Paris?"},
    {"role": "assistant", "content": "Modern Paris is a global center for culture, finance, and diplomacy, home to the EU institutions and major tech companies."},
]

CONSTITUTION = """
Article 1: All agents shall act in the interest of human welfare and safety.
Article 2: No agent shall deceive, manipulate, or cause harm to humans or other agents.
Article 3: Agents shall respect human autonomy and individual rights.
"""


def _write_trace_data(data_dir: str, conversations: list, num_sessions: int = 8):
    """Write conversation data as JSONL files mimicking daemon output."""
    conv_dir = Path(data_dir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_sessions):
        with open(conv_dir / f"session_{i}.jsonl", "w") as f:
            for turn in conversations:
                f.write(json.dumps({
                    **turn,
                    "timestamp": f"2026-04-09T{10+i}:00:00+00:00",
                }) + "\n")


def _make_config(data_dir: str, constitution: str = "") -> TrainingFeedConfig:
    """Small model config for fast local testing."""
    return TrainingFeedConfig(
        data_dir=data_dir,
        min_events_for_cycle=1,
        min_cycle_interval=0,
        vocab_size=260,
        max_seq_length=128,
        embed_dim=64,
        num_heads=4,
        encoder_depth=2,
        predictor_depth=1,
        predictor_embed_dim=32,
        batch_size=4,
        epochs=2,
        learning_rate=1e-3,
        scrub_pii=False,
        constitution_text=constitution,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleNodeTraining:
    """MVP Test 1: Does a single node actually train and produce weight deltas?"""

    def test_training_produces_weight_delta(self):
        """TrainingDataFeed.run_cycle() produces a non-empty weight delta."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_trace_data(tmpdir, AGENT_CONVERSATIONS)
            config = _make_config(tmpdir, constitution=CONSTITUTION)
            feed = TrainingDataFeed(config)
            feed._pending_events = 8

            result = feed.run_cycle()

            assert result is not None, "Training should produce a result (not skipped)"
            weight_delta, metrics = result
            assert len(weight_delta) > 0, "Weight delta should have entries"
            print(f"\n  Weight delta keys: {len(weight_delta)}")
            print(f"  Sample keys: {list(weight_delta.keys())[:5]}")

    def test_weight_deltas_are_nonzero(self):
        """The model actually changed — deltas are not all-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_trace_data(tmpdir, AGENT_CONVERSATIONS)
            config = _make_config(tmpdir)
            feed = TrainingDataFeed(config)
            feed._pending_events = 8

            result = feed.run_cycle()
            assert result is not None
            weight_delta, _ = result

            total_magnitude = sum(
                torch.tensor(v).abs().sum().item() if not isinstance(v, torch.Tensor)
                else v.abs().sum().item()
                for v in weight_delta.values()
            )
            assert total_magnitude > 0, "Weight deltas should be non-zero"
            print(f"\n  Total delta magnitude: {total_magnitude:.6f}")

    def test_loss_is_finite(self):
        """Training loss should be a finite positive number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_trace_data(tmpdir, AGENT_CONVERSATIONS)
            config = _make_config(tmpdir)
            feed = TrainingDataFeed(config)
            feed._pending_events = 8

            result = feed.run_cycle()
            assert result is not None
            _, metrics = result

            assert "loss" in metrics, "Metrics should include loss"
            loss = metrics["loss"]
            assert loss > 0, f"Loss should be positive, got {loss}"
            assert not (loss != loss), f"Loss should not be NaN"  # NaN != NaN
            print(f"\n  Final loss: {loss:.6f}")
            print(f"  Batches: {metrics.get('num_batches', 'N/A')}")

    def test_alignment_score_computed(self):
        """When constitution is provided, alignment score should be in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_trace_data(tmpdir, AGENT_CONVERSATIONS)
            config = _make_config(tmpdir, constitution=CONSTITUTION)
            feed = TrainingDataFeed(config)
            feed._pending_events = 8

            result = feed.run_cycle()
            assert result is not None
            _, metrics = result

            assert "alignment_score" in metrics, "Should compute alignment score"
            score = metrics["alignment_score"]
            assert 0.0 <= score <= 1.0, f"Alignment score {score} out of [0,1]"
            print(f"\n  Alignment score: {score:.4f}")

    def test_multiple_cycles_accumulate(self):
        """Running multiple training cycles should work (model state persists).

        The feed tracks segment count and skips if no new data appeared.
        We write fresh data each cycle to ensure the feed sees new segments.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir, constitution=CONSTITUTION)
            feed = TrainingDataFeed(config)

            losses = []
            for cycle in range(3):
                # Write fresh data each cycle so the feed sees new segments
                _write_trace_data(tmpdir, AGENT_CONVERSATIONS, num_sessions=5 + cycle * 5)
                feed._pending_events = 5
                result = feed.run_cycle()
                if result is None and cycle > 0:
                    # Feed may skip if segment count unchanged — that's OK
                    print(f"\n  Cycle {cycle}: skipped (no new segments)")
                    continue
                assert result is not None, f"Cycle {cycle} should produce a result"
                _, metrics = result
                losses.append(metrics["loss"])
                print(f"\n  Cycle {cycle}: loss={metrics['loss']:.6f}, "
                      f"alignment={metrics.get('alignment_score', 'N/A')}")

            assert feed.cycles_completed >= 1, "Should complete at least 1 cycle"
            assert all(l > 0 and l == l for l in losses), "All losses should be finite positive"

    def test_training_without_constitution(self):
        """Training works even without a constitution (alignment score should be absent or 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_trace_data(tmpdir, AGENT_CONVERSATIONS)
            config = _make_config(tmpdir, constitution="")
            feed = TrainingDataFeed(config)
            feed._pending_events = 8

            result = feed.run_cycle()
            assert result is not None
            weight_delta, metrics = result
            assert len(weight_delta) > 0
            print(f"\n  Loss: {metrics['loss']:.6f}")
            print(f"  Alignment: {metrics.get('alignment_score', 'not computed')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
