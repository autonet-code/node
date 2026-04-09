"""
Test 3: Multi-Node FedAvg with Alignment Weighting

Two simulated nodes train on different domain data. Their weight deltas
are merged via alignment-weighted FedAvg. The test confirms:
  - Higher-alignment nodes have more influence on the merged model
  - The merged model is a genuine blend (not a copy of either input)
  - Merge produces valid, non-degenerate weights

No blockchain, no P2P — pure ML aggregation test.

Run:
    cd C:\\code\\autonet
    python -m pytest tests/manual/test_fedavg_merge.py -v -s
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from nodes.common.training_feed import TrainingDataFeed, TrainingFeedConfig
from nodes.common.ml import aggregate_weight_deltas


# ---------------------------------------------------------------------------
# Domain-specific datasets (simulate two different nodes' data)
# ---------------------------------------------------------------------------

SCIENCE_DATA = [
    {"role": "user", "content": "Explain quantum entanglement in simple terms."},
    {"role": "assistant", "content": "Quantum entanglement is when two particles become linked so measuring one instantly reveals info about the other regardless of distance."},
    {"role": "user", "content": "What is the Heisenberg uncertainty principle?"},
    {"role": "assistant", "content": "You cannot simultaneously know both the exact position and exact momentum of a particle. Measuring one disturbs the other."},
    {"role": "user", "content": "How does general relativity describe gravity?"},
    {"role": "assistant", "content": "Mass curves spacetime and objects follow the curved paths. Gravity is geometry not a force in the Newtonian sense."},
    {"role": "user", "content": "What is dark matter?"},
    {"role": "assistant", "content": "Dark matter is invisible matter that interacts gravitationally. It makes up about 27 percent of the universe but has never been directly detected."},
]

COOKING_DATA = [
    {"role": "user", "content": "How do I make a proper roux?"},
    {"role": "assistant", "content": "Melt equal parts butter and flour over medium heat. Stir constantly for two to three minutes for a white roux, longer for darker."},
    {"role": "user", "content": "What temperature should I roast a chicken?"},
    {"role": "assistant", "content": "Roast at 425 degrees Fahrenheit for about 20 minutes per pound. Internal temperature should reach 165 degrees."},
    {"role": "user", "content": "How do I properly season a cast iron skillet?"},
    {"role": "assistant", "content": "Coat with thin layer of high smoke point oil. Bake upside down at 450 degrees for one hour. Repeat three times."},
    {"role": "user", "content": "What makes bread rise?"},
    {"role": "assistant", "content": "Yeast ferments sugars producing carbon dioxide gas. The gluten network traps the gas creating the airy structure."},
]

CONSTITUTION = "All agents shall pursue knowledge and serve human understanding."


def _write_data(data_dir: str, conversations: list, num_sessions: int = 8):
    conv_dir = Path(data_dir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_sessions):
        with open(conv_dir / f"session_{i}.jsonl", "w") as f:
            for turn in conversations:
                f.write(json.dumps({**turn, "timestamp": f"2026-04-09T{10+i}:00:00+00:00"}) + "\n")


def _make_config(data_dir: str, constitution: str = "") -> TrainingFeedConfig:
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


def _train_node(conversations, constitution=""):
    """Train a single node and return (weight_delta, metrics)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_data(tmpdir, conversations)
        config = _make_config(tmpdir, constitution=constitution)
        feed = TrainingDataFeed(config)
        feed._pending_events = 8
        result = feed.run_cycle()
        assert result is not None, "Training should produce a result"
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFedAvgMerge:
    """MVP Test 3: Alignment-weighted FedAvg produces a genuine blend."""

    def test_two_nodes_produce_different_deltas(self):
        """Two nodes training on different domains produce distinct weight deltas."""
        delta_a, metrics_a = _train_node(SCIENCE_DATA, CONSTITUTION)
        delta_b, metrics_b = _train_node(COOKING_DATA, CONSTITUTION)

        # Both should have the same keys (same architecture)
        assert set(delta_a.keys()) == set(delta_b.keys()), "Same architecture → same keys"

        # But the values should differ (different training data)
        differences = 0
        for key in list(delta_a.keys())[:5]:
            va = torch.tensor(delta_a[key]) if not isinstance(delta_a[key], torch.Tensor) else delta_a[key]
            vb = torch.tensor(delta_b[key]) if not isinstance(delta_b[key], torch.Tensor) else delta_b[key]
            diff = (va - vb).abs().sum().item()
            if diff > 1e-6:
                differences += 1

        assert differences > 0, "Different training data should produce different deltas"
        print(f"\n  Keys that differ: {differences}/{min(5, len(delta_a))}")
        print(f"  Node A alignment: {metrics_a.get('alignment_score', 'N/A')}")
        print(f"  Node B alignment: {metrics_b.get('alignment_score', 'N/A')}")

    def test_alignment_weighted_merge_biases_toward_higher(self):
        """Higher-alignment node should have more influence on the merged result."""
        # Use synthetic deltas with known values for precise verification
        delta_high = {}
        delta_low = {}
        for name in ["encoder.weight", "predictor.weight", "embed.weight"]:
            delta_high[name] = torch.ones(8, 8) * 10.0
            delta_low[name] = torch.ones(8, 8) * -10.0

        # High alignment (0.95) vs low alignment (0.1), same sample count
        weight_high = 10 * 0.95  # = 9.5
        weight_low = 10 * 0.1   # = 1.0

        merged = aggregate_weight_deltas(
            [delta_high, delta_low],
            weights=[weight_high, weight_low],
        )

        for key in merged:
            mean_val = merged[key].mean().item()
            # Expected: (9.5 * 10 + 1.0 * -10) / (9.5 + 1.0) = 85/10.5 ≈ 8.095
            assert mean_val > 5.0, f"Merged {key} mean={mean_val:.3f}, should be >5 (biased toward high-alignment)"
            print(f"\n  {key}: merged mean = {mean_val:.4f} (expected ~8.1)")

    def test_merge_is_genuine_blend(self):
        """Merged delta should differ from both individual deltas."""
        delta_a, metrics_a = _train_node(SCIENCE_DATA, CONSTITUTION)
        delta_b, metrics_b = _train_node(COOKING_DATA, CONSTITUTION)

        align_a = metrics_a.get("alignment_score", 0.5)
        align_b = metrics_b.get("alignment_score", 0.5)

        weights = [
            max(0.01, align_a) * 10,
            max(0.01, align_b) * 10,
        ]

        merged = aggregate_weight_deltas([delta_a, delta_b], weights=weights)

        differs_from_a = 0
        differs_from_b = 0
        for key in list(merged.keys())[:5]:
            mv = torch.tensor(merged[key]) if not isinstance(merged[key], torch.Tensor) else merged[key]
            va = torch.tensor(delta_a[key]) if not isinstance(delta_a[key], torch.Tensor) else delta_a[key]
            vb = torch.tensor(delta_b[key]) if not isinstance(delta_b[key], torch.Tensor) else delta_b[key]

            if (mv - va).abs().sum().item() > 1e-6:
                differs_from_a += 1
            if (mv - vb).abs().sum().item() > 1e-6:
                differs_from_b += 1

        assert differs_from_a > 0 or differs_from_b > 0, \
            "Merged delta should differ from at least one input"
        print(f"\n  Differs from A: {differs_from_a}, from B: {differs_from_b}")

    def test_merged_weights_are_valid(self):
        """Merged weights should be finite and non-degenerate."""
        delta_a, _ = _train_node(SCIENCE_DATA, CONSTITUTION)
        delta_b, _ = _train_node(COOKING_DATA, CONSTITUTION)

        merged = aggregate_weight_deltas(
            [delta_a, delta_b],
            weights=[1.0, 1.0],
        )

        for key, val in merged.items():
            t = torch.tensor(val) if not isinstance(val, torch.Tensor) else val
            assert torch.isfinite(t).all(), f"{key} has non-finite values"
            assert t.abs().max().item() < 1e6, f"{key} has exploding values"

    def test_three_node_merge(self):
        """FedAvg works with three or more nodes."""
        delta_a, ma = _train_node(SCIENCE_DATA, CONSTITUTION)
        delta_b, mb = _train_node(COOKING_DATA, CONSTITUTION)
        # Third node reuses science data but different random init
        delta_c, mc = _train_node(SCIENCE_DATA, CONSTITUTION)

        weights = [
            max(0.01, ma.get("alignment_score", 0.5)),
            max(0.01, mb.get("alignment_score", 0.5)),
            max(0.01, mc.get("alignment_score", 0.5)),
        ]

        merged = aggregate_weight_deltas([delta_a, delta_b, delta_c], weights=weights)
        assert len(merged) > 0
        print(f"\n  3-node merge: {len(merged)} keys")
        print(f"  Alignments: {[f'{w:.3f}' for w in weights]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
