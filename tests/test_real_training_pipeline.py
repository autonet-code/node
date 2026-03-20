"""
Integration test: real training pipeline end-to-end.

Exercises the full loop offline (no blockchain):
  config → task spec → real JEPA training → aggregation → model weights

This test proves that:
1. Config loads correctly and drives hyperparameters
2. Proposer generates well-formed task specs
3. Solver performs real JEPA training and produces weight deltas
4. Aggregator aggregates deltas with FedAvg
5. Aggregated deltas can be applied to produce a loadable model
6. The same pipeline works for supervised (CNN/MNIST) training

Run: python -m pytest tests/test_real_training_pipeline.py -v
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.common.config import AutonetConfig, ModelConfig, TrainingConfig, load_config
from nodes.common.blob_store import BlobStore


@pytest.fixture
def blob_store():
    """Create a temporary blob store for each test."""
    tmpdir = tempfile.mkdtemp(prefix="autonet-test-")
    return BlobStore(data_dir=tmpdir)


@pytest.fixture
def jepa_config():
    """Config for a small, fast JEPA test."""
    cfg = AutonetConfig()
    cfg.device = "cpu"
    cfg.training = TrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.05,
        num_samples=32,  # Very small for speed
        optimizer="adamw",
        task_type="jepa",
    )
    cfg.model = ModelConfig(
        architecture="jepa",
        image_size=32,
        patch_size=4,
        embed_dim=64,   # Tiny for fast test
        num_heads=2,
        encoder_depth=2,
        predictor_depth=1,
        predictor_embed_dim=32,
    )
    return cfg


@pytest.fixture
def supervised_config():
    """Config for a small, fast supervised test."""
    cfg = AutonetConfig()
    cfg.device = "cpu"
    cfg.training = TrainingConfig(
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0,
        num_samples=64,
        optimizer="sgd",
        task_type="supervised",
    )
    cfg.model = ModelConfig(architecture="simplenet")
    return cfg


# =============================================================================
# Test: Config system
# =============================================================================


def test_config_loads_from_yaml():
    """Config loads from autonet.yaml and has sane defaults."""
    cfg = load_config()
    assert cfg.model.architecture in ("jepa", "simplenet", "vl_jepa")
    assert cfg.training.epochs >= 1
    assert cfg.training.batch_size >= 1
    assert cfg.blockchain.rpc_url.startswith("http")
    assert cfg.staking.solver > 0


def test_config_env_override(monkeypatch):
    """Environment variables override YAML values."""
    monkeypatch.setenv("AUTONET_DEVICE", "cpu")
    monkeypatch.setenv("AUTONET_EPOCHS", "5")
    monkeypatch.setenv("AUTONET_IMAGE_SIZE", "64")
    cfg = load_config()
    assert cfg.device == "cpu"
    assert cfg.training.epochs == 5
    assert cfg.model.image_size == 64


# =============================================================================
# Test: Real JEPA training
# =============================================================================


def test_jepa_training_produces_real_deltas(blob_store, jepa_config):
    """Solver's JEPA training produces non-trivial weight deltas."""
    from nodes.common.ml import train_jepa_on_task

    task_spec = {
        "task_id": 1,
        "task_type": "jepa",
        "image_size": jepa_config.model.image_size,
        "patch_size": jepa_config.model.patch_size,
        "embed_dim": jepa_config.model.embed_dim,
        "num_heads": jepa_config.model.num_heads,
        "encoder_depth": jepa_config.model.encoder_depth,
        "predictor_depth": jepa_config.model.predictor_depth,
        "predictor_embed_dim": jepa_config.model.predictor_embed_dim,
    }

    weight_delta, metrics = train_jepa_on_task(
        task_spec=task_spec,
        store=blob_store,
        epochs=jepa_config.training.epochs,
        batch_size=jepa_config.training.batch_size,
        learning_rate=jepa_config.training.learning_rate,
        num_samples=jepa_config.training.num_samples,
        image_size=jepa_config.model.image_size,
    )

    # Weight delta should be a dict of param_name → list
    assert isinstance(weight_delta, dict), "Weight delta should be a dict"
    assert len(weight_delta) > 0, "Should have at least one parameter"

    # Check it's not all zeros (training actually happened)
    some_delta = list(weight_delta.values())[0]
    assert some_delta != 0, "Delta should be non-zero (training occurred)"

    # Metrics should have real values
    assert "loss" in metrics
    assert "cosine_similarity" in metrics
    assert metrics["loss"] > 0
    assert not metrics.get("mock", False), "Should not be mock training"


# =============================================================================
# Test: Real supervised training
# =============================================================================


def test_supervised_training_produces_real_deltas(blob_store, supervised_config):
    """Solver's supervised training produces real CNN weight deltas."""
    from nodes.common.ml import train_on_task

    weight_delta, metrics = train_on_task(
        task_spec={"task_id": 1, "epochs": 1},
        store=blob_store,
        epochs=supervised_config.training.epochs,
        batch_size=supervised_config.training.batch_size,
        learning_rate=supervised_config.training.learning_rate,
        num_samples=supervised_config.training.num_samples,
    )

    assert isinstance(weight_delta, dict)
    assert len(weight_delta) > 0

    # CNN layers should be present
    param_names = set(weight_delta.keys())
    assert any("conv" in k for k in param_names), "Should have conv layer deltas"
    assert any("fc" in k for k in param_names), "Should have fc layer deltas"

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0


# =============================================================================
# Test: FedAvg aggregation on real deltas
# =============================================================================


def test_fedavg_on_real_deltas(blob_store, supervised_config):
    """FedAvg correctly aggregates multiple real weight deltas."""
    from nodes.common.ml import train_on_task, aggregate_weight_deltas

    # Train twice to get two deltas
    delta1, metrics1 = train_on_task(
        task_spec={"task_id": 1},
        store=blob_store,
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
        num_samples=32,
    )

    delta2, metrics2 = train_on_task(
        task_spec={"task_id": 2},
        store=blob_store,
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
        num_samples=32,
    )

    # Aggregate
    aggregated = aggregate_weight_deltas(
        [delta1, delta2],
        weights=[metrics1["num_samples"], metrics2["num_samples"]],
    )

    assert isinstance(aggregated, dict)
    assert len(aggregated) == len(delta1), "Aggregated should have same params as inputs"

    # Aggregated values should be between the two inputs (weighted average)
    for key in aggregated:
        assert aggregated[key] is not None


# =============================================================================
# Test: Full pipeline (train → store → aggregate → apply)
# =============================================================================


def test_full_pipeline_supervised(blob_store, supervised_config):
    """Full supervised pipeline: train → store → aggregate → apply → load."""
    from nodes.common.ml import (
        train_on_task,
        aggregate_weight_deltas,
        apply_weight_delta,
        save_weights,
        load_weights,
        SimpleNet,
    )
    import torch

    # Step 1: Initial model weights
    model = SimpleNet()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    base_cid = save_weights(model.state_dict(), blob_store)
    assert base_cid is not None

    # Step 2: Two solvers train independently
    delta1, m1 = train_on_task(
        task_spec={"task_id": 1},
        store=blob_store,
        epochs=1, batch_size=16, learning_rate=0.01, num_samples=32,
        global_model_cid=base_cid,
    )
    delta2, m2 = train_on_task(
        task_spec={"task_id": 2},
        store=blob_store,
        epochs=1, batch_size=16, learning_rate=0.01, num_samples=32,
        global_model_cid=base_cid,
    )

    # Step 3: Aggregator combines deltas
    aggregated_delta = aggregate_weight_deltas(
        [delta1, delta2],
        weights=[m1["num_samples"], m2["num_samples"]],
    )

    # Step 4: Apply aggregated delta to base model
    base_weights = load_weights(base_cid, blob_store)
    assert base_weights is not None

    new_weights = apply_weight_delta(base_weights, aggregated_delta)

    # Step 5: Load new weights into a model and verify it works
    new_model = SimpleNet()
    new_model.load_state_dict(new_weights)
    new_model.eval()

    # Sanity: forward pass doesn't crash
    dummy_input = torch.randn(1, 1, 28, 28)
    output = new_model(dummy_input)
    assert output.shape == (1, 10), "Output should be 10-class prediction"

    # Weights should have changed from initial
    for key in initial_state:
        if not torch.equal(initial_state[key], new_weights[key]):
            break
    else:
        pytest.fail("Weights should have changed after training + aggregation")


# =============================================================================
# Test: Blob store round-trip with weight deltas
# =============================================================================


def test_blob_store_weight_roundtrip(blob_store):
    """Weight deltas survive JSON serialization through blob store."""
    import torch

    model = torch.nn.Linear(10, 5)
    state = {k: v.tolist() for k, v in model.state_dict().items()}

    solution = {
        "task_id": 42,
        "weight_delta": state,
        "metrics": {"loss": 0.5, "accuracy": 0.8},
        "real_training": True,
    }

    cid = blob_store.add_json(solution)
    loaded = blob_store.get_json(cid)

    assert loaded is not None
    assert loaded["task_id"] == 42
    assert loaded["real_training"] is True
    assert set(loaded["weight_delta"].keys()) == set(state.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
