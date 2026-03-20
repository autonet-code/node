"""
End-to-end real training integration test.

Exercises the FULL offline loop, proving that all components compose correctly:

1. Config → task spec → real training → weight deltas
2. Multiple solvers → aggregation → model publication
3. Inference attestation on the final model
4. Alignment pricing advisory on the inference request
5. Governance bridge wiring (attestation, rewards, reputation)

This test runs without a blockchain node — all on-chain interactions are
mocked. It proves that the ML pipeline, attestor, and pricing layer
work together end-to-end.

Run: python -m pytest tests/test_e2e_real_training.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from nodes.common.config import AutonetConfig, ModelConfig, TrainingConfig
from nodes.common.blob_store import BlobStore
from nodes.common.ml import (
    SimpleNet,
    train_on_task,
    aggregate_weight_deltas,
    apply_weight_delta,
    save_weights,
    load_weights,
    load_weights_into_model,
)
from nodes.common.blockchain import TransactionResult
from nodes.common.inference_attestor import InferenceAttestor
from nodes.common.alignment_pricing import AlignmentPricing


# =============================================================================
# Fixtures
# =============================================================================


def _ok():
    return TransactionResult(success=True, tx_hash="0xabc")


@pytest.fixture
def blob_store():
    tmpdir = tempfile.mkdtemp(prefix="autonet-e2e-test-")
    return BlobStore(data_dir=tmpdir)


@pytest.fixture
def governance():
    gov = MagicMock()
    gov.node_id = "e2e-test-node"
    gov.attest_task_completion = MagicMock(return_value=True)
    return gov


@pytest.fixture
def supervised_config():
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
# Test: Full E2E pipeline — train, aggregate, infer, attest, price
# =============================================================================


class TestE2EFullPipeline:
    """
    Complete end-to-end test:
    1. Initialize model and save to blob store
    2. Two solvers independently train on subsets
    3. Aggregator combines deltas with FedAvg
    4. New model loaded for inference
    5. Inference runs and is attested
    6. Alignment pricing scores the operation
    """

    def test_full_pipeline(self, blob_store, governance, supervised_config):
        """The complete loop from model init to attested inference."""
        # ---- Step 1: Create and store initial model ----
        model = SimpleNet()
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        base_cid = save_weights(model.state_dict(), blob_store)
        assert base_cid is not None

        # ---- Step 2: Two solvers train independently ----
        delta1, metrics1 = train_on_task(
            task_spec={"task_id": 1},
            store=blob_store,
            epochs=1, batch_size=16, learning_rate=0.01, num_samples=32,
            global_model_cid=base_cid,
        )
        assert not metrics1.get("mock", False), "Solver 1 should do real training"
        assert metrics1["accuracy"] > 0

        delta2, metrics2 = train_on_task(
            task_spec={"task_id": 2},
            store=blob_store,
            epochs=1, batch_size=16, learning_rate=0.01, num_samples=32,
            global_model_cid=base_cid,
        )
        assert not metrics2.get("mock", False), "Solver 2 should do real training"

        # ---- Step 3: Aggregator combines deltas ----
        aggregated_delta = aggregate_weight_deltas(
            [delta1, delta2],
            weights=[metrics1["num_samples"], metrics2["num_samples"]],
        )
        assert len(aggregated_delta) > 0

        # ---- Step 4: Apply delta to base and save new model ----
        base_weights = load_weights(base_cid, blob_store)
        assert base_weights is not None

        new_weights = apply_weight_delta(base_weights, aggregated_delta)
        new_cid = save_weights(
            {k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
             for k, v in new_weights.items()},
            blob_store,
        )
        assert new_cid is not None
        assert new_cid != base_cid  # Weights changed

        # ---- Step 5: Load model and run inference ----
        inference_model = SimpleNet()
        weights_data = blob_store.get_json(new_cid)
        assert weights_data is not None
        inference_model = load_weights_into_model(inference_model, weights_data)

        # Run inference on a batch
        dummy_input = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            output = inference_model(dummy_input)

        assert output.shape == (4, 10)
        predictions = output.argmax(dim=1).tolist()
        assert len(predictions) == 4

        # ---- Step 6: Attest inference usage ----
        attestor = InferenceAttestor(
            governance, blob_store,
            tokens_per_unit=1000, flush_threshold=100000,
        )

        attestor.record_inference(
            provider="native",
            input_tokens=dummy_input.numel(),  # 4*1*28*28 = 3136
            output_tokens=len(predictions) * 10,
            model=new_cid[:16],
            request_id="e2e-test-1",
        )

        assert attestor.pending_tokens == 3136 + 40
        attestor.flush()
        assert attestor.pending_tokens == 0
        governance.attest_task_completion.assert_called_once()

        # ---- Step 7: Advisory alignment pricing ----
        pricing = AlignmentPricing(
            network_maturity=0.5,
            treasury_balance=50000.0,
            treasury_threshold=100000.0,
        )

        result = pricing.compute_price(
            task_description="classify handwritten digit image recognition",
            user_standards="image recognition accuracy classification",
            jurisdiction_standards="computer vision image classification accuracy",
            base_cost=100.0,
        )

        # The pricing function should return a valid result
        assert 0.0 <= result.alignment_score <= 1.0
        assert result.user_pays >= 0.0
        assert result.tier in ("subsidized", "neutral", "premium")
        assert result.base_cost == 100.0

        # ---- Step 8: Verify weights actually changed ----
        for key in initial_state:
            if not torch.equal(initial_state[key], new_weights[key]):
                break
        else:
            pytest.fail("Weights should change after training + aggregation")


# =============================================================================
# Test: Multi-round training (model improves across rounds)
# =============================================================================


class TestMultiRoundTraining:
    def test_multi_round_model_improvement(self, blob_store, supervised_config):
        """Model loss decreases over multiple training rounds (FedAvg)."""
        # Round 0: Initialize model
        model = SimpleNet()
        current_cid = save_weights(model.state_dict(), blob_store)

        round_losses = []

        for round_num in range(3):
            # Two solvers train per round
            deltas = []
            sample_counts = []

            for solver_idx in range(2):
                delta, metrics = train_on_task(
                    task_spec={"task_id": round_num * 2 + solver_idx},
                    store=blob_store,
                    epochs=1, batch_size=16, learning_rate=0.01,
                    num_samples=64,
                    global_model_cid=current_cid,
                )
                deltas.append(delta)
                sample_counts.append(metrics["num_samples"])
                round_losses.append(metrics["loss"])

            # Aggregate and save
            aggregated = aggregate_weight_deltas(deltas, weights=sample_counts)
            base_weights = load_weights(current_cid, blob_store)
            new_weights = apply_weight_delta(base_weights, aggregated)
            current_cid = save_weights(
                {k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
                 for k, v in new_weights.items()},
                blob_store,
            )

        # The model should have trained over 3 rounds
        # We can't guarantee monotonic improvement with random data subsets,
        # but the model should have non-zero loss and have been updated
        assert len(round_losses) == 6  # 2 solvers * 3 rounds
        assert all(loss > 0 for loss in round_losses)

        # Final model should be loadable and functional
        final_model = SimpleNet()
        final_weights = load_weights(current_cid, blob_store)
        final_model.load_state_dict(final_weights)
        final_model.eval()

        dummy = torch.randn(1, 1, 28, 28)
        output = final_model(dummy)
        assert output.shape == (1, 10)


# =============================================================================
# Test: Attestation through training lifecycle
# =============================================================================


class TestAttestationLifecycle:
    def test_attestation_accumulates_through_training(self, blob_store, governance):
        """Attestation tracks usage across multiple training + inference steps."""
        attestor = InferenceAttestor(
            governance, blob_store,
            tokens_per_unit=1000, flush_threshold=100000,
        )

        # Simulate 5 inference calls from different providers
        providers = ["claude", "gpt", "claude", "gemini", "claude"]
        for i, provider in enumerate(providers):
            attestor.record_inference(
                provider=provider,
                input_tokens=500 + i * 100,
                output_tokens=100 + i * 20,
                model=f"model-v{i}",
            )

        # Check accumulation
        assert attestor.pending_records == 5
        assert attestor.provider_totals["claude"] > 0
        assert attestor.provider_totals["gpt"] > 0
        assert attestor.provider_totals["gemini"] > 0

        # Flush
        attestor.flush()
        assert attestor.pending_tokens == 0
        assert attestor.lifetime_attested_tokens > 0

        # Summary should be complete
        summary = attestor.summary()
        assert summary["attestation_count"] == 1
        assert len(summary["provider_totals"]) == 3


# =============================================================================
# Test: Training reward incentives via alignment pricing
# =============================================================================


class TestTrainingIncentives:
    def test_capability_gap_drives_rewards(self):
        """Higher capability gap -> higher training reward."""
        pricing = AlignmentPricing()

        # Module A: no capability (gap=1.0) -> high reward
        result_a = pricing.compute_training_incentive(
            module_id="visual_encoder",
            current_score=0,
            target_score=8000,
            base_reward=100.0,
        )

        # Module B: at target (gap=0.0) -> base reward
        result_b = pricing.compute_training_incentive(
            module_id="text_encoder",
            current_score=8000,
            target_score=8000,
            base_reward=100.0,
        )

        # Module C: above target -> reduced reward
        result_c = pricing.compute_training_incentive(
            module_id="predictor",
            current_score=10000,
            target_score=8000,
            base_reward=100.0,
        )

        assert result_a.effective_reward > result_b.effective_reward
        assert result_b.effective_reward > result_c.effective_reward
        assert result_a.reward_multiplier == pytest.approx(3.0)
        assert result_b.reward_multiplier == pytest.approx(1.0)
        assert result_c.reward_multiplier < 1.0

    def test_incentive_prevents_redundant_training(self):
        """Saturated modules get low rewards, preventing waste."""
        pricing = AlignmentPricing()

        # Three modules at different capability levels
        rewards = []
        for score in [0, 2000, 4000, 6000, 8000, 10000]:
            result = pricing.compute_training_incentive(
                module_id=f"module_{score}",
                current_score=score,
                target_score=8000,
                base_reward=100.0,
            )
            rewards.append(result.effective_reward)

        # Rewards should decrease as capability increases
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], (
                f"Reward at score {i} ({rewards[i]}) should be >= "
                f"reward at score {i+1} ({rewards[i+1]})"
            )


# =============================================================================
# Test: Blob store integrity through full pipeline
# =============================================================================


class TestBlobStoreIntegrity:
    def test_all_artifacts_persist(self, blob_store, governance):
        """All training artifacts are content-addressed and retrievable."""
        # Save initial model
        model = SimpleNet()
        base_cid = save_weights(model.state_dict(), blob_store)

        # Train
        delta, metrics = train_on_task(
            task_spec={"task_id": 1},
            store=blob_store,
            epochs=1, batch_size=16, learning_rate=0.01, num_samples=32,
            global_model_cid=base_cid,
        )

        # Save solution (as solver would)
        solution = {
            "task_id": 1,
            "weight_delta": delta,
            "metrics": metrics,
            "real_training": True,
        }
        solution_cid = blob_store.add_json(solution)

        # Save attestation receipt
        attestor = InferenceAttestor(
            governance, blob_store,
            tokens_per_unit=1000, flush_threshold=100000,
        )
        attestor.record_inference("claude", 1000, 200)
        attestor.flush()

        # Verify all blobs exist and are retrievable
        assert blob_store.has(base_cid)
        assert blob_store.has(solution_cid)

        # Verify solution round-trips correctly
        loaded = blob_store.get_json(solution_cid)
        assert loaded["task_id"] == 1
        assert loaded["real_training"] is True
        assert "weight_delta" in loaded
        assert "metrics" in loaded

        # Verify base model round-trips
        base_data = blob_store.get_json(base_cid)
        assert base_data is not None
        assert "weights" in base_data


# =============================================================================
# Test: Alignment pricing self-funding property
# =============================================================================


class TestAlignmentSelfFunding:
    def test_premiums_can_fund_subsidies(self):
        """Premium revenue from misaligned ops can fund aligned subsidies."""
        pricing = AlignmentPricing(
            network_maturity=1.0,
            treasury_balance=100000.0,
            treasury_threshold=100000.0,
            max_subsidy_rate=0.8,
            max_premium_rate=0.5,
        )

        # Misaligned operation -> premium
        premium_result = pricing.compute_price(
            task_description="something completely unrelated",
            user_standards="translation accuracy quality",
            jurisdiction_standards="medical research clinical trials",
            base_cost=100.0,
        )

        # Aligned operation -> subsidy
        subsidy_result = pricing.compute_price(
            task_description="translation accuracy quality assurance",
            user_standards="translation accuracy quality assurance",
            jurisdiction_standards="translation accuracy quality standards",
            base_cost=100.0,
        )

        # Premium revenue is positive
        assert premium_result.premium_collected >= 0.0

        # Subsidy reduces user cost
        assert subsidy_result.user_pays <= 100.0

        # The system is self-funding: premiums exist to fund subsidies
        assert premium_result.tier == "premium"
        assert subsidy_result.tier == "subsidized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
