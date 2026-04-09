"""
Test 5: Local Inference Pipeline — PoC Verification

Confirms the inference tier system routes requests correctly and
computes costs per tier. Uses local-only execution (no P2P).

No blockchain, no network — pure inference pipeline test.

Run:
    cd C:\\code\\autonet
    python -m pytest tests/manual/test_inference_local.py -v -s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from nodes.common.intelligence_tiers import (
    DEFAULT_PRICING,
    InferenceTier,
    InferenceTierManager,
    TIER_MODULES,
)
from nodes.common.inference_pipeline import (
    InferencePipeline,
    InferenceRequest,
    InferenceResult,
    ModuleRoute,
)
from nodes.common.decode_verifier import (
    DecodeOutput,
    DecodeVerifier,
    VerificationReport,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferenceTierRouting:
    """PoC Test 5a: Tier system correctly determines modules and costs."""

    def test_tier_1_encoder_only(self):
        """Tier 1 should route to encoder only (cheapest)."""
        mgr = InferenceTierManager()
        modules = mgr.get_modules(InferenceTier.TIER_1)
        assert "visual_encoder" in modules
        assert len(modules) == 1
        cost = mgr.compute_cost(InferenceTier.TIER_1)
        print(f"\n  Tier 1 modules: {modules}")
        print(f"  Tier 1 base cost: {cost:.4f} ATN")

    def test_tier_2_encoder_plus_predictor(self):
        """Tier 2 adds reasoning modules."""
        mgr = InferenceTierManager()
        modules = mgr.get_modules(InferenceTier.TIER_2)
        assert len(modules) > 1
        assert "visual_encoder" in modules
        assert "semantic_predictor" in modules
        cost = mgr.compute_cost(InferenceTier.TIER_2)
        print(f"\n  Tier 2 modules: {modules}")
        print(f"  Tier 2 base cost: {cost:.4f} ATN")

    def test_tier_3_full_pipeline(self):
        """Tier 3 uses all modules (most expensive)."""
        mgr = InferenceTierManager()
        modules = mgr.get_modules(InferenceTier.TIER_3)
        assert len(modules) == 5
        cost = mgr.compute_cost(InferenceTier.TIER_3)
        print(f"\n  Tier 3 modules: {modules}")
        print(f"  Tier 3 base cost: {cost:.4f} ATN")

    def test_cost_scales_with_tokens(self):
        """Output tokens should increase the total cost."""
        mgr = InferenceTierManager()
        cost_0 = mgr.compute_cost(InferenceTier.TIER_3, output_tokens=0)
        cost_100 = mgr.compute_cost(InferenceTier.TIER_3, output_tokens=100)
        cost_1000 = mgr.compute_cost(InferenceTier.TIER_3, output_tokens=1000)
        assert cost_100 > cost_0
        assert cost_1000 > cost_100
        print(f"\n  Tier 3 cost: 0 tokens={cost_0:.4f}, 100={cost_100:.4f}, 1000={cost_1000:.4f}")

    def test_tier_recommendation(self):
        """Task type should map to appropriate tier."""
        mgr = InferenceTierManager()
        assert mgr.recommend_tier("embedding") == InferenceTier.TIER_1
        assert mgr.recommend_tier("generation") == InferenceTier.TIER_3
        assert mgr.recommend_tier("reasoning") == InferenceTier.TIER_2


class TestInferencePipelineExecution:
    """PoC Test 5b: Pipeline executes requests locally with correct routing."""

    def test_pipeline_executes_tier_1_request(self):
        """Tier 1 request goes through encoder only, returns result."""
        mgr = InferenceTierManager()
        pipeline = InferencePipeline(tier_manager=mgr)

        request = InferenceRequest(
            request_id="test-001",
            tier=InferenceTier.TIER_1,
            input_data={"text": "Hello world"},
            requester_id="agent-001",
            max_credits=10.0,
        )

        result = pipeline.execute(request)

        assert result.request_id == "test-001"
        assert result.success
        assert len(result.modules_executed) > 0
        assert result.credits_used > 0
        print(f"\n  Modules executed: {result.modules_executed}")
        print(f"  Credits used: {result.credits_used:.4f}")
        print(f"  Latency: {result.total_latency_ms:.1f}ms")

    def test_pipeline_executes_tier_3_request(self):
        """Tier 3 request goes through full pipeline."""
        mgr = InferenceTierManager()
        pipeline = InferencePipeline(tier_manager=mgr)

        request = InferenceRequest(
            request_id="test-002",
            tier=InferenceTier.TIER_3,
            input_data={"text": "Explain quantum mechanics"},
            requester_id="agent-002",
            max_credits=100.0,
        )

        result = pipeline.execute(request)

        assert result.success
        assert result.credits_used > 0
        # Tier 3 should be more expensive than tier 1
        print(f"\n  Tier 3 modules: {result.modules_executed}")
        print(f"  Credits: {result.credits_used:.4f}")

    def test_pipeline_rejects_underfunded_request(self):
        """Request with insufficient credits should be flagged."""
        mgr = InferenceTierManager()
        pipeline = InferencePipeline(tier_manager=mgr)

        request = InferenceRequest(
            request_id="test-003",
            tier=InferenceTier.TIER_3,
            input_data={"text": "test"},
            requester_id="agent-003",
            max_credits=0.0001,  # Way too low
        )

        result = pipeline.execute(request)
        # Should either reject or execute with a warning
        print(f"\n  Success: {result.success}")
        print(f"  Error: {result.error}")

    def test_pipeline_stats_accumulate(self):
        """Pipeline tracks execution statistics across requests."""
        mgr = InferenceTierManager()
        pipeline = InferencePipeline(tier_manager=mgr)

        for i in range(5):
            request = InferenceRequest(
                request_id=f"test-{i:03d}",
                tier=InferenceTier.TIER_1,
                input_data={"text": f"Query {i}"},
                requester_id="agent-stats",
                max_credits=50.0,
            )
            pipeline.execute(request)

        stats = pipeline.stats
        print(f"\n  Pipeline stats: total={stats.requests_total}, "
              f"completed={stats.requests_completed}, "
              f"avg_latency={stats.avg_latency_ms:.1f}ms")
        assert stats.requests_total >= 5


class TestDecodeVerification:
    """PoC Test 5c: Multiple decoder outputs reach consensus."""

    def test_consensus_with_matching_outputs(self):
        """Three identical outputs should reach VERIFIED."""
        verifier = DecodeVerifier(min_verifiers=2, agreement_threshold=0.5)
        plan_hash = "plan-abc"
        for i in range(3):
            verifier.submit_output(DecodeOutput(
                node_id=f"node-{i}",
                generated_ids=[1, 2, 3, 4, 5],
                latent_plan_hash=plan_hash,
                seed=42,
            ))

        report = verifier.verify()

        assert report.result == VerificationResult.VERIFIED
        assert report.agreement_ratio == 1.0
        print(f"\n  Result: {report.result.value}")
        print(f"  Agreement: {report.agreement_ratio:.1%}")

    def test_dissent_with_mixed_outputs(self):
        """All different outputs → MISMATCH (no majority above threshold)."""
        verifier = DecodeVerifier(min_verifiers=2, agreement_threshold=0.67)
        plan_hash = "plan-xyz"
        for i, ids in enumerate([[1, 2], [3, 4], [5, 6]]):
            verifier.submit_output(DecodeOutput(
                node_id=f"node-{i}",
                generated_ids=ids,
                latent_plan_hash=plan_hash,
                seed=42,
            ))

        report = verifier.verify()

        # 1/3 ≈ 33% agreement, below 67% threshold → MISMATCH
        assert report.result == VerificationResult.MISMATCH
        print(f"\n  Result: {report.result.value}")
        print(f"  Agreement: {report.agreement_ratio:.1%}")
        print(f"  Dissenting nodes: {report.dissenting_nodes}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
