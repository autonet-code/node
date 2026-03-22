"""Tests for Epic 9 — Inference pipeline, tiers, and decode verification."""

import hashlib
import unittest

from nodes.common.intelligence_tiers import (
    DEFAULT_PRICING,
    InferenceTier,
    InferenceTierManager,
    TIER_MODULES,
)
from nodes.common.decode_verifier import (
    DecodeOutput,
    DecodeVerifier,
    VerificationReport,
    VerificationResult,
)
from nodes.common.inference_pipeline import (
    InferencePipeline,
    InferenceRequest,
    InferenceResult,
    ModuleRoute,
    PipelineStats,
)


# =========================================================================
# Intelligence Tiers (Story 9.3)
# =========================================================================


class TestInferenceTier(unittest.TestCase):
    """Tests for InferenceTier enum."""

    def test_tier_values(self):
        self.assertEqual(InferenceTier.TIER_1.value, 1)
        self.assertEqual(InferenceTier.TIER_2.value, 2)
        self.assertEqual(InferenceTier.TIER_3.value, 3)

    def test_tier_modules_defined(self):
        for tier in InferenceTier:
            self.assertIn(tier, TIER_MODULES)
            self.assertGreater(len(TIER_MODULES[tier]), 0)

    def test_tier_1_is_encoder_only(self):
        self.assertEqual(TIER_MODULES[InferenceTier.TIER_1], ["visual_encoder"])

    def test_tier_3_includes_all_modules(self):
        self.assertEqual(len(TIER_MODULES[InferenceTier.TIER_3]), 5)

    def test_tier_module_counts_increase(self):
        t1 = len(TIER_MODULES[InferenceTier.TIER_1])
        t2 = len(TIER_MODULES[InferenceTier.TIER_2])
        t3 = len(TIER_MODULES[InferenceTier.TIER_3])
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)


class TestInferenceTierManager(unittest.TestCase):
    """Tests for InferenceTierManager."""

    def setUp(self):
        self.mgr = InferenceTierManager()

    def test_get_pricing(self):
        p = self.mgr.get_pricing(InferenceTier.TIER_1)
        self.assertEqual(p.tier, InferenceTier.TIER_1)
        self.assertGreater(p.base_cost_atn, 0)

    def test_get_modules(self):
        modules = self.mgr.get_modules(InferenceTier.TIER_2)
        self.assertIn("visual_encoder", modules)
        self.assertIn("semantic_predictor", modules)

    def test_compute_cost_base_only(self):
        cost = self.mgr.compute_cost(InferenceTier.TIER_1, output_tokens=0)
        self.assertEqual(cost, DEFAULT_PRICING[InferenceTier.TIER_1].base_cost_atn)

    def test_compute_cost_with_tokens(self):
        cost = self.mgr.compute_cost(InferenceTier.TIER_3, output_tokens=100)
        expected = DEFAULT_PRICING[InferenceTier.TIER_3].base_cost_atn + (
            DEFAULT_PRICING[InferenceTier.TIER_3].cost_per_token * 100
        )
        self.assertAlmostEqual(cost, expected)

    def test_cost_increases_with_tier(self):
        c1 = self.mgr.compute_cost(InferenceTier.TIER_1)
        c2 = self.mgr.compute_cost(InferenceTier.TIER_2)
        c3 = self.mgr.compute_cost(InferenceTier.TIER_3)
        self.assertLess(c1, c2)
        self.assertLess(c2, c3)

    def test_recommend_tier_embedding(self):
        self.assertEqual(self.mgr.recommend_tier("embedding"), InferenceTier.TIER_1)

    def test_recommend_tier_generation(self):
        self.assertEqual(self.mgr.recommend_tier("generation"), InferenceTier.TIER_3)

    def test_recommend_tier_unknown(self):
        self.assertEqual(self.mgr.recommend_tier("unknown_task"), InferenceTier.TIER_2)

    def test_tier_summary(self):
        summary = self.mgr.tier_summary()
        self.assertIn("Tier 1", summary)
        self.assertIn("Tier 3", summary)
        self.assertIn("visual_encoder", summary)

    def test_invalid_tier(self):
        with self.assertRaises(ValueError):
            self.mgr.get_pricing(99)


# =========================================================================
# Decode Verifier (Story 9.4)
# =========================================================================


class TestDecodeOutput(unittest.TestCase):
    """Tests for DecodeOutput."""

    def test_output_hash_computed(self):
        out = DecodeOutput(
            node_id="node-1",
            generated_ids=[1, 2, 3],
            latent_plan_hash="abc123",
        )
        self.assertNotEqual(out.output_hash, "")
        self.assertEqual(len(out.output_hash), 64)  # SHA-256 hex

    def test_same_ids_same_hash(self):
        out1 = DecodeOutput(node_id="n1", generated_ids=[1, 2, 3], latent_plan_hash="x")
        out2 = DecodeOutput(node_id="n2", generated_ids=[1, 2, 3], latent_plan_hash="x")
        self.assertEqual(out1.output_hash, out2.output_hash)

    def test_different_ids_different_hash(self):
        out1 = DecodeOutput(node_id="n1", generated_ids=[1, 2, 3], latent_plan_hash="x")
        out2 = DecodeOutput(node_id="n2", generated_ids=[1, 2, 4], latent_plan_hash="x")
        self.assertNotEqual(out1.output_hash, out2.output_hash)


class TestDecodeVerifier(unittest.TestCase):
    """Tests for DecodeVerifier."""

    def _make_output(self, node_id, ids, plan_hash="plan_abc"):
        return DecodeOutput(
            node_id=node_id,
            generated_ids=ids,
            latent_plan_hash=plan_hash,
        )

    def test_insufficient_verifiers(self):
        v = DecodeVerifier(min_verifiers=3)
        v.submit_output(self._make_output("n1", [1, 2, 3]))
        report = v.verify()
        self.assertEqual(report.result, VerificationResult.INSUFFICIENT_VERIFIERS)
        self.assertFalse(report.is_verified)

    def test_unanimous_agreement(self):
        v = DecodeVerifier(min_verifiers=3, agreement_threshold=0.67)
        v.submit_output(self._make_output("n1", [1, 2, 3]))
        v.submit_output(self._make_output("n2", [1, 2, 3]))
        v.submit_output(self._make_output("n3", [1, 2, 3]))
        report = v.verify()
        self.assertEqual(report.result, VerificationResult.VERIFIED)
        self.assertTrue(report.is_verified)
        self.assertAlmostEqual(report.agreement_ratio, 1.0)
        self.assertEqual(report.dissenting_nodes, [])

    def test_majority_with_dissenter(self):
        v = DecodeVerifier(min_verifiers=3, agreement_threshold=0.66)
        v.submit_output(self._make_output("n1", [1, 2, 3]))
        v.submit_output(self._make_output("n2", [1, 2, 3]))
        v.submit_output(self._make_output("n3", [9, 9, 9]))  # Dissenter
        report = v.verify()
        self.assertEqual(report.result, VerificationResult.VERIFIED)
        self.assertAlmostEqual(report.agreement_ratio, 2 / 3, places=2)
        self.assertEqual(report.dissenting_nodes, ["n3"])

    def test_mismatch(self):
        v = DecodeVerifier(min_verifiers=2, agreement_threshold=0.67)
        v.submit_output(self._make_output("n1", [1, 2, 3]))
        v.submit_output(self._make_output("n2", [4, 5, 6]))
        v.submit_output(self._make_output("n3", [7, 8, 9]))
        report = v.verify()
        self.assertEqual(report.result, VerificationResult.MISMATCH)
        self.assertFalse(report.is_verified)

    def test_mixed_plan_hashes(self):
        v = DecodeVerifier(min_verifiers=2)
        v.submit_output(self._make_output("n1", [1, 2], plan_hash="plan_a"))
        v.submit_output(self._make_output("n2", [1, 2], plan_hash="plan_b"))
        report = v.verify()
        self.assertEqual(report.result, VerificationResult.ERROR)

    def test_reset(self):
        v = DecodeVerifier(min_verifiers=2)
        v.submit_output(self._make_output("n1", [1]))
        self.assertEqual(len(v._outputs), 1)
        v.reset()
        self.assertEqual(len(v._outputs), 0)

    def test_two_verifiers_sufficient(self):
        v = DecodeVerifier(min_verifiers=2, agreement_threshold=1.0)
        v.submit_output(self._make_output("n1", [42, 43]))
        v.submit_output(self._make_output("n2", [42, 43]))
        report = v.verify()
        self.assertTrue(report.is_verified)


# =========================================================================
# Inference Pipeline (Story 9.1)
# =========================================================================


class TestInferenceRequest(unittest.TestCase):
    """Tests for InferenceRequest."""

    def test_create_request(self):
        req = InferenceRequest(
            request_id="req-1",
            tier=InferenceTier.TIER_2,
            input_data={"images": "placeholder"},
            requester_id="user-1",
        )
        self.assertEqual(req.request_id, "req-1")
        self.assertGreater(req.timestamp, 0)


class TestInferencePipeline(unittest.TestCase):
    """Tests for InferencePipeline."""

    def setUp(self):
        self.pipeline = InferencePipeline()

    def test_plan_route_tier_1(self):
        req = InferenceRequest(
            request_id="r1", tier=InferenceTier.TIER_1,
            input_data={}, requester_id="u1",
        )
        routes = self.pipeline.plan_route(req)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].module_name, "visual_encoder")

    def test_plan_route_tier_3(self):
        req = InferenceRequest(
            request_id="r1", tier=InferenceTier.TIER_3,
            input_data={}, requester_id="u1",
        )
        routes = self.pipeline.plan_route(req)
        self.assertEqual(len(routes), 5)

    def test_execute_local(self):
        """Execute in local/stub mode."""
        req = InferenceRequest(
            request_id="r1", tier=InferenceTier.TIER_1,
            input_data={"images": "test_data"}, requester_id="u1",
        )
        result = self.pipeline.execute(req)
        self.assertIsInstance(result, InferenceResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.modules_executed), 1)
        self.assertGreater(result.total_latency_ms, 0)

    def test_execute_tier_2(self):
        req = InferenceRequest(
            request_id="r2", tier=InferenceTier.TIER_2,
            input_data={}, requester_id="u1",
        )
        result = self.pipeline.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(len(result.modules_executed), 4)

    def test_stats_tracking(self):
        req = InferenceRequest(
            request_id="r1", tier=InferenceTier.TIER_1,
            input_data={}, requester_id="u1",
        )
        self.pipeline.execute(req)
        self.pipeline.execute(req)
        self.assertEqual(self.pipeline.stats.requests_total, 2)
        self.assertEqual(self.pipeline.stats.requests_completed, 2)

    def test_pipeline_stats_avg_latency(self):
        stats = PipelineStats(
            requests_total=10, requests_completed=10,
            requests_failed=0, total_latency_ms=1000.0,
        )
        self.assertAlmostEqual(stats.avg_latency_ms, 100.0)

    def test_pipeline_stats_success_rate(self):
        stats = PipelineStats(
            requests_total=10, requests_completed=8,
            requests_failed=2, total_latency_ms=0,
        )
        self.assertAlmostEqual(stats.success_rate, 0.8)

    def test_pipeline_stats_empty(self):
        stats = PipelineStats()
        self.assertEqual(stats.avg_latency_ms, 0.0)
        self.assertEqual(stats.success_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
