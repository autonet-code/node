"""
Integration tests for the ATN latent reasoning pipeline.

Tests the full pipeline:
    TraceEncoder → VL-JEPA → LatentReasoningEngine → ActionDecoder
as well as module interface compatibility, collapse detection, and OOD routing.

Run with:
    python -m pytest nodes/common/test_integration.py -v
or from nodes/common/:
    python -m pytest test_integration.py -v
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
import torch.nn as nn

# Ensure the project root is on sys.path so both relative and absolute imports
# work regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nodes.common.trace_encoder import TraceEncoder, TraceEncoderConfig
from nodes.common.vl_jepa import (
    VLJEPA,
    VLJEPAConfig,
    LatentDynamicsPredictor,
    get_vljepa_preset,
)
from nodes.common.action_decoder import (
    ActionDecoder,
    ActionDecoderConfig,
    DecoderOutput,
    ToolCallOutput,
)
from nodes.common.latent_reasoning import (
    LatentReasoningEngine,
    LatentReasoningConfig,
    ReasoningTrace,
)
from nodes.common.two_speed import TwoSpeedEngine, TwoSpeedConfig
from nodes.common.knowledge_acq import (
    KnowledgeAcquisitionEngine,
    KnowledgeAcqConfig,
    OODConfig,
    AcquisitionPath,
    UncertaintyType,
)
from nodes.common.tokenizer import SimpleTokenizer

# ---------------------------------------------------------------------------
# Constants for the small model config
# ---------------------------------------------------------------------------
SMALL_DIM = 256    # embed_dim for the "small" VLJEPA preset
SMALL_K = 16       # num_latent_vectors for "small"
SMALL_PATCHES = 64 # (image_size=64 // patch_size=8) ** 2

# ---------------------------------------------------------------------------
# Synthetic trace fixture (3 turns, succeeds — passes quality filter)
# ---------------------------------------------------------------------------
SYNTHETIC_TRACE: Dict[str, Any] = {
    "session_id": "integ-test-001",
    "agent_id": "test-agent",
    "agent_type": "solver",
    "timestamp": "2026-03-28T00:00:00Z",
    "turns": [
        {
            "role": "user",
            "content": "Find all Python files containing the string 'TraceEncoder'.",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "I will search for the class definition.",
            "tool_calls": [
                {
                    "tool": "grep",
                    "args": {"pattern": "TraceEncoder", "path": "./"},
                    "result": "nodes/common/trace_encoder.py:498:class TraceEncoder",
                    "success": True,
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Found in nodes/common/trace_encoder.py at line 498.",
            "tool_calls": [],
        },
    ],
    "outcome": {"success": True, "task_completed": True, "errors": []},
}


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def vljepa_small() -> VLJEPA:
    """VLJEPA with the 'small' preset — CPU, random weights."""
    model = VLJEPA(get_vljepa_preset("small"))
    model.eval()
    return model


@pytest.fixture(scope="module")
def trace_encoder_small() -> TraceEncoder:
    """TraceEncoder with embed_dim and num_heads matching the small VLJEPA.

    num_heads must divide embed_dim evenly; VLJEPA small uses num_heads=8 for
    embed_dim=256 (256 // 8 = 32 head_dim).
    """
    enc = TraceEncoder(TraceEncoderConfig(embed_dim=SMALL_DIM, num_heads=8))
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def action_decoder_small() -> ActionDecoder:
    """ActionDecoder with jepa_dim matching the small VLJEPA."""
    cfg = ActionDecoderConfig(
        jepa_dim=SMALL_DIM,
        decoder_dim=128,
        num_heads=4,
        decoder_depth=2,
        adapter_hidden_dim=256,
        max_output_length=32,
    )
    dec = ActionDecoder(cfg)
    dec.eval()
    return dec


@pytest.fixture(scope="module")
def reasoning_engine_small(vljepa_small: VLJEPA) -> LatentReasoningEngine:
    """LatentReasoningEngine using the small model's dynamics predictor."""
    cfg = LatentReasoningConfig(
        max_steps=5,
        replan_interval=0,       # no scheduled replans for basic tests
        fallback_on_collapse=True,
    )
    return LatentReasoningEngine(
        dynamics_predictor=vljepa_small.dynamics_predictor,
        config=cfg,
    )


@pytest.fixture(scope="module")
def tokenizer_small() -> SimpleTokenizer:
    return SimpleTokenizer(vocab_size=260)


def _make_token_ids(text: str, max_length: int = 64):
    """Tokenise text → (1, max_length) LongTensor + (1, max_length) bool mask."""
    tok = SimpleTokenizer(vocab_size=260)
    ids, mask = tok.batch_encode([text], max_length=max_length)
    return ids, mask  # each (1, max_length)


# ===========================================================================
# 1. End-to-End Pipeline Test
# ===========================================================================


class TestEndToEndPipeline:
    """Smoke-test for the full trace → decode pipeline."""

    def test_full_pipeline(
        self,
        trace_encoder_small: TraceEncoder,
        vljepa_small: VLJEPA,
        reasoning_engine_small: LatentReasoningEngine,
        action_decoder_small: ActionDecoder,
    ):
        """Synthetic trace → TraceEncoder → VL-JEPA → reason → ActionDecoder."""
        # --- Step 1: Encode the synthetic trace ---
        with torch.no_grad():
            encoded = trace_encoder_small.encode_trace(SYNTHETIC_TRACE)

        assert encoded is not None, "Trace failed quality filter unexpectedly"
        turn_embs = encoded["embeddings"]    # (max_turns, D)
        turn_mask = encoded["turn_mask"]     # (max_turns,)

        assert turn_embs.shape == (trace_encoder_small.config.max_turns, SMALL_DIM)
        assert turn_mask.dtype == torch.bool
        assert turn_mask.any(), "At least one real turn must be present"

        # --- Step 2: Use turn embeddings as visual context for VL-JEPA ---
        # (max_turns=64 == num_patches=64 for small preset — by design)
        visual_ctx = turn_embs.unsqueeze(0)   # (1, 64, 256)
        assert visual_ctx.shape == (1, SMALL_PATCHES, SMALL_DIM)

        # --- Step 3: Encode a text query ---
        token_ids, attn_mask = _make_token_ids("Which tool searches file content?")

        with torch.no_grad():
            text_embs = vljepa_small.encode_text(token_ids, attn_mask)  # (1, S, D)
            fused = vljepa_small.fuse(visual_ctx, text_embs)            # (1, ?, D)
            latent_plan = vljepa_small.predict_semantic(fused)          # (1, K, D)

        assert latent_plan.shape == (1, SMALL_K, SMALL_DIM)

        # --- Step 4: Latent reasoning (3 steps) ---
        trace = reasoning_engine_small.reason(latent_plan, num_steps=3)

        assert isinstance(trace, ReasoningTrace)
        assert trace.final_plan is not None
        assert trace.final_plan.shape == (1, SMALL_K, SMALL_DIM)

        # --- Step 5: Action decoding ---
        with torch.no_grad():
            # ActionDecoder.decode() accepts (B, S, jepa_dim) where S = K latent vecs
            decoder_out = action_decoder_small.decode(trace.final_plan)

        assert isinstance(decoder_out, DecoderOutput)
        assert len(decoder_out.tool_calls) == 1
        tc = decoder_out.tool_calls[0]
        assert isinstance(tc, ToolCallOutput)
        assert isinstance(tc.tool, str)
        assert isinstance(tc.args, dict)

    def test_trace_quality_filter_rejects_short(
        self, trace_encoder_small: TraceEncoder
    ):
        """Traces with fewer than min_turns should be filtered out."""
        one_turn_trace = {**SYNTHETIC_TRACE, "turns": SYNTHETIC_TRACE["turns"][:1]}
        result = trace_encoder_small.encode_trace(one_turn_trace)
        assert result is None, "Single-turn trace should be rejected"

    def test_trace_quality_filter_rejects_errored(
        self, trace_encoder_small: TraceEncoder
    ):
        """Traces with errors should be rejected when include_errored=False."""
        errored_trace = {
            **SYNTHETIC_TRACE,
            "outcome": {"success": False, "task_completed": False, "errors": ["oops"]},
        }
        result = trace_encoder_small.encode_trace(errored_trace)
        # Default config has include_errored=False
        assert result is None


# ===========================================================================
# 2. Module Interface Compatibility Tests
# ===========================================================================


class TestModuleInterfaces:
    """Verify output shapes / types match the next module's expected inputs."""

    def test_trace_encoder_output_dim_matches_vljepa(
        self,
        trace_encoder_small: TraceEncoder,
        vljepa_small: VLJEPA,
    ):
        """TraceEncoder embed_dim must equal VL-JEPA embed_dim."""
        assert trace_encoder_small.config.embed_dim == vljepa_small.config.embed_dim, (
            f"TraceEncoder.embed_dim={trace_encoder_small.config.embed_dim} "
            f"!= VLJEPA.embed_dim={vljepa_small.config.embed_dim}"
        )

    def test_trace_encoder_output_is_vljepa_compatible(
        self,
        trace_encoder_small: TraceEncoder,
        vljepa_small: VLJEPA,
    ):
        """TraceEncoder turn embeddings can feed directly into VLJEPA.fuse()."""
        with torch.no_grad():
            encoded = trace_encoder_small.encode_trace(SYNTHETIC_TRACE)
        assert encoded is not None

        visual_ctx = encoded["embeddings"].unsqueeze(0)  # (1, T, D)
        token_ids, attn_mask = _make_token_ids("test query")

        with torch.no_grad():
            text_embs = vljepa_small.encode_text(token_ids, attn_mask)
            fused = vljepa_small.fuse(visual_ctx, text_embs)
            latent_plan = vljepa_small.predict_semantic(fused)

        assert latent_plan.shape[-1] == SMALL_DIM
        assert latent_plan.shape[-2] == SMALL_K

    def test_vljepa_rollout_output_matches_reasoning_engine_input(
        self, vljepa_small: VLJEPA
    ):
        """VLJEPA.rollout() predictions are (B, K, D) — matching LatentReasoningEngine."""
        B, K, D = 1, SMALL_K, SMALL_DIM
        plan = torch.randn(B, K, D)

        with torch.no_grad():
            rollout = vljepa_small.rollout(plan, num_steps=3)

        preds = rollout["predictions"]
        assert len(preds) == 3
        for p in preds:
            assert p.shape == (B, K, D), f"Unexpected prediction shape: {p.shape}"

        # LatentReasoningEngine.reason() takes (B, K, D) — verify it accepts the prediction
        cfg = LatentReasoningConfig(max_steps=2, replan_interval=0)
        engine = LatentReasoningEngine(
            dynamics_predictor=vljepa_small.dynamics_predictor,
            config=cfg,
        )
        trace = engine.reason(preds[-1], num_steps=2)
        assert trace.final_plan is not None
        assert trace.final_plan.shape == (B, K, D)

    def test_reasoning_engine_output_matches_action_decoder_input(
        self,
        vljepa_small: VLJEPA,
        action_decoder_small: ActionDecoder,
    ):
        """LatentReasoningEngine.final_plan is (B, K, D) — ActionDecoder accepts (B, S, D)."""
        plan = torch.randn(1, SMALL_K, SMALL_DIM)
        cfg = LatentReasoningConfig(max_steps=2, replan_interval=0)
        engine = LatentReasoningEngine(
            dynamics_predictor=vljepa_small.dynamics_predictor,
            config=cfg,
        )
        trace = engine.reason(plan, num_steps=2)

        final_plan = trace.final_plan  # (1, K, D) = (1, 16, 256)
        assert final_plan is not None

        # ActionDecoder.decode() accepts (B, S, jepa_dim); here S=K, jepa_dim=D
        with torch.no_grad():
            out = action_decoder_small.decode(final_plan)

        assert isinstance(out, DecoderOutput)
        assert out.input_embeddings.shape[-1] == action_decoder_small.config.decoder_dim

    def test_two_speed_engine_routing(self, vljepa_small: VLJEPA):
        """TwoSpeedEngine correctly routes between fast and slow paths."""
        tok = SimpleTokenizer(vocab_size=260)
        cfg = TwoSpeedConfig(
            complexity_threshold=0.6,
            confidence_threshold=0.5,
            cache_size=16,
            rollout_steps=1,   # keep it fast for tests
            fast_max_len=8,
            slow_max_len=16,
            temperature=0.0,
        )
        engine = TwoSpeedEngine.from_vljepa(vljepa_small, config=cfg, tokenizer=tok)

        query = "search for Python files"
        token_ids, attn_mask = _make_token_ids(query, max_length=32)

        # First call: cold cache → slow path
        result1 = engine.infer_sync(token_ids, attn_mask, query_text=query)
        assert result1.path_taken == "slow", (
            "First call with cold cache should take the slow path"
        )
        assert result1.latent_plan.shape == (1, SMALL_K, SMALL_DIM)
        assert isinstance(result1.confidence, float)
        assert 0.0 <= result1.confidence <= 1.0
        assert isinstance(result1.complexity_score, float)

        # Second call: warm cache with identical query → fast path
        result2 = engine.infer_sync(token_ids, attn_mask, query_text=query)
        assert result2.path_taken == "fast", (
            "Second call with warm cache and same query should take the fast path"
        )


# ===========================================================================
# 3. Collapse Detection Test
# ===========================================================================


class _ConstantPredictor(nn.Module):
    """Mock dynamics predictor that always returns its initial input unchanged.

    Produces cosine_sim_to_prev ≈ 1.0 (stagnation) and effective_rank ≈ 1
    (all K vectors identical after first step), both triggering collapse.
    """

    def __init__(self, constant: torch.Tensor):
        super().__init__()
        # Register as buffer so it follows .eval() / .to() correctly
        self.register_buffer("_const", constant.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self._const.expand_as(x)


class TestCollapseDetection:
    """Verify the LatentReasoningEngine detects and handles embedding collapse."""

    def test_constant_embeddings_trigger_collapse(self):
        """Constant DynamicsPredictor output → collapse detected at step 1."""
        B, K, D = 1, SMALL_K, SMALL_DIM

        # Use a non-trivial initial plan so baseline spread is non-zero
        initial_plan = torch.randn(B, K, D)
        # Predictor always returns the SAME constant tensor
        constant_out = torch.zeros(B, K, D)  # all-zero → rank=1, stagnates
        predictor = _ConstantPredictor(constant_out)

        cfg = LatentReasoningConfig(
            max_steps=5,
            replan_interval=0,
            fallback_on_collapse=True,
            cosine_sim_ceiling=0.98,
            spectral_rank_min=4,
        )
        engine = LatentReasoningEngine(dynamics_predictor=predictor, config=cfg)
        trace = engine.reason(initial_plan, num_steps=5)

        # Engine should have terminated early due to collapse
        assert trace.terminated_early, (
            "Constant predictor should trigger collapse and early termination"
        )
        assert len(trace.steps) >= 1, "At least one step should be recorded"

        # The last recorded step must be flagged as collapsed
        collapsed_steps = [s for s in trace.steps if s.is_collapsed]
        assert len(collapsed_steps) >= 1, (
            "At least one step must be marked is_collapsed=True"
        )

    def test_collapse_detection_fires_before_max_steps(self):
        """Collapse terminates before reaching max_steps."""
        B, K, D = 1, 8, 64
        initial_plan = torch.randn(B, K, D)
        predictor = _ConstantPredictor(torch.zeros(B, K, D))

        cfg = LatentReasoningConfig(max_steps=10, replan_interval=0, fallback_on_collapse=True)
        engine = LatentReasoningEngine(dynamics_predictor=predictor, config=cfg)
        trace = engine.reason(initial_plan, num_steps=10)

        assert trace.terminated_early
        assert len(trace.steps) < 10, (
            "Chain should terminate before max_steps when collapse is detected"
        )

    def test_replan_callback_fires_on_scheduled_interval(self):
        """replan_fn is called on scheduled replan checkpoints."""
        B, K, D = 1, SMALL_K, SMALL_DIM
        initial_plan = torch.randn(B, K, D)

        replan_calls: List[torch.Tensor] = []

        def replan_fn(plan: torch.Tensor) -> torch.Tensor:
            replan_calls.append(plan.detach().clone())
            # Return a fresh random plan to break stagnation
            return torch.randn_like(plan)

        # Standard (non-constant) predictor from VLJEPA small
        cfg_vljepa = get_vljepa_preset("small")
        predictor = LatentDynamicsPredictor(cfg_vljepa)

        cfg = LatentReasoningConfig(
            max_steps=6,
            replan_interval=2,        # replan every 2 steps
            fallback_on_collapse=False,  # don't stop on collapse
            decode_on_replan=True,
        )
        engine = LatentReasoningEngine(
            dynamics_predictor=predictor,
            config=cfg,
            replan_fn=replan_fn,
        )
        trace = engine.reason(initial_plan, num_steps=6)

        assert len(replan_calls) >= 1, (
            "replan_fn should have been called at least once with replan_interval=2"
        )
        assert len(trace.replan_events) >= 1, "ReplanEvents should be recorded"

    def test_degradation_curves_populated(self, vljepa_small: VLJEPA):
        """ReasoningTrace.degradation_curves() returns all 6 expected metrics."""
        plan = torch.randn(1, SMALL_K, SMALL_DIM)
        cfg = LatentReasoningConfig(max_steps=4, replan_interval=0, fallback_on_collapse=False)
        engine = LatentReasoningEngine(
            dynamics_predictor=vljepa_small.dynamics_predictor,
            config=cfg,
        )
        trace = engine.reason(plan, num_steps=4)
        curves = trace.degradation_curves()

        expected_keys = {
            "embedding_variance",
            "spread",
            "effective_rank",
            "cosine_sim_to_prev",
            "uncertainty",
            "quality",
        }
        assert set(curves.keys()) == expected_keys, (
            f"Missing metrics: {expected_keys - set(curves.keys())}"
        )
        for key, vals in curves.items():
            assert len(vals) == len(trace.steps), (
                f"Curve '{key}' length {len(vals)} != num_steps {len(trace.steps)}"
            )
            assert all(isinstance(v, float) for v in vals), (
                f"All values in curve '{key}' should be floats"
            )


# ===========================================================================
# 4. OOD Detection Test
# ===========================================================================


class TestOODDetection:
    """Verify KnowledgeAcquisitionEngine routes OOD queries to token-space fallback."""

    def test_empty_index_always_epistemic(self, vljepa_small: VLJEPA):
        """With no in-distribution embeddings, all queries are epistemic OOD."""
        engine = KnowledgeAcquisitionEngine(
            model=vljepa_small,
            config=KnowledgeAcqConfig(
                ood=OODConfig(epistemic_threshold=0.65),
            ),
            provider_fn=None,  # stub — no real LLM needed
        )

        ood_embedding = torch.randn(SMALL_DIM)

        result = engine.acquire(
            query_embedding=ood_embedding,
            query_text="completely unknown query",
        )

        assert result.ood_result.uncertainty_type == UncertaintyType.EPISTEMIC, (
            "Empty KNN index should classify all queries as EPISTEMIC"
        )
        assert result.path_taken == AcquisitionPath.TOKEN_SPACE, (
            "Epistemic OOD + empty store should route to TOKEN_SPACE"
        )

    def test_in_distribution_queries_not_epistemic(self, vljepa_small: VLJEPA):
        """Queries close to the in-distribution index should not be epistemic."""
        engine = KnowledgeAcquisitionEngine(
            model=vljepa_small,
            config=KnowledgeAcqConfig(
                ood=OODConfig(
                    epistemic_threshold=0.65,
                    knn_k=1,
                ),
            ),
            provider_fn=None,
        )

        # Populate with 20 in-distribution embeddings
        D = SMALL_DIM
        ref_embeddings = torch.randn(20, D)
        ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)

        for emb in ref_embeddings:
            engine.ood_detector.add_in_distribution(emb)

        # Query close to one of the reference embeddings
        close_query = ref_embeddings[0] + torch.randn(D) * 0.01  # tiny perturbation
        close_query = close_query / close_query.norm()

        ood_result = engine.ood_detector.detect(close_query)
        # KNN distance should be very small → not epistemic
        assert ood_result.uncertainty_type != UncertaintyType.EPISTEMIC, (
            "Query very close to training data should not be epistemic"
        )

    def test_ood_routing_uses_token_space_fallback(self, vljepa_small: VLJEPA):
        """OOD queries with empty store route to TOKEN_SPACE and record path."""
        fallback_calls: List[str] = []

        def stub_provider(query: str) -> str:
            fallback_calls.append(query)
            return f"mock answer for: {query}"

        engine = KnowledgeAcquisitionEngine(
            model=vljepa_small,
            config=KnowledgeAcqConfig(
                ood=OODConfig(epistemic_threshold=0.65),
            ),
            provider_fn=stub_provider,
        )

        result = engine.acquire(
            query_embedding=torch.randn(SMALL_DIM),
            query_text="novel unknown topic",
        )

        assert result.path_taken == AcquisitionPath.TOKEN_SPACE
        assert engine.fallback.total_calls == 1

    def test_ood_result_fields_populated(self, vljepa_small: VLJEPA):
        """OODResult has all expected fields with values in valid ranges."""
        engine = KnowledgeAcquisitionEngine(
            model=vljepa_small,
            provider_fn=None,
        )

        result = engine.acquire(
            query_embedding=torch.randn(SMALL_DIM),
            query_text="",
        )

        r = result.ood_result
        assert 0.0 <= r.variance_score <= 1.0
        assert 0.0 <= r.knn_distance_score <= 1.0
        assert 0.0 <= r.combined_score <= 1.0
        assert isinstance(r.uncertainty_type, UncertaintyType)
        assert r.latency_ms >= 0.0
