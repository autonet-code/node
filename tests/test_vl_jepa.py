"""
Tests for VL-JEPA model and modular abstraction.

Tests cover:
- VL-JEPA forward pass shapes (latent plan output)
- Individual component isolation (TextEncoder, CrossModalFusion, etc.)
- Self-supervised training loss
- Selective decoding (based on latent plan change)
- EMA target encoder updates
- Modular split correctness
- Pipeline vs monolithic equivalence
- Module serialization / weight loading
- EMA blending
- Intelligence tiers (partial pipelines)
- Local-only decoder flag
- Tokenizer round-trip
"""

import copy
import math
import pytest
import torch
import torch.nn as nn

from nodes.common.jepa import JEPAConfig, VisionTransformerEncoder, get_1d_sincos_pos_embed
from nodes.common.vl_jepa import (
    VLJEPA,
    VLJEPAConfig,
    TextEncoder,
    CrossModalFusion,
    SemanticPredictor,
    TextDecoder,
    CrossAttention,
)
from nodes.common.model_module import (
    ModelModule,
    ModularVLJEPA,
    ModuleSpec,
    MODULE_NAMES,
)
from nodes.common.tokenizer import SimpleTokenizer, PAD_ID, BOS_ID, EOS_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(**overrides) -> VLJEPAConfig:
    """Small config for fast tests."""
    defaults = dict(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_heads=4,
        encoder_depth=2,
        mlp_ratio=2.0,
        vocab_size=260,
        max_seq_length=16,
        text_encoder_depth=2,
        fusion_depth=2,
        semantic_predictor_depth=2,
        num_latent_vectors=8,  # K=8 for fast tests
        decoder_depth=2,
        semantic_change_threshold=0.1,
    )
    defaults.update(overrides)
    return VLJEPAConfig(**defaults)


def make_inputs(config: VLJEPAConfig, batch_size: int = 2):
    """Create dummy inputs matching config."""
    images = torch.randn(batch_size, config.in_channels, config.image_size, config.image_size)
    seq_len = 8  # shorter than max for variety
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    target_token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    return images, token_ids, attention_mask, target_token_ids


# ===========================================================================
# VL-JEPA Model Tests
# ===========================================================================

class TestVLJEPA:

    def test_forward_shapes(self):
        """Full model forward produces correct output shapes."""
        cfg = make_config()
        model = VLJEPA(cfg)
        images, token_ids, attn_mask, target_ids = make_inputs(cfg)

        result = model(images, token_ids, attn_mask, target_token_ids=target_ids)

        assert "latent_plan" in result
        assert result["latent_plan"].shape == (2, cfg.num_latent_vectors, cfg.embed_dim)
        assert "decoder_logits" in result
        assert result["decoder_logits"].shape == (2, target_ids.shape[1], cfg.vocab_size)

    def test_forward_without_decoder(self):
        """Forward without target_token_ids skips decoder."""
        cfg = make_config()
        model = VLJEPA(cfg)
        images, token_ids, attn_mask, _ = make_inputs(cfg)

        result = model(images, token_ids, attn_mask)

        assert "latent_plan" in result
        assert "decoder_logits" not in result

    def test_visual_encoder_only(self):
        """Visual encoder alone produces patch embeddings."""
        cfg = make_config()
        model = VLJEPA(cfg)
        images = torch.randn(2, 3, 32, 32)

        embeds = model.encode_visual(images)

        num_patches = (32 // 8) ** 2  # = 16
        assert embeds.shape == (2, num_patches, cfg.embed_dim)

    def test_text_encoder_only(self):
        """Text encoder produces sequence embeddings."""
        cfg = make_config()
        model = VLJEPA(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 10))

        embeds = model.encode_text(token_ids)

        assert embeds.shape == (2, 10, cfg.embed_dim)

    def test_cross_modal_fusion_shapes(self):
        """Fusion combines visual + text correctly."""
        cfg = make_config()
        model = VLJEPA(cfg)
        visual = torch.randn(2, 16, cfg.embed_dim)
        text = torch.randn(2, 8, cfg.embed_dim)

        fused = model.fuse(visual, text)

        # Fusion output has same seq len as text input
        assert fused.shape == (2, 8, cfg.embed_dim)

    def test_semantic_prediction_shape(self):
        """Predictor produces latent plan (K vectors)."""
        cfg = make_config()
        model = VLJEPA(cfg)
        fused = torch.randn(2, 8, cfg.embed_dim)

        latent_plan = model.predict_semantic(fused)

        assert latent_plan.shape == (2, cfg.num_latent_vectors, cfg.embed_dim)

    def test_text_generation(self):
        """Decoder generates token sequence from latent plan."""
        cfg = make_config()
        model = VLJEPA(cfg)
        images, token_ids, attn_mask, _ = make_inputs(cfg)

        generated = model.generate(images, token_ids, attn_mask, max_len=12)

        assert generated.shape[0] == 2
        assert generated.shape[1] <= 12
        # First token should be BOS
        assert (generated[:, 0] == BOS_ID).all()

    def test_selective_decoding_first_call(self):
        """First call to should_decode always returns True."""
        cfg = make_config()
        model = VLJEPA(cfg)
        plan = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)

        assert model.should_decode(plan) is True

    def test_selective_decoding_no_change(self):
        """Identical latent plan -> should_decode returns False."""
        cfg = make_config()
        model = VLJEPA(cfg)
        plan = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)

        model.should_decode(plan)  # first call
        assert model.should_decode(plan) is False

    def test_selective_decoding_with_change(self):
        """Sufficiently different latent plan -> should_decode returns True."""
        cfg = make_config(semantic_change_threshold=0.01)
        model = VLJEPA(cfg)
        plan1 = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)
        plan2 = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)

        model.should_decode(plan1)
        assert model.should_decode(plan2) is True

    def test_self_supervised_loss(self):
        """Self-supervised loss computes and is finite."""
        cfg = make_config()
        model = VLJEPA(cfg)
        images, token_ids, attn_mask, _ = make_inputs(cfg)

        result = model.self_supervised_loss(images, token_ids, attn_mask)

        assert "loss" in result
        assert "visual_loss" in result
        assert "latent_plan" in result
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() >= 0
        assert result["latent_plan"].shape == (2, cfg.num_latent_vectors, cfg.embed_dim)

    def test_self_supervised_training_step(self):
        """Training loop runs without error and loss remains finite."""
        cfg = make_config()
        model = VLJEPA(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        images, token_ids, attn_mask, _ = make_inputs(cfg, batch_size=4)

        losses = []
        for _ in range(5):
            result = model.self_supervised_loss(images, token_ids, attn_mask)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder()
            losses.append(loss.item())

        # All losses should be finite
        assert all(math.isfinite(l) for l in losses)

    def test_ema_target_encoder_update(self):
        """Target encoder tracks context encoder via EMA."""
        cfg = make_config()
        model = VLJEPA(cfg)

        # Change visual encoder weights
        with torch.no_grad():
            for p in model.visual_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Before update: target != visual
        ve_params = list(model.visual_encoder.parameters())
        te_params = list(model.target_encoder.parameters())
        diff_before = sum(
            (a - b).abs().sum().item()
            for a, b in zip(ve_params, te_params)
        )
        assert diff_before > 0

        # After EMA update: target moves toward visual
        model.update_target_encoder(momentum=0.5)
        diff_after = sum(
            (a - b).abs().sum().item()
            for a, b in zip(
                model.visual_encoder.parameters(),
                model.target_encoder.parameters(),
            )
        )
        assert diff_after < diff_before

    def test_latent_plan_bytes(self):
        """Config correctly computes latent plan transfer size."""
        cfg = make_config()
        assert cfg.latent_plan_bytes == cfg.num_latent_vectors * cfg.embed_dim * 2
        # For test config: 8 * 64 * 2 = 1024 bytes
        assert cfg.latent_plan_bytes == 1024


# ===========================================================================
# Component Tests
# ===========================================================================

class TestComponents:

    def test_cross_attention(self):
        """CrossAttention produces correct shapes."""
        ca = CrossAttention(embed_dim=64, num_heads=4)
        q = torch.randn(2, 8, 64)
        ctx = torch.randn(2, 16, 64)
        out = ca(q, ctx)
        assert out.shape == (2, 8, 64)

    def test_text_encoder_standalone(self):
        """TextEncoder works in isolation."""
        cfg = make_config()
        enc = TextEncoder(cfg)
        ids = torch.randint(0, 260, (2, 10))
        out = enc(ids)
        assert out.shape == (2, 10, cfg.embed_dim)

    def test_semantic_predictor_standalone(self):
        """SemanticPredictor produces K-vector latent plan."""
        cfg = make_config()
        pred = SemanticPredictor(cfg)
        x = torch.randn(2, 8, cfg.embed_dim)
        out = pred(x)
        assert out.shape == (2, cfg.num_latent_vectors, cfg.embed_dim)

    def test_text_decoder_forward(self):
        """TextDecoder teacher-forced forward with latent plan."""
        cfg = make_config()
        dec = TextDecoder(cfg)
        latent_plan = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)
        target = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = dec(latent_plan, target)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_text_decoder_generate(self):
        """TextDecoder autoregressive generation from latent plan."""
        cfg = make_config()
        dec = TextDecoder(cfg)
        latent_plan = torch.randn(2, cfg.num_latent_vectors, cfg.embed_dim)
        generated = dec.generate(latent_plan, max_len=10, temperature=1.0)
        assert generated.shape[0] == 2
        assert generated.shape[1] <= 10

    def test_1d_sincos_pos_embed(self):
        """1D sinusoidal pos embed has correct shape."""
        embed = get_1d_sincos_pos_embed(64, 32)
        assert embed.shape == (32, 64)

    def test_forward_layers(self):
        """VisionTransformerEncoder.forward_layers runs subset."""
        cfg = make_config()
        enc = VisionTransformerEncoder(cfg)
        images = torch.randn(2, 3, 32, 32)

        # Full forward
        full_out = enc(images)

        # Get pre-embedded input
        x = enc.patch_embed(images)
        x = x + enc.pos_embed

        # Run all layers via forward_layers
        layers_out = enc.forward_layers(x, 0, None)

        assert torch.allclose(full_out, layers_out, atol=1e-5)

    def test_decoder_cross_attends_to_full_plan(self):
        """Decoder cross-attention uses all K vectors, not just one."""
        cfg = make_config(num_latent_vectors=4)
        dec = TextDecoder(cfg)
        target = torch.randint(0, cfg.vocab_size, (1, 4))

        # Compare output with K=4 plan vs single-vector plan repeated
        plan_k4 = torch.randn(1, 4, cfg.embed_dim)
        plan_k1 = plan_k4[:, :1, :].expand(-1, 4, -1)  # same vector repeated

        logits_k4 = dec(plan_k4, target)
        logits_k1 = dec(plan_k1, target)

        # Outputs should differ — decoder uses the full K-vector context
        assert not torch.allclose(logits_k4, logits_k1, atol=1e-3)


# ===========================================================================
# Modular VL-JEPA Tests
# ===========================================================================

class TestModularVLJEPA:

    def test_split_into_modules(self):
        """Model splits into correct number of modules."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        assert len(modular.modules) == 5
        for name in MODULE_NAMES:
            assert name in modular.modules

    def test_module_specs(self):
        """Module specs have valid metadata."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        specs = modular.get_module_specs()
        assert len(specs) == 5

        for i, spec in enumerate(specs):
            assert spec.module_id == i
            assert spec.name == MODULE_NAMES[i]
            assert len(spec.input_names) > 0
            assert len(spec.output_names) > 0

    def test_decoder_is_local_only(self):
        """Text decoder is marked as local_only."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        decoder_spec = modular.get_module("text_decoder").spec
        assert decoder_spec.local_only is True

        # All other modules are network modules
        for name in MODULE_NAMES[:-1]:
            assert modular.get_module(name).spec.local_only is False

    def test_module_forward_visual(self):
        """Visual encoder module forward produces correct shapes."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        images = torch.randn(2, 3, 32, 32)
        out = modular.run_module("visual_encoder", images=images)

        num_patches = (32 // 8) ** 2
        assert "visual_embeds" in out
        assert out["visual_embeds"].shape == (2, num_patches, cfg.embed_dim)

    def test_module_forward_text(self):
        """Text encoder module forward produces correct shapes."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        ids = torch.randint(0, 260, (2, 8))
        out = modular.run_module("text_encoder", token_ids=ids)

        assert "text_embeds" in out
        assert out["text_embeds"].shape == (2, 8, cfg.embed_dim)

    def test_module_forward_predictor(self):
        """Semantic predictor outputs latent_plan with correct shape."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)

        fused = torch.randn(2, 8, cfg.embed_dim)
        out = modular.run_module("semantic_predictor", fused_repr=fused)

        assert "latent_plan" in out
        assert out["latent_plan"].shape == (2, cfg.num_latent_vectors, cfg.embed_dim)

    def test_pipeline_full(self):
        """Full pipeline produces latent_plan and decoder_logits."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        images, token_ids, attn_mask, target_ids = make_inputs(cfg)

        result = modular.run_pipeline(
            images, token_ids, attn_mask, target_token_ids=target_ids
        )

        assert "visual_embeds" in result
        assert "text_embeds" in result
        assert "fused_repr" in result
        assert "latent_plan" in result
        assert "decoder_logits" in result

    def test_pipeline_matches_monolithic(self):
        """Running modules in sequence equals full model forward."""
        cfg = make_config()
        model = VLJEPA(cfg)
        model.eval()
        modular = ModularVLJEPA(model=model)

        images, token_ids, attn_mask, target_ids = make_inputs(cfg)

        with torch.no_grad():
            mono = model(images, token_ids, attn_mask, target_token_ids=target_ids)
            pipe = modular.run_pipeline(
                images, token_ids, attn_mask, target_token_ids=target_ids
            )

        assert torch.allclose(
            mono["latent_plan"], pipe["latent_plan"], atol=1e-5
        )
        assert torch.allclose(
            mono["decoder_logits"], pipe["decoder_logits"], atol=1e-5
        )

    def test_partial_pipeline_tier1(self):
        """Tier 1: visual encoder only."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        images = torch.randn(2, 3, 32, 32)

        result = modular.run_pipeline(
            images,
            token_ids=torch.zeros(2, 1, dtype=torch.long),  # dummy
            modules=["visual_encoder"],
        )

        assert "visual_embeds" in result
        assert "latent_plan" not in result

    def test_partial_pipeline_network_only(self):
        """Network modules only (no local decoder) — produces latent plan."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        images, token_ids, attn_mask, _ = make_inputs(cfg)

        result = modular.run_pipeline(
            images,
            token_ids,
            attn_mask,
            modules=["visual_encoder", "text_encoder", "cross_modal_fusion", "semantic_predictor"],
        )

        assert "latent_plan" in result
        assert result["latent_plan"].shape == (2, cfg.num_latent_vectors, cfg.embed_dim)
        assert "decoder_logits" not in result

    def test_module_serialization(self):
        """Modules can be saved/loaded independently."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        module = modular.get_module("text_encoder")

        # Save weights
        weights = module.save_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

        # Modify weights
        with torch.no_grad():
            for p in module.layers.parameters():
                p.zero_()

        # Load weights back
        module.load_weights(weights)

        # Verify restored
        for key in weights:
            assert torch.allclose(
                module.layers.state_dict()[key],
                weights[key],
                atol=1e-6,
            )

    def test_ema_blend(self):
        """EMA blending smoothly interpolates weights."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        module = modular.get_module("visual_encoder")

        old_weights = module.save_weights()

        # Create new weights (all ones)
        new_weights = {k: torch.ones_like(v) for k, v in old_weights.items()}

        # Blend with alpha=0.5
        module.ema_blend(new_weights, alpha=0.5)

        # Result should be midpoint
        for key in old_weights:
            expected = 0.5 * old_weights[key] + 0.5 * new_weights[key]
            actual = module.layers.state_dict()[key]
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch in {key}"

    def test_weights_hash(self):
        """Content hash changes when weights change."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        module = modular.get_module("semantic_predictor")

        hash1 = module.get_weights_hash()
        assert len(hash1) == 64  # SHA-256 hex

        # Modify weights
        with torch.no_grad():
            for p in module.layers.parameters():
                p.add_(1.0)

        hash2 = module.get_weights_hash()
        assert hash1 != hash2

    def test_module_spec_serialization(self):
        """ModuleSpec round-trips through dict (including local_only)."""
        spec = ModuleSpec(
            module_id=4,
            name="text_decoder",
            input_names=["latent_plan", "target_token_ids"],
            output_names=["decoder_logits"],
            input_shapes={"latent_plan": (8, 64), "target_token_ids": (16,)},
            output_shapes={"decoder_logits": (16, 260)},
            ema_alpha=0.998,
            local_only=True,
        )
        d = spec.to_dict()
        restored = ModuleSpec.from_dict(d)

        assert restored.module_id == spec.module_id
        assert restored.name == spec.name
        assert restored.input_shapes == spec.input_shapes
        assert restored.output_shapes == spec.output_shapes
        assert restored.ema_alpha == spec.ema_alpha
        assert restored.local_only == spec.local_only

    def test_module_summary(self):
        """Module summary shows network/LOCAL labels and latent plan size."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        summary = modular.module_summary()
        assert "visual_encoder" in summary
        assert "text_decoder" in summary
        assert "LOCAL" in summary
        assert "network" in summary
        assert "Latent plan:" in summary
        assert "Total:" in summary

    def test_total_params(self):
        """Total params across modules equals model params."""
        cfg = make_config()
        model = VLJEPA(cfg)
        modular = ModularVLJEPA(model=model)

        # Count model params (excluding target encoder since it's EMA copy)
        model_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "target_encoder" not in name
        )
        modular_params = modular.total_params()

        assert modular_params == model_params

    def test_pipeline_generation_mode(self):
        """Pipeline in generation mode (no target_token_ids)."""
        cfg = make_config()
        modular = ModularVLJEPA(config=cfg)
        images, token_ids, attn_mask, _ = make_inputs(cfg)

        result = modular.run_pipeline(images, token_ids, attn_mask)

        assert "generated_ids" in result
        assert result["generated_ids"].shape[0] == 2

    def test_different_k_values(self):
        """Model works with different num_latent_vectors (K)."""
        for k in [1, 4, 16, 64]:
            cfg = make_config(num_latent_vectors=k)
            model = VLJEPA(cfg)
            images, token_ids, attn_mask, target_ids = make_inputs(cfg)

            result = model(images, token_ids, attn_mask, target_token_ids=target_ids)

            assert result["latent_plan"].shape == (2, k, cfg.embed_dim)
            assert result["decoder_logits"].shape == (2, target_ids.shape[1], cfg.vocab_size)


# ===========================================================================
# Tokenizer Tests
# ===========================================================================

class TestTokenizer:

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding produces original text."""
        tok = SimpleTokenizer()
        text = "Hello, world!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_has_bos_eos(self):
        """Encoded sequence starts with BOS and ends with EOS."""
        tok = SimpleTokenizer()
        ids = tok.encode("test")
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

    def test_encode_no_special(self):
        """Can encode without BOS/EOS."""
        tok = SimpleTokenizer()
        ids = tok.encode("ab", add_bos=False, add_eos=False)
        assert ids[0] != BOS_ID
        assert ids[-1] != EOS_ID
        assert len(ids) == 2  # just the two bytes

    def test_batch_encode(self):
        """Batch encoding produces padded tensors."""
        tok = SimpleTokenizer()
        texts = ["short", "a bit longer text"]
        ids, mask = tok.batch_encode(texts, max_length=64)

        assert ids.shape[0] == 2
        assert mask.shape[0] == 2
        assert ids.shape == mask.shape

        # Shorter sequence should have padding (mask=False)
        assert mask[0].sum() < mask[1].sum()

    def test_batch_encode_truncation(self):
        """Batch encoding truncates to max_length."""
        tok = SimpleTokenizer()
        long_text = "x" * 1000
        ids, mask = tok.batch_encode([long_text], max_length=32)

        assert ids.shape[1] <= 32

    def test_unicode_roundtrip(self):
        """Unicode text survives encode/decode."""
        tok = SimpleTokenizer()
        text = "日本語テスト"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_empty_string(self):
        """Empty string encodes to just BOS + EOS."""
        tok = SimpleTokenizer()
        ids = tok.encode("")
        assert ids == [BOS_ID, EOS_ID]
        assert tok.decode(ids) == ""
