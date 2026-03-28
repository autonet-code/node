"""
VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

Extends JEPA with language capabilities:
- TextEncoder: encodes text queries into shared embedding space
- CrossModalFusion: cross-attention between visual and text representations
- SemanticPredictor: maps joint representation to a latent plan (K vectors)
- TextDecoder: autoregressive text generation from latent plan (runs locally)

Key properties:
- Self-supervised: predicts masked embeddings, no labeled data needed
- Embedding-based: reasons in continuous space, not discrete tokens
- Latent plan: predictor outputs K vectors, not one — rich enough for
  coherent generation, compact enough for network transfer (~50KB)
- Local decoding: decoder runs on requesting node, not the network.
  Only one distributed round-trip needed regardless of output length.
- Selective decoding: text generation only when latent plan changes
- Modular: each component can run independently on a separate node

Reuses ViT components from jepa.py (TransformerBlock, PatchEmbed, etc.)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import (
    JEPAConfig,
    TransformerBlock,
    VisionTransformerEncoder,
    get_1d_sincos_pos_embed,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VLJEPAConfig(JEPAConfig):
    """Configuration for VL-JEPA (extends JEPAConfig with language settings)."""

    # Text encoder
    vocab_size: int = 260  # byte-level: 4 special + 256 bytes
    max_seq_length: int = 512
    text_encoder_depth: int = 6

    # Cross-modal fusion
    fusion_depth: int = 6

    # Semantic predictor
    semantic_predictor_depth: int = 4
    num_latent_vectors: int = 32  # K: number of latent plan vectors

    # Text decoder (runs locally on requesting node)
    decoder_depth: int = 6
    decoder_embed_dim: int = 0  # 0 = same as embed_dim

    # Selective decoding
    semantic_change_threshold: float = 0.1  # cosine distance threshold

    # Dynamics predictor (for multi-step rollout in embedding space)
    dynamics_depth: int = 4  # transformer layers in LatentDynamicsPredictor

    # SIGReg: Spectral Instance-Global Regularization
    # From LeJEPA (Balestriero & LeCun, 2025, arXiv 2511.08544).
    # Prevents embedding collapse by enforcing isotropic Gaussian structure.
    # Set sigreg_weight=0.0 to disable.
    sigreg_weight: float = 0.1
    sigreg_num_projections: int = 128

    def __post_init__(self):
        if self.decoder_embed_dim == 0:
            self.decoder_embed_dim = self.embed_dim

    @property
    def latent_plan_bytes(self) -> int:
        """Size of latent plan in bytes (float16)."""
        return self.num_latent_vectors * self.embed_dim * 2


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder(nn.Module):
    """
    Encode text tokens into the shared embedding space.

    Uses the same TransformerBlock architecture as the visual encoder
    for consistency in the shared representation space.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional embedding (1D sinusoidal, initialized below)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.text_encoder_depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.config.embed_dim, self.config.max_seq_length
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text tokens.

        Args:
            token_ids: (B, seq_len) LongTensor
            attention_mask: (B, seq_len) BoolTensor — True = real token

        Returns:
            Text embeddings (B, seq_len, embed_dim)
        """
        B, S = token_ids.shape

        # Token + positional embeddings
        x = self.token_embed(token_ids)  # (B, S, embed_dim)
        x = x + self.pos_embed[:, :S, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


# =============================================================================
# Cross-Modal Fusion
# =============================================================================

class CrossAttention(nn.Module):
    """Cross-attention: queries from one modality attend to keys/values from another."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, D) — the sequence being updated
            context: (B, Nc, D) — the sequence providing information

        Returns:
            Updated query (B, Nq, D)
        """
        B, Nq, D = query.shape
        Nc = context.shape[1]

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)


class FusionBlock(nn.Module):
    """Single fusion block: cross-attention + self-attention + MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # Cross-attention (text attends to visual)
        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        # Self-attention on fused sequence
        self.self_attn_block = TransformerBlock(embed_dim, num_heads, mlp_ratio)

    def forward(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: (B, St, D) — text being fused with visual info
            visual_embeds: (B, Sv, D) — visual context

        Returns:
            Fused text embeddings (B, St, D)
        """
        # Cross-attention: text queries attend to visual keys/values
        x = text_embeds + self.cross_attn(
            self.norm_cross_q(text_embeds),
            self.norm_cross_kv(visual_embeds),
        )
        # Self-attention + MLP
        x = self.self_attn_block(x)
        return x


class CrossModalFusion(nn.Module):
    """
    Fuse visual and text representations via stacked cross-attention.

    Architecture: text embeddings attend to visual embeddings through
    multiple fusion blocks. The output is a sequence that combines
    information from both modalities.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.fusion_depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual and text representations.

        Args:
            visual_embeds: (B, Sv, D) from visual encoder
            text_embeds: (B, St, D) from text encoder

        Returns:
            Fused representation (B, St, D) — text enriched with visual info
        """
        x = text_embeds
        for block in self.blocks:
            x = block(x, visual_embeds)
        return self.norm(x)


# =============================================================================
# Semantic Predictor
# =============================================================================

class SemanticPredictor(nn.Module):
    """
    Predict a latent plan from the fused representation.

    Uses K learned query tokens that cross-attend to the fused sequence,
    producing a fixed-length latent plan (B, K, D). This is the representation
    that gets transferred over the network — compact enough for fast transfer,
    rich enough for coherent text generation.

    Similar to Perceiver's latent array or BLIP-2's Q-Former.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        K = config.num_latent_vectors

        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.semantic_predictor_depth)
        ])

        # K learned query tokens for latent plan extraction
        self.latent_queries = nn.Parameter(torch.zeros(1, K, config.embed_dim))
        self.cross_attn = CrossAttention(config.embed_dim, config.num_heads)
        self.norm = nn.LayerNorm(config.embed_dim)

        nn.init.normal_(self.latent_queries, std=0.02)

    def forward(self, fused_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_repr: (B, S, D) from cross-modal fusion

        Returns:
            Latent plan (B, K, D)
        """
        x = fused_repr
        for block in self.blocks:
            x = block(x)

        # K queries cross-attend to the processed sequence
        B = x.shape[0]
        queries = self.latent_queries.expand(B, -1, -1)  # (B, K, D)
        latent_plan = self.cross_attn(queries, x)  # (B, K, D)
        return self.norm(latent_plan)


# =============================================================================
# SIGReg: Spectral Instance-Global Regularization
# =============================================================================

class SIGReg(nn.Module):
    """Spectral Instance-Global Regularization (SIGReg).

    From: "LeJEPA: Provable and Scalable Self-Supervised Learning Without
    the Heuristics" — Balestriero & LeCun, 2025 (arXiv 2511.08544).
    Also used in: "LeWorldModel" — Maes, Balestriero, LeCun et al., 2026
    (arXiv 2603.19312), where it is the key mechanism that enables stable
    end-to-end JEPA training without EMA or stop-gradient.

    **Why this matters for VL-JEPA scale-up:**
    The 18M validation model collapsed in all four real-data runs (Runs 1-4
    in VALIDATION_FINDINGS.md). The root cause was insufficient latent plan
    capacity combined with embedding collapse. SIGReg addresses the collapse
    component by constraining the embedding distribution.

    **How it works:**
    The Cramér-Wold theorem guarantees that a d-dimensional distribution is
    uniquely determined by all its 1D projections. SIGReg enforces that each
    1D random projection of the embeddings follows N(0,1), which implies the
    full joint distribution is isotropic Gaussian — the theoretical optimum
    for self-supervised embedding objectives.

    Practically:
    1. Sample P fixed random unit-norm directions from the unit sphere.
    2. Project embeddings onto each direction: z_p = Z @ r_p (scalar per sample).
    3. Penalize deviation of the empirical distribution of z_p from N(0,1)
       via moment matching (mean=0, variance=1, skewness=0, kurtosis=3).

    Properties:
    - Single hyperparameter (sigreg_weight).
    - O(N * P) time and memory — linear in batch size and projections.
    - No stop-gradient, no EMA target, no adversarial component.
    - Fixed random directions (not trained) — reproducible across runs.
    """

    def __init__(self, embed_dim: int, num_projections: int = 128):
        super().__init__()
        # Fixed unit-norm projection directions — not trained, reproducible
        directions = F.normalize(torch.randn(num_projections, embed_dim), dim=1)
        self.register_buffer("directions", directions)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss for a batch of embeddings.

        Args:
            z: Embeddings of any shape (..., embed_dim). Flattened to (N, D).

        Returns:
            Scalar regularization loss. Add to training loss weighted by
            config.sigreg_weight.
        """
        flat = z.reshape(-1, z.shape[-1])   # (N, D)

        # 1D projections onto each random direction: (N, P)
        proj = flat @ self.directions.T

        # --- Instance-level terms (properties of the projection distribution) ---
        # Mean of each projection should be 0 (zero-centered)
        mean_loss = proj.mean(dim=0).pow(2).mean()

        # Variance of each projection should be 1 (non-collapsed, unit spread)
        var_loss = (proj.var(dim=0) - 1.0).pow(2).mean()

        # --- Global-level terms (shape of distribution, normalized) ---
        # Standardize for higher-order moment checks
        proj_std = proj.std(dim=0, keepdim=True).clamp(min=1e-6)
        proj_norm = (proj - proj.mean(dim=0, keepdim=True)) / proj_std

        # Third standardized moment (skewness) should be 0
        skew_loss = proj_norm.pow(3).mean(dim=0).pow(2).mean()

        # Fourth standardized moment (kurtosis) should be 3 for Gaussian
        kurt_loss = (proj_norm.pow(4).mean(dim=0) - 3.0).pow(2).mean()

        return mean_loss + var_loss + skew_loss + kurt_loss


# =============================================================================
# Latent Dynamics Predictor (multi-step rollout)
# =============================================================================

class LatentDynamicsPredictor(nn.Module):
    """Predicts the next-step latent plan from the current latent plan.

    This is the "world model" component for multi-step rollout in embedding
    space. Given a latent plan (B, K, D) representing one reasoning step, it
    predicts the latent plan (B, K, D) for the *next* reasoning step.

    Analogous to V-JEPA 2-AC's predictor (Bardes et al., 2025) but applied
    to agent-trace latent plans rather than robot video frames.

    Architecture:
    - Stacked TransformerBlocks that let the K latent vectors attend to
      each other (capturing inter-token dependencies within the plan).
    - Residual output projection initialized to zero (identity at init),
      so training starts from "predict no change" — stable initialization.

    The residual design means early in training the model simply echoes the
    input plan, and the transformer gradually learns to predict meaningful
    dynamics. This mirrors the initialization strategy recommended in
    LeWorldModel (Maes et al., 2026).
    """

    def __init__(self, config: 'VLJEPAConfig'):
        super().__init__()
        d = config.embed_dim

        self.blocks = nn.ModuleList([
            TransformerBlock(d, config.num_heads, config.mlp_ratio)
            for _ in range(config.dynamics_depth)
        ])
        self.norm = nn.LayerNorm(d)

        # Residual delta projection — zero-initialized for stable training start
        self.delta_proj = nn.Linear(d, d)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, latent_plan: torch.Tensor) -> torch.Tensor:
        """Predict the next-step latent plan.

        Args:
            latent_plan: (B, K, D) — current latent plan

        Returns:
            next_plan: (B, K, D) — predicted next-step latent plan
        """
        x = latent_plan
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Residual: output = input + predicted_delta (zero at initialization)
        delta = self.delta_proj(x)
        return latent_plan + delta


# =============================================================================
# Text Decoder
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Self-attention with causal mask for autoregressive generation.

    Supports prefix tokens: when n_prefix > 0, the first n_prefix positions
    are visible to all subsequent positions (bidirectional among themselves,
    and visible to all future tokens). This enables latent plan prefix injection.
    """

    def __init__(self, embed_dim: int, num_heads: int, max_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (registered as buffer so it moves with the model)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, n_prefix: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:N, :N]
        if n_prefix > 0:
            # Allow all positions to attend to prefix tokens (columns 0..n_prefix-1)
            # and allow prefix tokens to attend to each other (rows 0..n_prefix-1)
            mask = mask.clone()
            mask[:, :n_prefix] = False  # all positions can see prefix
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: generates scale & shift from conditioning."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)
        # Initialize to identity transform (scale=1, shift=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:feat_dim] = 1.0  # scale init = 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) features to modulate
            cond: (B, D) conditioning vector
        Returns:
            Modulated features (B, S, D)
        """
        gamma_beta = self.proj(cond)  # (B, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, D)
        return x * gamma.unsqueeze(1) + beta.unsqueeze(1)


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention + FiLM + MLP.

    Uses three conditioning mechanisms:
    1. Causal self-attention with prefix tokens (plan vectors in sequence)
    2. Cross-attention to full latent plan
    3. FiLM conditioning: plan modulates layer norm outputs multiplicatively
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, max_len: int = 512):
        super().__init__()
        from .jepa import MLP

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = CausalSelfAttention(embed_dim, num_heads, max_len)

        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

        # FiLM: plan conditions the post-norm features before self-attn and MLP
        self.film_sa = FiLMLayer(embed_dim, embed_dim)
        self.film_mlp = FiLMLayer(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                plan_cond: torch.Tensor, n_prefix: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) — decoder token sequence (may include prefix)
            context: (B, K, D) — latent plan for cross-attention
            plan_cond: (B, D) — mean-pooled plan for FiLM conditioning
            n_prefix: number of prefix tokens at start of sequence

        Returns:
            (B, S, D)
        """
        # Causal self-attention with FiLM modulation
        normed = self.norm1(x)
        normed = self.film_sa(normed, plan_cond)
        x = x + self.self_attn(normed, n_prefix=n_prefix)
        # Cross-attention to semantic embedding
        x = x + self.cross_attn(self.norm_cross_q(x), self.norm_cross_kv(context))
        # MLP with FiLM modulation
        normed = self.norm2(x)
        normed = self.film_mlp(normed, plan_cond)
        x = x + self.mlp(normed)
        return x


class TextDecoder(nn.Module):
    """
    Autoregressive text decoder conditioned on a latent plan.

    Uses THREE conditioning mechanisms to prevent the decoder from ignoring
    the latent plan (which caused mode collapse in runs 2-3):

    1. PREFIX TOKENS: The K latent plan vectors are prepended to the decoder
       sequence. Self-attention is forced to see them — they're part of the
       sequence, not an optional cross-attention source. This is how Flamingo,
       BLIP-2, and PaLM-E all condition their decoders.

    2. FiLM CONDITIONING: Each decoder block uses the mean-pooled plan to
       generate scale/shift parameters for layer normalization. This
       multiplicatively modulates activations — the decoder literally cannot
       produce features without the plan's involvement.

    3. CROSS-ATTENTION: Standard cross-attention to the full latent plan
       (retained from previous architecture for additional conditioning signal).

    Additionally, TOKEN EMBEDDING DROPOUT (word_dropout) randomly zeros out
    token embeddings during training to weaken the language model prior and
    force the decoder to rely on the plan for content.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config
        d = config.decoder_embed_dim
        K = config.num_latent_vectors

        self.token_embed = nn.Embedding(config.vocab_size, d)
        # +K for prefix positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length + K, d)
        )

        # Project latent plan vectors to decoder space (for prefix injection)
        self.plan_to_prefix = nn.Linear(d, d)

        # Token embedding dropout (word dropout): zero out entire token embeddings
        # during training to weaken the LM prior and force plan dependence
        self.word_dropout = nn.Dropout(0.3)

        self.blocks = nn.ModuleList([
            DecoderBlock(d, config.num_heads, config.mlp_ratio,
                         config.max_seq_length + K)
            for _ in range(config.decoder_depth)
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.config.decoder_embed_dim,
            self.config.max_seq_length + self.config.num_latent_vectors,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(
        self,
        latent_plan: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.

        Args:
            latent_plan: (B, K, D) — latent plan from semantic predictor
            target_token_ids: (B, S) — target token IDs (shifted right)

        Returns:
            logits: (B, S, vocab_size) — logits for text positions only
        """
        B, S = target_token_ids.shape
        K = latent_plan.shape[1]

        # --- Prefix: project latent plan to prefix token embeddings ---
        prefix = self.plan_to_prefix(latent_plan)  # (B, K, D)

        # --- Token embeddings with word dropout ---
        tok_embed = self.token_embed(target_token_ids)  # (B, S, D)
        if self.training:
            tok_embed = self.word_dropout(tok_embed)

        # --- Concatenate: [prefix | tokens] ---
        x = torch.cat([prefix, tok_embed], dim=1)  # (B, K+S, D)
        x = x + self.pos_embed[:, :K + S, :]

        # --- Mean-pooled plan for FiLM conditioning ---
        plan_cond = latent_plan.mean(dim=1)  # (B, D)

        # --- Decoder blocks ---
        for block in self.blocks:
            x = block(x, latent_plan, plan_cond, n_prefix=K)

        x = self.norm(x)

        # Only return logits for text token positions (skip prefix)
        text_features = x[:, K:, :]  # (B, S, D)
        return self.head(text_features)

    @torch.no_grad()
    def generate(
        self,
        latent_plan: torch.Tensor,
        max_len: int = 128,
        temperature: float = 1.0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation from latent plan.

        Args:
            latent_plan: (B, K, D) — latent plan from semantic predictor
            max_len: Maximum tokens to generate (clamped to max_seq_length)
            temperature: Sampling temperature (0 = greedy)
            bos_id: Beginning-of-sequence token ID
            eos_id: End-of-sequence token ID

        Returns:
            Generated token IDs (B, generated_len)
        """
        B = latent_plan.shape[0]
        K = latent_plan.shape[1]
        device = latent_plan.device

        # Clamp to positional embedding length (minus K prefix positions)
        pos_len = self.pos_embed.shape[1] - K
        max_len = min(max_len, pos_len)

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(latent_plan, generated)  # (B, current_len, vocab)
            next_logits = logits[:, -1, :]  # (B, vocab)

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # Mask finished sequences to pad
            next_token = next_token.masked_fill(finished.unsqueeze(1), 0)
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return generated


# =============================================================================
# VL-JEPA Model
# =============================================================================

class VLJEPA(nn.Module):
    """
    Vision-Language JEPA — unified multimodal model.

    Components (distributed on network):
    1. Visual encoder (ViT) — encodes image patches
    2. Text encoder — encodes text query
    3. Cross-modal fusion — fuses visual + text via cross-attention
    4. Semantic predictor — produces latent plan (K vectors)

    Component (local on requesting node):
    5. Text decoder — generates text from latent plan

    Architecture:
    - Network produces a latent plan (B, K, D) in one round-trip
    - Local decoder expands latent plan to text autoregressively
    - Latency = one network pass + local decode (independent of output length)

    For self-supervised training:
    - Uses JEPA-style masking on visual patches
    - Target encoder (EMA) provides supervision signal
    - No labeled data needed

    For inference:
    - Encode image + text query → latent plan → local decode to text
    """

    def __init__(self, config: Optional[VLJEPAConfig] = None):
        super().__init__()
        self.config = config or VLJEPAConfig()

        # Visual pathway
        self.visual_encoder = VisionTransformerEncoder(self.config)
        self.target_encoder = VisionTransformerEncoder(self.config)
        self.target_encoder.load_state_dict(self.visual_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Language pathway
        self.text_encoder = TextEncoder(self.config)
        self.cross_modal_fusion = CrossModalFusion(self.config)
        self.semantic_predictor = SemanticPredictor(self.config)
        self.text_decoder = TextDecoder(self.config)

        # Multi-step rollout: predicts next-step latent plan from current plan
        self.dynamics_predictor = LatentDynamicsPredictor(self.config)

        # Collapse prevention: SIGReg regularization on latent plan embeddings
        self.sigreg = SIGReg(
            self.config.embed_dim,
            self.config.sigreg_num_projections,
        )

        # Track previous semantic embedding for selective decoding
        self._prev_semantic: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update target encoder with EMA of visual encoder."""
        momentum = momentum or self.config.ema_momentum
        for param_q, param_k in zip(
            self.visual_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual embeddings. (B, C, H, W) -> (B, N, D)"""
        return self.visual_encoder(images)

    def encode_text(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to text embeddings. (B, S) -> (B, S, D)"""
        return self.text_encoder(token_ids, attention_mask)

    def fuse(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-modal fusion. Returns fused representation (B, St, D)."""
        return self.cross_modal_fusion(visual_embeds, text_embeds)

    def predict_semantic(self, fused_repr: torch.Tensor) -> torch.Tensor:
        """Predict latent plan. (B, S, D) -> (B, K, D)"""
        return self.semantic_predictor(fused_repr)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            images: (B, C, H, W)
            token_ids: (B, S) — query text tokens
            attention_mask: (B, S) — query attention mask
            target_token_ids: (B, St) — target text for decoder training (optional)

        Returns:
            Dict with:
            - latent_plan: (B, K, D) — latent plan from semantic predictor
            - decoder_logits: (B, St, vocab) — if target_token_ids provided
        """
        visual = self.encode_visual(images)
        text = self.encode_text(token_ids, attention_mask)
        fused = self.fuse(visual, text)
        latent_plan = self.predict_semantic(fused)

        result = {"latent_plan": latent_plan}

        if target_token_ids is not None:
            logits = self.text_decoder(latent_plan, target_token_ids)
            result["decoder_logits"] = logits

        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate text from image + query."""
        self.eval()
        result = self.forward(images, token_ids, attention_mask)
        return self.text_decoder.generate(
            result["latent_plan"],
            max_len=max_len,
            temperature=temperature,
        )

    def should_decode(self, latent_plan: torch.Tensor) -> bool:
        """
        Selective decoding: check if latent plan changed enough to warrant
        running the decoder.

        Compares mean-pooled latent plan to previous via cosine distance.
        Returns True if decode is needed, False if previous output is still valid.
        """
        # Mean-pool over K vectors for comparison
        current = latent_plan.mean(dim=1)  # (B, D)

        if self._prev_semantic is None:
            self._prev_semantic = current.detach()
            return True

        cos_dist = 1.0 - F.cosine_similarity(
            current, self._prev_semantic, dim=-1
        ).mean().item()

        if cos_dist > self.config.semantic_change_threshold:
            self._prev_semantic = current.detach()
            return True

        return False

    @torch.no_grad()
    def rollout(
        self,
        context_embeddings: torch.Tensor,
        num_steps: int,
    ) -> Dict[str, object]:
        """Autoregressive rollout in latent embedding space.

        Given a starting latent plan, uses the LatentDynamicsPredictor to
        iteratively predict future latent plans — one step at a time.

        This is the core multi-step prediction capability for Phase B:
        given agent-trace embeddings for turns 1..N, predict the embedding
        for turn N+1, then use that to predict N+2, and so on.

        Architecture rationale (from jepa_reasoning_survey.md):
        - V-JEPA 2-AC trains with a T=2 rollout loss and uses MPC replanning
          to avoid long open-loop chains. We start from T=1 and extend.
        - LeWorldModel (Maes et al., 2026) confirms that SIGReg + residual
          prediction enables stable chaining; adding it here mirrors that
          design.
        - Uncertainty tracking follows the survey recommendation: "Monitor
          embedding variance during chained prediction as an early warning
          signal — if variance drops >50% over 5 steps, the chain is
          collapsing."

        Args:
            context_embeddings: (B, K, D) — latent plan at step 0. Typically
                the output of ``predict_semantic()`` for the current turn.
            num_steps: Number of future steps to predict (>= 1).

        Returns:
            Dict with:
            - ``predictions``:  list of ``num_steps`` tensors, each (B, K, D).
                                predictions[i] is the predicted plan at step i+1.
            - ``uncertainties``: list of ``num_steps`` floats in [0, 1].
                                 0 = confident/diverse, 1 = collapsed/uncertain.
                                 Derived from inter-vector spread of each plan:
                                 higher spread = more information = lower uncertainty.
            - ``embedding_variances``: list of ``num_steps`` floats — mean
                                       variance of each step's plan across
                                       embed_dim. Use as collapse diagnostic:
                                       sharp drop signals representation collapse.
        """
        predictions: List[torch.Tensor] = []
        uncertainties: List[float] = []
        embedding_variances: List[float] = []

        current = context_embeddings  # (B, K, D)

        # Compute baseline spread from the context plan (step 0) for reference
        baseline_spread = self._plan_spread(current)

        for _ in range(num_steps):
            next_plan = self.dynamics_predictor(current)       # (B, K, D)
            predictions.append(next_plan)

            # Embedding variance: mean variance across embed_dim for each token,
            # then averaged over tokens and batch. Sharp drop = collapse.
            emb_var = next_plan.var(dim=-1).mean().item()      # scalar
            embedding_variances.append(emb_var)

            # Uncertainty: 1 - normalized inter-vector spread.
            # Inter-vector spread = how much the K latent vectors differ from
            # each other within the plan. Collapsed plan → K vectors converge
            # to the same point → zero spread → uncertainty = 1.
            spread = self._plan_spread(next_plan)
            # Normalize by baseline so the first step is near 0 uncertainty
            normalized_spread = spread / (baseline_spread + 1e-8)
            uncertainty = 1.0 / (1.0 + normalized_spread)
            uncertainties.append(float(uncertainty))

            current = next_plan

        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "embedding_variances": embedding_variances,
        }

    @staticmethod
    def _plan_spread(plan: torch.Tensor) -> float:
        """Mean squared deviation of K latent vectors from their centroid.

        High spread = K vectors are diverse = plan carries more information.
        """
        centroid = plan.mean(dim=1, keepdim=True)          # (B, 1, D)
        spread = (plan - centroid).pow(2).mean().item()
        return spread

    def self_supervised_loss(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Self-supervised training loss (JEPA-style).

        Uses a masking strategy: drop a random subset of visual patches before
        encoding with the online encoder, then compare the resulting latent plan
        against the target encoder's latent plan computed from all patches.
        This forces the model to learn predictive representations.

        Loss = smooth_l1 between online and target latent plans.
        """
        B = images.shape[0]
        num_patches = self.config.num_patches

        # Target path (no gradient): full visual encoding
        with torch.no_grad():
            target_visual = self.target_encoder(images)
            target_text = self.text_encoder(token_ids, attention_mask)
            target_fused = self.cross_modal_fusion(target_visual, target_text)
            target_plan = self.semantic_predictor(target_fused)  # (B, K, D)

        # Online path: mask ~25% of visual patches by zeroing them out
        online_visual = self.visual_encoder(images)  # (B, N, D)
        mask_ratio = 0.25
        num_mask = max(1, int(num_patches * mask_ratio))
        noise = torch.rand(B, num_patches, device=images.device)
        mask_indices = noise.argsort(dim=1)[:, :num_mask]
        online_visual_masked = online_visual.clone()
        for b in range(B):
            online_visual_masked[b, mask_indices[b]] = 0.0

        # Online latent plan from masked visual
        online_text = self.text_encoder(token_ids, attention_mask)
        online_fused = self.cross_modal_fusion(online_visual_masked, online_text)
        online_plan = self.semantic_predictor(online_fused)  # (B, K, D)

        # Primary loss: online plan should match target plan
        visual_loss = F.smooth_l1_loss(online_plan, target_plan.detach())

        # SIGReg: spectral regularization to prevent embedding collapse.
        # Applied to the online latent plan so the dynamics predictor and
        # downstream decoder always receive a well-distributed embedding.
        # Per LeWorldModel (Maes et al., 2026): "two loss terms — MSE + SIGReg
        # — are sufficient for stable end-to-end JEPA training from pixels."
        # Weight=0 disables it entirely (backward-compatible default: 0.1).
        if self.config.sigreg_weight > 0.0:
            sigreg_loss = self.sigreg(online_plan)
            total_loss = visual_loss + self.config.sigreg_weight * sigreg_loss
        else:
            sigreg_loss = torch.zeros(1, device=images.device)
            total_loss = visual_loss

        result = {
            "latent_plan": online_plan,
            "visual_loss": visual_loss,
            "sigreg_loss": sigreg_loss,
            "loss": total_loss,
        }
        return result


# =============================================================================
# Configuration Presets
# =============================================================================

# Parameter estimates below are approximate (exclude frozen target encoder).
# The target encoder is a frozen EMA copy of the visual encoder; it contributes
# to checkpoint size but not to the gradient computation graph.
#
# Calculation methodology:
#   TransformerBlock(D, H): ~12 * D^2 params (attn QKV+O + MLP)
#   DecoderBlock(D, H):     ~16 * D^2 params (adds cross-attn + 2x FiLM)
#   FusionBlock(D, H):      ~10 * D^2 params (cross-attn + self-attn)
#   Patch embed (C→D):      patch_size^2 * 3 * D
#   Token embed (V×D):      vocab_size * D (≈ 260 * D)

VLJEPA_PRESETS: Dict[str, VLJEPAConfig] = {
    # -------------------------------------------------------------------------
    # small — validated 18M configuration from VALIDATION_FINDINGS.md.
    # Runs 1-4 used this config. Confirmed to work on synthetic data (100%
    # shape+color accuracy) but collapses on real COCO images.
    # Use for unit testing and integration debugging only.
    # Approximate trainable parameters: ~18M
    # -------------------------------------------------------------------------
    "small": VLJEPAConfig(
        # Vision encoder
        image_size=64,
        patch_size=8,
        embed_dim=256,
        num_heads=8,
        encoder_depth=4,
        # Language encoder + fusion
        text_encoder_depth=4,
        fusion_depth=4,
        semantic_predictor_depth=2,
        # Latent plan bottleneck
        num_latent_vectors=16,
        # Text decoder
        decoder_depth=4,
        # Dynamics predictor
        dynamics_depth=2,
        # SIGReg
        sigreg_weight=0.1,
        sigreg_num_projections=64,
    ),

    # -------------------------------------------------------------------------
    # medium — minimum viable scale for real-world agent traces.
    # The JEPA survey (jepa_reasoning_survey.md) sets the floor at 200M params
    # for useful latent reasoning. This preset targets ~200M total parameters
    # (~155M trainable, ~55M in frozen target encoder).
    #
    # Key scaling choices vs. small:
    #  - embed_dim 256→768: 9x capacity per layer
    #  - K 16→64: 4x richer latent plan (was the primary failure mode)
    #  - image_size 64→224: full COCO resolution; patch 8→16 keeps seq length ≈ 196
    #  - encoder_depth 4→6: moderate depth to encode 224px images
    #  - sigreg_weight: increased to 0.2 for more aggressive collapse prevention
    #    at higher K (more vectors = more collapse surface area)
    #
    # Approximate trainable parameters: ~155M
    # -------------------------------------------------------------------------
    "medium": VLJEPAConfig(
        # Vision encoder
        image_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        encoder_depth=6,
        # Language encoder + fusion
        text_encoder_depth=4,
        fusion_depth=4,
        semantic_predictor_depth=3,
        # Latent plan bottleneck
        num_latent_vectors=64,
        # Text decoder
        decoder_depth=6,
        # Dynamics predictor
        dynamics_depth=4,
        # SIGReg — slightly higher weight at this scale
        sigreg_weight=0.2,
        sigreg_num_projections=128,
    ),

    # -------------------------------------------------------------------------
    # large — Phase B target configuration.
    # The JEPA survey notes Meta's VL-JEPA achieves strong results at 1.6B
    # on general VL tasks; agent traces are a tighter distribution. This
    # preset targets ~500M total params (~380M trainable) — the survey's
    # minimum for multi-step latent reasoning.
    #
    # Key scaling choices vs. medium:
    #  - embed_dim 768→1024: 1.8x capacity per layer
    #  - K 64→128: 2x richer latent plan for longer agent traces
    #  - encoder_depth 6→8: deeper visual encoding for 224px
    #  - text_encoder_depth 4→6: deeper text understanding
    #  - fusion_depth 4→6: more cross-modal refinement passes
    #  - semantic_predictor_depth 3→4: richer latent plan extraction
    #  - decoder_depth 6→8: more expressive generation
    #  - dynamics_depth 4→6: more capacity for multi-step dynamics modeling
    #
    # Approximate trainable parameters: ~380M
    # -------------------------------------------------------------------------
    "large": VLJEPAConfig(
        # Vision encoder
        image_size=224,
        patch_size=16,
        embed_dim=1024,
        num_heads=16,
        encoder_depth=8,
        # Language encoder + fusion
        text_encoder_depth=6,
        fusion_depth=6,
        semantic_predictor_depth=4,
        # Latent plan bottleneck
        num_latent_vectors=128,
        # Text decoder
        decoder_depth=8,
        # Dynamics predictor
        dynamics_depth=6,
        # SIGReg — same relative weight at large scale
        sigreg_weight=0.2,
        sigreg_num_projections=256,
    ),
}


def get_vljepa_preset(name: str, **overrides) -> VLJEPAConfig:
    """Return a named VLJEPAConfig preset, optionally with field overrides.

    Available presets:
        ``small``  — ~18M params. Validated on synthetic data (VALIDATION_FINDINGS.md).
                     Use for testing only.
        ``medium`` — ~155M trainable params. Minimum viable for real agent traces.
        ``large``  — ~380M trainable params. Phase B target.

    Args:
        name: Preset name — ``"small"``, ``"medium"``, or ``"large"``.
        **overrides: Optional field overrides applied after preset defaults.
            Example: ``get_vljepa_preset("medium", sigreg_weight=0.5)``

    Returns:
        A fresh :class:`VLJEPAConfig` instance.

    Raises:
        ValueError: If ``name`` is not a known preset.

    Example::

        cfg = get_vljepa_preset("medium")
        model = VLJEPA(cfg)

        # Override a single field for an experiment
        cfg_exp = get_vljepa_preset("medium", num_latent_vectors=32)
    """
    if name not in VLJEPA_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {list(VLJEPA_PRESETS)}"
        )
    # Shallow-copy the preset so mutations don't affect the global dict
    cfg = VLJEPA_PRESETS[name]
    if overrides:
        import dataclasses
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg
