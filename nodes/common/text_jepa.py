"""
Text JEPA — Self-Supervised Text Encoder Training

Applies the JEPA (Joint Embedding Predictive Architecture) training paradigm
to text sequences.  Instead of masking rectangular patches of an image, we
mask contiguous spans of tokens and predict their embeddings from context.

Architecture mirrors vision JEPA exactly:
  - Context encoder: encodes visible (unmasked) tokens
  - Target encoder: EMA copy (no gradients)
  - Predictor: predicts target embeddings from context + target positions
  - Loss: smooth L1 in embedding space (NOT token space)

Reuses components from vl_jepa.py (TextEncoder, TransformerBlock) and
text_data.py (TextMasker).  Produces weight deltas compatible with FedAvg
aggregation — same as vision JEPA.

Story 1.3 (BACKLOG_TRAINING_DATA.md)
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import TransformerBlock, get_1d_sincos_pos_embed
from .tokenizer import SimpleTokenizer, PAD_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TextJEPAConfig:
    """Configuration for text-only JEPA training."""
    # Vocabulary
    vocab_size: int = 260           # byte-level: 4 special + 256 bytes
    max_seq_length: int = 2048

    # Encoder
    embed_dim: int = 384
    num_heads: int = 6
    encoder_depth: int = 6
    mlp_ratio: float = 4.0

    # Predictor
    predictor_depth: int = 3
    predictor_embed_dim: int = 192

    # Training
    ema_momentum: float = 0.996


# ---------------------------------------------------------------------------
# Text Encoder with masking support
# ---------------------------------------------------------------------------

class MaskedTextEncoder(nn.Module):
    """Text encoder that supports JEPA-style masking.

    When a mask is provided, only the visible (True) positions are encoded.
    This is the text analogue of VisionTransformerEncoder with mask support.
    """

    def __init__(self, config: TextJEPAConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.encoder_depth)
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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode tokens, optionally keeping only visible (masked) positions.

        Args:
            token_ids: (B, S) LongTensor
            mask: (B, S) BoolTensor — True = KEEP (visible).
                  If None, process all tokens.

        Returns:
            Embeddings (B, num_visible, embed_dim)
        """
        B, S = token_ids.shape

        x = self.token_embed(token_ids)           # (B, S, D)
        x = x + self.pos_embed[:, :S, :]

        if mask is not None:
            # Keep only visible positions (same approach as VisionTransformerEncoder)
            x = torch.stack([x[b][mask[b]] for b in range(B)])  # (B, num_visible, D)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


# ---------------------------------------------------------------------------
# 1D Predictor (text positions instead of 2D grid)
# ---------------------------------------------------------------------------

class TextJEPAPredictor(nn.Module):
    """Predictor for text JEPA — 1D positional embeddings.

    Takes context embeddings + target positions and predicts target embeddings.
    Architecturally identical to JEPAPredictor but uses 1D sinusoidal
    positional embeddings instead of 2D grid.
    """

    def __init__(self, config: TextJEPAConfig):
        super().__init__()
        self.config = config

        self.proj_in = nn.Linear(config.embed_dim, config.predictor_embed_dim)

        self.target_pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.predictor_embed_dim)
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.predictor_embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.predictor_embed_dim,
                max(1, config.num_heads // 2),
                config.mlp_ratio,
            )
            for _ in range(config.predictor_depth)
        ])
        self.norm = nn.LayerNorm(config.predictor_embed_dim)
        self.proj_out = nn.Linear(config.predictor_embed_dim, config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.config.predictor_embed_dim, self.config.max_seq_length
        )
        self.target_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target embeddings from context.

        Args:
            context_embeddings: (B, num_context, embed_dim)
            context_indices:    (B, num_context)  — positions in sequence
            target_indices:     (B, num_targets)  — positions to predict

        Returns:
            Predicted embeddings (B, num_targets, embed_dim)
        """
        B = context_embeddings.shape[0]
        num_targets = target_indices.shape[1]

        x = self.proj_in(context_embeddings)

        # Add positional info for context positions
        context_pos = torch.gather(
            self.target_pos_embed.expand(B, -1, -1),
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )
        x = x + context_pos

        # Create mask tokens with target positional info
        mask_tokens = self.mask_token.expand(B, num_targets, -1)
        target_pos = torch.gather(
            self.target_pos_embed.expand(B, -1, -1),
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, mask_tokens.shape[-1]),
        )
        mask_tokens = mask_tokens + target_pos

        # Concatenate and process
        x = torch.cat([x, mask_tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        predictions = x[:, -num_targets:, :]
        return self.proj_out(predictions)


# ---------------------------------------------------------------------------
# TextJEPA Model
# ---------------------------------------------------------------------------

class TextJEPA(nn.Module):
    """Text JEPA — self-supervised text representation learning.

    Architecture:
        context encoder → encode visible tokens
        target encoder  → encode all tokens (EMA, frozen)
        predictor       → predict masked token embeddings from context

    Loss:
        smooth_l1(predicted_embeddings, target_embeddings)

    This is the text analogue of JEPA (vision). Same training interface,
    same weight delta format, same verification metrics. FedAvg doesn't
    care which modality the deltas came from.
    """

    def __init__(self, config: Optional[TextJEPAConfig] = None):
        super().__init__()
        self.config = config or TextJEPAConfig()

        self.context_encoder = MaskedTextEncoder(self.config)
        self.target_encoder = MaskedTextEncoder(self.config)
        # Copy weights and freeze target
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = TextJEPAPredictor(self.config)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """EMA update of target encoder."""
        momentum = momentum or self.config.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: list[torch.Tensor],
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for text JEPA training.

        Args:
            token_ids:      (B, S) token IDs
            attention_mask:  (B, S) True = real token
            context_mask:   (B, S) True = visible (context)
            target_masks:   list of (B, S) True = target
            return_loss:    whether to compute loss

        Returns:
            Dict with predicted_embeddings, target_embeddings, and optionally loss
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Merge all target masks into one
        all_targets = torch.zeros(B, S, dtype=torch.bool, device=device)
        for tm in target_masks:
            all_targets |= tm

        # Build uniform-sized index tensors across batch
        # Context indices
        ctx_counts = context_mask.sum(dim=1)
        tgt_counts = all_targets.sum(dim=1)
        min_ctx = ctx_counts.min().item()
        min_tgt = max(1, tgt_counts.min().item())

        context_indices_list = []
        target_indices_list = []

        for b in range(B):
            ctx_idx = context_mask[b].nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(ctx_idx), device=device)[:min_ctx]
            context_indices_list.append(ctx_idx[perm])

            tgt_idx = all_targets[b].nonzero(as_tuple=True)[0]
            if len(tgt_idx) == 0:
                # No targets for this sample — use dummy
                tgt_idx = torch.zeros(min_tgt, dtype=torch.long, device=device)
            else:
                perm = torch.randperm(len(tgt_idx), device=device)[:min_tgt]
                tgt_idx = tgt_idx[perm]
            target_indices_list.append(tgt_idx)

        context_indices = torch.stack(context_indices_list)   # (B, min_ctx)
        target_indices = torch.stack(target_indices_list)     # (B, min_tgt)

        # Build uniform context mask from selected indices
        uniform_context_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        for b in range(B):
            uniform_context_mask[b, context_indices[b]] = True

        # Encode context (visible tokens only)
        context_embeddings = self.context_encoder(token_ids, uniform_context_mask)

        # Encode targets (full sequence, no masking, no gradient)
        with torch.no_grad():
            all_embeddings = self.target_encoder(token_ids)  # (B, S, D)
            target_embeddings = torch.gather(
                all_embeddings, 1,
                target_indices.unsqueeze(-1).expand(-1, -1, all_embeddings.shape[-1]),
            )

        # Predict target embeddings
        predicted_embeddings = self.predictor(
            context_embeddings, context_indices, target_indices,
        )

        result = {
            "predicted_embeddings": predicted_embeddings,
            "target_embeddings": target_embeddings,
        }

        if return_loss:
            result["loss"] = F.smooth_l1_loss(
                predicted_embeddings, target_embeddings
            )

        return result


# ---------------------------------------------------------------------------
# Trainer (same interface as JEPATrainer)
# ---------------------------------------------------------------------------

class TextJEPATrainer:
    """Text JEPA trainer — same interface as JEPATrainer for Autonet integration.

    Usage:
        trainer = TextJEPATrainer(config)
        for batch in text_data_source:
            metrics = trainer.train_step(batch, optimizer)
    """

    def __init__(
        self,
        config: Optional[TextJEPAConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config or TextJEPAConfig()
        self.device = device
        self.model = TextJEPA(self.config).to(device)
        self._masker = None

    @property
    def masker(self):
        if self._masker is None:
            from .text_data import TextMasker
            self._masker = TextMasker(mask_ratio=0.20, num_targets=4)
        return self._masker

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: dict with token_ids (B, S) and attention_mask (B, S)
            optimizer: optimizer

        Returns:
            Dict with loss and cosine_similarity metrics
        """
        self.model.train()
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Generate masks
        context_mask, target_masks = self.masker(attention_mask)

        # Forward
        outputs = self.model(
            token_ids, attention_mask, context_mask, target_masks,
            return_loss=True,
        )
        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        self.model.update_target_encoder()

        # Metrics
        with torch.no_grad():
            pred = outputs["predicted_embeddings"]
            target = outputs["target_embeddings"]
            cosine_sim = F.cosine_similarity(
                pred.mean(dim=1), target.mean(dim=1), dim=-1,
            ).mean().item()

        return {
            "loss": loss.item(),
            "cosine_similarity": cosine_sim,
        }

    @torch.no_grad()
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on a batch (for Autonet verification)."""
        self.model.eval()
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        context_mask, target_masks = self.masker(attention_mask)
        outputs = self.model(
            token_ids, attention_mask, context_mask, target_masks,
            return_loss=True,
        )

        pred = outputs["predicted_embeddings"]
        target = outputs["target_embeddings"]

        l2_distance = F.mse_loss(pred, target).item()
        cosine_sim = F.cosine_similarity(
            pred.mean(dim=1), target.mean(dim=1), dim=-1,
        ).mean().item()

        return {
            "l2_distance": l2_distance,
            "cosine_similarity": cosine_sim,
            "smooth_l1_loss": outputs["loss"].item(),
            "embedding_energy": l2_distance,
        }

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights for aggregation."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load model weights."""
        self.model.load_state_dict(weights)
        self.model.target_encoder.load_state_dict(
            self.model.context_encoder.state_dict()
        )
