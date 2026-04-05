"""
LLM Backbone Encoder — uses a frozen open-weights LLM as the JEPA encoder.

Instead of training a tiny TextJEPA encoder from scratch (64-dim, 2 layers,
byte-level tokens), this module loads a real LLM (Llama, Qwen, Gemma, etc.)
and uses its hidden states as rich semantic embeddings.  Only a small
projection head is trained — it compresses the LLM's representations into
the K*D latent plan vectors that travel the network via FedAvg.

Architecture:
    Text → LLM Tokenizer → Frozen LLM → Hidden States (B, S, hidden_dim)
         → Trainable Projection Head → (B, S, embed_dim)
         → JEPA Predictor (existing) → predicted target embeddings
         → smooth_l1 loss against target embeddings

The projection head is the only trainable component.  It maps from the
LLM's hidden_dim (e.g. 2048 for Llama-1B) to the JEPA's embed_dim
(e.g. 256-768).  This is what gets distributed via FedAvg — the LLM
weights never leave the node.

What travels the network:
    - Projection head weights (~2-8M params, <50MB)
    - Predictor weights (same as before)
    - NOT the LLM weights (those stay local)

Story 7.2: LLM backbone encoder for JEPA training.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class BackboneConfig:
    """Configuration for the LLM backbone encoder."""
    # Model selection (from local_models.ModelSpec)
    model_id: str = ""                  # HuggingFace repo ID
    hidden_dim: int = 2048              # LLM hidden state size (from ModelSpec)
    extract_layer: int = -1             # Layer to extract from (-1 = auto)
    num_layers: int = 16                # Total LLM layers (for auto extract_layer)

    # Projection head
    embed_dim: int = 256                # Output dim (JEPA latent space)
    projection_depth: int = 2           # Number of MLP layers in projection head
    projection_dropout: float = 0.1
    use_layer_mix: bool = True          # Mix multiple layers (weighted sum)
    num_mix_layers: int = 4             # How many layers to mix

    # JEPA predictor (reused from existing architecture)
    num_heads: int = 4
    predictor_depth: int = 2
    predictor_embed_dim: int = 128

    # Training
    max_seq_length: int = 2048
    mask_ratio: float = 0.20
    ema_momentum: float = 0.996

    # Device
    device: str = "auto"
    dtype: str = "float16"              # "float16", "bfloat16", "float32"

    @property
    def resolved_extract_layer(self) -> int:
        if self.extract_layer >= 0:
            return self.extract_layer
        # Default: extract from 2/3 depth (sweet spot for representations)
        return int(self.num_layers * 2 / 3)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.dtype, torch.float16)


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Maps LLM hidden states to JEPA embedding space.

    This is the trainable component.  It's a small MLP that projects
    from the LLM's hidden_dim (e.g. 2048) down to the JEPA's embed_dim
    (e.g. 256).

    When use_layer_mix=True, also learns a weighted combination of hidden
    states from multiple LLM layers (like a mini version of the "learned
    layer aggregation" technique from ELMo/BERT probing).
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        # Layer mixing weights (softmax-normalized)
        if config.use_layer_mix and config.num_mix_layers > 1:
            self.layer_weights = nn.Parameter(
                torch.zeros(config.num_mix_layers)
            )
        else:
            self.layer_weights = None

        # MLP projection: hidden_dim → embed_dim
        layers: list[nn.Module] = []
        in_dim = config.hidden_dim
        for i in range(config.projection_depth):
            out_dim = config.embed_dim if i == config.projection_depth - 1 else in_dim // 2
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.projection_depth - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.projection_dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Project LLM hidden states to JEPA embedding space.

        Args:
            hidden_states: (B, S, hidden_dim) from the primary extraction layer.
            all_hidden_states: Optional list of (B, S, hidden_dim) from multiple
                               layers, for layer mixing.

        Returns:
            (B, S, embed_dim) projected embeddings.
        """
        if self.layer_weights is not None and all_hidden_states is not None:
            # Learned weighted sum of multiple layers
            weights = F.softmax(self.layer_weights, dim=0)
            mixed = torch.zeros_like(hidden_states)
            for w, h in zip(weights, all_hidden_states):
                mixed = mixed + w * h
            hidden_states = mixed

        return self.norm(self.mlp(hidden_states))


# ---------------------------------------------------------------------------
# LLM Backbone Encoder
# ---------------------------------------------------------------------------

class LLMBackboneEncoder(nn.Module):
    """Frozen LLM + trainable projection head for JEPA training.

    The LLM is loaded in inference mode (no gradients, half precision).
    Only the projection head and JEPA predictor are trained.

    Usage:
        encoder = LLMBackboneEncoder(config)
        encoder.load_backbone()  # Downloads/loads the LLM

        # Training loop
        embeddings = encoder(token_ids, attention_mask)  # (B, S, embed_dim)
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.projection = ProjectionHead(config)

        # Positional embedding for the projected space
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # The LLM backbone (loaded lazily via load_backbone())
        self._backbone = None
        self._tokenizer = None
        self._device = None

    @property
    def backbone_loaded(self) -> bool:
        return self._backbone is not None

    def load_backbone(self, device: Optional[str] = None) -> None:
        """Load the LLM backbone from HuggingFace.

        Downloads the model on first call, then caches locally.
        The model is set to eval mode with no gradients.
        """
        if self._backbone is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for LLM backbone. "
                "Install with: pip install transformers"
            )

        model_id = self.config.model_id
        if not model_id:
            raise ValueError("model_id must be set in BackboneConfig")

        # Resolve device
        if device:
            self._device = torch.device(device)
        elif self.config.device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.config.device)

        logger.info(
            "Loading LLM backbone: %s (device=%s, dtype=%s)",
            model_id, self._device, self.config.dtype,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model — output_hidden_states=True so we can extract from any layer
        self._backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.config.torch_dtype,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self._backbone = self._backbone.to(self._device)
        self._backbone.eval()

        # Freeze all backbone parameters
        for param in self._backbone.parameters():
            param.requires_grad = False

        # Move projection head to same device
        self.projection = self.projection.to(self._device)
        self.pos_embed = nn.Parameter(self.pos_embed.to(self._device))

        param_count = sum(p.numel() for p in self._backbone.parameters())
        trainable = sum(p.numel() for p in self.projection.parameters())
        logger.info(
            "Backbone loaded: %s — %.1fM frozen params, %.2fM trainable projection",
            model_id,
            param_count / 1e6,
            trainable / 1e6,
        )

    @property
    def tokenizer(self):
        """Access the backbone's tokenizer (load_backbone() must be called first)."""
        if self._tokenizer is None:
            raise RuntimeError("Call load_backbone() before accessing tokenizer")
        return self._tokenizer

    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts using the backbone's tokenizer.

        Returns dict with 'input_ids' and 'attention_mask' on the correct device.
        """
        max_length = max_length or self.config.max_seq_length
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    @torch.no_grad()
    def _extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run frozen LLM and extract hidden states.

        Returns:
            primary: (B, S, hidden_dim) from the extraction layer
            mix_layers: list of (B, S, hidden_dim) from the last N layers
                        (only if use_layer_mix is True)
        """
        outputs = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        all_hidden = outputs.hidden_states  # tuple of (B, S, hidden_dim)
        extract_idx = self.config.resolved_extract_layer

        # Clamp to valid range
        extract_idx = min(extract_idx, len(all_hidden) - 1)
        primary = all_hidden[extract_idx]

        # Collect layers for mixing
        mix_layers = None
        if self.config.use_layer_mix and self.config.num_mix_layers > 1:
            n = self.config.num_mix_layers
            # Take the last N layers up to and including extract_idx
            start = max(0, extract_idx - n + 1)
            mix_layers = [all_hidden[i] for i in range(start, extract_idx + 1)]
            # Pad if we don't have enough layers
            while len(mix_layers) < n:
                mix_layers.insert(0, mix_layers[0])

        return primary, mix_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text through frozen LLM + trainable projection.

        Args:
            input_ids: (B, S) token IDs from the backbone's tokenizer.
            attention_mask: (B, S) attention mask.

        Returns:
            (B, S, embed_dim) projected embeddings in JEPA latent space.
        """
        if self._backbone is None:
            raise RuntimeError("Call load_backbone() before forward()")

        # Extract hidden states (no grad on LLM)
        primary, mix_layers = self._extract_hidden_states(input_ids, attention_mask)

        # Cast to float32 for projection head if backbone is in half precision
        if primary.dtype != torch.float32:
            primary = primary.float()
            if mix_layers is not None:
                mix_layers = [h.float() for h in mix_layers]

        # Project to JEPA embedding space (this is the trainable part)
        projected = self.projection(primary, mix_layers)

        # Add positional embeddings
        seq_len = projected.shape[1]
        projected = projected + self.pos_embed[:, :seq_len, :]

        return projected

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Convenience: tokenize + encode in one call.

        Returns mean-pooled embeddings: (B, embed_dim).
        """
        encoded = self.tokenize(texts)
        embeddings = self.forward(encoded["input_ids"], encoded["attention_mask"])

        # Mean pool over non-padding positions
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, S, 1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def get_trainable_state_dict(self) -> Dict[str, Any]:
        """Return only the trainable parameters (projection head + pos_embed).

        This is what gets distributed via FedAvg.
        """
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data.clone()
        return state

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load trainable parameters from a state dict (e.g. from FedAvg)."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        backbone_params = sum(
            p.numel() for p in self._backbone.parameters()
        ) if self._backbone else 0
        return backbone_params + sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# JEPA Trainer with LLM Backbone
# ---------------------------------------------------------------------------

class BackboneTextMasker:
    """Generates span masks for backbone-encoded text.

    Same masking strategy as TextMasker but works with the backbone's
    tokenizer tokens instead of byte-level tokens.
    """

    def __init__(
        self,
        mask_ratio: float = 0.20,
        num_targets: int = 4,
        min_span: int = 3,
        max_span: int = 15,
    ):
        self.mask_ratio = mask_ratio
        self.num_targets = num_targets
        self.min_span = min_span
        self.max_span = max_span

    def __call__(
        self,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate context and target masks.

        Args:
            attention_mask: (B, S) bool tensor, True = real token.

        Returns:
            context_mask: (B, S) bool, True = visible to context encoder.
            target_masks: list of (B, S) bool, True = target to predict.
        """
        B, S = attention_mask.shape
        device = attention_mask.device

        # Number of real tokens per sample
        real_lengths = attention_mask.sum(dim=1)  # (B,)

        target_masks = []
        all_targets = torch.zeros(B, S, dtype=torch.bool, device=device)

        for _ in range(self.num_targets):
            mask = torch.zeros(B, S, dtype=torch.bool, device=device)
            for b in range(B):
                real_len = real_lengths[b].item()
                if real_len < self.min_span + 1:
                    continue

                n_mask = max(self.min_span, int(real_len * self.mask_ratio / self.num_targets))
                span_len = min(n_mask, self.max_span, real_len - 1)
                start = torch.randint(0, max(1, real_len - span_len), (1,)).item()
                mask[b, start:start + span_len] = True

            # Don't overlap with previous targets
            mask = mask & ~all_targets & attention_mask
            all_targets = all_targets | mask
            target_masks.append(mask)

        # Context = everything that's real and not a target
        context_mask = attention_mask & ~all_targets

        return context_mask, target_masks


class BackboneJEPATrainer:
    """Trains the projection head using JEPA objectives on LLM hidden states.

    The LLM backbone provides rich semantic embeddings.  We mask spans of
    the projected embeddings and train the predictor to reconstruct the
    target encoder's output at the masked positions.

    Architecture:
        Context path: LLM → ProjectionHead → mask context → Predictor → predicted targets
        Target path:  LLM → ProjectionHead_EMA → full sequence → target embeddings
        Loss: smooth_l1(predicted, target) at masked positions

    Only the projection head and predictor are trained.  The LLM is frozen.
    The target projection head is an EMA copy (same as standard JEPA).
    """

    def __init__(self, config: BackboneConfig, device: str = "auto"):
        self.config = config

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Online encoder (projection head — trainable)
        self.encoder = LLMBackboneEncoder(config)

        # Target encoder (EMA copy of projection head — frozen)
        self.target_encoder = LLMBackboneEncoder(config)

        # Predictor
        from .text_jepa import TextJEPAConfig, TextJEPAPredictor
        predictor_config = TextJEPAConfig(
            embed_dim=config.embed_dim,
            max_seq_length=config.max_seq_length,
            num_heads=config.num_heads,
            predictor_depth=config.predictor_depth,
            predictor_embed_dim=config.predictor_embed_dim,
        )
        self.predictor = TextJEPAPredictor(predictor_config).to(self._device)

        # Masker
        self.masker = BackboneTextMasker(
            mask_ratio=config.mask_ratio,
        )

    def load_backbone(self, device: Optional[str] = None) -> None:
        """Load the LLM backbone for both online and target encoders."""
        dev = device or str(self._device)
        self.encoder.load_backbone(dev)

        # Target encoder shares the same frozen LLM backbone
        # but has its own projection head (EMA copy)
        self.target_encoder._backbone = self.encoder._backbone
        self.target_encoder._tokenizer = self.encoder._tokenizer
        self.target_encoder._device = self.encoder._device
        self.target_encoder.projection = self.target_encoder.projection.to(self._device)
        self.target_encoder.pos_embed = nn.Parameter(
            self.target_encoder.pos_embed.to(self._device)
        )

        # Initialize target projection as copy of online
        self._sync_target()

        # Freeze target projection
        for param in self.target_encoder.projection.parameters():
            param.requires_grad = False
        self.target_encoder.pos_embed.requires_grad = False

    def _sync_target(self) -> None:
        """Copy online projection weights to target (hard copy)."""
        self.target_encoder.projection.load_state_dict(
            self.encoder.projection.state_dict()
        )
        self.target_encoder.pos_embed.data.copy_(self.encoder.pos_embed.data)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        """EMA update of target projection head."""
        m = momentum or self.config.ema_momentum
        for p_online, p_target in zip(
            self.encoder.projection.parameters(),
            self.target_encoder.projection.parameters(),
        ):
            p_target.data.mul_(m).add_(p_online.data, alpha=1 - m)

        # Also EMA update pos_embed
        self.target_encoder.pos_embed.data.mul_(m).add_(
            self.encoder.pos_embed.data, alpha=1 - m,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Run one training step.

        Args:
            input_ids: (B, S) token IDs from the backbone tokenizer.
            attention_mask: (B, S) attention mask.
            optimizer: Optimizer for projection head + predictor.

        Returns:
            Dict with loss, cosine_similarity, etc.
        """
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        # Generate masks
        context_mask, target_masks = self.masker(attention_mask)

        # Target embeddings (no grad — from EMA projection)
        with torch.no_grad():
            target_embeds = self.target_encoder(input_ids, attention_mask)

        # Context embeddings (trainable projection)
        context_embeds = self.encoder(input_ids, attention_mask)

        # Gather context indices and embeddings
        context_indices = context_mask.nonzero(as_tuple=False)
        # We need per-sample indices, so we work sample-by-sample
        B, S = input_ids.shape
        total_loss = 0.0
        total_cos = 0.0
        n_targets = 0

        for target_mask in target_masks:
            for b in range(B):
                ctx_idx = context_mask[b].nonzero(as_tuple=True)[0]  # (num_ctx,)
                tgt_idx = target_mask[b].nonzero(as_tuple=True)[0]   # (num_tgt,)

                if len(ctx_idx) == 0 or len(tgt_idx) == 0:
                    continue

                ctx_emb = context_embeds[b:b+1, ctx_idx, :]        # (1, num_ctx, D)
                tgt_emb_true = target_embeds[b:b+1, tgt_idx, :]    # (1, num_tgt, D)

                # Predict target embeddings from context
                predicted = self.predictor(
                    ctx_emb,
                    ctx_idx.unsqueeze(0),
                    tgt_idx.unsqueeze(0),
                )  # (1, num_tgt, D)

                loss = F.smooth_l1_loss(predicted, tgt_emb_true.detach())
                total_loss = total_loss + loss

                with torch.no_grad():
                    cos = F.cosine_similarity(
                        predicted.mean(dim=1),
                        tgt_emb_true.mean(dim=1),
                    ).item()
                    total_cos += cos
                    n_targets += 1

        if n_targets > 0:
            avg_loss = total_loss / n_targets
        else:
            avg_loss = torch.tensor(0.0, device=self._device, requires_grad=True)

        optimizer.zero_grad()
        avg_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.projection.parameters())
            + [self.encoder.pos_embed]
            + list(self.predictor.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        # EMA update
        self.update_target_encoder()

        return {
            "loss": avg_loss.item(),
            "cosine_similarity": total_cos / max(n_targets, 1),
            "num_targets": n_targets,
        }

    def get_weights(self) -> Dict[str, Any]:
        """Return trainable weights for FedAvg."""
        state = {}
        # Projection head
        for name, param in self.encoder.projection.named_parameters():
            state[f"projection.{name}"] = param.data.cpu()
        # Positional embeddings
        state["pos_embed"] = self.encoder.pos_embed.data.cpu()
        # Predictor
        for name, param in self.predictor.named_parameters():
            state[f"predictor.{name}"] = param.data.cpu()
        return state

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load trainable weights (e.g. from FedAvg global model)."""
        for name, param in state_dict.items():
            if name.startswith("projection."):
                key = name[len("projection."):]
                proj_state = self.encoder.projection.state_dict()
                if key in proj_state:
                    proj_state[key].copy_(param)
            elif name == "pos_embed":
                self.encoder.pos_embed.data.copy_(param)
            elif name.startswith("predictor."):
                key = name[len("predictor."):]
                pred_state = self.predictor.state_dict()
                if key in pred_state:
                    pred_state[key].copy_(param)

        # Re-sync target
        self._sync_target()

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate without gradients."""
        self.encoder.projection.eval()
        self.predictor.eval()
        with torch.no_grad():
            # Use a dummy optimizer
            metrics = self._eval_forward(input_ids, attention_mask)
        self.encoder.projection.train()
        self.predictor.train()
        return metrics

    @torch.no_grad()
    def _eval_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Forward pass for evaluation (no gradients, no optimizer step)."""
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        context_mask, target_masks = self.masker(attention_mask)
        target_embeds = self.target_encoder(input_ids, attention_mask)
        context_embeds = self.encoder(input_ids, attention_mask)

        B = input_ids.shape[0]
        total_loss = 0.0
        total_cos = 0.0
        n = 0

        for target_mask in target_masks:
            for b in range(B):
                ctx_idx = context_mask[b].nonzero(as_tuple=True)[0]
                tgt_idx = target_mask[b].nonzero(as_tuple=True)[0]
                if len(ctx_idx) == 0 or len(tgt_idx) == 0:
                    continue

                predicted = self.predictor(
                    context_embeds[b:b+1, ctx_idx, :],
                    ctx_idx.unsqueeze(0),
                    tgt_idx.unsqueeze(0),
                )
                true = target_embeds[b:b+1, tgt_idx, :]
                total_loss += F.smooth_l1_loss(predicted, true).item()
                total_cos += F.cosine_similarity(
                    predicted.mean(dim=1), true.mean(dim=1),
                ).item()
                n += 1

        return {
            "loss": total_loss / max(n, 1),
            "cosine_similarity": total_cos / max(n, 1),
        }
