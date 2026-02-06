"""
JEPA (Joint Embedding Predictive Architecture) Training Module for Autonet

Implements JEPA-style self-supervised learning:
- Predicts masked embeddings in representation space (not raw tokens/pixels)
- Works with any encoder backbone (ViT, CNN, Transformer-LM)
- Context/target masking strategy
- Predictor network for embedding prediction
- EMA (Exponential Moving Average) target encoder

This enables self-supervised training without labeled data,
perfect for decentralized training where labels are expensive.

Supported Modalities:
- Vision: Image patches via ViT encoder (I-JEPA style)
- Text: Token sequences via Transformer encoder (planned)
- Video: Spatiotemporal patches (V-JEPA style, planned)

The key insight: JEPA predicts in EMBEDDING space, not output space.
This makes verification simple: compare embedding similarity.

For Absolute Zero loop integration:
- Ground truth = Target encoder embeddings
- Verification = Cosine similarity of embeddings
- FedAvg = Standard weight averaging on context encoder

References:
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
"""

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JEPAConfig:
    """
    Configuration for JEPA training.

    Supports multiple modalities:
    - "vision": Images/video (default) - uses ViT encoder
    - "text": Text sequences - uses Transformer encoder (future)
    """
    # Modality
    modality: str = "vision"  # "vision" or "text"

    # Vision-specific settings (used when modality="vision")
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Text-specific settings (used when modality="text", future)
    vocab_size: int = 32000  # For text modality
    max_seq_length: int = 512  # For text modality

    # Encoder settings (shared across modalities)
    embed_dim: int = 384  # Small ViT / Small Transformer
    num_heads: int = 6
    encoder_depth: int = 12
    mlp_ratio: float = 4.0

    # Predictor settings
    predictor_depth: int = 6
    predictor_embed_dim: int = 192

    # Masking settings
    num_targets: int = 4  # Number of target blocks
    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    target_scale: Tuple[float, float] = (0.15, 0.2)  # Size of target blocks
    context_scale: Tuple[float, float] = (0.85, 1.0)  # Size of context

    # Training settings
    ema_momentum: float = 0.996  # EMA for target encoder

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


# =============================================================================
# Vision Transformer Components
# =============================================================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP block for transformer."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder for JEPA."""

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(config)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.encoder_depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings with sinusoidal pattern
        pos_embed = get_2d_sincos_pos_embed(
            self.config.embed_dim,
            self.config.grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional masking.

        Args:
            x: Input image (B, C, H, W)
            mask: Boolean mask of patches to KEEP (B, num_patches)
                  If None, process all patches

        Returns:
            Patch embeddings (B, num_visible_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply mask if provided (keep only visible patches)
        if mask is not None:
            B, N, D = x.shape
            # mask: (B, N) boolean - True means KEEP
            # All samples must have the same number of True values (enforced by caller)
            # Use list comprehension to select per sample, then stack
            x = torch.stack([x[b][mask[b]] for b in range(B)])  # (B, num_visible, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class JEPAPredictor(nn.Module):
    """
    Predictor network for JEPA.

    Takes context embeddings + target positions and predicts target embeddings.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # Project from encoder dim to predictor dim
        self.proj_in = nn.Linear(config.embed_dim, config.predictor_embed_dim)

        # Positional embedding for target positions
        self.target_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.predictor_embed_dim)
        )

        # Mask token for targets
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.predictor_embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.predictor_embed_dim,
                config.num_heads // 2,  # Fewer heads for smaller predictor
                config.mlp_ratio
            )
            for _ in range(config.predictor_depth)
        ])

        self.norm = nn.LayerNorm(config.predictor_embed_dim)

        # Project back to encoder dim for loss
        self.proj_out = nn.Linear(config.predictor_embed_dim, config.embed_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.config.predictor_embed_dim,
            self.config.grid_size
        )
        self.target_pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict target embeddings from context.

        Args:
            context_embeddings: Encoded context patches (B, num_context, embed_dim)
            context_indices: Indices of context patches (B, num_context)
            target_indices: Indices of target patches to predict (B, num_targets)

        Returns:
            Predicted embeddings for targets (B, num_targets, embed_dim)
        """
        B = context_embeddings.shape[0]
        num_targets = target_indices.shape[1]

        # Project context to predictor dim
        x = self.proj_in(context_embeddings)  # (B, num_context, pred_dim)

        # Add positional embeddings for context
        context_pos = self.target_pos_embed[:, :, :].expand(B, -1, -1)
        context_pos = torch.gather(
            context_pos, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x = x + context_pos

        # Create mask tokens for targets with positional info
        mask_tokens = self.mask_token.expand(B, num_targets, -1)
        target_pos = torch.gather(
            self.target_pos_embed.expand(B, -1, -1), 1,
            target_indices.unsqueeze(-1).expand(-1, -1, mask_tokens.shape[-1])
        )
        mask_tokens = mask_tokens + target_pos

        # Concatenate context and mask tokens
        x = torch.cat([x, mask_tokens], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract predictions for target positions (last num_targets tokens)
        predictions = x[:, -num_targets:, :]

        # Project back to encoder dim
        predictions = self.proj_out(predictions)

        return predictions


# =============================================================================
# JEPA Model
# =============================================================================

class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture for self-supervised learning.

    Architecture:
    - Context encoder: Encodes visible (unmasked) patches
    - Target encoder: EMA copy of context encoder (no gradients)
    - Predictor: Predicts target embeddings from context

    Training:
    1. Sample context and target blocks from image
    2. Encode context with context encoder
    3. Encode targets with target encoder (frozen EMA)
    4. Predict target embeddings from context
    5. Minimize L2 loss between predicted and actual target embeddings
    """

    def __init__(self, config: Optional[JEPAConfig] = None):
        super().__init__()
        self.config = config or JEPAConfig()

        # Context encoder (trained)
        self.context_encoder = VisionTransformerEncoder(self.config)

        # Target encoder (EMA of context encoder, no gradients)
        self.target_encoder = VisionTransformerEncoder(self.config)
        # Copy weights and freeze
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = JEPAPredictor(self.config)

        # Masking module
        self.masker = JEPAMasker(self.config)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update target encoder with EMA of context encoder."""
        momentum = momentum or self.config.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def forward(
        self,
        images: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA training.

        Args:
            images: Input images (B, C, H, W)
            return_loss: Whether to compute and return loss

        Returns:
            Dict with:
            - loss: JEPA loss (if return_loss)
            - predicted_embeddings: Predictor outputs
            - target_embeddings: Target encoder outputs
        """
        B = images.shape[0]
        device = images.device

        # Generate masks
        context_mask, target_masks = self.masker(B, device)

        # Get uniform sizes across batch (use minimum to ensure all have enough)
        num_context_per_sample = context_mask.sum(dim=1)  # (B,)
        num_target_per_sample = target_masks[0].sum(dim=1)  # (B,)
        min_context = num_context_per_sample.min().item()
        min_target = num_target_per_sample.min().item()

        # Extract indices per sample, subsampling to uniform size
        context_indices_list = []
        target_indices_list = []

        for b in range(B):
            # Get all context indices for this sample
            ctx_idx = context_mask[b].nonzero(as_tuple=True)[0]
            # Randomly select min_context indices (shuffle and take first)
            perm = torch.randperm(len(ctx_idx), device=device)[:min_context]
            context_indices_list.append(ctx_idx[perm])

            # Get all target indices for this sample
            tgt_idx = target_masks[0][b].nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(tgt_idx), device=device)[:min_target]
            target_indices_list.append(tgt_idx[perm])

        context_indices = torch.stack(context_indices_list)  # (B, min_context)
        target_indices = torch.stack(target_indices_list)    # (B, min_target)

        # Create uniform masks from selected indices
        uniform_context_mask = torch.zeros(B, self.config.num_patches, dtype=torch.bool, device=device)
        for b in range(B):
            uniform_context_mask[b, context_indices[b]] = True

        # Encode context (visible patches)
        context_embeddings = self.context_encoder(images, uniform_context_mask)

        # Encode targets (no gradient)
        with torch.no_grad():
            # Get full patch embeddings
            all_embeddings = self.target_encoder(images)
            # Extract target embeddings using gather
            target_embeddings = torch.gather(
                all_embeddings, 1,
                target_indices.unsqueeze(-1).expand(-1, -1, all_embeddings.shape[-1])
            )

        # Predict target embeddings
        predicted_embeddings = self.predictor(
            context_embeddings, context_indices, target_indices
        )

        result = {
            'predicted_embeddings': predicted_embeddings,
            'target_embeddings': target_embeddings,
        }

        if return_loss:
            # Smooth L1 loss in embedding space
            loss = F.smooth_l1_loss(predicted_embeddings, target_embeddings)
            result['loss'] = loss

        return result

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings (for inference/downstream tasks).
        Uses target encoder for stable representations.
        """
        with torch.no_grad():
            embeddings = self.target_encoder(images)
        return embeddings


# =============================================================================
# Masking Strategy
# =============================================================================

class JEPAMasker:
    """
    Generates context and target masks for JEPA training.

    Strategy (from I-JEPA paper):
    - Sample multiple small target blocks
    - Context is everything except targets
    - Targets should be spatially distributed but not too small
    """

    def __init__(self, config: JEPAConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.num_patches = config.num_patches

    def __call__(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate masks for a batch.

        Returns:
            context_mask: (B, num_patches) boolean - True = context (visible)
            target_masks: List of (B, num_patches) boolean - True = target
        """
        context_masks = []
        all_target_masks = []

        for _ in range(batch_size):
            # Sample target blocks
            target_indices_list = []
            used_positions = set()

            for target_num in range(self.config.num_targets):
                # Sample target block dimensions
                aspect = torch.empty(1).uniform_(*self.config.target_aspect_ratio).item()
                scale = torch.empty(1).uniform_(*self.config.target_scale).item()

                target_area = self.num_patches * scale
                h = int(math.sqrt(target_area / aspect))
                w = int(h * aspect)
                h = max(1, min(h, self.grid_size))
                w = max(1, min(w, self.grid_size))

                # Sample position
                max_attempts = 50  # Increased attempts
                success = False
                for _ in range(max_attempts):
                    top = torch.randint(0, max(1, self.grid_size - h + 1), (1,)).item()
                    left = torch.randint(0, max(1, self.grid_size - w + 1), (1,)).item()

                    # Get patch indices for this block
                    indices = []
                    for i in range(h):
                        for j in range(w):
                            idx = (top + i) * self.grid_size + (left + j)
                            indices.append(idx)

                    # Check overlap with existing targets
                    if not any(idx in used_positions for idx in indices):
                        target_indices_list.append(indices)
                        used_positions.update(indices)
                        success = True
                        break

                # Fallback: if no valid position found, pick random unused patches
                if not success:
                    available = [i for i in range(self.num_patches) if i not in used_positions]
                    if available:
                        num_to_pick = min(4, len(available))  # Small fallback block
                        indices = available[:num_to_pick]
                        target_indices_list.append(indices)
                        used_positions.update(indices)

            # Create target masks
            sample_target_masks = []
            for indices in target_indices_list:
                mask = torch.zeros(self.num_patches, dtype=torch.bool, device=device)
                mask[indices] = True
                sample_target_masks.append(mask)

            # Context is everything except targets
            context_mask = torch.ones(self.num_patches, dtype=torch.bool, device=device)
            context_mask[list(used_positions)] = False

            context_masks.append(context_mask)
            all_target_masks.append(sample_target_masks)

        # Stack batch
        context_mask = torch.stack(context_masks)  # (B, num_patches)

        # Reorganize target masks: list of (B, num_patches)
        num_targets = len(all_target_masks[0])
        target_masks = [
            torch.stack([all_target_masks[b][t] for b in range(batch_size)])
            for t in range(num_targets)
        ]

        return context_mask, target_masks


# =============================================================================
# Utilities
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> 'np.ndarray':
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (H/patch_size)

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    import numpy as np

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: 'np.ndarray') -> 'np.ndarray':
    """Generate positional embeddings from a grid."""
    import numpy as np

    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: 'np.ndarray') -> 'np.ndarray':
    """Generate 1D sinusoidal positional embeddings."""
    import numpy as np

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, embed_dim/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb


# =============================================================================
# Training Utilities for Autonet Integration
# =============================================================================

class JEPATrainer:
    """
    JEPA trainer compatible with Autonet's training loop.

    Key differences from supervised training:
    - No labels needed (self-supervised)
    - Verification uses embedding distance instead of accuracy
    - EMA update of target encoder each step
    """

    def __init__(
        self,
        config: Optional[JEPAConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config or JEPAConfig()
        self.device = device
        self.model = JEPA(self.config).to(device)

    def train_step(
        self,
        images: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            images: Batch of images (B, C, H, W)
            optimizer: Optimizer

        Returns:
            Dict with training metrics
        """
        self.model.train()
        images = images.to(self.device)

        # Forward pass
        outputs = self.model(images, return_loss=True)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target encoder (EMA)
        self.model.update_target_encoder()

        # Compute embedding distance for monitoring
        with torch.no_grad():
            pred = outputs['predicted_embeddings']
            target = outputs['target_embeddings']
            cosine_sim = F.cosine_similarity(
                pred.mean(dim=1),
                target.mean(dim=1),
                dim=-1
            ).mean().item()

        return {
            'loss': loss.item(),
            'cosine_similarity': cosine_sim,
        }

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on a batch (for verification).

        Returns embedding distance metrics used for Autonet verification.
        """
        self.model.eval()
        images = images.to(self.device)

        outputs = self.model(images, return_loss=True)

        pred = outputs['predicted_embeddings']
        target = outputs['target_embeddings']

        # Multiple metrics for robust verification
        l2_distance = F.mse_loss(pred, target).item()
        cosine_sim = F.cosine_similarity(
            pred.mean(dim=1),
            target.mean(dim=1),
            dim=-1
        ).mean().item()
        smooth_l1 = outputs['loss'].item()

        return {
            'l2_distance': l2_distance,
            'cosine_similarity': cosine_sim,
            'smooth_l1_loss': smooth_l1,
            'embedding_energy': l2_distance,  # Lower = better (like JEPA energy)
        }

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights for aggregation."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load model weights."""
        self.model.load_state_dict(weights)
        # Also update target encoder after loading
        self.model.target_encoder.load_state_dict(
            self.model.context_encoder.state_dict()
        )


# =============================================================================
# Autonet Task Specification for JEPA
# =============================================================================

@dataclass
class JEPATaskSpec:
    """
    Task specification for JEPA training in Autonet.

    Unlike supervised tasks, JEPA tasks specify:
    - Data source (unlabeled images/video)
    - Masking strategy parameters
    - Target encoder checkpoint (for verification consistency)
    """
    data_cid: str  # CID of unlabeled training data
    validation_cid: str  # CID of validation data for verification
    config: JEPAConfig
    target_encoder_cid: Optional[str] = None  # For consistent verification
    min_cosine_similarity: float = 0.5  # Verification threshold
    max_embedding_energy: float = 1.0  # Verification threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'type': 'jepa',
            'data_cid': self.data_cid,
            'validation_cid': self.validation_cid,
            'config': {
                'image_size': self.config.image_size,
                'patch_size': self.config.patch_size,
                'embed_dim': self.config.embed_dim,
                'encoder_depth': self.config.encoder_depth,
                'predictor_depth': self.config.predictor_depth,
            },
            'target_encoder_cid': self.target_encoder_cid,
            'min_cosine_similarity': self.min_cosine_similarity,
            'max_embedding_energy': self.max_embedding_energy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JEPATaskSpec':
        """Deserialize from dict."""
        config_data = data.get('config', {})
        config = JEPAConfig(
            image_size=config_data.get('image_size', 224),
            patch_size=config_data.get('patch_size', 16),
            embed_dim=config_data.get('embed_dim', 384),
            encoder_depth=config_data.get('encoder_depth', 12),
            predictor_depth=config_data.get('predictor_depth', 6),
        )
        return cls(
            data_cid=data['data_cid'],
            validation_cid=data['validation_cid'],
            config=config,
            target_encoder_cid=data.get('target_encoder_cid'),
            min_cosine_similarity=data.get('min_cosine_similarity', 0.5),
            max_embedding_energy=data.get('max_embedding_energy', 1.0),
        )
