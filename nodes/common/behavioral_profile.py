"""
Behavioral Semantic Profile — EMA accumulator for alignment pricing.

After each training cycle, the node's agent interaction data is encoded into
K-vectors (the same latent representation used by VL-JEPA's SemanticPredictor).
These are accumulated into an exponential moving average (EMA) that represents
the node's behavioral history.

    profile_t = decay * profile_{t-1} + (1 - decay) * current_embeddings

With decay=0.998 and daily updates, old behavior decays to 1/e in ~500 days.
This prevents gaming — you can't flip your agent prompts today and get cheap
inference tomorrow.

The profile is:
- A single [K, D] tensor (~50KB at K=32, D=384)
- Persisted locally across restarts
- Published as a hash on-chain per epoch (privacy-preserving)
- Used at inference time for K-NN alignment scoring (Story 5.3)

Story 5.1 (BACKLOG_TRAINING_DATA.md)
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BehavioralProfile:
    """Accumulated behavioral semantic profile for a node.

    Maintains an EMA over K-vector representations of the node's training
    data. Updated after each training cycle with the mean-pooled embeddings
    from that cycle's data.

    Usage:
        profile = BehavioralProfile(K=32, D=384)
        # After each training cycle:
        profile.update(embeddings)  # embeddings: (num_samples, K, D)
        # Persist:
        profile.save("~/.atn/profile.pt")
        # On-chain attestation:
        profile_hash = profile.hash()
    """

    def __init__(
        self,
        K: int = 32,
        D: int = 384,
        decay: float = 0.998,
        profile_path: Optional[str] = None,
    ):
        """
        Args:
            K: Number of latent vectors (matches SemanticPredictor.num_latent_vectors)
            D: Embedding dimension (matches VLJEPAConfig.embed_dim)
            decay: EMA decay factor. 0.998 with daily updates → ~500 day half-life.
            profile_path: Path to load/save persisted profile.
        """
        self.K = K
        self.D = D
        self.decay = decay
        self.profile_path = profile_path

        # The accumulated profile — starts as zeros (no history)
        self._profile: torch.Tensor = torch.zeros(K, D)
        self._initialized: bool = False
        self._update_count: int = 0

        # Try to load persisted profile
        if profile_path:
            self._load(profile_path)

    @property
    def profile(self) -> torch.Tensor:
        """Current behavioral profile tensor [K, D]."""
        return self._profile

    @property
    def initialized(self) -> bool:
        """Whether the profile has received at least one update."""
        return self._initialized

    @property
    def update_count(self) -> int:
        """Number of updates applied to this profile."""
        return self._update_count

    def update(self, embeddings: torch.Tensor) -> None:
        """Update the behavioral profile with new training cycle embeddings.

        Args:
            embeddings: Tensor of shape (N, K, D) or (K, D).
                N = number of samples from this training cycle.
                If (N, K, D), mean-pools over N first.
                K and D must match profile dimensions.
        """
        if embeddings.dim() == 3:
            # (N, K, D) → mean over samples → (K, D)
            current = embeddings.mean(dim=0)
        elif embeddings.dim() == 2:
            current = embeddings
        else:
            raise ValueError(
                f"Expected 2D or 3D tensor, got shape {embeddings.shape}"
            )

        if current.shape != (self.K, self.D):
            raise ValueError(
                f"Embedding shape {current.shape} doesn't match profile "
                f"({self.K}, {self.D})"
            )

        current = current.detach().cpu()

        if not self._initialized:
            # First update: initialize directly (no decay of zeros)
            self._profile = current.clone()
            self._initialized = True
        else:
            # EMA: profile_t = decay * profile_{t-1} + (1 - decay) * current
            self._profile = self.decay * self._profile + (1 - self.decay) * current

        self._update_count += 1

    def similarity_to(self, other: "BehavioralProfile") -> float:
        """Cosine similarity between this profile and another.

        Args:
            other: Another BehavioralProfile to compare against.

        Returns:
            Cosine similarity in [-1, 1]. Higher = more similar behavior.
        """
        if not self._initialized or not other._initialized:
            return 0.0

        # Flatten to 1D for cosine similarity
        a = self._profile.flatten()
        b = other._profile.flatten()
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def distance_to_embedding(self, embedding: torch.Tensor) -> float:
        """Cosine distance from this profile to a single embedding.

        Used for K-NN alignment scoring at inference time (Story 5.3):
        - profile ↔ jurisdiction standards
        - profile ↔ request semantics

        Args:
            embedding: Tensor of shape (K, D) — e.g. jurisdiction standards
                encoded through the model, or an inference request's K-vectors.

        Returns:
            Cosine similarity in [-1, 1].
        """
        if not self._initialized:
            return 0.0

        if embedding.dim() == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)

        a = self._profile.flatten()
        b = embedding.detach().cpu().flatten()
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def hash(self) -> str:
        """Compute a deterministic hash of the profile for on-chain attestation.

        The hash is published on-chain per epoch — it links training activity
        to behavioral signature without revealing the profile itself.

        Returns:
            Hex string (SHA-256 of profile tensor bytes).
        """
        # Quantize to float16 for deterministic hashing across platforms
        quantized = self._profile.half()
        raw_bytes = quantized.numpy().tobytes()
        return hashlib.sha256(raw_bytes).hexdigest()

    def save(self, path: Optional[str] = None) -> None:
        """Persist profile to disk.

        Args:
            path: File path. Uses self.profile_path if not specified.
        """
        save_path = Path(path or self.profile_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "profile": self._profile,
                "K": self.K,
                "D": self.D,
                "decay": self.decay,
                "initialized": self._initialized,
                "update_count": self._update_count,
            },
            save_path,
        )
        logger.info("Saved behavioral profile to %s (%d updates)", save_path, self._update_count)

    def _load(self, path: str) -> None:
        """Load profile from disk if it exists."""
        p = Path(path)
        if not p.exists():
            logger.debug("No persisted profile at %s — starting fresh", path)
            return

        try:
            data = torch.load(p, map_location="cpu", weights_only=True)
            if data["K"] != self.K or data["D"] != self.D:
                logger.warning(
                    "Profile dimensions mismatch: saved (%d, %d) vs expected (%d, %d). "
                    "Starting fresh.",
                    data["K"], data["D"], self.K, self.D,
                )
                return

            self._profile = data["profile"]
            self._initialized = data["initialized"]
            self._update_count = data["update_count"]
            logger.info(
                "Loaded behavioral profile from %s (%d updates, decay=%.4f)",
                path, self._update_count, self.decay,
            )
        except Exception as e:
            logger.warning("Failed to load profile from %s: %s", path, e)

    def to_dict(self) -> Dict:
        """Serialize profile metadata (not the tensor) for reporting."""
        return {
            "K": self.K,
            "D": self.D,
            "decay": self.decay,
            "initialized": self._initialized,
            "update_count": self._update_count,
            "hash": self.hash() if self._initialized else None,
            "profile_norm": self._profile.norm().item(),
        }


def compute_training_embeddings(
    trainer,
    data_source,
    max_batches: int = 50,
) -> torch.Tensor:
    """Extract K-vector embeddings from training data using a trained model.

    After a training cycle completes, this function runs the trained model
    on the same data to produce K-vector embeddings that represent the
    semantic content of the training data. These embeddings are then used
    to update the behavioral profile.

    Args:
        trainer: A TextJEPATrainer (or JEPATrainer) with a trained model.
        data_source: Iterable yielding training batches.
        max_batches: Maximum batches to process (limits compute cost).

    Returns:
        Tensor of shape (N, D) where N = total samples processed, D = embed_dim.
        Each row is the mean-pooled context encoder output for one sample.
    """
    trainer.model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(data_source):
            if i >= max_batches:
                break

            token_ids = batch["token_ids"].to(next(trainer.model.parameters()).device)

            # Get context encoder output (full sequence, no masking)
            embeddings = trainer.model.context_encoder(token_ids)  # (B, S, D)

            # Mean-pool over sequence length → (B, D)
            pooled = embeddings.mean(dim=1)
            all_embeddings.append(pooled.cpu())

    if not all_embeddings:
        return torch.zeros(0)

    return torch.cat(all_embeddings, dim=0)  # (N, D)
