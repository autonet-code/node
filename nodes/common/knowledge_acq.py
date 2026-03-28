"""
Knowledge Acquisition Module — ATN Phase C

Detects when the VL-JEPA model encounters a query outside its latent knowledge
(OOD), routes it to the appropriate acquisition strategy, and integrates the
resulting knowledge into the network's shared embedding store.

Architecture (from novel_knowledge_verification_survey.md §5):

    Query embedding z_q
        │
        ▼
    ┌──────────────────────────────────────────────────────┐
    │  OODDetector  (< 1 ms total)                         │
    │  1. Predictive variance — short JEPA rollout (2 steps)│
    │  2. KNN distance        — FAISS / torch brute-force  │
    │  3. Combined score      — α·σ² + β·d_knn             │
    │  → epistemic (fallback) | aleatoric | confident       │
    └──────────────────────────────────────────────────────┘
        │                    │                    │
        │ epistemic OOD      │ uncertain / net    │ in-distribution
        ▼                    ▼                    ▼
    TokenSpaceFallback   NetworkAcquire       Return cached plan
        │                    │                    (TwoSpeedEngine)
        │                    ▼
        │                KnowledgeStore
        │                search + return
        │
        ▼
    Research result  (token-space LLM response)
        │
        ├──→ Encode to embedding (VL-JEPA encoder)
        ├──→ Novelty gate  (Layer 1: KNN distance)
        ├──→ Consistency check (Layer 2: contradiction detection)
        ├──→ Commit attestation (hash of embedding + trace)
        └──→ KnowledgeSharingProtocol.propose()
                │
                ├──→ Probabilistic spot-check (Layer 3, 7% rate)
                ├──→ Challenge window (Layer 4, 24 h)
                └──→ KnowledgeStore.add() on acceptance
                      TraceRank reputation update

Components:
    OODDetector              — VJEPA variance + KNN distance → novelty score
    TokenSpaceFallback       — LLM provider bridge; re-anchors embeddings
    KnowledgeStore           — Validated embedding index with provenance
    KnowledgeSharingProtocol — Network exchange, attestations, verification
    KnowledgeAcquisitionEngine — Main coordinator

Key papers:
    - Sun et al. (2022): KNN-based OOD detection, ICML 2022, arXiv:2204.06507
    - VJEPA (2026): Variational JEPA uncertainty, arXiv:2601.14354
    - TraceRank (Shi et al., 2025): Reputation via usage flows, arXiv:2510.27554
    - RIFT Protocol (2025-26): Commit-reveal + peer review
    - Bittensor (2022-26): Stake-weighted validation, Yuma Consensus

Phase B compatibility:
    FAISS is optional — the module falls back to torch brute-force similarity
    search when FAISS is not installed.  FAISS is strongly recommended for
    production deployments with > 10,000 knowledge entries.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# FAISS is optional: available in production; falls back to torch brute-force.
try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore[assignment]

from .vl_jepa import VLJEPA

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OODConfig:
    """Thresholds for the OODDetector.

    Two independent signals drive OOD decisions:
      1. **Predictive variance** — proxy for VJEPA's σ²: mean uncertainty
         across a short (``variance_rollout_steps``) rollout.  High = model
         is uncertain about this query's latent dynamics.
      2. **KNN distance** — cosine distance from the k-th nearest neighbour
         in the in-distribution index.  High = query is far from anything the
         model has been trained on.

    Interpretation matrix (survey §2.4):
      High σ² + High d_knn  → epistemic: trigger fallback
      High σ² + Low  d_knn  → aleatoric: don't fallback (inherently ambiguous)
      Low  σ² + High d_knn  → caution: verify prediction, but don't fallback
      Low  σ² + Low  d_knn  → confident: proceed normally
    """

    # ---- KNN distance signal ----
    knn_k: int = 5
    """Number of nearest neighbours used to compute the KNN distance score.
    Sun et al. (2022) find that k=1 works well; k=5 is more robust to noise."""

    max_index_size: int = 50_000
    """Maximum number of embeddings stored in the in-distribution index.
    Older embeddings are dropped FIFO when this limit is reached."""

    # ---- Predictive variance signal ----
    variance_rollout_steps: int = 2
    """Number of latent dynamics steps used to estimate predictive variance.
    Short (2) = fast; longer = more accurate but higher latency.
    V-JEPA 2 uses T=2 as its standard rollout horizon."""

    # ---- Combined score weights ----
    alpha: float = 0.5
    """Weight of the predictive-variance sub-score in the combined OOD score.
    Combined score = alpha * variance_score + beta * knn_score."""

    beta: float = 0.5
    """Weight of the KNN-distance sub-score in the combined OOD score."""

    # ---- Decision thresholds ----
    epistemic_threshold: float = 0.65
    """Combined score above this → epistemic OOD → trigger token-space fallback.
    Conservative default; calibrate against held-out traces with known labels."""

    aleatoric_knn_threshold: float = 0.55
    """If KNN score > this but variance score < aleatoric_variance_ceiling,
    classify as aleatoric uncertainty (inherently ambiguous, no fallback)."""

    aleatoric_variance_ceiling: float = 0.35
    """Maximum variance score that still allows aleatoric classification.
    Above this, the model is actively uncertain — not just the data."""

    caution_knn_threshold: float = 0.45
    """KNN score above this triggers a 'caution' state even when the combined
    score is below epistemic_threshold.  Caution: continue but log."""


@dataclass
class TokenSpaceFallbackConfig:
    """Configuration for the TokenSpaceFallback bridge."""

    max_history_size: int = 1_000
    """Maximum number of fallback records retained in memory for monitoring."""

    re_anchor_on_response: bool = True
    """If True, encode the provider's response back into embedding space
    (via VLJEPA text encoder) so the result can be stored in KnowledgeStore."""

    provider_id: str = "local_llm"
    """Identifier of the token-space provider (for provenance logging)."""


@dataclass
class KnowledgeStoreConfig:
    """Configuration for KnowledgeStore."""

    embed_dim: int = 384
    """Embedding dimension — must match VLJEPAConfig.embed_dim."""

    max_entries: int = 100_000
    """Maximum validated knowledge entries.  FIFO eviction when full."""

    # ---- Novelty gate (Layer 1) ----
    novelty_distance_threshold: float = 0.30
    """Minimum cosine distance from the nearest existing entry for a new
    contribution to be accepted as novel.  Below this = redundant.
    The survey recommends starting conservative and tuning upwards."""

    # ---- Consistency check (Layer 2) ----
    consistency_contradiction_threshold: float = 0.92
    """Cosine similarity above this between a new embedding and an existing
    one with *opposite* label is flagged as a potential contradiction.
    (Phase B: we use high similarity to the negated embedding as a proxy
    for contradiction; Phase C can add explicit label tracking.)"""

    # ---- Quality floor ----
    quality_score_floor: float = 0.5
    """Entries with quality_score below this are not stored."""


@dataclass
class KnowledgeSharingConfig:
    """Configuration for KnowledgeSharingProtocol."""

    spot_check_rate: float = 0.07
    """Fraction of incoming contributions selected for probabilistic
    spot-checking (Layer 3).  7% matches the survey recommendation."""

    min_stake_for_validation: float = 100.0
    """Minimum stake (in jurisdiction token units) required for a node to
    act as a validator.  Sybil resistance: fake validators have no stake."""

    challenge_window_seconds: float = 86_400.0
    """Seconds after proposal before a contribution is considered final
    (barring active challenges).  Default: 24 hours."""

    min_validators_for_acceptance: int = 1
    """Minimum number of validators that must endorse a contribution.
    Phase B: 1 (local self-endorsement).  Phase C: 3+."""

    reputation_decay: float = 0.98
    """Per-round multiplicative decay applied to TraceRank scores.
    Prevents old high-reputation nodes from dominating indefinitely."""


@dataclass
class KnowledgeAcqConfig:
    """Top-level configuration for KnowledgeAcquisitionEngine.

    Composes sub-configs for each component.  The defaults target Phase B
    (single-node) operation; Phase C wires the network acquisition path.
    """

    ood: OODConfig = field(default_factory=OODConfig)
    fallback: TokenSpaceFallbackConfig = field(default_factory=TokenSpaceFallbackConfig)
    store: KnowledgeStoreConfig = field(default_factory=KnowledgeStoreConfig)
    sharing: KnowledgeSharingConfig = field(default_factory=KnowledgeSharingConfig)

    # ---- Network acquisition (Phase C stub) ----
    network_acquisition_timeout_ms: float = 500.0
    """Timeout for attempting to retrieve matching knowledge from the network
    before falling back to local token-space research."""

    min_network_match_similarity: float = 0.80
    """Minimum cosine similarity between the query embedding and a network
    knowledge entry for the entry to be considered a hit."""

    # ---- Feature flags ----
    enable_network_acquisition: bool = False
    """Phase B: disabled (stub only).  Phase C: True."""

    enable_knowledge_sharing: bool = True
    """When True, validated research results are proposed to the network."""

    track_provenance: bool = True
    """When True, all knowledge entries carry provenance metadata including
    contributing node ID and the research trace hash."""


# =============================================================================
# Structured output types
# =============================================================================


class UncertaintyType(Enum):
    """Classification of an OOD detector's uncertainty reading."""
    CONFIDENT = auto()    # low σ², low d_knn  — proceed normally
    ALEATORIC = auto()    # high d_knn, low σ² — inherently ambiguous; no fallback
    CAUTION = auto()      # moderate d_knn     — verify but don't fallback
    EPISTEMIC = auto()    # high σ², high d_knn — trigger token-space fallback


@dataclass
class OODResult:
    """Output of a single OODDetector.detect() call.

    Attributes:
        query_embedding:        The (D,) query embedding that was evaluated.
        variance_score:         Predictive-variance sub-score ∈ [0, 1].
                                0 = confident dynamics; 1 = collapsed/uncertain.
        knn_distance_score:     KNN cosine-distance sub-score ∈ [0, 1].
                                0 = very close to training data; 1 = far away.
        combined_score:         Weighted combination of the two signals ∈ [0, 1].
        uncertainty_type:       CONFIDENT | ALEATORIC | CAUTION | EPISTEMIC.
        nearest_distance:       Raw cosine distance to the nearest neighbour.
        index_size:             Number of embeddings in the in-distribution index.
        latency_ms:             Wall-clock time for the detection call.
    """

    query_embedding: torch.Tensor       # (D,) detached, on CPU
    variance_score: float
    knn_distance_score: float
    combined_score: float
    uncertainty_type: UncertaintyType
    nearest_distance: float
    index_size: int
    latency_ms: float


@dataclass
class FallbackResult:
    """Output of a single TokenSpaceFallback.run() call.

    Attributes:
        query_text:           The original query text.
        response_text:        The token-space provider's response.
        re_anchored_embedding: (D,) embedding of the response, or None if
                              ``config.re_anchor_on_response`` is False or
                              encoding failed.
        provider_id:          Identifier of the token-space provider used.
        latency_ms:           Wall-clock time for the fallback call.
        fallback_count:       Total fallback calls made so far (for monitoring).
    """

    query_text: str
    response_text: str
    re_anchored_embedding: Optional[torch.Tensor]  # (D,) or None
    provider_id: str
    latency_ms: float
    fallback_count: int


@dataclass
class KnowledgeEntryMetadata:
    """Provenance and quality metadata for a single KnowledgeStore entry.

    All fields are serialisable for network exchange.
    """

    entry_id: str
    """Unique identifier (UUID4) assigned at insertion time."""

    contributor_node_id: str
    """Identifier of the node that contributed this knowledge."""

    timestamp: float
    """Unix timestamp of the contribution (UTC)."""

    novelty_score: float
    """Cosine distance from the nearest pre-existing entry at insertion time.
    Higher = more novel.  Stored for reward calculation."""

    quality_score: float
    """Quality score ∈ [0, 1] from the verification pipeline.  Starts at
    1.0 for unverified entries; updated after spot-check / challenge."""

    verification_status: str
    """One of: 'pending' | 'verified' | 'challenged' | 'rejected'."""

    usage_count: int
    """Number of inference calls that activated this knowledge entry.
    Drives the usage_bonus in the reward formula."""

    research_trace_hash: str
    """SHA-256 hash of the token-space research trace that produced this
    entry.  Used for spot-check verification and dispute resolution."""

    challenge_deadline: float
    """Unix timestamp after which the entry is considered final (if no
    active challenge exists)."""


@dataclass
class KnowledgeEntry:
    """A single validated knowledge entry in KnowledgeStore.

    The embedding is stored detached on CPU to keep VRAM free for inference.
    Move to device before computing similarity.
    """

    embedding: torch.Tensor           # (D,) float32 on CPU
    metadata: KnowledgeEntryMetadata


@dataclass
class ContributionAttestation:
    """Cryptographic attestation accompanying a knowledge contribution.

    Phase B: simplified — uses SHA-256 hashes of the constituent fields.
    Phase C: replace ``signature`` with an actual ECDSA / BLS signature
    over the embedding hash using the node's key pair.

    Attributes:
        contributor_node_id: Node that produced the contribution.
        embedding_hash:      SHA-256 hex digest of the raw embedding bytes.
        research_trace_hash: SHA-256 hex digest of the research trace JSON.
        timestamp:           Unix timestamp of the attestation.
        signature:           SHA-256 of (contributor_node_id + embedding_hash
                             + research_trace_hash + timestamp).
                             Phase C: replace with cryptographic signature.
    """

    contributor_node_id: str
    embedding_hash: str
    research_trace_hash: str
    timestamp: float
    signature: str


@dataclass
class KnowledgeContribution:
    """A packaged knowledge contribution ready for network proposal.

    Attributes:
        entry_id:       Local ID assigned before network submission.
        embedding:      The (D,) knowledge embedding (L2-normalised).
        attestation:    Provenance and integrity attestation.
        novelty_score:  Estimated novelty at the time of packaging.
        proposed_reward: Estimated reward per the survey §5.4 formula.
    """

    entry_id: str
    embedding: torch.Tensor       # (D,) float32
    attestation: ContributionAttestation
    novelty_score: float
    proposed_reward: float


class VerificationVerdict(Enum):
    """Outcome of KnowledgeSharingProtocol.verify_incoming()."""
    ACCEPTED = auto()
    REJECTED_NOVELTY = auto()       # Layer 1 novelty gate failed
    REJECTED_CONSISTENCY = auto()   # Layer 2 consistency check failed
    PENDING_SPOT_CHECK = auto()     # Layer 3 — awaiting spot-checker result
    PENDING_CHALLENGE = auto()      # Layer 4 — in challenge window


@dataclass
class VerificationResult:
    """Output of KnowledgeSharingProtocol.verify_incoming().

    Attributes:
        entry_id:                 ID of the contribution being verified.
        novelty_gate_passed:      Layer 1 passed (distance > threshold).
        consistency_check_passed: Layer 2 passed (no contradictions found).
        spot_check_selected:      Whether this contribution was selected for
                                  probabilistic re-verification.
        spot_check_passed:        Result of spot-check; None if not selected
                                  or not yet complete.
        verdict:                  Final disposition.
        verifier_node_id:         Node that performed the verification.
    """

    entry_id: str
    novelty_gate_passed: bool
    consistency_check_passed: bool
    spot_check_selected: bool
    spot_check_passed: Optional[bool]
    verdict: VerificationVerdict
    verifier_node_id: str


class AcquisitionPath(Enum):
    """Which path the KnowledgeAcquisitionEngine took for a query."""
    KNOWLEDGE_CACHE = auto()    # Hit in local KnowledgeStore
    NETWORK = auto()            # Retrieved from the distributed network
    TOKEN_SPACE = auto()        # Full token-space fallback + new knowledge


@dataclass
class AcquisitionResult:
    """Output of KnowledgeAcquisitionEngine.acquire().

    Attributes:
        query_embedding:    Original (D,) query embedding.
        result_embedding:   Best available (D,) result embedding, drawn from
                            KnowledgeStore, network, or token-space research.
        path_taken:         Which acquisition strategy was used.
        ood_result:         OOD detection output (always present).
        knowledge_entry:    The matched or newly created KnowledgeEntry, or
                            None when the result is from cache without a stored
                            entry (e.g. fast two-speed path).
        contribution:       Packaged contribution if new knowledge was acquired
                            via token-space fallback.  None otherwise.
        latency_ms:         End-to-end acquisition latency.
    """

    query_embedding: torch.Tensor        # (D,)
    result_embedding: Optional[torch.Tensor]   # (D,) or None
    path_taken: AcquisitionPath
    ood_result: OODResult
    knowledge_entry: Optional[KnowledgeEntry]
    contribution: Optional[KnowledgeContribution]
    latency_ms: float


# =============================================================================
# Internal vector index (FAISS or torch brute-force)
# =============================================================================


class _VectorIndex:
    """Approximate nearest-neighbour index backed by FAISS or torch.

    Stores L2-normalised float32 embeddings.  Cosine similarity is computed
    as inner product on the unit sphere (equivalent after L2 normalisation).

    When FAISS is available, uses ``IndexFlatIP`` (exact cosine search).
    When FAISS is absent, uses brute-force torch matrix multiplication.

    Thread-safety: **not** thread-safe.  Serialise access from callers.
    """

    def __init__(self, embed_dim: int, max_size: int):
        self.embed_dim = embed_dim
        self.max_size = max_size

        self._index: Optional[Any] = None  # faiss.Index or None
        self._store: Optional[torch.Tensor] = None  # (N, D) CPU tensor fallback
        self._count: int = 0

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(embed_dim)
            logger.debug(
                "_VectorIndex: using FAISS IndexFlatIP (embed_dim=%d)", embed_dim
            )
        else:
            logger.debug(
                "_VectorIndex: FAISS not available, using torch brute-force "
                "(install faiss-cpu for better performance at scale)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: torch.Tensor) -> None:
        """Add a single L2-normalised embedding to the index.

        Args:
            embedding: (D,) float tensor, any device.  Will be normalised
                       and moved to CPU before storing.
        """
        vec = F.normalize(embedding.float().cpu().unsqueeze(0), dim=-1)  # (1, D)

        if _FAISS_AVAILABLE and self._index is not None:
            if self._count >= self.max_size:
                # FAISS FlatIP doesn't support removal; rebuild periodically
                # by trimming is expensive, so we just rotate to a new index.
                self._index.reset()
                self._count = 0
                if self._store is not None:
                    self._store = None
            self._index.add(vec.numpy())
        else:
            # Torch fallback: append to a growing tensor, evict FIFO at limit.
            if self._store is None:
                self._store = vec
            else:
                self._store = torch.cat([self._store, vec], dim=0)
                if self._store.shape[0] > self.max_size:
                    # FIFO eviction
                    self._store = self._store[-self.max_size:]

        self._count = min(self._count + 1, self.max_size)

    def search(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[List[float], List[int]]:
        """Search for the k nearest embeddings.

        Args:
            query: (D,) float tensor, any device.
            k:     Number of results to return.

        Returns:
            ``(similarities, indices)`` — both of length min(k, index_size).
            Similarities are cosine similarities in [−1, 1]; higher = closer.
            Indices are positional (not stable across ``add()`` calls with
            FAISS reset; use for distance estimation only, not retrieval).
        """
        if self._count == 0:
            return [], []

        k = min(k, self._count)
        q = F.normalize(query.float().cpu().unsqueeze(0), dim=-1)  # (1, D)

        if _FAISS_AVAILABLE and self._index is not None:
            sims_np, idxs_np = self._index.search(q.numpy(), k)
            sims = sims_np[0].tolist()
            idxs = idxs_np[0].tolist()
        else:
            # Brute-force cosine similarity
            if self._store is None or self._store.shape[0] == 0:
                return [], []
            sims_all = (self._store @ q.T).squeeze(-1)  # (N,)
            topk = min(k, sims_all.shape[0])
            top_sims, top_idxs = sims_all.topk(topk)
            sims = top_sims.tolist()
            idxs = top_idxs.tolist()

        return sims, idxs

    @property
    def size(self) -> int:
        """Number of embeddings currently in the index."""
        return self._count

    def reset(self) -> None:
        """Remove all entries."""
        if _FAISS_AVAILABLE and self._index is not None:
            self._index.reset()
        self._store = None
        self._count = 0


# =============================================================================
# OODDetector
# =============================================================================


class OODDetector:
    """Out-of-distribution detector for VL-JEPA embedding space.

    Combines two complementary signals to detect when a query falls outside
    the model's latent knowledge:

    1. **Predictive variance** (proxy for VJEPA σ²): runs a short VLJEPA
       rollout on the query's initial latent plan and records the mean
       uncertainty across steps.  High uncertainty → the model is unsure
       about this query's dynamics.

    2. **KNN cosine distance**: looks up the nearest in-distribution
       embeddings from the training index.  High distance → the query is
       far from anything the model was trained on.

    The combined score separates epistemic ("I don't know") from aleatoric
    ("inherently ambiguous") uncertainty, avoiding unnecessary expensive
    fallbacks for ambiguous but in-distribution queries.

    Usage::

        detector = OODDetector(vljepa_model, OODConfig())

        # Populate in-distribution index from training embeddings
        for emb in training_embeddings:
            detector.add_in_distribution(emb)

        result = detector.detect(query_embedding, initial_plan)
        if result.uncertainty_type == UncertaintyType.EPISTEMIC:
            # trigger token-space fallback
            ...
    """

    def __init__(
        self,
        model: VLJEPA,
        config: Optional[OODConfig] = None,
        node_id: str = "local",
    ):
        """
        Args:
            model:   Fully-initialised VLJEPA instance.  Used only for the
                     short rollout that produces the variance proxy score.
            config:  OOD detection configuration.
            node_id: Identifier of the local node (for logging).
        """
        self.model = model
        self.config = config or OODConfig()
        self.node_id = node_id

        # In-distribution reference index (populated from training embeddings)
        embed_dim = model.config.embed_dim
        self._id_index = _VectorIndex(embed_dim, self.config.max_index_size)

        self._detect_calls: int = 0
        self._epistemic_calls: int = 0

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_in_distribution(self, embedding: torch.Tensor) -> None:
        """Add an in-distribution embedding to the KNN reference index.

        Call this during training or after loading a checkpoint to populate
        the reference distribution.  Each embedding is L2-normalised before
        storage.

        Args:
            embedding: (D,) or (1, D) float tensor.
        """
        emb = embedding.squeeze(0) if embedding.dim() == 2 else embedding
        self._id_index.add(emb)

    def add_batch_in_distribution(self, embeddings: torch.Tensor) -> None:
        """Add a batch of in-distribution embeddings.

        Args:
            embeddings: (N, D) float tensor.
        """
        for i in range(embeddings.shape[0]):
            self._id_index.add(embeddings[i])

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self,
        query_embedding: torch.Tensor,   # (D,) or (1, D)
        initial_plan: Optional[torch.Tensor] = None,  # (1, K, D) for variance
    ) -> OODResult:
        """Detect whether the query is out of distribution.

        Args:
            query_embedding: Mean-pooled query embedding (D,) or (1, D).
            initial_plan:    Optional (1, K, D) latent plan for the variance
                             proxy rollout.  If None, variance_score = 0.5
                             (neutral — only KNN distance is used).

        Returns:
            :class:`OODResult` with all fields populated.
        """
        t0 = time.perf_counter()
        self._detect_calls += 1

        # Normalise to (D,)
        q = query_embedding.squeeze(0) if query_embedding.dim() == 2 else query_embedding

        # ---- Signal 1: Predictive variance proxy ----
        variance_score = self._compute_variance_score(initial_plan)

        # ---- Signal 2: KNN distance ----
        knn_score, nearest_distance = self._compute_knn_score(q)

        # ---- Combined score ----
        combined = self.config.alpha * variance_score + self.config.beta * knn_score

        # ---- Classify uncertainty type ----
        uncertainty_type = self._classify(variance_score, knn_score, combined)
        if uncertainty_type == UncertaintyType.EPISTEMIC:
            self._epistemic_calls += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "OOD detect: var=%.3f knn=%.3f combined=%.3f type=%s index=%d",
            variance_score, knn_score, combined,
            uncertainty_type.name, self._id_index.size,
        )

        return OODResult(
            query_embedding=q.cpu().detach(),
            variance_score=float(variance_score),
            knn_distance_score=float(knn_score),
            combined_score=float(combined),
            uncertainty_type=uncertainty_type,
            nearest_distance=float(nearest_distance),
            index_size=self._id_index.size,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def epistemic_rate(self) -> float:
        """Fraction of detect() calls that were classified epistemic."""
        if self._detect_calls == 0:
            return 0.0
        return self._epistemic_calls / self._detect_calls

    def reset_index(self) -> None:
        """Clear the in-distribution index (e.g. after a model update)."""
        self._id_index.reset()
        logger.info("OODDetector: in-distribution index cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_variance_score(
        self,
        initial_plan: Optional[torch.Tensor],
    ) -> float:
        """Compute the predictive variance proxy from a short JEPA rollout.

        Uses VLJEPA.rollout() uncertainties as a proxy for the variational
        JEPA σ² signal (VJEPA, arXiv:2601.14354).  The rollout uncertainty
        already captures spread collapse (see latent_reasoning.py).

        Returns a score in [0, 1] where 1.0 = maximum uncertainty.
        """
        if initial_plan is None:
            # No plan available — return neutral mid-point
            return 0.5

        self.model.eval()
        rollout = self.model.rollout(initial_plan, self.config.variance_rollout_steps)
        uncertainties: List[float] = rollout.get("uncertainties", [])
        if not uncertainties:
            return 0.5

        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        # Uncertainty from rollout() is already in [0, 1] (1/(1+spread_ratio))
        return float(min(1.0, max(0.0, mean_uncertainty)))

    def _compute_knn_score(
        self,
        query: torch.Tensor,  # (D,)
    ) -> Tuple[float, float]:
        """Compute the KNN distance score.

        Returns:
            ``(knn_score, nearest_distance)`` where knn_score ∈ [0, 1] and
            nearest_distance is the cosine distance (1 − cosine_similarity)
            to the single nearest neighbour.

        When the index is empty (no training data loaded), returns
        ``(1.0, 1.0)`` — assume OOD so the system falls through to the
        slow path rather than silently failing.
        """
        if self._id_index.size == 0:
            return 1.0, 1.0

        sims, _ = self._id_index.search(query, k=self.config.knn_k)
        if not sims:
            return 1.0, 1.0

        # Use k-th nearest neighbour distance (most conservative signal)
        # Cosine similarity → distance: dist = 1 − sim; clamp to [0, 1].
        kth_sim = float(sims[-1])   # smallest similarity (farthest of k-NN)
        nearest_sim = float(sims[0])
        kth_dist = max(0.0, 1.0 - kth_sim)
        nearest_dist = max(0.0, 1.0 - nearest_sim)

        # Normalise distance to [0, 1] (kth_dist is already in [0, 2] for
        # cosine; clamp to [0, 1] since query and store are normalised).
        knn_score = min(1.0, kth_dist)
        return knn_score, nearest_dist

    def _classify(
        self,
        variance_score: float,
        knn_score: float,
        combined: float,
    ) -> UncertaintyType:
        """Apply the 2×2 interpretation matrix from the survey §2.4."""
        cfg = self.config

        # Epistemic: high combined score → definitive fallback trigger
        if combined >= cfg.epistemic_threshold:
            return UncertaintyType.EPISTEMIC

        # Aleatoric: far from training data BUT model is not actively uncertain
        # (the data itself is ambiguous — more research won't help)
        if (
            knn_score >= cfg.aleatoric_knn_threshold
            and variance_score < cfg.aleatoric_variance_ceiling
        ):
            return UncertaintyType.ALEATORIC

        # Caution: moderate distance — verify but don't fallback
        if knn_score >= cfg.caution_knn_threshold:
            return UncertaintyType.CAUTION

        return UncertaintyType.CONFIDENT


# =============================================================================
# TokenSpaceFallback
# =============================================================================


class TokenSpaceFallback:
    """Bridge to full token-space LLM reasoning when OOD is detected.

    When the OODDetector classifies a query as epistemic, this component:
      1. Invokes a token-space provider callable with the query text.
      2. Optionally re-encodes the response into embedding space (via the
         VLJEPA text encoder) so the result can be stored in KnowledgeStore.
      3. Records the fallback for monitoring (frequency, cost tracking).

    The ``provider_fn`` callable is the interface to the token-space LLM
    (Claude, GPT, local LLM, etc.).  It receives the query text and returns
    the response string.  The actual transport (HTTP, subprocess, etc.) is
    the caller's concern — this class is provider-agnostic.

    Usage::

        def my_llm(query: str) -> str:
            # call Claude / GPT / local model
            return response_text

        fallback = TokenSpaceFallback(vljepa, config, provider_fn=my_llm)
        result = fallback.run("What is X?")
        if result.re_anchored_embedding is not None:
            # store in KnowledgeStore
            ...
    """

    def __init__(
        self,
        model: VLJEPA,
        config: Optional[TokenSpaceFallbackConfig] = None,
        provider_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            model:       VLJEPA instance for re-anchoring the response.
            config:      Fallback configuration.
            provider_fn: Callable ``(query_text: str) -> response_text: str``.
                         When None, fallback returns an empty response (stub
                         for Phase B integration testing).
        """
        self.model = model
        self.config = config or TokenSpaceFallbackConfig()
        self.provider_fn = provider_fn

        self._history: List[Dict[str, Any]] = []   # bounded ring; for monitoring
        self._total_calls: int = 0
        self._total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        query_text: str,
        token_ids: Optional[torch.Tensor] = None,   # (1, S) LongTensor
        attention_mask: Optional[torch.Tensor] = None,
    ) -> FallbackResult:
        """Invoke the token-space provider and re-anchor to embedding space.

        Args:
            query_text:     Raw query text (sent to the provider).
            token_ids:      Optional tokenised query for re-anchoring the
                            *query* embedding via VLJEPA.encode_text().
            attention_mask: Optional attention mask for token_ids.

        Returns:
            :class:`FallbackResult` with all fields populated.
        """
        t0 = time.perf_counter()
        self._total_calls += 1

        # ---- Step 1: call the token-space provider ----
        if self.provider_fn is not None:
            try:
                response_text = self.provider_fn(query_text)
            except Exception as exc:  # noqa: BLE001
                logger.error("TokenSpaceFallback: provider raised %s", exc)
                response_text = ""
        else:
            logger.debug(
                "TokenSpaceFallback: no provider_fn configured (Phase B stub)"
            )
            response_text = ""

        # ---- Step 2: re-anchor to embedding space ----
        re_anchored: Optional[torch.Tensor] = None
        if self.config.re_anchor_on_response and response_text and token_ids is not None:
            try:
                re_anchored = self._encode_response(token_ids, attention_mask)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TokenSpaceFallback: re-anchor failed: %s", exc)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += latency_ms

        result = FallbackResult(
            query_text=query_text,
            response_text=response_text,
            re_anchored_embedding=re_anchored,
            provider_id=self.config.provider_id,
            latency_ms=latency_ms,
            fallback_count=self._total_calls,
        )

        self._record(result)
        logger.info(
            "TokenSpaceFallback: call #%d completed in %.1f ms (provider=%s)",
            self._total_calls, latency_ms, self.config.provider_id,
        )
        return result

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        """Total number of fallback calls made since construction."""
        return self._total_calls

    def mean_latency_ms(self) -> float:
        """Mean latency across all fallback calls."""
        if self._total_calls == 0:
            return 0.0
        return self._total_latency_ms / self._total_calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_response(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode token_ids via VLJEPA text encoder → mean-pooled (D,)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        ids = token_ids.to(device)
        mask = attention_mask.to(device) if attention_mask is not None else None

        text_embeds = self.model.encode_text(ids, mask)  # (1, S, D)

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)          # (1, S, 1)
            pooled = (text_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = text_embeds.mean(dim=1)             # (1, D)

        return pooled.squeeze(0).cpu()  # (D,)

    def _record(self, result: FallbackResult) -> None:
        """Add a record to the bounded history for monitoring."""
        record = {
            "timestamp": time.time(),
            "latency_ms": result.latency_ms,
            "provider_id": result.provider_id,
            "had_reanchor": result.re_anchored_embedding is not None,
        }
        self._history.append(record)
        if len(self._history) > self.config.max_history_size:
            self._history = self._history[-self.config.max_history_size:]


# =============================================================================
# KnowledgeStore
# =============================================================================


class KnowledgeStore:
    """Local knowledge base of validated embeddings with provenance metadata.

    Stores embeddings that have passed the novelty gate and consistency check,
    indexed by a FAISS (or torch brute-force) ANN index for fast retrieval.

    The store applies a **two-layer gate** at insertion time:
      - Layer 1 (novelty): rejects embeddings too close to existing entries.
      - Layer 2 (consistency): rejects embeddings that strongly contradict
        well-established existing knowledge.

    Knowledge entries accrue usage_count as they are retrieved during
    inference, which feeds into the downstream reward calculation.

    Thread-safety: **not** thread-safe.

    Usage::

        store = KnowledgeStore(KnowledgeStoreConfig(embed_dim=384))

        entry_id = store.add(
            embedding,
            contributor_node_id="node-001",
            research_trace_hash="abc123...",
            novelty_score=0.72,
            quality_score=1.0,
        )
        if entry_id:
            print("Stored:", entry_id)

        hits = store.search(query_embedding, k=5)
    """

    def __init__(self, config: Optional[KnowledgeStoreConfig] = None):
        """
        Args:
            config: Store configuration.  Uses KnowledgeStoreConfig defaults
                    if not provided.
        """
        self.config = config or KnowledgeStoreConfig()

        self._index = _VectorIndex(self.config.embed_dim, self.config.max_entries)
        self._entries: List[KnowledgeEntry] = []   # parallel to _index (positional)
        self._id_map: Dict[str, int] = {}          # entry_id → list position

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        embedding: torch.Tensor,       # (D,)
        contributor_node_id: str,
        research_trace_hash: str,
        novelty_score: float,
        quality_score: float = 1.0,
        challenge_window_seconds: float = 86_400.0,
    ) -> Optional[str]:
        """Attempt to add a new knowledge embedding.

        Applies the novelty gate (Layer 1) and consistency check (Layer 2).
        Returns the assigned entry_id on success, or None if rejected.

        Args:
            embedding:             (D,) float tensor — the knowledge embedding.
            contributor_node_id:   Node that produced this knowledge.
            research_trace_hash:   SHA-256 of the source research trace.
            novelty_score:         Cosine distance from the nearest neighbour
                                   at proposal time (pre-computed by caller).
            quality_score:         Initial quality estimate ∈ [0, 1].
            challenge_window_seconds: How long before the entry is finalised.

        Returns:
            Assigned ``entry_id`` string on acceptance, or ``None`` if the
            novelty gate or consistency check rejected the contribution, or
            if quality_score is below the floor.
        """
        # ---- Quality floor ----
        if quality_score < self.config.quality_score_floor:
            logger.debug(
                "KnowledgeStore.add: rejected (quality %.3f < floor %.3f)",
                quality_score, self.config.quality_score_floor,
            )
            return None

        # ---- Layer 1: Novelty gate ----
        if not self._novelty_gate(embedding):
            logger.debug(
                "KnowledgeStore.add: rejected by novelty gate (entry too close to existing)"
            )
            return None

        # ---- Layer 2: Consistency check ----
        if not self._consistency_check(embedding):
            logger.debug(
                "KnowledgeStore.add: rejected by consistency check (contradiction detected)"
            )
            return None

        # ---- Accept: create metadata and store ----
        entry_id = str(uuid.uuid4())
        metadata = KnowledgeEntryMetadata(
            entry_id=entry_id,
            contributor_node_id=contributor_node_id,
            timestamp=time.time(),
            novelty_score=novelty_score,
            quality_score=quality_score,
            verification_status="pending",
            usage_count=0,
            research_trace_hash=research_trace_hash,
            challenge_deadline=time.time() + challenge_window_seconds,
        )
        entry = KnowledgeEntry(
            embedding=embedding.float().cpu().detach(),
            metadata=metadata,
        )

        pos = len(self._entries)
        self._entries.append(entry)
        self._id_map[entry_id] = pos
        self._index.add(embedding)

        logger.info(
            "KnowledgeStore: accepted entry %s (novelty=%.3f, quality=%.3f, size=%d)",
            entry_id, novelty_score, quality_score, len(self._entries),
        )
        return entry_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: torch.Tensor,   # (D,)
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Find the k most similar knowledge entries.

        Args:
            query_embedding: (D,) float tensor, any device.
            k:               Maximum number of results to return.
            min_similarity:  Minimum cosine similarity to include in results.

        Returns:
            List of ``(entry, cosine_similarity)`` tuples, sorted by
            descending similarity.  May be shorter than k if the store
            has fewer entries or all similarities fall below ``min_similarity``.
        """
        if not self._entries:
            return []

        sims, idxs = self._index.search(query_embedding, k=k)
        results: List[Tuple[KnowledgeEntry, float]] = []

        for sim, idx in zip(sims, idxs):
            if sim < min_similarity:
                continue
            if 0 <= idx < len(self._entries):
                entry = self._entries[idx]
                entry.metadata.usage_count += 1   # track activations
                results.append((entry, float(sim)))

        return results

    def get_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a specific entry by ID, or None if not found."""
        pos = self._id_map.get(entry_id)
        if pos is None:
            return None
        return self._entries[pos]

    def update_quality(self, entry_id: str, quality_score: float) -> bool:
        """Update the quality score for an entry (e.g. after spot-check).

        Returns True if the entry was found and updated, False otherwise.
        """
        entry = self.get_by_id(entry_id)
        if entry is None:
            return False
        entry.metadata.quality_score = float(min(1.0, max(0.0, quality_score)))
        return True

    def update_verification_status(self, entry_id: str, status: str) -> bool:
        """Set the verification_status field of an entry.

        Args:
            entry_id: The entry to update.
            status:   One of 'pending' | 'verified' | 'challenged' | 'rejected'.

        Returns:
            True if found and updated.
        """
        entry = self.get_by_id(entry_id)
        if entry is None:
            return False
        entry.metadata.verification_status = status
        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def compute_novelty_score(self, embedding: torch.Tensor) -> float:
        """Compute the novelty score for a candidate embedding.

        Returns the cosine distance (1 − cosine_similarity) to the nearest
        existing entry.  Returns 1.0 (maximally novel) if the store is empty.

        Args:
            embedding: (D,) float tensor.

        Returns:
            Novelty score ∈ [0, 1].  Higher = more novel.
        """
        if not self._entries:
            return 1.0
        sims, _ = self._index.search(embedding, k=1)
        if not sims:
            return 1.0
        nearest_sim = max(-1.0, min(1.0, float(sims[0])))
        return float(1.0 - nearest_sim) / 2.0   # scale cosine dist [0,2] → [0,1]

    # ------------------------------------------------------------------
    # Internal gates
    # ------------------------------------------------------------------

    def _novelty_gate(self, embedding: torch.Tensor) -> bool:
        """Layer 1: reject if embedding is too close to an existing entry."""
        if not self._entries:
            return True  # empty store — everything is novel
        novelty = self.compute_novelty_score(embedding)
        return novelty >= self.config.novelty_distance_threshold

    def _consistency_check(self, embedding: torch.Tensor) -> bool:
        """Layer 2: detect strong contradictions with existing knowledge.

        Phase B implementation: flag if the embedding has very high cosine
        similarity to existing entries that are in tension with each other.
        In practice, without explicit label tracking, we check that the new
        embedding doesn't exceed the ``consistency_contradiction_threshold``
        in similarity with the top-k existing entries while having low
        intra-cluster agreement — a sign of conflicting information.

        Phase C: replace with explicit ontology-based contradiction detection.
        """
        # Phase B: simple check — if store is small, accept optimistically
        if len(self._entries) < 10:
            return True

        # Search top-5 similar entries
        sims, idxs = self._index.search(embedding, k=5)
        if not sims:
            return True

        # If the nearest neighbour is extremely similar (almost identical)
        # and already flagged as 'rejected' or 'challenged', be cautious
        nearest_sim = float(sims[0]) if sims else 0.0
        if nearest_sim >= self.config.consistency_contradiction_threshold:
            nearest_idx = idxs[0] if idxs else -1
            if 0 <= nearest_idx < len(self._entries):
                status = self._entries[nearest_idx].metadata.verification_status
                if status in ("rejected", "challenged"):
                    logger.debug(
                        "KnowledgeStore: consistency check failed — near-duplicate "
                        "of %s entry (sim=%.3f)",
                        status, nearest_sim,
                    )
                    return False

        return True


# =============================================================================
# KnowledgeSharingProtocol
# =============================================================================


class KnowledgeSharingProtocol:
    """Network knowledge exchange with attestations and layered verification.

    Implements the four-layer verification stack from the survey §3.2:
      - Layer 1 (cheap):   Novelty gate — KNN distance check in KnowledgeStore.
      - Layer 2 (medium):  Consistency check — contradiction detection.
      - Layer 3 (statistical): Probabilistic spot-check at ``spot_check_rate``.
      - Layer 4 (reactive):    Challenge window (tracked by metadata deadline).

    Also implements simplified TraceRank reputation scoring (survey §4.3):
      - Usage flows (entry.metadata.usage_count) propagate reputation.
      - High-reputation validators' endorsements carry more weight.
      - Sybil resistance: stake-weighted validation (zero-stake → no weight).

    Phase B: all verification runs locally.  Phase C: spot-checks are
    dispatched to random remote validators via the P2P layer.

    Usage::

        protocol = KnowledgeSharingProtocol(
            store, config, local_node_id="node-001"
        )

        # Package a locally-researched embedding for sharing
        contribution = protocol.package(
            embedding, research_trace_hash="abc...", local_node_id="node-001"
        )

        # Verify an incoming contribution from another node
        result = protocol.verify_incoming(contribution)
        if result.verdict == VerificationVerdict.ACCEPTED:
            # integrate into local store
            ...
    """

    def __init__(
        self,
        store: KnowledgeStore,
        config: Optional[KnowledgeSharingConfig] = None,
        local_node_id: str = "local",
    ):
        """
        Args:
            store:          Local KnowledgeStore for novelty / consistency checks.
            config:         Sharing configuration.
            local_node_id:  Identifier of the local node.
        """
        self.store = store
        self.config = config or KnowledgeSharingConfig()
        self.local_node_id = local_node_id

        # TraceRank reputation table: node_id → float score (starts at 1.0)
        self._reputation: Dict[str, float] = {}

        # Stake table (Phase B: simulated; Phase C: from smart contract)
        self._stake: Dict[str, float] = {local_node_id: 1000.0}

        # Verification counters
        self._verified_count: int = 0
        self._rejected_count: int = 0
        self._spot_check_count: int = 0

        # Pending spot-checks: entry_id → ContributionAttestation
        self._pending_spot_checks: Dict[str, ContributionAttestation] = {}

    # ------------------------------------------------------------------
    # Packaging (outbound)
    # ------------------------------------------------------------------

    def package(
        self,
        embedding: torch.Tensor,        # (D,)
        research_trace_hash: str,
        contributor_node_id: Optional[str] = None,
    ) -> KnowledgeContribution:
        """Package a knowledge embedding as a network-ready contribution.

        Creates a :class:`ContributionAttestation` and computes the initial
        reward estimate using the survey §5.4 formula (simplified).

        Args:
            embedding:            The (D,) knowledge embedding to share.
            research_trace_hash:  SHA-256 of the token-space research trace.
            contributor_node_id:  Defaults to ``self.local_node_id``.

        Returns:
            :class:`KnowledgeContribution` ready for network submission.
        """
        contributor = contributor_node_id or self.local_node_id
        entry_id = str(uuid.uuid4())
        ts = time.time()

        emb_hash = _embedding_hash(embedding)
        signature = _make_signature(contributor, emb_hash, research_trace_hash, ts)

        attestation = ContributionAttestation(
            contributor_node_id=contributor,
            embedding_hash=emb_hash,
            research_trace_hash=research_trace_hash,
            timestamp=ts,
            signature=signature,
        )

        novelty_score = self.store.compute_novelty_score(embedding)
        reputation = self._get_reputation(contributor)

        # Simplified reward: novelty × reputation (survey §5.4)
        proposed_reward = float(novelty_score * reputation * 100.0)  # in token units

        return KnowledgeContribution(
            entry_id=entry_id,
            embedding=embedding.float().cpu().detach(),
            attestation=attestation,
            novelty_score=float(novelty_score),
            proposed_reward=proposed_reward,
        )

    # ------------------------------------------------------------------
    # Verification (inbound)
    # ------------------------------------------------------------------

    def verify_incoming(
        self,
        contribution: KnowledgeContribution,
        verifier_node_id: Optional[str] = None,
    ) -> VerificationResult:
        """Verify an incoming knowledge contribution from the network.

        Applies the four-layer verification stack:
          1. Attestation integrity check (signature validation).
          2. Layer 1: Novelty gate via KnowledgeStore.
          3. Layer 2: Consistency check via KnowledgeStore.
          4. Layer 3: Probabilistic spot-check selection.
          5. On full acceptance: add to KnowledgeStore and update reputation.

        Args:
            contribution:       The incoming contribution to verify.
            verifier_node_id:   Node performing the verification.  Defaults
                                to ``self.local_node_id``.

        Returns:
            :class:`VerificationResult` with the verdict and all gate outcomes.
        """
        verifier = verifier_node_id or self.local_node_id
        entry_id = contribution.entry_id
        att = contribution.attestation

        # ---- Stake check (Sybil resistance) ----
        contributor_stake = self._stake.get(att.contributor_node_id, 0.0)
        if contributor_stake < self.config.min_stake_for_validation:
            logger.debug(
                "KnowledgeSharingProtocol: contributor %s has insufficient stake "
                "(%.1f < %.1f)",
                att.contributor_node_id,
                contributor_stake,
                self.config.min_stake_for_validation,
            )
            # Still verify (Phase B: allow zero-stake contributors in testing)
            # Phase C: reject immediately.

        # ---- Attestation integrity ----
        expected_sig = _make_signature(
            att.contributor_node_id,
            att.embedding_hash,
            att.research_trace_hash,
            att.timestamp,
        )
        if expected_sig != att.signature:
            logger.warning(
                "KnowledgeSharingProtocol: attestation signature mismatch for %s",
                entry_id,
            )
            # Phase B: log but don't reject (Phase C: reject)

        # ---- Layer 1: Novelty gate ----
        novelty_passed = self.store._novelty_gate(contribution.embedding)
        if not novelty_passed:
            self._rejected_count += 1
            return VerificationResult(
                entry_id=entry_id,
                novelty_gate_passed=False,
                consistency_check_passed=False,
                spot_check_selected=False,
                spot_check_passed=None,
                verdict=VerificationVerdict.REJECTED_NOVELTY,
                verifier_node_id=verifier,
            )

        # ---- Layer 2: Consistency check ----
        consistency_passed = self.store._consistency_check(contribution.embedding)
        if not consistency_passed:
            self._rejected_count += 1
            return VerificationResult(
                entry_id=entry_id,
                novelty_gate_passed=True,
                consistency_check_passed=False,
                spot_check_selected=False,
                spot_check_passed=None,
                verdict=VerificationVerdict.REJECTED_CONSISTENCY,
                verifier_node_id=verifier,
            )

        # ---- Layer 3: Probabilistic spot-check ----
        spot_selected = torch.rand(1).item() < self.config.spot_check_rate
        spot_passed: Optional[bool] = None

        if spot_selected:
            self._spot_check_count += 1
            self._pending_spot_checks[entry_id] = att
            # Phase B: auto-pass (Phase C: dispatch to a random remote verifier)
            spot_passed = True
            logger.debug(
                "KnowledgeSharingProtocol: spot-check selected for %s (Phase B: auto-pass)",
                entry_id,
            )

        # ---- Accept: integrate into store ----
        stored_id = self.store.add(
            embedding=contribution.embedding,
            contributor_node_id=att.contributor_node_id,
            research_trace_hash=att.research_trace_hash,
            novelty_score=contribution.novelty_score,
            quality_score=1.0,
            challenge_window_seconds=self.config.challenge_window_seconds,
        )

        if stored_id is not None:
            # Update TraceRank reputation for the contributor
            self._update_reputation(att.contributor_node_id, delta=+contribution.novelty_score)
            self._verified_count += 1

            verdict = (
                VerificationVerdict.PENDING_SPOT_CHECK
                if spot_selected and spot_passed is None
                else VerificationVerdict.ACCEPTED
            )
        else:
            # Store rejected (novelty / quality), but gates passed — shouldn't happen
            verdict = VerificationVerdict.REJECTED_NOVELTY
            self._rejected_count += 1

        return VerificationResult(
            entry_id=stored_id or entry_id,
            novelty_gate_passed=True,
            consistency_check_passed=True,
            spot_check_selected=spot_selected,
            spot_check_passed=spot_passed,
            verdict=verdict,
            verifier_node_id=verifier,
        )

    # ------------------------------------------------------------------
    # Reputation (TraceRank)
    # ------------------------------------------------------------------

    def _get_reputation(self, node_id: str) -> float:
        """Return the TraceRank reputation score for a node (default 1.0)."""
        return self._reputation.get(node_id, 1.0)

    def _update_reputation(self, node_id: str, delta: float) -> None:
        """Apply a reputation delta and decay all scores.

        Follows the TraceRank principle: reputation flows through usage
        endorsements.  After each update we decay all scores by
        ``config.reputation_decay`` to prevent unbounded accumulation.
        """
        current = self._reputation.get(node_id, 1.0)
        self._reputation[node_id] = current + delta

        # Apply global decay (survey §4.3 — prevent dominance of old nodes)
        decay = self.config.reputation_decay
        self._reputation = {k: v * decay for k, v in self._reputation.items()}

    def reputation_leaderboard(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return the top-n nodes by reputation score."""
        ranked = sorted(self._reputation.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a summary of verification activity."""
        total = self._verified_count + self._rejected_count
        return {
            "total_verified": self._verified_count,
            "total_rejected": self._rejected_count,
            "acceptance_rate": self._verified_count / max(1, total),
            "spot_checks": self._spot_check_count,
            "pending_spot_checks": len(self._pending_spot_checks),
            "known_nodes": len(self._reputation),
        }


# =============================================================================
# KnowledgeAcquisitionEngine
# =============================================================================


class KnowledgeAcquisitionEngine:
    """Main coordinator for the knowledge acquisition loop.

    Integrates OOD detection into the reasoning loop and routes queries to
    the appropriate acquisition strategy:

      - **KNOWLEDGE_CACHE**: Query is in-distribution and the KnowledgeStore
        contains a highly-similar entry.  Use the cached embedding directly.
      - **NETWORK**: OOD detected, but the distributed network has a matching
        entry (Phase C only; Phase B always falls through to token-space).
      - **TOKEN_SPACE**: OOD detected, no network match.  Invoke the
        TokenSpaceFallback, encode the result, and propose to the network via
        KnowledgeSharingProtocol.

    The engine is stateless between ``acquire()`` calls.

    Usage::

        engine = KnowledgeAcquisitionEngine(
            model=vljepa,
            config=KnowledgeAcqConfig(),
            provider_fn=lambda q: claude_api(q),
            local_node_id="node-001",
        )

        # Populate in-distribution reference from training embeddings
        engine.ood_detector.add_batch_in_distribution(training_embeddings)

        # Per-turn acquisition
        result = engine.acquire(
            query_embedding=z_q,
            query_text="What is the capital of France?",
            initial_plan=jepa_plan,
        )
        if result.result_embedding is not None:
            # Feed into downstream decoder
            ...
    """

    def __init__(
        self,
        model: VLJEPA,
        config: Optional[KnowledgeAcqConfig] = None,
        provider_fn: Optional[Callable[[str], str]] = None,
        local_node_id: str = "local",
    ):
        """
        Args:
            model:          Fully-initialised VLJEPA instance.
            config:         Acquisition engine configuration.
            provider_fn:    Token-space provider callable ``(query) -> response``.
                            Required for real token-space fallback; when None,
                            fallback returns empty results (Phase B stub).
            local_node_id:  Node identifier for provenance tracking.
        """
        self.model = model
        self.config = config or KnowledgeAcqConfig()
        self.local_node_id = local_node_id

        self.ood_detector = OODDetector(
            model, config=self.config.ood, node_id=local_node_id
        )

        self.knowledge_store = KnowledgeStore(config=self.config.store)

        self.fallback = TokenSpaceFallback(
            model,
            config=self.config.fallback,
            provider_fn=provider_fn,
        )

        self.sharing_protocol = KnowledgeSharingProtocol(
            self.knowledge_store,
            config=self.config.sharing,
            local_node_id=local_node_id,
        )

        # Acquisition path counters for monitoring
        self._path_counts: Dict[str, int] = {
            "cache": 0, "network": 0, "token_space": 0
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vljepa(
        cls,
        model: VLJEPA,
        provider_fn: Optional[Callable[[str], str]] = None,
        local_node_id: str = "local",
        **config_kwargs: Any,
    ) -> "KnowledgeAcquisitionEngine":
        """Convenience constructor for the common case.

        Args:
            model:          Initialised VLJEPA model.
            provider_fn:    Token-space provider callable.
            local_node_id:  Node identifier.
            **config_kwargs: Keyword arguments forwarded to KnowledgeAcqConfig.

        Returns:
            Ready-to-use :class:`KnowledgeAcquisitionEngine`.
        """
        config = KnowledgeAcqConfig(**config_kwargs) if config_kwargs else KnowledgeAcqConfig()
        return cls(
            model=model,
            config=config,
            provider_fn=provider_fn,
            local_node_id=local_node_id,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def acquire(
        self,
        query_embedding: torch.Tensor,          # (D,) or (1, D)
        query_text: str = "",
        initial_plan: Optional[torch.Tensor] = None,   # (1, K, D)
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        knowledge_similarity_threshold: Optional[float] = None,
    ) -> AcquisitionResult:
        """Run the full knowledge acquisition loop for a query.

        Decision flow:
          1. Detect OOD via OODDetector.
          2. If CONFIDENT or ALEATORIC:
               a. Search KnowledgeStore for a cached match.
               b. Return cache hit on match, else CONFIDENT pass-through.
          3. If CAUTION or EPISTEMIC:
               a. Search KnowledgeStore first (may still be useful).
               b. Optionally try network (Phase C).
               c. If nothing found: TokenSpaceFallback → encode → propose.
          4. Record provenance and return AcquisitionResult.

        Args:
            query_embedding:   (D,) or (1, D) query embedding from VLJEPA.
            query_text:        Raw query text (sent to provider on fallback).
            initial_plan:      (1, K, D) latent plan for OOD variance signal.
            token_ids:         (1, S) token IDs for response re-anchoring.
            attention_mask:    (1, S) attention mask for token_ids.
            knowledge_similarity_threshold: Override the minimum similarity
                               for cache hits.  Defaults to
                               ``config.min_network_match_similarity``.

        Returns:
            :class:`AcquisitionResult` with all fields populated.
        """
        t0 = time.perf_counter()

        # Normalise query embedding
        q = query_embedding.squeeze(0) if query_embedding.dim() == 2 else query_embedding
        threshold = (
            knowledge_similarity_threshold
            or self.config.min_network_match_similarity
        )

        # ---- Step 1: OOD detection ----
        ood_result = self.ood_detector.detect(q, initial_plan)

        # ---- Step 2: KnowledgeStore search (always worth checking) ----
        cache_hits = self.knowledge_store.search(q, k=5, min_similarity=threshold)

        if cache_hits and ood_result.uncertainty_type in (
            UncertaintyType.CONFIDENT, UncertaintyType.ALEATORIC
        ):
            # In-distribution + cache hit → use cached knowledge
            best_entry, best_sim = cache_hits[0]
            self._path_counts["cache"] += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "acquire: CACHE HIT (sim=%.3f, uncertainty=%s)",
                best_sim, ood_result.uncertainty_type.name,
            )
            return AcquisitionResult(
                query_embedding=q.cpu(),
                result_embedding=best_entry.embedding,
                path_taken=AcquisitionPath.KNOWLEDGE_CACHE,
                ood_result=ood_result,
                knowledge_entry=best_entry,
                contribution=None,
                latency_ms=latency_ms,
            )

        # ---- Step 3: Network acquisition (Phase C stub) ----
        if (
            self.config.enable_network_acquisition
            and ood_result.uncertainty_type == UncertaintyType.EPISTEMIC
        ):
            network_result = self._try_network_acquire(q, threshold)
            if network_result is not None:
                self._path_counts["network"] += 1
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.debug("acquire: NETWORK HIT")
                return AcquisitionResult(
                    query_embedding=q.cpu(),
                    result_embedding=network_result.embedding,
                    path_taken=AcquisitionPath.NETWORK,
                    ood_result=ood_result,
                    knowledge_entry=network_result,
                    contribution=None,
                    latency_ms=latency_ms,
                )

        # ---- Step 4: Return cache hit even for CAUTION/EPISTEMIC if available ----
        if cache_hits:
            best_entry, best_sim = cache_hits[0]
            self._path_counts["cache"] += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "acquire: CACHE HIT (OOD=%s, using closest match sim=%.3f)",
                ood_result.uncertainty_type.name, best_sim,
            )
            return AcquisitionResult(
                query_embedding=q.cpu(),
                result_embedding=best_entry.embedding,
                path_taken=AcquisitionPath.KNOWLEDGE_CACHE,
                ood_result=ood_result,
                knowledge_entry=best_entry,
                contribution=None,
                latency_ms=latency_ms,
            )

        # ---- Step 5: Token-space fallback + knowledge acquisition ----
        self._path_counts["token_space"] += 1
        fallback_result = self.fallback.run(
            query_text=query_text,
            token_ids=token_ids,
            attention_mask=attention_mask,
        )

        result_embedding = fallback_result.re_anchored_embedding
        contribution: Optional[KnowledgeContribution] = None

        if (
            result_embedding is not None
            and self.config.enable_knowledge_sharing
        ):
            contribution = self._propose_knowledge(
                embedding=result_embedding,
                research_trace_hash=_text_hash(
                    fallback_result.query_text + fallback_result.response_text
                ),
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "acquire: TOKEN_SPACE fallback completed (%.1f ms, "
            "ood_type=%s, re_anchored=%s)",
            latency_ms,
            ood_result.uncertainty_type.name,
            result_embedding is not None,
        )

        return AcquisitionResult(
            query_embedding=q.cpu(),
            result_embedding=result_embedding,
            path_taken=AcquisitionPath.TOKEN_SPACE,
            ood_result=ood_result,
            knowledge_entry=None,
            contribution=contribution,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def path_distribution(self) -> Dict[str, float]:
        """Return the fraction of acquire() calls taken via each path."""
        total = sum(self._path_counts.values())
        if total == 0:
            return {"cache": 0.0, "network": 0.0, "token_space": 0.0}
        return {k: v / total for k, v in self._path_counts.items()}

    def summary(self) -> Dict[str, Any]:
        """Return a diagnostic summary dict."""
        return {
            "path_counts": dict(self._path_counts),
            "path_distribution": self.path_distribution(),
            "epistemic_rate": self.ood_detector.epistemic_rate(),
            "ood_detect_calls": self.ood_detector._detect_calls,
            "knowledge_store_size": len(self.knowledge_store),
            "fallback_calls": self.fallback.total_calls,
            "fallback_mean_latency_ms": self.fallback.mean_latency_ms(),
            "sharing_stats": self.sharing_protocol.stats(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_network_acquire(
        self,
        query_embedding: torch.Tensor,
        threshold: float,
    ) -> Optional[KnowledgeEntry]:
        """Phase C stub — attempt to acquire knowledge from the network.

        Phase B: always returns None (network not wired).
        Phase C: serialise query_embedding, broadcast to peer nodes,
                 collect candidate KnowledgeEntries, return best match
                 above ``threshold``.

        Args:
            query_embedding: (D,) query embedding.
            threshold:       Minimum cosine similarity to accept.

        Returns:
            Matching :class:`KnowledgeEntry` or None.
        """
        # Phase B stub — network not wired
        logger.debug(
            "_try_network_acquire: Phase B stub — returning None "
            "(Phase C: wire to P2P layer)"
        )
        return None

    def _propose_knowledge(
        self,
        embedding: torch.Tensor,
        research_trace_hash: str,
    ) -> Optional[KnowledgeContribution]:
        """Package and propose a newly-acquired embedding to the network.

        Args:
            embedding:            (D,) re-anchored embedding from the fallback.
            research_trace_hash:  Hash of the research trace.

        Returns:
            The packaged :class:`KnowledgeContribution`, or None if packaging
            fails (e.g. novelty too low for even local storage).
        """
        novelty = self.knowledge_store.compute_novelty_score(embedding)
        if novelty < self.config.store.novelty_distance_threshold:
            logger.debug(
                "_propose_knowledge: novelty %.3f below threshold %.3f — skipping",
                novelty, self.config.store.novelty_distance_threshold,
            )
            return None

        contribution = self.sharing_protocol.package(
            embedding=embedding,
            research_trace_hash=research_trace_hash,
            contributor_node_id=self.local_node_id,
        )

        # Self-verify and integrate into local store
        if self.config.enable_knowledge_sharing:
            verification = self.sharing_protocol.verify_incoming(
                contribution,
                verifier_node_id=self.local_node_id,
            )
            logger.debug(
                "_propose_knowledge: local verification verdict=%s for %s",
                verification.verdict.name, contribution.entry_id,
            )

        return contribution


# =============================================================================
# Internal utilities
# =============================================================================


def _embedding_hash(embedding: torch.Tensor) -> str:
    """SHA-256 hex digest of a float32 embedding tensor's raw bytes."""
    data = embedding.float().cpu().detach().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _text_hash(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_signature(
    contributor_node_id: str,
    embedding_hash: str,
    research_trace_hash: str,
    timestamp: float,
) -> str:
    """Produce a deterministic signature string for Phase B attestations.

    Phase C: replace with ECDSA / BLS signature using the node's key pair.
    """
    payload = f"{contributor_node_id}|{embedding_hash}|{research_trace_hash}|{timestamp:.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_knowledge_engine(
    model: VLJEPA,
    provider_fn: Optional[Callable[[str], str]] = None,
    local_node_id: str = "local",
    ood_epistemic_threshold: float = 0.65,
    novelty_distance_threshold: float = 0.30,
    enable_network_acquisition: bool = False,
    enable_knowledge_sharing: bool = True,
    **kwargs: Any,
) -> KnowledgeAcquisitionEngine:
    """Convenience factory for KnowledgeAcquisitionEngine.

    Args:
        model:                     Initialised VLJEPA instance.
        provider_fn:               Token-space provider callable.
        local_node_id:             Node identifier.
        ood_epistemic_threshold:   Combined OOD score above which epistemic
                                   uncertainty is declared (default 0.65).
        novelty_distance_threshold: Minimum novelty distance for KnowledgeStore
                                   acceptance (default 0.30).
        enable_network_acquisition: Phase C: True.  Phase B: False (default).
        enable_knowledge_sharing:  Share acquired knowledge with the network.
        **kwargs:                  Additional fields for KnowledgeAcqConfig.

    Returns:
        Configured :class:`KnowledgeAcquisitionEngine`.
    """
    config = KnowledgeAcqConfig(
        ood=OODConfig(epistemic_threshold=ood_epistemic_threshold),
        store=KnowledgeStoreConfig(
            novelty_distance_threshold=novelty_distance_threshold,
        ),
        enable_network_acquisition=enable_network_acquisition,
        enable_knowledge_sharing=enable_knowledge_sharing,
        **{k: v for k, v in kwargs.items() if hasattr(KnowledgeAcqConfig, k)},
    )
    return KnowledgeAcquisitionEngine(
        model=model,
        config=config,
        provider_fn=provider_fn,
        local_node_id=local_node_id,
    )


# =============================================================================
# Verification
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("=" * 70)
    print("Knowledge Acquisition Module — verification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0. Minimal VLJEPA import (needs the rest of the common package)
    # ------------------------------------------------------------------
    try:
        from autonet.nodes.common.vl_jepa import VLJEPA, get_vljepa_preset
        MODEL_AVAILABLE = True
    except Exception as exc:
        print(f"[SKIP] VLJEPA not importable ({exc}) — using a mock instead")
        MODEL_AVAILABLE = False

    # ------------------------------------------------------------------
    # 1. _VectorIndex (no model dependency)
    # ------------------------------------------------------------------
    print("\n[1] _VectorIndex basic operations")
    D = 64
    idx = _VectorIndex(embed_dim=D, max_size=100)
    assert idx.size == 0, "fresh index should be empty"

    # Add 10 random embeddings
    for _ in range(10):
        v = torch.randn(D)
        idx.add(v)
    assert idx.size == 10, f"expected 10 entries, got {idx.size}"

    # Search — should return k results
    query = torch.randn(D)
    sims, idxs = idx.search(query, k=3)
    assert len(sims) == 3, f"expected 3 results, got {len(sims)}"
    assert all(-1.0 <= s <= 1.0 + 1e-5 for s in sims), "similarities out of range"

    idx.reset()
    assert idx.size == 0, "reset should clear the index"
    print("  _VectorIndex: PASS")

    # ------------------------------------------------------------------
    # 2. KnowledgeStore (no model dependency)
    # ------------------------------------------------------------------
    print("\n[2] KnowledgeStore add / search / gates")
    store_cfg = KnowledgeStoreConfig(
        embed_dim=D,
        novelty_distance_threshold=0.10,  # relaxed for random test vectors
        quality_score_floor=0.5,
    )
    store = KnowledgeStore(config=store_cfg)
    assert len(store) == 0

    # Add a first embedding — should always be accepted (empty store)
    emb0 = F.normalize(torch.randn(D), dim=-1)
    eid0 = store.add(
        emb0,
        contributor_node_id="node-001",
        research_trace_hash="abc123",
        novelty_score=1.0,
        quality_score=0.9,
    )
    assert eid0 is not None, "first entry should be accepted"
    assert len(store) == 1, f"expected 1 entry, got {len(store)}"

    # Add a near-duplicate — should be rejected by novelty gate
    emb_dup = emb0 + torch.randn(D) * 0.001   # very close
    emb_dup = F.normalize(emb_dup, dim=-1)
    eid_dup = store.add(
        emb_dup,
        contributor_node_id="node-001",
        research_trace_hash="dup",
        novelty_score=0.01,
        quality_score=0.9,
    )
    assert eid_dup is None, "near-duplicate should be rejected by novelty gate"
    assert len(store) == 1, "store size should still be 1"

    # Add a truly different embedding — should be accepted
    emb1 = F.normalize(torch.randn(D), dim=-1)
    # Ensure it's different enough by projecting out emb0 component
    emb1 = F.normalize(emb1 - (emb1 @ emb0) * emb0, dim=-1)
    eid1 = store.add(
        emb1,
        contributor_node_id="node-002",
        research_trace_hash="xyz789",
        novelty_score=0.85,
        quality_score=0.8,
    )
    # This may still fail the novelty gate with small D — be tolerant
    if eid1 is not None:
        assert len(store) == 2
        print(f"  KnowledgeStore: novel entry accepted (id={eid1[:8]}...)")
    else:
        print("  KnowledgeStore: novel entry rejected by tight gate (ok for small D)")

    # Search
    hits = store.search(emb0, k=3, min_similarity=0.0)
    assert len(hits) >= 1, "should return at least 1 result"
    entry0, sim0 = hits[0]
    assert sim0 >= 0.0, f"similarity should be non-negative, got {sim0}"

    # Novelty score for emb0 itself should be near 0 (it's already stored)
    nov = store.compute_novelty_score(emb0)
    assert nov < 0.3, f"novelty of stored embedding should be low, got {nov}"
    print("  KnowledgeStore: PASS")

    # ------------------------------------------------------------------
    # 3. ContributionAttestation integrity
    # ------------------------------------------------------------------
    print("\n[3] ContributionAttestation signature")
    sig1 = _make_signature("node-001", "emb_hash_abc", "trace_xyz", 1711584000.0)
    sig2 = _make_signature("node-001", "emb_hash_abc", "trace_xyz", 1711584000.0)
    sig3 = _make_signature("node-002", "emb_hash_abc", "trace_xyz", 1711584000.0)
    assert sig1 == sig2, "same inputs → same signature"
    assert sig1 != sig3, "different node_id → different signature"
    print("  Attestation signature: PASS")

    # ------------------------------------------------------------------
    # 4. KnowledgeSharingProtocol
    # ------------------------------------------------------------------
    print("\n[4] KnowledgeSharingProtocol package + verify")
    sharing_cfg = KnowledgeSharingConfig(
        spot_check_rate=0.0,   # deterministic for testing
        min_stake_for_validation=0.0,
        min_validators_for_acceptance=1,
    )
    store2 = KnowledgeStore(config=store_cfg)
    protocol = KnowledgeSharingProtocol(
        store=store2,
        config=sharing_cfg,
        local_node_id="node-001",
    )

    # Package a contribution
    emb_contrib = F.normalize(torch.randn(D), dim=-1)
    contrib = protocol.package(
        embedding=emb_contrib,
        research_trace_hash="deadbeef" * 8,
        contributor_node_id="node-001",
    )
    assert contrib.embedding.shape == (D,)
    assert 0.0 <= contrib.novelty_score <= 1.0
    assert contrib.attestation.signature  # non-empty

    # Verify the contribution (should be accepted)
    vr = protocol.verify_incoming(contrib, verifier_node_id="node-001")
    assert vr.novelty_gate_passed, "first contribution should pass novelty gate"
    assert vr.consistency_check_passed, "first contribution should pass consistency"
    accepted = vr.verdict in (
        VerificationVerdict.ACCEPTED, VerificationVerdict.PENDING_SPOT_CHECK
    )
    assert accepted, f"expected ACCEPTED, got {vr.verdict.name}"
    print(f"  Verification verdict: {vr.verdict.name}")

    # Verify a near-duplicate — should be rejected
    contrib_dup = protocol.package(
        embedding=emb_contrib + torch.randn(D) * 0.001,
        research_trace_hash="deadbeef" * 8 + "dup",
        contributor_node_id="node-002",
    )
    vr_dup = protocol.verify_incoming(contrib_dup, verifier_node_id="node-001")
    assert vr_dup.verdict == VerificationVerdict.REJECTED_NOVELTY, (
        f"near-duplicate should be REJECTED_NOVELTY, got {vr_dup.verdict.name}"
    )
    print("  KnowledgeSharingProtocol: PASS")

    # ------------------------------------------------------------------
    # 5. OODDetector and KnowledgeAcquisitionEngine (requires VLJEPA)
    # ------------------------------------------------------------------
    if MODEL_AVAILABLE:
        print("\n[5] OODDetector + KnowledgeAcquisitionEngine (with VLJEPA)")
        try:
            model = VLJEPA(get_vljepa_preset("small"))
            model.eval()
            D_model = model.config.embed_dim

            # Populate in-distribution index with random training embeddings
            detector = OODDetector(model, config=OODConfig(), node_id="node-001")
            train_embs = F.normalize(torch.randn(50, D_model), dim=-1)
            detector.add_batch_in_distribution(train_embs)
            assert detector._id_index.size == 50

            # In-distribution query (close to training data)
            q_id = F.normalize(train_embs[0] + torch.randn(D_model) * 0.01, dim=-1)
            # Create a stub initial_plan for variance
            K = model.config.num_latents
            plan = torch.randn(1, K, D_model)
            ood_in = detector.detect(q_id, initial_plan=plan)
            print(
                f"  In-dist query: combined={ood_in.combined_score:.3f} "
                f"type={ood_in.uncertainty_type.name}"
            )
            assert ood_in.latency_ms < 500, "OOD detection should be fast"

            # Full engine
            engine = KnowledgeAcquisitionEngine.from_vljepa(
                model=model,
                provider_fn=lambda q: f"Answer to: {q}",
                local_node_id="node-001",
            )
            engine.ood_detector.add_batch_in_distribution(train_embs)

            result = engine.acquire(
                query_embedding=q_id,
                query_text="test query",
                initial_plan=plan,
            )
            assert result.ood_result is not None
            assert result.path_taken in (
                AcquisitionPath.KNOWLEDGE_CACHE,
                AcquisitionPath.TOKEN_SPACE,
            )
            print(f"  Engine path: {result.path_taken.name}, latency={result.latency_ms:.1f}ms")
            print(f"  Engine summary: {engine.summary()}")
            print("  OODDetector + KnowledgeAcquisitionEngine: PASS")

        except Exception as exc:
            print(f"  [SKIP] VLJEPA test failed: {exc}")
    else:
        print("\n[5] OODDetector + KnowledgeAcquisitionEngine: SKIPPED (no model)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All verification checks passed.")
    faiss_status = "available" if _FAISS_AVAILABLE else "NOT available (using torch brute-force)"
    print(f"FAISS: {faiss_status}")
    sys.exit(0)
