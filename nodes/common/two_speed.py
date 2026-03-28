"""
Two-Speed Inference Engine — ATN Phase B

Coordinates between two inference paths operating at different speeds
and cost levels:

  Fast path (local, low-latency):
      Uses the VLJEPA decoder with cached/local latent embeddings.
      No new JEPA reasoning required — decodes from a previously
      computed (or cache-retrieved) latent plan.  Typical latency:
      <100 ms on a local GPU.

  Slow path (deep reasoning, higher latency):
      Runs the full VLJEPA rollout() in latent space for multi-step
      reasoning, producing a fresh latent plan before decoding.
      Typical latency: 200–2000 ms depending on rollout depth.

Routing policy:
  1. Estimate query complexity → complexity score ∈ [0, 1].
  2. If score ≥ config.complexity_threshold → go slow immediately.
  3. Otherwise → try fast path (cache lookup + local decode).
  4. If fast-path confidence < config.confidence_threshold → fall back
     to slow path and update the cache with the result.
  5. Return the response (and path taken) to the caller.

Phase B note:
  The distributed "network" is not yet wired up.  The slow path runs
  locally on the same node using the full VLJEPA model.  The
  NetworkStub interface documents what a real remote call would look
  like so that Phase C can drop in an actual implementation.

Components:
  ComplexityEstimator  — scores a query as [0, 1] using length and
                         embedding-space novelty signals.
  EmbeddingCache       — LRU cache mapping query embeddings to cached
                         latent plans from previous reasoning turns.
  FastPath             — decodes directly from the nearest cached plan.
  SlowPath             — rolls out latent dynamics then decodes.
  NetworkStub          — Phase B shim; Phase C replaces with real net.
  TwoSpeedEngine       — orchestrates routing, fall-back, cache updates.

Usage::

    from autonet.nodes.common.vl_jepa import VLJEPA, get_vljepa_preset
    from autonet.nodes.common.two_speed import TwoSpeedConfig, TwoSpeedEngine

    cfg = TwoSpeedConfig(complexity_threshold=0.6, confidence_threshold=0.5)
    model = VLJEPA(get_vljepa_preset("small"))
    engine = TwoSpeedEngine.from_vljepa(model, config=cfg)

    # async
    result = await engine.infer(token_ids=ids, attention_mask=mask)

    # sync wrapper
    result = engine.infer_sync(token_ids=ids, attention_mask=mask)

    print(result.path_taken)   # "fast" or "slow"
    print(result.confidence)   # float in [0, 1]
    print(result.text)         # decoded text (requires tokenizer)
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

from .vl_jepa import VLJEPA

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TwoSpeedConfig:
    """Tunable thresholds and limits for the Two-Speed Engine.

    All thresholds are in [0, 1] unless otherwise noted.

    Routing thresholds:
        ``complexity_threshold``  — queries with complexity ≥ this bypass the
                                    fast path entirely and go straight to the
                                    slow path.
        ``confidence_threshold``  — fast-path results with confidence below
                                    this trigger a slow-path fallback.

    Embedding cache:
        ``cache_size``            — max entries; LRU eviction when full.
        ``cache_similarity_threshold`` — minimum cosine similarity for a cache
                                    hit to be accepted.

    Slow-path rollout:
        ``rollout_steps``         — number of latent-dynamics steps for the
                                    slow path.  Higher = richer reasoning,
                                    longer latency.

    Decoder:
        ``fast_max_len``          — token budget for fast-path decoding.
        ``slow_max_len``          — token budget for slow-path decoding.
        ``temperature``           — sampling temperature (0.0 = greedy argmax).

    Complexity signals:
        ``length_weight``         — weight of the query-length sub-score.
        ``novelty_weight``        — weight of the embedding-novelty sub-score.
        ``length_reference``      — token count treated as "short" (score ≈ 0).
                                    Queries twice this length score ≈ 1.0 on
                                    the length sub-signal.
    """

    # Routing
    complexity_threshold: float = 0.6
    confidence_threshold: float = 0.5

    # Embedding cache
    cache_size: int = 256
    cache_similarity_threshold: float = 0.75

    # Slow-path rollout
    rollout_steps: int = 4

    # Decoder
    fast_max_len: int = 256
    slow_max_len: int = 512
    temperature: float = 0.8

    # Complexity estimation weights
    length_weight: float = 0.3
    novelty_weight: float = 0.7
    length_reference: int = 64


# =============================================================================
# Tokenizer protocol (duck-typed; no hard dependency on a specific tokenizer)
# =============================================================================


class TokenizerProtocol(Protocol):
    """Minimal interface expected by TwoSpeedEngine from an attached tokenizer."""

    def decode(self, token_ids: List[int]) -> str: ...


# =============================================================================
# Embedding Cache
# =============================================================================


@dataclass
class _CacheEntry:
    """A single cached latent plan with provenance metadata."""
    latent_plan: torch.Tensor       # (1, K, D) stored on CPU
    query_embedding: torch.Tensor   # (D,) mean-pooled query, for similarity
    hits: int = 0
    created_at: float = field(default_factory=time.time)


class EmbeddingCache:
    """LRU cache of latent plans indexed by query embedding similarity.

    Stores up to ``config.cache_size`` entries, evicting the
    least-recently-used entry when capacity is exceeded.

    Cache lookup uses cosine similarity between the probe embedding and
    all stored query embeddings; the most-similar entry above
    ``config.cache_similarity_threshold`` is returned.

    Thread-safety: **not** thread-safe.  Callers must serialise access
    if running the engine from multiple threads simultaneously.
    """

    def __init__(self, config: TwoSpeedConfig):
        self.config = config
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        query_embedding: torch.Tensor,  # (D,)
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Return the best-matching cached latent plan.

        Args:
            query_embedding: Mean-pooled query vector (D,), any device.

        Returns:
            ``(latent_plan, similarity)`` when a hit above threshold is
            found; ``(None, best_similarity)`` on a miss.  The returned
            latent plan is on CPU and must be moved to the target device
            by the caller.
        """
        if not self._store:
            return None, 0.0

        q = F.normalize(query_embedding.float().cpu().unsqueeze(0), dim=-1)  # (1, D)

        best_key: Optional[str] = None
        best_sim: float = -1.0

        for key, entry in self._store.items():
            k = F.normalize(entry.query_embedding.unsqueeze(0), dim=-1)
            sim = float(F.cosine_similarity(q, k, dim=-1).item())
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is None or best_sim < self.config.cache_similarity_threshold:
            return None, max(0.0, best_sim)

        # Mark as most-recently used
        self._store.move_to_end(best_key)
        self._store[best_key].hits += 1
        return self._store[best_key].latent_plan, best_sim

    def store(
        self,
        query_embedding: torch.Tensor,  # (D,)
        latent_plan: torch.Tensor,       # (1, K, D)
    ) -> None:
        """Insert or refresh a cache entry.

        Evicts the least-recently-used entry if the cache is at capacity.
        """
        key = _embedding_key(query_embedding)
        entry = _CacheEntry(
            latent_plan=latent_plan.cpu().detach(),
            query_embedding=query_embedding.cpu().detach().float(),
        )
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry

        while len(self._store) > self.config.cache_size:
            self._store.popitem(last=False)

    def centroid(self) -> Optional[torch.Tensor]:
        """Mean of all cached query embeddings (D,) on CPU, or None if empty."""
        if not self._store:
            return None
        vecs = torch.stack([e.query_embedding for e in self._store.values()])
        return vecs.mean(dim=0)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _embedding_key(embedding: torch.Tensor) -> str:
    """Produce a stable string key from an embedding tensor (MD5 of bytes)."""
    data = embedding.cpu().detach().float().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


# =============================================================================
# Complexity Estimator
# =============================================================================


class ComplexityEstimator:
    """Estimate the cognitive complexity of a query as a score in [0, 1].

    Combines two lightweight signals:

    1. **Length signal** — longer queries tend to require more elaborate
       responses.  Saturates at twice the ``config.length_reference`` token
       count.

    2. **Novelty signal** — queries far from the centroid of previously-seen
       (cached) queries are more likely to be out-of-distribution and
       therefore harder for a cached plan to handle.  On a cold cache
       (no entries yet) novelty is set to 1.0 so the first few queries
       populate the cache via the slow path.

    Both sub-scores are combined as a weighted sum using
    ``config.length_weight`` and ``config.novelty_weight``.
    """

    def __init__(self, config: TwoSpeedConfig, cache: EmbeddingCache):
        self.config = config
        self.cache = cache

    def estimate(
        self,
        token_ids: torch.Tensor,        # (1, S) LongTensor
        query_embedding: torch.Tensor,  # (D,) float
    ) -> float:
        """Return complexity score ∈ [0, 1].

        Args:
            token_ids: Tokenised query (used for length heuristic).
            query_embedding: Mean-pooled query embedding.

        Returns:
            Scalar complexity estimate clipped to [0, 1].
        """
        length_score = self._length_score(token_ids)
        novelty_score = self._novelty_score(query_embedding)
        score = (
            self.config.length_weight * length_score
            + self.config.novelty_weight * novelty_score
        )
        return float(min(1.0, max(0.0, score)))

    def _length_score(self, token_ids: torch.Tensor) -> float:
        seq_len = token_ids.shape[-1]
        ref = max(1, self.config.length_reference)
        return min(1.0, seq_len / (2.0 * ref))

    def _novelty_score(self, query_embedding: torch.Tensor) -> float:
        """Cosine distance from the cached centroid; 1.0 when cache is empty."""
        centroid = self.cache.centroid()
        if centroid is None:
            return 1.0
        q = F.normalize(query_embedding.float().cpu().unsqueeze(0), dim=-1)
        c = F.normalize(centroid.unsqueeze(0), dim=-1)
        cos_sim = float(F.cosine_similarity(q, c, dim=-1).item())
        # Convert cosine similarity → distance ∈ [0, 1]
        return (1.0 - cos_sim) / 2.0


# =============================================================================
# Inference result
# =============================================================================


@dataclass
class TwoSpeedResult:
    """Output from a single TwoSpeedEngine.infer() call.

    Attributes:
        text:            Decoded text response, or ``None`` if no tokenizer
                         was attached to the engine.
        token_ids:       Generated token IDs as (1, L) LongTensor.
        latent_plan:     The (1, K, D) latent plan used for decoding.
        confidence:      Float in [0, 1].  Fast-path hits carry the cache
                         similarity as confidence; slow-path results carry
                         ``1 - mean_uncertainty`` from the rollout.
        path_taken:      ``"fast"`` or ``"slow"``.
        complexity_score: Complexity estimate from ComplexityEstimator.
        latency_ms:      Wall-clock latency from call entry to return.
    """
    text: Optional[str]
    token_ids: torch.Tensor         # (1, L)
    latent_plan: torch.Tensor       # (1, K, D)
    confidence: float
    path_taken: str                 # "fast" | "slow"
    complexity_score: float
    latency_ms: float


# =============================================================================
# Network stub  (Phase B: local single-node mode)
# =============================================================================


class NetworkStub:
    """Stub for the distributed VL-JEPA network interface (Phase B shim).

    In Phase B the slow path runs locally using the same VLJEPA model.
    In Phase C this class will be replaced by an actual network client
    that serialises a context latent plan to bytes, sends it to the
    distributed VL-JEPA cluster, and deserialises the returned K-vector
    guidance bundle.

    The interface deliberately mirrors what a real remote call would look
    like — same inputs, same output dict schema — so that the transition
    to Phase C is a drop-in replacement rather than a refactor.

    Phase B behaviour:
        ``reason()`` calls ``VLJEPA.rollout()`` locally and wraps the
        result in an ``asyncio``-compatible coroutine.  No I/O occurs.
    """

    def __init__(self, model: VLJEPA):
        self._model = model

    async def reason(
        self,
        context_plan: torch.Tensor,  # (B, K, D)
        num_steps: int,
    ) -> Dict[str, Any]:
        """Run multi-step latent reasoning and return updated plan info.

        Phase B: delegates to local ``VLJEPA.rollout()``.
        Phase C: would transmit ``context_plan`` over the P2P network and
                 await the distributed reasoning result.

        Returns:
            Dict matching ``VLJEPA.rollout()`` output:
              ``predictions``         — list of ``num_steps`` tensors (B, K, D)
              ``uncertainties``       — list of ``num_steps`` floats
              ``embedding_variances`` — list of ``num_steps`` floats
        """
        # Phase B: no network I/O — run the dynamics predictor locally.
        return self._model.rollout(context_plan, num_steps)

    @property
    def is_available(self) -> bool:
        """Whether the network (or stub) is ready to accept requests."""
        return True  # always True in single-node Phase B mode


# =============================================================================
# Fast Path
# =============================================================================


class FastPath:
    """Local inference from the embedding cache.

    Looks up the query embedding in the EmbeddingCache.  On a cache hit
    above the similarity threshold, returns the stored latent plan and
    treats the hit similarity as confidence.  The caller (TwoSpeedEngine)
    then decides whether that confidence is sufficient or a slow-path
    fallback is needed.

    The FastPath does **not** run any JEPA encoder or dynamics predictor —
    its entire cost is a cache lookup (O(cache_size) dot products) plus
    the TextDecoder forward pass.
    """

    def __init__(
        self,
        cache: EmbeddingCache,
        config: TwoSpeedConfig,
    ):
        self.cache = cache
        self.config = config

    def run(
        self,
        query_embedding: torch.Tensor,  # (D,)
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Attempt to retrieve a cached latent plan.

        Args:
            query_embedding: Mean-pooled query embedding (D,).
            device: Target device for the returned tensor.

        Returns:
            ``(latent_plan, confidence)`` on hit; ``(None, 0.0)`` on miss.
            ``latent_plan`` is on ``device`` when returned.
        """
        cached_plan, similarity = self.cache.lookup(query_embedding)
        if cached_plan is None:
            return None, 0.0
        return cached_plan.to(device), float(similarity)


# =============================================================================
# Slow Path
# =============================================================================


class SlowPath:
    """Full JEPA multi-step latent reasoning.

    Accepts an initial latent plan (output of the VLJEPA encoder +
    semantic predictor for the current query) and rolls it forward in
    latent space via the LatentDynamicsPredictor.  The final step's
    plan is used for decoding.

    The network interface is abstracted through ``NetworkStub`` so that
    Phase C can supply a real distributed implementation without changing
    this class.
    """

    def __init__(
        self,
        network: NetworkStub,
        config: TwoSpeedConfig,
    ):
        self.network = network
        self.config = config

    async def run(
        self,
        initial_plan: torch.Tensor,  # (1, K, D)
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """Roll out latent dynamics and return the final plan + confidence.

        Confidence is derived from the rollout's uncertainty estimates:
        ``1 - mean_uncertainty``.  A fully-collapsed rollout (uncertainty=1
        throughout) yields confidence=0; a highly-diverse rollout yields
        confidence≈1.

        Args:
            initial_plan: Starting plan from ``predict_semantic()``.
            device: Target device for the returned plan tensor.

        Returns:
            ``(final_plan, confidence)`` where ``final_plan`` is (1, K, D).
        """
        rollout = await self.network.reason(initial_plan, self.config.rollout_steps)
        predictions: List[torch.Tensor] = rollout["predictions"]
        uncertainties: List[float] = rollout["uncertainties"]

        final_plan = predictions[-1].to(device)
        mean_uncertainty = sum(uncertainties) / max(1, len(uncertainties))
        confidence = 1.0 - mean_uncertainty
        return final_plan, float(confidence)


# =============================================================================
# Two-Speed Engine
# =============================================================================


class TwoSpeedEngine:
    """Orchestrates fast and slow inference paths for ATN latent reasoning.

    Routing logic (per ``infer()`` call):
      1. Encode query text → mean-pooled query embedding via VLJEPA text
         encoder.
      2. Estimate complexity from length + cache-novelty signals.
      3. If complexity ≥ ``config.complexity_threshold`` → slow path only.
      4. Otherwise → try fast path (cache lookup).
         a. Hit and confidence ≥ ``config.confidence_threshold`` → fast.
         b. Miss or low confidence → slow path fallback.
      5. After slow path: update the embedding cache with the new plan so
         that similar future queries can be served by the fast path.
      6. Decode tokens from the selected latent plan via the VLJEPA decoder.
      7. Return ``TwoSpeedResult`` with text, diagnostics, and metadata.

    Phase B (single-node mode):
      ``NetworkStub`` routes slow-path calls to ``VLJEPA.rollout()`` locally.
      No network I/O occurs.  The routing, caching, and fallback logic is
      fully exercised so that Phase C only requires wiring up a real network
      client.

    Usage::

        from autonet.nodes.common.vl_jepa import VLJEPA, get_vljepa_preset
        from autonet.nodes.common.two_speed import TwoSpeedConfig, TwoSpeedEngine

        model = VLJEPA(get_vljepa_preset("small"))
        engine = TwoSpeedEngine.from_vljepa(model)

        # async context
        result = await engine.infer(token_ids=ids, attention_mask=mask)

        # synchronous context
        result = engine.infer_sync(token_ids=ids, attention_mask=mask)
    """

    def __init__(
        self,
        model: VLJEPA,
        config: Optional[TwoSpeedConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            model: Fully-initialised VLJEPA instance.  Weights may be random
                   for testing; trained weights are required for quality output.
            config: Routing / caching configuration.  Uses ``TwoSpeedConfig``
                    defaults if ``None``.
            tokenizer: Optional object with a ``decode(List[int]) -> str``
                       method.  If ``None``, ``TwoSpeedResult.text`` is
                       ``None``.  The engine accepts any tokenizer that
                       satisfies this duck-typed interface.
        """
        self.model = model
        self.config = config or TwoSpeedConfig()
        self.tokenizer = tokenizer

        self.cache = EmbeddingCache(self.config)
        self.complexity_estimator = ComplexityEstimator(self.config, self.cache)
        self.fast_path = FastPath(self.cache, self.config)
        self.network = NetworkStub(self.model)
        self.slow_path = SlowPath(self.network, self.config)

        self._device: torch.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vljepa(
        cls,
        model: VLJEPA,
        config: Optional[TwoSpeedConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> "TwoSpeedEngine":
        """Construct a TwoSpeedEngine from an existing VLJEPA model.

        Args:
            model: Initialised VLJEPA instance.
            config: Optional engine config; defaults apply if ``None``.
            tokenizer: Optional tokenizer with ``decode()`` method.

        Returns:
            Ready-to-use ``TwoSpeedEngine``.
        """
        return cls(model, config=config, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Main entry point (async)
    # ------------------------------------------------------------------

    async def infer(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_text: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
    ) -> TwoSpeedResult:
        """Run two-speed inference for a single query.

        Args:
            token_ids: (1, S) LongTensor of tokenised query.
            attention_mask: (1, S) BoolTensor — ``True`` for real tokens.
            query_text: Raw text string used only for debug logging.
            images: (1, C, H, W) optional image input.  If ``None``, the
                    engine synthesises a zero visual embedding so the text
                    pathway is exercised without a visual forward pass
                    (Phase B text-only mode).

        Returns:
            :class:`TwoSpeedResult` with all fields populated.
        """
        t0 = time.perf_counter()
        device = self._device

        token_ids = token_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Step 1: encode query → mean-pooled embedding
        query_embedding = self._encode_query(token_ids, attention_mask)  # (D,)

        # Step 2: estimate complexity
        complexity = self.complexity_estimator.estimate(token_ids, query_embedding)
        logger.debug(
            "infer: complexity=%.3f threshold=%.3f cache_size=%d query=%r",
            complexity,
            self.config.complexity_threshold,
            len(self.cache),
            (query_text or "")[:80],
        )

        # Steps 3–5: route to fast or slow path
        if complexity >= self.config.complexity_threshold:
            latent_plan, confidence, path = await self._run_slow(
                token_ids, attention_mask, images, query_embedding, device
            )
        else:
            latent_plan, confidence = self.fast_path.run(query_embedding, device)
            if latent_plan is not None and confidence >= self.config.confidence_threshold:
                path = "fast"
                logger.debug("fast path hit (confidence=%.3f)", confidence)
            else:
                logger.debug(
                    "fast path miss/low-confidence (%.3f) → slow path", confidence
                )
                latent_plan, confidence, path = await self._run_slow(
                    token_ids, attention_mask, images, query_embedding, device
                )

        # Step 6: decode tokens from latent plan
        generated_ids = self._decode(latent_plan, path)  # (1, L)

        # Step 7: convert to text if tokenizer is available
        text = self._ids_to_text(generated_ids)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "infer done: path=%s confidence=%.3f latency=%.1fms",
            path, confidence, latency_ms,
        )

        return TwoSpeedResult(
            text=text,
            token_ids=generated_ids,
            latent_plan=latent_plan,
            confidence=confidence,
            path_taken=path,
            complexity_score=complexity,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Synchronous wrapper
    # ------------------------------------------------------------------

    def infer_sync(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_text: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
    ) -> TwoSpeedResult:
        """Blocking wrapper around :meth:`infer` for non-async callers.

        Uses ``asyncio.run()`` when no event loop is active.  When called
        from within a running event loop (e.g. Jupyter) it offloads to a
        ``ThreadPoolExecutor`` to avoid blocking the loop.
        """
        coro = self.infer(token_ids, attention_mask, query_text, images)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if loop.is_running():
            # Running inside an existing event loop — run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                return exe.submit(asyncio.run, coro).result()

        return loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_query(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode query tokens → mean-pooled query embedding (D,).

        Uses the VLJEPA text encoder.  Masked mean-pooling skips padding
        positions when an attention mask is provided.
        """
        self.model.eval()
        text_embeds = self.model.encode_text(token_ids, attention_mask)  # (1, S, D)

        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(-1)  # (1, S, 1)
            pooled = (text_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = text_embeds.mean(dim=1)  # (1, D)

        return pooled.squeeze(0)  # (D,)

    @torch.no_grad()
    def _compute_initial_plan(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the full VLJEPA encoder + predictor to get an initial latent plan.

        Text-only mode (``images=None``): synthesises a zero visual embedding
        (1, num_patches, embed_dim) so the encoder–predictor pipeline
        receives a valid tensor and produces a coherent initial plan driven
        entirely by the text pathway.

        Returns:
            Latent plan (1, K, D) on engine device.
        """
        device = self._device
        self.model.eval()

        if images is not None:
            visual_embeds = self.model.encode_visual(images.to(device))
        else:
            D = self.model.config.embed_dim
            N = self.model.config.num_patches
            visual_embeds = torch.zeros(1, N, D, device=device)

        text_embeds = self.model.encode_text(token_ids, attention_mask)
        fused = self.model.fuse(visual_embeds, text_embeds)
        return self.model.predict_semantic(fused)  # (1, K, D)

    async def _run_slow(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float, str]:
        """Compute a fresh initial plan, run the slow path, and update the cache.

        Returns:
            ``(final_plan, confidence, "slow")``.
        """
        initial_plan = self._compute_initial_plan(token_ids, attention_mask, images)
        final_plan, confidence = await self.slow_path.run(initial_plan, device)

        # Populate cache so future similar queries can use the fast path
        self.cache.store(query_embedding, final_plan)
        logger.debug("slow path done (confidence=%.3f), cache updated", confidence)

        return final_plan, confidence, "slow"

    @torch.no_grad()
    def _decode(
        self,
        latent_plan: torch.Tensor,  # (1, K, D)
        path: str,
    ) -> torch.Tensor:
        """Decode token IDs from a latent plan via the VLJEPA text decoder.

        ``path`` selects the token budget: fast path uses ``fast_max_len``,
        slow path uses ``slow_max_len``.

        Returns:
            Generated token IDs (1, L).
        """
        self.model.eval()
        max_len = (
            self.config.fast_max_len if path == "fast" else self.config.slow_max_len
        )
        return self.model.text_decoder.generate(
            latent_plan,
            max_len=max_len,
            temperature=self.config.temperature,
        )

    def _ids_to_text(self, token_ids: torch.Tensor) -> Optional[str]:
        """Convert generated token IDs to text using the attached tokenizer.

        Returns ``None`` if no tokenizer was provided at construction time.
        """
        if self.tokenizer is None:
            return None
        ids: List[int] = token_ids.squeeze(0).tolist()
        return self.tokenizer.decode(ids)
