"""
Latent Reasoning Engine for multi-step reasoning in embedding space.

Implements a reasoning loop that chains VL-JEPA's LatentDynamicsPredictor
to "think" in embedding space without generating intermediate tokens.  The
local decoder fires only when the engine needs to act (tool calls, user
responses) or re-anchor mid-chain.

Architecture follows the "plan short, replan often" pattern from world model
literature (MuZero, Dreamer, LeWorldModel): short forward predictions with
periodic re-anchoring in token space to prevent error accumulation.

Key design decisions:
- Collapse detection is the primary safety mechanism (Risk #1, score 20/25
  in FEASIBILITY_POSITION_PAPER.md).
- Replan interval defaults to 8 steps (ThinkJEPA dense branch horizon).
- Degradation curves are recorded per-step for Gate 2 validation.
- Token-space fallback is a first-class feature, not an afterthought.

References:
- V-JEPA 2-AC (Bardes et al., 2025, arXiv:2506.09985): T=2 rollout loss + MPC.
- LeWorldModel (Maes et al., 2026, arXiv:2603.19312): SIGReg + residual stability.
- ThinkJEPA (2026, arXiv:2603.22281): VLM-guided rollouts at H=4,8,16,32.
- Coconut (Meta FAIR, ICLR 2025, arXiv:2412.06769): latent CoT vectors.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .vl_jepa import LatentDynamicsPredictor

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LatentReasoningConfig:
    """Hyperparameters for the Latent Reasoning Engine.

    All thresholds are tuned against the risk profile in the feasibility
    position paper.  The defaults represent a conservative posture that
    prefers falling back to token space over silently degrading.
    """

    # ---- Rollout budget ----
    max_steps: int = 32
    """Maximum reasoning steps per call.  32 matches ThinkJEPA's longest
    validated horizon.  Raise with caution — error accumulates with depth."""

    # ---- Replanning schedule ----
    replan_interval: int = 8
    """Steps between token-space re-anchor checkpoints.  Set to 0 to disable
    periodic replanning (useful for ablations).  ThinkJEPA's dense branch
    operates at 4-8 step horizons, so 8 is a safe starting point."""

    decode_on_replan: bool = True
    """If True, call the replan_fn at each checkpoint to get a fresh latent
    plan grounded in decoded token space.  If False, just reset the collapse
    detector's baseline without token-space decode."""

    # ---- Collapse detection thresholds ----
    collapse_threshold: float = 0.5
    """Fraction of initial embedding variance below which we declare collapse.
    Per the JEPA survey recommendation: "if variance drops >50% over 5 steps,
    the chain is collapsing."  Lower values are more tolerant."""

    variance_window: int = 3
    """Number of recent steps used to compute the rolling variance mean for
    collapse detection.  Wider windows smooth out transient dips."""

    min_spread_ratio: float = 0.2
    """Minimum inter-vector spread relative to the step-0 baseline.  When the
    K latent vectors converge to a single point (spread → 0), the plan carries
    no information — hard collapse.  0.2 = stop when <20% of initial spread."""

    cosine_sim_ceiling: float = 0.98
    """If the mean cosine similarity between consecutive steps exceeds this,
    the chain is stagnating (predicting identity).  The residual delta
    projection in LatentDynamicsPredictor initializes to zero, so this fires
    mainly for under-trained models."""

    spectral_rank_min: int = 4
    """Minimum effective rank (exp-entropy of singular values) for the K×D
    latent plan matrix.  Collapse drives rank toward 1 (all vectors aligned);
    we fall back when rank < this threshold."""

    # ---- Fallback behaviour ----
    fallback_on_collapse: bool = True
    """When True, stop the chain and flag the trace if collapse is detected.
    When False, continue but mark each step as degraded (useful for studying
    collapse curves)."""

    # ---- Per-step quality floor for Gate 2 ----
    quality_floor: float = 0.0
    """Optional hard floor on per-step quality score (0–1).  Steps below this
    are marked degraded in the trace.  0.0 = disabled."""


# =============================================================================
# Per-step record
# =============================================================================

@dataclass
class ReasoningStep:
    """Snapshot of a single step in the latent reasoning chain.

    All per-step metrics are scalar floats so the trace can be serialised
    and plotted as degradation curves for Gate 2 validation.
    """

    step_idx: int
    """0-based index within the current reasoning chain segment."""

    latent_plan: torch.Tensor
    """Predicted latent plan at this step, shape (B, K, D).  Stored
    detached; do not backpropagate through the trace."""

    # ---- Quality metrics (higher is generally better) ----
    embedding_variance: float
    """Mean variance of the plan across the embedding dimension.
    Computed as plan.var(dim=-1).mean() — a scalar per batch, then
    batch-averaged.  Sharp drop signals collapse."""

    spread: float
    """Mean squared deviation of the K latent vectors from their centroid.
    High spread = K vectors are diverse = plan carries rich information.
    Zero spread = complete collapse."""

    effective_rank: float
    """Spectral effective rank of the (B*K) × D plan matrix, computed as
    exp(Shannon entropy of normalised singular values).  Ranges from 1
    (rank-1, fully collapsed) to D (isotropic).  Tracks diversity
    orthogonally to spread."""

    cosine_sim_to_prev: float
    """Mean cosine similarity between this step's plan and the previous
    step's plan (mean-pooled over K vectors, then batch-averaged).
    Values near 1.0 indicate stagnation."""

    uncertainty: float
    """Derived uncertainty score in [0, 1].
    0 = confident / diverse embedding.
    1 = collapsed / stagnant embedding.
    Aggregates spread and cosine sim into a single signal."""

    quality: float
    """Overall quality score in [0, 1] combining all per-step metrics.
    Used to populate the degradation curve for Gate 2."""

    # ---- Flags ----
    is_replan: bool = False
    """True if this step was produced after a token-space re-anchoring."""

    is_collapsed: bool = False
    """True if collapse was detected at this step."""

    is_degraded: bool = False
    """True if quality fell below LatentReasoningConfig.quality_floor."""

    decoded_tokens: Optional[torch.Tensor] = None
    """Token IDs produced at a replan checkpoint, shape (B, T).  None for
    all non-replan steps."""


# =============================================================================
# Collapse detector
# =============================================================================

class CollapseDetector:
    """Monitors embedding quality across reasoning steps.

    Maintains rolling statistics and flags collapse using four independent
    signals, any one of which triggers a collapse event:

    1. **Variance drop** — embedding variance falls below
       ``config.collapse_threshold`` × initial variance.
    2. **Spread collapse** — inter-vector spread falls below
       ``config.min_spread_ratio`` × initial spread.
    3. **Cosine stagnation** — consecutive steps are too similar (>ceiling).
    4. **Spectral rank collapse** — effective rank falls below
       ``config.spectral_rank_min``.

    The detector is reset each time the reasoning engine re-anchors with a
    fresh latent plan (replan checkpoint).
    """

    def __init__(self, config: LatentReasoningConfig):
        self.config = config
        self._variance_history: List[float] = []
        self._baseline_variance: Optional[float] = None
        self._baseline_spread: Optional[float] = None

    def reset(self, baseline_plan: torch.Tensor) -> None:
        """Initialise or reset baselines from a fresh latent plan.

        Call this at step 0 and at every replan checkpoint.

        Args:
            baseline_plan: (B, K, D) — the anchor latent plan.
        """
        self._variance_history = []
        self._baseline_variance = float(baseline_plan.var(dim=-1).mean().item())
        self._baseline_spread = _plan_spread(baseline_plan)

    def check(
        self,
        plan: torch.Tensor,
        cosine_sim_to_prev: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """Check a single step for collapse.

        Args:
            plan: (B, K, D) — the current step's latent plan.
            cosine_sim_to_prev: cosine similarity to the previous step's plan.

        Returns:
            (collapsed, metrics) where collapsed is True if any signal fired,
            and metrics is a dict of per-signal values for the trace record.
        """
        assert self._baseline_variance is not None, (
            "CollapseDetector.reset() must be called before check()"
        )

        emb_var = float(plan.var(dim=-1).mean().item())
        spread = _plan_spread(plan)
        eff_rank = _effective_rank(plan)

        self._variance_history.append(emb_var)

        # Rolling mean over recent window
        window = self._variance_history[-self.config.variance_window:]
        rolling_var = sum(window) / len(window)

        metrics = {
            "embedding_variance": emb_var,
            "spread": spread,
            "effective_rank": eff_rank,
            "cosine_sim_to_prev": cosine_sim_to_prev,
            "rolling_variance": rolling_var,
        }

        # --- Signal 1: variance drop ---
        var_collapsed = (
            self._baseline_variance > 0
            and rolling_var < self.config.collapse_threshold * self._baseline_variance
        )

        # --- Signal 2: spread collapse ---
        spread_collapsed = (
            self._baseline_spread is not None
            and self._baseline_spread > 1e-8
            and spread < self.config.min_spread_ratio * self._baseline_spread
        )

        # --- Signal 3: cosine stagnation ---
        stagnated = cosine_sim_to_prev > self.config.cosine_sim_ceiling

        # --- Signal 4: spectral rank ---
        rank_collapsed = eff_rank < self.config.spectral_rank_min

        collapsed = var_collapsed or spread_collapsed or stagnated or rank_collapsed

        if collapsed:
            reasons = []
            if var_collapsed:
                reasons.append(
                    f"variance_drop({rolling_var:.4f}<"
                    f"{self.config.collapse_threshold * self._baseline_variance:.4f})"
                )
            if spread_collapsed:
                reasons.append(
                    f"spread_collapse({spread:.4f}<"
                    f"{self.config.min_spread_ratio * self._baseline_spread:.4f})"
                )
            if stagnated:
                reasons.append(f"cosine_stagnation({cosine_sim_to_prev:.4f})")
            if rank_collapsed:
                reasons.append(f"rank_collapse({eff_rank:.2f}<{self.config.spectral_rank_min})")
            logger.warning("Collapse detected: %s", ", ".join(reasons))

        return collapsed, metrics

    @property
    def baseline_variance(self) -> Optional[float]:
        return self._baseline_variance

    @property
    def baseline_spread(self) -> Optional[float]:
        return self._baseline_spread


# =============================================================================
# Replanning strategy
# =============================================================================

class ReplanTrigger(Enum):
    """Reason why a replan was triggered."""
    SCHEDULED = auto()       # periodic interval reached
    COLLAPSE = auto()        # collapse detector fired
    QUALITY_FLOOR = auto()   # quality fell below config.quality_floor


@dataclass
class ReplanEvent:
    """Record of a single replan event within a reasoning trace."""
    step_idx: int
    trigger: ReplanTrigger
    plan_before: torch.Tensor   # (B, K, D) plan at the replan point
    plan_after: torch.Tensor    # (B, K, D) fresh plan from replan_fn
    decoded_tokens: Optional[torch.Tensor] = None  # (B, T) if decode_on_replan


class ReplanningStrategy:
    """Decides when and how to re-anchor the latent reasoning chain.

    Implements "plan short, replan often": every ``replan_interval`` steps
    (or immediately on collapse), calls ``replan_fn`` to get a fresh latent
    plan grounded in decoded token space.

    If ``replan_fn`` is None, replanning is a no-op that only resets the
    collapse detector's baseline (useful for studying collapse without the
    confound of re-anchoring).
    """

    def __init__(
        self,
        config: LatentReasoningConfig,
        replan_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Args:
            config: Reasoning configuration.
            replan_fn: Optional callable ``(latent_plan) -> latent_plan``.
                       Receives the current (B, K, D) plan (possibly degraded)
                       and returns a fresh (B, K, D) plan after decode +
                       re-encode.  If None, replanning resets baselines only.
        """
        self.config = config
        self.replan_fn = replan_fn
        self.events: List[ReplanEvent] = []

    def should_replan(
        self,
        step_idx: int,
        collapsed: bool,
        quality: float,
    ) -> Optional[ReplanTrigger]:
        """Return the trigger reason if a replan is needed, else None.

        Args:
            step_idx: Current step index (0-based within the chain).
            collapsed: Whether the collapse detector fired this step.
            quality: Per-step quality score (0–1).

        Returns:
            ReplanTrigger enum value, or None if no replan needed.
        """
        if collapsed and self.config.fallback_on_collapse:
            return ReplanTrigger.COLLAPSE

        if (
            self.config.quality_floor > 0.0
            and quality < self.config.quality_floor
        ):
            return ReplanTrigger.QUALITY_FLOOR

        if (
            self.config.replan_interval > 0
            and self.config.enable_replan  # checked below via config attr
            and (step_idx + 1) % self.config.replan_interval == 0
        ):
            return ReplanTrigger.SCHEDULED

        return None

    def execute(
        self,
        step_idx: int,
        trigger: ReplanTrigger,
        current_plan: torch.Tensor,
        collapse_detector: CollapseDetector,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute a replan and reset the collapse detector.

        Args:
            step_idx: Step index at which the replan occurs.
            trigger: Why the replan was triggered.
            current_plan: (B, K, D) — plan at replan time.
            collapse_detector: Detector to reset with the fresh plan.

        Returns:
            (fresh_plan, decoded_tokens) where decoded_tokens is non-None
            only when replan_fn produces tokens (i.e., wraps a TextDecoder).
        """
        fresh_plan = current_plan
        decoded_tokens: Optional[torch.Tensor] = None

        if self.replan_fn is not None and self.config.decode_on_replan:
            fresh_plan = self.replan_fn(current_plan)
            logger.debug(
                "Replan at step %d (%s): plan re-anchored via replan_fn",
                step_idx, trigger.name,
            )

        # Record the event
        self.events.append(ReplanEvent(
            step_idx=step_idx,
            trigger=trigger,
            plan_before=current_plan.detach(),
            plan_after=fresh_plan.detach(),
            decoded_tokens=decoded_tokens,
        ))

        # Reset collapse detector baselines to the fresh plan
        collapse_detector.reset(fresh_plan)

        return fresh_plan, decoded_tokens

    @property
    def enable_replan(self) -> bool:
        return self.config.replan_interval > 0


# Patch ReplanningStrategy to access enable_replan from config via attribute
# (avoids circular reference in should_replan)
LatentReasoningConfig.enable_replan = property(lambda self: self.replan_interval > 0)  # type: ignore[attr-defined]


# =============================================================================
# Reasoning trace
# =============================================================================

@dataclass
class ReasoningTrace:
    """Full record of a latent reasoning chain.

    Provides the per-step degradation curves needed for Gate 2 validation
    and the final state needed for downstream decoding.

    Attributes:
        steps: Ordered list of ReasoningStep records.
        replan_events: Ordered list of ReplanEvent records.
        initial_plan: The (B, K, D) plan passed to the engine.
        final_plan: The (B, K, D) plan at the last completed step.
        terminated_early: True if the chain stopped before max_steps due to
                          collapse (and fallback_on_collapse=True).
        fallback_to_tokens: True if the engine returned a token-space result
                            rather than a latent plan (reserved for future
                            use; not currently set by the engine itself).
        config: The LatentReasoningConfig that produced this trace.
    """

    steps: List[ReasoningStep] = field(default_factory=list)
    replan_events: List[ReplanEvent] = field(default_factory=list)
    initial_plan: Optional[torch.Tensor] = None
    final_plan: Optional[torch.Tensor] = None
    terminated_early: bool = False
    fallback_to_tokens: bool = False
    config: Optional[LatentReasoningConfig] = None

    # ---- Degradation curve accessors ----

    def degradation_curves(self) -> Dict[str, List[float]]:
        """Return all per-step metrics as lists for plotting / Gate 2 logging.

        Returns:
            Dict mapping metric name to list of per-step values.  All lists
            have the same length (equal to len(self.steps)).
        """
        return {
            "embedding_variance": [s.embedding_variance for s in self.steps],
            "spread": [s.spread for s in self.steps],
            "effective_rank": [s.effective_rank for s in self.steps],
            "cosine_sim_to_prev": [s.cosine_sim_to_prev for s in self.steps],
            "uncertainty": [s.uncertainty for s in self.steps],
            "quality": [s.quality for s in self.steps],
        }

    def quality_at_step(self, k: int) -> Optional[float]:
        """Return the quality score at step k, or None if step doesn't exist."""
        if 0 <= k < len(self.steps):
            return self.steps[k].quality
        return None

    def relative_quality_drop(self) -> Optional[float]:
        """Fraction of initial quality lost by the final step.

        Returns None if the trace has fewer than 2 steps.  A value of 0.5
        means quality halved — the go/no-go threshold from the research brief.
        """
        if len(self.steps) < 2:
            return None
        q0 = self.steps[0].quality
        qn = self.steps[-1].quality
        if q0 < 1e-8:
            return None
        return max(0.0, 1.0 - qn / q0)

    def summary(self) -> Dict[str, object]:
        """Compact summary dict for logging."""
        return {
            "num_steps": len(self.steps),
            "terminated_early": self.terminated_early,
            "num_replans": len(self.replan_events),
            "final_quality": self.steps[-1].quality if self.steps else None,
            "relative_quality_drop": self.relative_quality_drop(),
            "num_collapsed": sum(1 for s in self.steps if s.is_collapsed),
        }


# =============================================================================
# Main engine
# =============================================================================

class LatentReasoningEngine:
    """Multi-step reasoning loop in VL-JEPA embedding space.

    The engine chains ``LatentDynamicsPredictor`` to advance a latent plan
    through ``num_steps`` reasoning steps.  At each step it:

    1. Predicts the next latent plan via the dynamics predictor.
    2. Computes per-step quality metrics (variance, spread, cosine sim, rank).
    3. Checks for embedding collapse via ``CollapseDetector``.
    4. Optionally re-anchors via the ``ReplanningStrategy``.
    5. Appends a ``ReasoningStep`` to the growing ``ReasoningTrace``.

    The engine is stateless between ``reason()`` calls — all state lives in
    the returned ``ReasoningTrace``.

    Usage::

        engine = LatentReasoningEngine(
            dynamics_predictor=vljepa.dynamics_predictor,
            config=LatentReasoningConfig(max_steps=16, replan_interval=8),
            replan_fn=my_replan_callback,
        )
        trace = engine.reason(initial_plan, num_steps=16)
        final_plan = trace.final_plan
    """

    def __init__(
        self,
        dynamics_predictor: LatentDynamicsPredictor,
        config: Optional[LatentReasoningConfig] = None,
        replan_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Args:
            dynamics_predictor: The ``LatentDynamicsPredictor`` from a
                ``VLJEPA`` model.  Must already be on the desired device.
            config: Reasoning configuration.  Defaults to
                ``LatentReasoningConfig()`` if not provided.
            replan_fn: Optional callable ``(latent_plan: Tensor) ->
                latent_plan: Tensor`` that returns a fresh (B, K, D) plan
                after decoding to token space and re-encoding.  When None,
                periodic replan checkpoints reset only the collapse detector
                baseline without token-space grounding.
        """
        self.dynamics_predictor = dynamics_predictor
        self.config = config or LatentReasoningConfig()
        self._collapse_detector = CollapseDetector(self.config)
        self._replan_strategy = ReplanningStrategy(self.config, replan_fn)

    @torch.no_grad()
    def reason(
        self,
        initial_plan: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> ReasoningTrace:
        """Run the latent reasoning loop.

        Args:
            initial_plan: Starting latent plan, shape (B, K, D).  Typically
                the output of ``VLJEPA.predict_semantic()`` for the current
                context turn.
            num_steps: Number of reasoning steps to attempt.  Clamped to
                ``config.max_steps``.  Defaults to ``config.max_steps``.

        Returns:
            ``ReasoningTrace`` containing all per-step records, replan events,
            and the degradation curves needed for Gate 2 validation.
        """
        if num_steps is None:
            num_steps = self.config.max_steps
        num_steps = min(num_steps, self.config.max_steps)

        trace = ReasoningTrace(
            initial_plan=initial_plan.detach().clone(),
            config=self.config,
        )

        # Fresh strategy / detector for this call
        self._replan_strategy = ReplanningStrategy(self.config, self._replan_strategy.replan_fn)
        self._collapse_detector = CollapseDetector(self.config)
        self._collapse_detector.reset(initial_plan)

        current_plan = initial_plan  # (B, K, D)
        prev_plan_mean: Optional[torch.Tensor] = None  # for cosine sim

        for step_idx in range(num_steps):
            # ---- 1. Predict next latent plan ----
            next_plan = self.dynamics_predictor(current_plan)  # (B, K, D)

            # ---- 2. Compute cosine similarity to previous step ----
            current_mean = next_plan.mean(dim=1)  # (B, D)
            if prev_plan_mean is None:
                cosine_sim = 0.0
            else:
                cosine_sim = float(
                    F.cosine_similarity(current_mean, prev_plan_mean, dim=-1)
                    .mean()
                    .item()
                )
            prev_plan_mean = current_mean.detach()

            # ---- 3. Collapse detection ----
            collapsed, metrics = self._collapse_detector.check(next_plan, cosine_sim)

            # ---- 4. Compute per-step quality and uncertainty scores ----
            quality, uncertainty = _quality_score(metrics, self._collapse_detector)

            # ---- 5. Check if replanning is needed ----
            trigger = self._replan_strategy.should_replan(step_idx, collapsed, quality)
            decoded_tokens: Optional[torch.Tensor] = None
            is_replan = trigger is not None

            if is_replan:
                if trigger is ReplanTrigger.COLLAPSE and self.config.fallback_on_collapse:
                    # Record the collapsed step, then stop
                    step = _make_step(
                        step_idx=step_idx,
                        plan=next_plan,
                        metrics=metrics,
                        cosine_sim=cosine_sim,
                        quality=quality,
                        uncertainty=uncertainty,
                        is_replan=False,
                        is_collapsed=True,
                        decoded_tokens=None,
                        config=self.config,
                    )
                    trace.steps.append(step)
                    trace.terminated_early = True
                    logger.warning(
                        "Latent reasoning terminated early at step %d (collapse).",
                        step_idx,
                    )
                    break

                # Non-collapse replan or collapse without fallback_on_collapse
                fresh_plan, decoded_tokens = self._replan_strategy.execute(
                    step_idx=step_idx,
                    trigger=trigger,
                    current_plan=next_plan,
                    collapse_detector=self._collapse_detector,
                )
                # Use fresh plan for recording and next iteration
                next_plan = fresh_plan
                trace.replan_events.extend(self._replan_strategy.events[-1:])

                # Recompute metrics after re-anchor for accurate trace record
                current_mean = next_plan.mean(dim=1)
                prev_plan_mean = current_mean.detach()
                collapsed_after, metrics = self._collapse_detector.check(next_plan, 0.0)
                quality, uncertainty = _quality_score(metrics, self._collapse_detector)
                collapsed = collapsed and collapsed_after  # only collapsed if still bad

            # ---- 6. Record step ----
            step = _make_step(
                step_idx=step_idx,
                plan=next_plan,
                metrics=metrics,
                cosine_sim=cosine_sim,
                quality=quality,
                uncertainty=uncertainty,
                is_replan=is_replan,
                is_collapsed=collapsed and not is_replan,
                decoded_tokens=decoded_tokens,
                config=self.config,
            )
            trace.steps.append(step)

            current_plan = next_plan

            logger.debug(
                "Step %d: var=%.4f spread=%.4f rank=%.1f cos=%.4f q=%.4f%s%s",
                step_idx,
                metrics["embedding_variance"],
                metrics["spread"],
                metrics["effective_rank"],
                cosine_sim,
                quality,
                " [replan]" if is_replan else "",
                " [collapsed]" if step.is_collapsed else "",
            )

        trace.final_plan = current_plan.detach()
        logger.info(
            "Latent reasoning completed: %d steps, %d replans, early_stop=%s",
            len(trace.steps),
            len(trace.replan_events),
            trace.terminated_early,
        )
        return trace


# =============================================================================
# Internal helpers
# =============================================================================

def _plan_spread(plan: torch.Tensor) -> float:
    """Mean squared deviation of the K latent vectors from their centroid.

    High spread = K vectors are diverse = plan carries more information.
    Zero spread = all K vectors identical = complete collapse.

    Args:
        plan: (B, K, D) latent plan.

    Returns:
        Scalar spread value (float).
    """
    centroid = plan.mean(dim=1, keepdim=True)  # (B, 1, D)
    return float((plan - centroid).pow(2).mean().item())


def _effective_rank(plan: torch.Tensor) -> float:
    """Spectral effective rank of the (B*K) × D plan matrix.

    Computed as exp(Shannon entropy of normalised singular values).
    Ranges from 1 (rank-1, fully collapsed) to D (isotropic).

    The spectral effective rank is the Roy & Vetterli (2007) measure; it
    captures whether the plan spans many dimensions in embedding space or
    has collapsed to a low-dimensional subspace.

    Args:
        plan: (B, K, D) latent plan.

    Returns:
        Effective rank as a float (≥ 1.0).
    """
    B, K, D = plan.shape
    flat = plan.reshape(B * K, D)  # (B*K, D)
    flat = flat - flat.mean(dim=0, keepdim=True)

    try:
        # torch.linalg.svdvals is faster than full SVD
        s = torch.linalg.svdvals(flat)  # (min(B*K, D),)
    except Exception:
        # Fallback for older PyTorch versions
        _, s, _ = torch.svd(flat)

    s = s / (s.sum() + 1e-8)
    entropy = -(s * (s + 1e-8).log()).sum()
    return float(entropy.exp().item())


def _cosine_sim_plans(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two latent plans (mean-pooled over K).

    Args:
        a: (B, K, D)
        b: (B, K, D)

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    a_mean = a.mean(dim=1)  # (B, D)
    b_mean = b.mean(dim=1)  # (B, D)
    return float(F.cosine_similarity(a_mean, b_mean, dim=-1).mean().item())


def _quality_score(
    metrics: Dict[str, float],
    detector: CollapseDetector,
) -> Tuple[float, float]:
    """Combine per-step metrics into quality and uncertainty scores in [0, 1].

    The quality score reflects how much of the original embedding structure is
    preserved at this step.  It is used to populate the degradation curve for
    Gate 2 validation.

    Weighting rationale (from feasibility paper):
    - Variance drop is the primary signal (50% weight).
    - Spread is the secondary signal (30% weight).
    - Cosine stagnation is a tertiary signal (20% weight).
    Effective rank is implicitly captured by the spread term.

    The uncertainty score mirrors VLJEPA.rollout(): 1/(1 + normalised_spread).

    Args:
        metrics: Dict from CollapseDetector.check().
        detector: Detector holding the baseline values.

    Returns:
        (quality, uncertainty) both in [0, 1].
    """
    baseline_var = detector.baseline_variance or 1.0
    baseline_spread = detector.baseline_spread or 1.0

    # Variance ratio (clamped to [0, 1])
    var_ratio = min(1.0, metrics["embedding_variance"] / max(baseline_var, 1e-8))

    # Spread ratio (clamped to [0, 1])
    spread_ratio = min(1.0, metrics["spread"] / max(baseline_spread, 1e-8))

    # Cosine stagnation penalty: quality = 1 when cos_sim = 0, 0 when cos_sim = 1
    cos_quality = 1.0 - max(0.0, metrics["cosine_sim_to_prev"])

    quality = 0.5 * var_ratio + 0.3 * spread_ratio + 0.2 * cos_quality

    # Uncertainty following VLJEPA.rollout() convention:
    # normalized_spread = spread / baseline; uncertainty = 1/(1 + normalized_spread)
    # Low spread → uncertainty → 1.0 (collapsed); high spread → uncertainty → 0.0
    normalized_spread = metrics["spread"] / max(baseline_spread, 1e-8)
    uncertainty = 1.0 / (1.0 + normalized_spread)

    return float(quality), float(uncertainty)


def _make_step(
    step_idx: int,
    plan: torch.Tensor,
    metrics: Dict[str, float],
    cosine_sim: float,
    quality: float,
    uncertainty: float,
    is_replan: bool,
    is_collapsed: bool,
    decoded_tokens: Optional[torch.Tensor],
    config: LatentReasoningConfig,
) -> ReasoningStep:
    """Construct a ReasoningStep from raw metrics."""
    return ReasoningStep(
        step_idx=step_idx,
        latent_plan=plan.detach(),
        embedding_variance=metrics["embedding_variance"],
        spread=metrics["spread"],
        effective_rank=metrics["effective_rank"],
        cosine_sim_to_prev=cosine_sim,
        uncertainty=uncertainty,
        quality=quality,
        is_replan=is_replan,
        is_collapsed=is_collapsed,
        is_degraded=config.quality_floor > 0.0 and quality < config.quality_floor,
        decoded_tokens=decoded_tokens,
    )


# =============================================================================
# Convenience factory
# =============================================================================

def make_reasoning_engine(
    dynamics_predictor: LatentDynamicsPredictor,
    max_steps: int = 32,
    replan_interval: int = 8,
    replan_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **config_kwargs,
) -> LatentReasoningEngine:
    """Convenience constructor for the common case.

    Args:
        dynamics_predictor: The LatentDynamicsPredictor from a VLJEPA model.
        max_steps: Maximum reasoning depth.
        replan_interval: Steps between replan checkpoints (0 = disabled).
        replan_fn: Optional callback for token-space re-anchoring.
        **config_kwargs: Additional keyword arguments forwarded to
            LatentReasoningConfig.

    Returns:
        Configured LatentReasoningEngine.
    """
    config = LatentReasoningConfig(
        max_steps=max_steps,
        replan_interval=replan_interval,
        **config_kwargs,
    )
    return LatentReasoningEngine(
        dynamics_predictor=dynamics_predictor,
        config=config,
        replan_fn=replan_fn,
    )
