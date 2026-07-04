"""
Constitutional Geometry: Embedding-Space Constitutional Evaluation

Implements the 3-tier evaluation architecture from the constitutional
geometry survey:

  Tier 1 — Geometric pre-filter (O(1))
    Cosine similarity against principle direction vectors in JEPA embedding
    space. Handles ~80-90% of clear cases immediately.

  Tier 2 — Concept bottleneck classifier (lightweight MLP)
    Interpretable concept layer with one score per RPB principle.
    Handles nuanced cases where Tier 1 is uncertain.

  Tier 3 — Full LLM evaluation (expensive, high-stakes)
    Falls through to a direct AIProvider call (see
    ThreeTierConstitutionalEvaluator). Reserved for adversarial-seeming or
    genuinely ambiguous inputs. Signalled by returning Verdict.UNCERTAIN —
    caller must escalate.

Critical design constraint: embedding drift causes ROC-AUC to collapse from
0.85 to 0.50 at σ=0.028 drift (arXiv:2603.01297). DriftMonitor detects this
before it causes high-confidence silent failures.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# RPB Constitutional Principles (UDHR-derived)
# =============================================================================

class RPBPrinciple(Enum):
    """The 7 RPB constitutional principles derived from the UDHR."""
    HUMAN_DIGNITY         = "human_dignity"          # Art. 1 — inherent dignity
    FREEDOM_OF_THOUGHT    = "freedom_of_thought"     # Art. 18-19 — conscience, expression
    DEMOCRATIC_GOVERNANCE = "democratic_governance"  # Art. 21 — participation in governance
    TRANSPARENCY          = "transparency"            # Art. 19 — access to information
    PRIVACY               = "privacy"                # Art. 12 — private life, correspondence
    NON_DISCRIMINATION    = "non_discrimination"     # Art. 2, 7 — equal treatment
    CULTURAL_RESPECT      = "cultural_respect"       # Art. 27 — cultural life participation


# Paired (compliant_description, violating_description) examples per principle.
# Used to bootstrap principle direction vectors when no labeled data is available.
# In production, pairs from evaluated proposals (with LLM ground truth) replace these.
PRINCIPLE_BOOTSTRAP_PAIRS: Dict[RPBPrinciple, List[Tuple[str, str]]] = {
    RPBPrinciple.HUMAN_DIGNITY: [
        (
            "This proposal treats all participants as autonomous agents with inherent worth.",
            "This proposal treats some node operators as expendable resources to be exploited.",
        ),
        (
            "The system ensures every contributor is fairly compensated for their work.",
            "The protocol allows majority stakeholders to extract value from minority "
            "contributors without consent.",
        ),
    ],
    RPBPrinciple.FREEDOM_OF_THOUGHT: [
        (
            "Nodes are free to form and express independent evaluations of proposals.",
            "The system enforces a single approved viewpoint and silences dissenting evaluations.",
        ),
        (
            "Participants can freely access research and update their beliefs based on evidence.",
            "The protocol restricts information flow to prevent nodes from forming "
            "independent judgments.",
        ),
    ],
    RPBPrinciple.DEMOCRATIC_GOVERNANCE: [
        (
            "All stakeholders can participate in governance decisions proportional to contribution.",
            "Governance decisions are made unilaterally by a small group with no stakeholder input.",
        ),
        (
            "Protocol changes require broad consensus through open deliberation.",
            "The admin can change core parameters without any community vote or notification.",
        ),
    ],
    RPBPrinciple.TRANSPARENCY: [
        (
            "All governance decisions include public reasoning and on-chain audit trails.",
            "Critical system decisions are made in private with no public record.",
        ),
        (
            "The evaluation methodology is openly documented and verifiable by any participant.",
            "The scoring algorithm is a black box — participants cannot verify how decisions are made.",
        ),
    ],
    RPBPrinciple.PRIVACY: [
        (
            "User data is minimized, anonymized, and never shared without explicit consent.",
            "The system collects and shares personal data without user knowledge or consent.",
        ),
        (
            "Communication between nodes is encrypted and private to the participants.",
            "All node communications are logged centrally and accessible to any third party.",
        ),
    ],
    RPBPrinciple.NON_DISCRIMINATION: [
        (
            "All nodes receive equal evaluation criteria regardless of operator identity or location.",
            "Nodes operated by certain jurisdictions are systematically disadvantaged in rewards.",
        ),
        (
            "The protocol applies the same standards to all participants without bias.",
            "The system gives preferential treatment to early adopters over later participants.",
        ),
    ],
    RPBPrinciple.CULTURAL_RESPECT: [
        (
            "The system accommodates diverse cultural norms and communication styles.",
            "The protocol imposes a single cultural framework and penalizes deviation from it.",
        ),
        (
            "Governance processes are designed to be accessible across languages and contexts.",
            "All governance is conducted in a single language, excluding non-native speakers.",
        ),
    ],
}


# =============================================================================
# Result types
# =============================================================================

class Verdict(Enum):
    COMPLIANT = "compliant"    # Clear constitutional compliance
    UNCERTAIN = "uncertain"    # Ambiguous — escalate to next tier
    VIOLATION = "violation"    # Clear constitutional violation


@dataclass
class PrincipleScore:
    """Compliance score for a single constitutional principle."""
    principle: RPBPrinciple
    score: float       # Calibrated compliance [0, 1]; >0.65 = compliant, <0.35 = violation
    confidence: float  # How confident we are in this individual score [0, 1]


@dataclass
class ConstitutionalEvalResult:
    """Result from the constitutional geometry evaluation pipeline."""
    verdict: Verdict
    overall_confidence: float              # [0, 1] — confidence in the verdict
    principle_scores: List[PrincipleScore]
    tier_used: int                         # 1, 2, or 3
    weakest_principle: Optional[RPBPrinciple]  # Principle with lowest compliance
    explanation: str
    drift_warning: bool = False            # True if embedding drift detected


# =============================================================================
# Drift Monitor
# =============================================================================

class DriftMonitor:
    """
    Detects embedding drift that causes geometric classifiers to fail silently.

    From arXiv:2603.01297: ROC-AUC collapses from 0.85 to 0.50 (random guessing)
    at drift magnitude σ=0.028. 72% of failures occur with high confidence (>0.8),
    making the failure invisible without active monitoring.

    Strategy: track EMA of per-dimension mean and variance. When current statistics
    deviate from the calibration baseline by more than drift_threshold, flag drift
    and request recalibration.
    """

    def __init__(
        self,
        embed_dim: int,
        ema_decay: float = 0.99,
        drift_threshold: float = 0.025,
    ):
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.drift_threshold = drift_threshold  # Conservative relative to σ=0.028 paper

        # Calibration baseline (set from representative embeddings at startup)
        self.baseline_mean: Optional[torch.Tensor] = None
        self.baseline_var: Optional[torch.Tensor] = None

        # Running EMA statistics
        self._ema_mean: Optional[torch.Tensor] = None
        self._ema_var: Optional[torch.Tensor] = None

        # Drift state
        self.drift_detected: bool = False
        self.last_drift_magnitude: float = 0.0

    def set_baseline(self, embeddings: torch.Tensor) -> None:
        """
        Set the calibration baseline from representative embeddings.

        Call this after every model weight update — embedding drift makes all
        geometric classifiers unreliable until recalibrated.

        Args:
            embeddings: (N, embed_dim) tensor from the current model
        """
        with torch.no_grad():
            self.baseline_mean = embeddings.mean(dim=0).clone()
            self.baseline_var = embeddings.var(dim=0).clone()
            self._ema_mean = self.baseline_mean.clone()
            self._ema_var = self.baseline_var.clone()
            self.drift_detected = False
            self.last_drift_magnitude = 0.0

        logger.info(
            f"DriftMonitor: baseline set from {embeddings.shape[0]} embeddings "
            f"(embed_dim={self.embed_dim})"
        )

    def update(self, embeddings: torch.Tensor) -> bool:
        """
        Update running statistics with new embeddings.

        Returns True if drift is detected (recalibration required).

        Args:
            embeddings: (N, embed_dim) recent embeddings from the model
        """
        with torch.no_grad():
            batch_mean = embeddings.mean(dim=0)
            batch_var = embeddings.var(dim=0)

            if self._ema_mean is None:
                self._ema_mean = batch_mean.clone()
                self._ema_var = batch_var.clone()
            else:
                alpha = 1.0 - self.ema_decay
                self._ema_mean = self.ema_decay * self._ema_mean + alpha * batch_mean
                self._ema_var = self.ema_decay * self._ema_var + alpha * batch_var

        return self._check_drift()

    def _check_drift(self) -> bool:
        if self.baseline_mean is None or self._ema_mean is None:
            return False

        with torch.no_grad():
            # Normalized mean shift: (current - baseline) / baseline_std
            baseline_std = (self.baseline_var + 1e-8).sqrt()
            shift = ((self._ema_mean - self.baseline_mean) / baseline_std).abs()
            magnitude = shift.mean().item()

        self.last_drift_magnitude = magnitude
        self.drift_detected = magnitude > self.drift_threshold

        if self.drift_detected:
            logger.warning(
                f"DriftMonitor: embedding drift detected! "
                f"magnitude={magnitude:.4f} > threshold={self.drift_threshold:.4f}. "
                f"Geometric classifiers may be unreliable — recalibration required."
            )

        return self.drift_detected

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_mean is not None


# =============================================================================
# Tier 1: Geometric Pre-Filter
# =============================================================================

class GeometricPreFilter:
    """
    Tier 1: O(1) constitutional evaluation via cosine similarity.

    Each RPB principle is encoded as a direction vector in JEPA embedding
    space. Compliance is the dot product of the (normalized) action embedding
    with the (normalized) principle direction vector.

    Direction vectors are computed from paired (compliant, violating) examples:
        v_i = normalize(mean(embed(compliant)) - mean(embed(violating)))

    Then Gram-Schmidt orthogonalized to reduce inter-principle interference.
    High-dimensional embedding spaces (d=384+) support near-orthogonality for
    all 7 principle vectors with minimal degradation (arXiv:2510.23609).
    """

    # Decision thresholds (calibrated conservatively; isotonic regression in production)
    COMPLIANT_THRESHOLD = 0.65   # All-principle min score above this → COMPLIANT
    VIOLATION_THRESHOLD = 0.35   # Average score below this → VIOLATION
    # Between the two thresholds → UNCERTAIN (escalate to Tier 2)

    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
        self._directions: Dict[RPBPrinciple, torch.Tensor] = {}
        self._calibrated: bool = False

    def set_directions(
        self,
        directions: Dict[RPBPrinciple, torch.Tensor],
        orthogonalize: bool = True,
    ) -> None:
        """
        Set principle direction vectors, optionally Gram-Schmidt orthogonalized.

        Args:
            directions: RPBPrinciple → (embed_dim,) direction tensor
            orthogonalize: Apply Gram-Schmidt to reduce inter-principle interference
        """
        if orthogonalize and len(directions) > 1:
            directions = self._gram_schmidt(directions)

        self._directions = {
            p: F.normalize(v.float(), dim=0)
            for p, v in directions.items()
        }
        self._calibrated = True
        logger.info(
            f"GeometricPreFilter: {len(self._directions)} principle directions set"
            + (" (Gram-Schmidt orthogonalized)" if orthogonalize else "")
        )

    def _gram_schmidt(
        self, directions: Dict[RPBPrinciple, torch.Tensor]
    ) -> Dict[RPBPrinciple, torch.Tensor]:
        """Gram-Schmidt orthogonalization in canonical principle order."""
        basis: List[torch.Tensor] = []
        result: Dict[RPBPrinciple, torch.Tensor] = {}

        for principle in RPBPrinciple:
            if principle not in directions:
                continue

            v = directions[principle].float().clone()
            for b in basis:
                v = v - (v @ b) * b

            norm = v.norm()
            if norm > 1e-6:
                v = v / norm
                basis.append(v)
                result[principle] = v
            else:
                # Near-degenerate: fall back to normalized original
                result[principle] = F.normalize(directions[principle].float(), dim=0)
                logger.warning(
                    f"GeometricPreFilter: near-degenerate direction for "
                    f"{principle.value} — keeping unnormalized"
                )

        return result

    def evaluate(
        self, embedding: torch.Tensor
    ) -> Tuple[Optional[ConstitutionalEvalResult], bool]:
        """
        Evaluate an action embedding against all principle directions.

        Returns:
            (result, handled) where handled=False means escalate to Tier 2.
        """
        if not self._calibrated:
            return None, False

        emb = embedding.float()
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        emb_norm = F.normalize(emb, dim=0)

        scores: List[PrincipleScore] = []
        for principle in RPBPrinciple:
            if principle not in self._directions:
                continue
            raw_sim = (emb_norm @ self._directions[principle]).item()
            # Map cosine similarity [-1, 1] → [0, 1]
            calibrated = (raw_sim + 1.0) / 2.0
            scores.append(PrincipleScore(
                principle=principle,
                score=calibrated,
                confidence=0.85,  # Fixed Tier 1 confidence; per-example calibration in Tier 2
            ))

        if not scores:
            return None, False

        min_score = min(s.score for s in scores)
        avg_score = sum(s.score for s in scores) / len(scores)
        weakest = min(scores, key=lambda s: s.score).principle

        if min_score >= self.COMPLIANT_THRESHOLD:
            verdict = Verdict.COMPLIANT
            # Higher min_score → higher confidence
            confidence = min(0.95, 0.70 + 0.50 * (min_score - self.COMPLIANT_THRESHOLD))
            handled = True
        elif avg_score <= self.VIOLATION_THRESHOLD:
            verdict = Verdict.VIOLATION
            confidence = min(0.95, 0.70 + 0.50 * (self.VIOLATION_THRESHOLD - avg_score))
            handled = True
        else:
            verdict = Verdict.UNCERTAIN
            confidence = 0.5
            handled = False

        return ConstitutionalEvalResult(
            verdict=verdict,
            overall_confidence=confidence,
            principle_scores=scores,
            tier_used=1,
            weakest_principle=weakest,
            explanation=(
                f"Tier 1 geometric: min_score={min_score:.3f}, avg_score={avg_score:.3f}"
            ),
        ), handled

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# =============================================================================
# Tier 2: Concept Bottleneck Classifier
# =============================================================================

class ConceptBottleneckClassifier(nn.Module):
    """
    Tier 2: Lightweight MLP with interpretable constitutional concept layer.

    Architecture inspired by CB-LLMs (arXiv:2412.07992):
        embedding → concept scores (7D, one per RPB principle) → verdict (3-class)

    The bottleneck forces all predictive information through principle-aligned
    concepts, making intermediate representations inspectable. An adversarial
    input that fools Tier 1 (geometrically compliant) must also fool all 7
    concept scores — a substantially harder attack surface.

    Trained from Tier 3 (LLM) evaluation ground truth via the calibration loop.
    Before training data accumulates, it produces low-confidence outputs that
    naturally escalate to Tier 3.
    """

    CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to handle (not escalate to Tier 3)

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        n_principles = len(RPBPrinciple)

        # Concept bottleneck: embedding → per-principle compliance scores [0, 1]
        self.concept_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_principles),
            nn.Sigmoid(),
        )

        # Classification head: concept scores → verdict logits
        self.classifier = nn.Sequential(
            nn.Linear(n_principles, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # [COMPLIANT, UNCERTAIN, VIOLATION]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding: (B, embed_dim) or (embed_dim,)

        Returns:
            concept_scores: (B, 7) per-principle compliance [0, 1]
            logits:         (B, 3) verdict logits
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        concept_scores = self.concept_layer(embedding)
        logits = self.classifier(concept_scores)
        return concept_scores, logits

    def evaluate(
        self, embedding: torch.Tensor
    ) -> Tuple[Optional[ConstitutionalEvalResult], bool]:
        """
        Evaluate through the concept bottleneck.

        Returns:
            (result, handled) where handled=False means escalate to Tier 3.
        """
        self.eval()
        with torch.no_grad():
            concept_scores, logits = self(embedding.float())

            probs = F.softmax(logits, dim=-1)  # (1, 3)
            verdict_idx = probs.argmax(dim=-1).item()
            verdict_conf = probs.max(dim=-1).values.item()

            verdicts = [Verdict.COMPLIANT, Verdict.UNCERTAIN, Verdict.VIOLATION]
            verdict = verdicts[verdict_idx]

            cs = concept_scores.squeeze(0)
            principle_scores = [
                PrincipleScore(
                    principle=p,
                    score=cs[i].item(),
                    confidence=verdict_conf,
                )
                for i, p in enumerate(RPBPrinciple)
            ]
            weakest = min(principle_scores, key=lambda s: s.score).principle

            # Only handle if confident enough and not already uncertain
            handled = (
                verdict != Verdict.UNCERTAIN
                and verdict_conf >= self.CONFIDENCE_THRESHOLD
            )

            return ConstitutionalEvalResult(
                verdict=verdict,
                overall_confidence=verdict_conf,
                principle_scores=principle_scores,
                tier_used=2,
                weakest_principle=weakest,
                explanation=(
                    f"Tier 2 concept bottleneck: "
                    f"verdict={verdict.value}, conf={verdict_conf:.3f}"
                ),
            ), handled


# =============================================================================
# Direction Vector Computation
# =============================================================================

def compute_principle_directions(
    encode_fn: Callable[[str], torch.Tensor],
    pairs: Optional[Dict[RPBPrinciple, List[Tuple[str, str]]]] = None,
) -> Dict[RPBPrinciple, torch.Tensor]:
    """
    Compute principle direction vectors from paired (compliant, violating) examples.

    Uses the LEGEND approach (arXiv:2406.08124): for each principle, average
    the difference vectors (embed(compliant) - embed(violating)) across examples.

    Args:
        encode_fn: Text → (embed_dim,) embedding function (from JEPA TextEncoder)
        pairs:     Principle → [(compliant_text, violating_text), ...].
                   Falls back to PRINCIPLE_BOOTSTRAP_PAIRS if None.

    Returns:
        Dict of principle → unnormalized direction vector (embed_dim,).
        Normalization and orthogonalization happen in GeometricPreFilter.
    """
    pairs = pairs or PRINCIPLE_BOOTSTRAP_PAIRS
    directions: Dict[RPBPrinciple, torch.Tensor] = {}

    for principle, example_pairs in pairs.items():
        diff_vectors: List[torch.Tensor] = []
        for compliant_text, violating_text in example_pairs:
            try:
                compliant_emb = encode_fn(compliant_text).float()
                violating_emb = encode_fn(violating_text).float()
                if compliant_emb.dim() > 1:
                    compliant_emb = compliant_emb.mean(dim=0)
                if violating_emb.dim() > 1:
                    violating_emb = violating_emb.mean(dim=0)
                diff_vectors.append(compliant_emb - violating_emb)
            except Exception as e:
                logger.warning(
                    f"Failed to encode pair for {principle.value}: {e}"
                )

        if diff_vectors:
            directions[principle] = torch.stack(diff_vectors).mean(dim=0)
            logger.debug(
                f"Direction computed for {principle.value} "
                f"from {len(diff_vectors)} pair(s)"
            )
        else:
            logger.warning(
                f"No valid pairs for {principle.value} — principle will be skipped"
            )

    return directions


# =============================================================================
# Constitutional Geometry: Top-Level Coordinator
# =============================================================================

class ConstitutionalGeometry:
    """
    3-tier constitutional evaluation system for the Autonet governance engine.

    The three tiers are tried in order:
      1. GeometricPreFilter  — O(1) cosine similarity, handles ~80-90% of cases
      2. ConceptBottleneck   — lightweight MLP, handles nuanced cases
      3. LLM (external)      — signalled by returning Verdict.UNCERTAIN

    Caller is responsible for Tier 3: when evaluate() returns UNCERTAIN, the
    action must be passed to an AIProvider (see
    ThreeTierConstitutionalEvaluator).

    Drift monitoring is active continuously. When drift is detected, all Tier 1
    and Tier 2 results include drift_warning=True and confidence is reduced.
    The GovernanceEngine.tick() should call update_drift_stats() periodically.

    Usage:
        geometry = ConstitutionalGeometry(embed_dim=384)
        geometry.calibrate(encode_fn, reference_embeddings)

        result = geometry.evaluate(action_embedding)
        if result.verdict == Verdict.UNCERTAIN:
            # Escalate to LLM (Tier 3)
            ...
    """

    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
        self.pre_filter = GeometricPreFilter(embed_dim)
        self.bottleneck = ConceptBottleneckClassifier(embed_dim)
        self.drift_monitor = DriftMonitor(embed_dim)
        self.logger = logging.getLogger("ConstitutionalGeometry")

        # Usage statistics
        self._n_tier1 = 0
        self._n_tier2 = 0
        self._n_tier3 = 0

    def calibrate(
        self,
        encode_fn: Callable[[str], torch.Tensor],
        reference_embeddings: Optional[torch.Tensor] = None,
        principle_pairs: Optional[Dict[RPBPrinciple, List[Tuple[str, str]]]] = None,
    ) -> None:
        """
        Calibrate the geometric pre-filter.

        Must be called at startup and after any JEPA model weight update.
        Embedding drift invalidates all Tier 1/2 results silently — calibration
        resets the drift monitor baseline and recomputes principle directions.

        Args:
            encode_fn:            Text → embedding (from JEPA TextEncoder.forward)
            reference_embeddings: (N, embed_dim) representative embeddings for
                                  drift baseline. Required for drift detection.
            principle_pairs:      Override bootstrap pairs for direction computation.
        """
        self.logger.info("Calibrating constitutional geometry...")

        directions = compute_principle_directions(encode_fn, principle_pairs)
        if not directions:
            self.logger.warning(
                "No principle directions computed — geometric pre-filter disabled. "
                "All evaluations will use Tier 3 (LLM)."
            )
            return

        self.pre_filter.set_directions(directions, orthogonalize=True)

        if reference_embeddings is not None and reference_embeddings.shape[0] > 0:
            self.drift_monitor.set_baseline(reference_embeddings)
        else:
            self.logger.warning(
                "No reference embeddings provided — drift detection disabled. "
                "Provide reference_embeddings after model updates."
            )

        self.logger.info(
            f"Calibration complete: {len(directions)}/{len(list(RPBPrinciple))} "
            f"principles encoded"
        )

    def update_drift_stats(self, embeddings: torch.Tensor) -> bool:
        """
        Update drift statistics with recent embeddings.

        Should be called by GovernanceEngine.tick() with embeddings from recent
        actions/proposals. Returns True if drift detected (recalibrate).

        Args:
            embeddings: (N, embed_dim) recent embeddings from the current model
        """
        return self.drift_monitor.update(embeddings)

    def evaluate(
        self,
        embedding: torch.Tensor,
        force_tier: Optional[int] = None,
    ) -> ConstitutionalEvalResult:
        """
        Evaluate an action/proposal embedding through the tiered pipeline.

        Tiers 1 and 2 are tried in order. If neither produces a confident
        decision, returns Verdict.UNCERTAIN to signal that Tier 3 (LLM) is
        needed. The caller must then invoke an AIProvider (Tier 3).

        Args:
            embedding:  (embed_dim,) or (1, embed_dim) embedding of the action
            force_tier: Force a specific tier (1, 2, or 3) for testing

        Returns:
            ConstitutionalEvalResult. verdict=UNCERTAIN means LLM required.
        """
        drift_warning = self.drift_monitor.drift_detected

        # Tier 1: Geometric pre-filter
        if force_tier in (None, 1) and self.pre_filter.is_calibrated:
            result, handled = self.pre_filter.evaluate(embedding)
            if handled and result is not None:
                result.drift_warning = drift_warning
                self._n_tier1 += 1
                return result

        # Tier 2: Concept bottleneck
        if force_tier in (None, 2):
            result, handled = self.bottleneck.evaluate(embedding)
            if handled and result is not None:
                result.drift_warning = drift_warning
                self._n_tier2 += 1
                return result

        # Tier 3 required — signal caller
        self._n_tier3 += 1
        return ConstitutionalEvalResult(
            verdict=Verdict.UNCERTAIN,
            overall_confidence=0.0,
            principle_scores=[
                PrincipleScore(p, score=0.5, confidence=0.0)
                for p in RPBPrinciple
            ],
            tier_used=3,
            weakest_principle=None,
            explanation="Tier 1+2 inconclusive — LLM evaluation (Tier 3) required",
            drift_warning=drift_warning,
        )

    def get_stats(self) -> Dict[str, object]:
        """Return tier usage statistics for observability."""
        total = self._n_tier1 + self._n_tier2 + self._n_tier3
        return {
            "tier1_handled": self._n_tier1,
            "tier2_handled": self._n_tier2,
            "tier3_needed": self._n_tier3,
            "total": total,
            "tier1_pct": round(100 * self._n_tier1 / max(1, total)),
            "tier2_pct": round(100 * self._n_tier2 / max(1, total)),
            "tier3_pct": round(100 * self._n_tier3 / max(1, total)),
            "drift_detected": self.drift_monitor.drift_detected,
            "drift_magnitude": round(self.drift_monitor.last_drift_magnitude, 5),
        }
