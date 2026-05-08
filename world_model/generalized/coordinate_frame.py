"""Coordinate-space reference frame and novelty probe.

A non-LLM concrete implementation of the abstractions in
``world_model.novelty.core``. Designed for domain-agnostic settings
where each observation is a point in ℝ^d and there is no language.

Design choices, with rationale:

  - **Similarity** sim(a, b) = exp(-||a - b||^2 / (2 σ^2)) where σ is
    a per-frame bandwidth. This is a Gaussian kernel on coordinates;
    yields [0, 1], symmetric, monotone in distance. Replaces the
    sentence-embedding cosine.

  - **Stance** σ(c, claim) requires a polarity axis attached to each
    claim. The claim has an anchor point a and a unit vector u. For an
    observation x, project (x - a) onto u:
        proj = ((x - a) · u) / ||x - a||  ∈ [-1, 1]
    Combined with topical guard (sim(x, a) >= θ_topic):
        proj > θ_pro     -> PRO with confidence proj
        proj < -θ_pro    -> CON with confidence -proj
        else             -> NEUTRAL
    If sim(x, a) < θ_topic     -> NEUTRAL (orthogonal)
    Replaces the NLI model.

  - **Containment** contains(x) = True if x is within ε distance of
    any already-integrated observation in the frame. Replaces near-
    paraphrase detection.

  - **Adjacency** get_adjacent(x) = the k claims with highest sim(x, ·)
    that have not yet been visited. Replaces Wikidata graph traversal.

  - **Absorption** monotonically reduces novelty: an observation added
    to the integrated set will hereafter contains() at full similarity.

This satisfies all axioms of the formalization (A1-A8) when the
coordinate space is metric, the frame's bandwidth is finite, and at
least one claim exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple
import math

from ..novelty.core import (
    Claim,
    Focus,
    NoveltyProbe,
    ParseResult,
    ReferenceFrame,
    Stance,
    Termination,
)
from .observation import Observation


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


@dataclass
class CoordinateClaim(Claim):
    """A claim anchored at a point in coordinate space.

    The polarity axis ``u`` is a unit vector; observations on the +u
    side of the anchor are PRO, on the -u side are CON. Magnitude of
    projection = confidence.

    Inherits ``content`` (the claim string), ``depth``, ``stake`` from
    the base Claim. Adds anchor and polarity_axis.
    """

    anchor: Tuple[float, ...] = ()        # point in coordinate space
    polarity_axis: Tuple[float, ...] = () # unit vector defining PRO direction
    children: List["CoordinateClaim"] = field(default_factory=list)
    integrated_observation_ids: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # Ensure polarity axis is unit length if provided
        if self.polarity_axis:
            n = math.sqrt(sum(x * x for x in self.polarity_axis))
            if n > 0:
                self.polarity_axis = tuple(x / n for x in self.polarity_axis)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euclid_sq(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _gaussian_sim(a: Tuple[float, ...], b: Tuple[float, ...], bandwidth: float) -> float:
    """Gaussian kernel similarity in [0, 1]. 1 when a == b, decays with distance."""
    d_sq = _euclid_sq(a, b)
    return math.exp(-d_sq / (2.0 * bandwidth * bandwidth + 1e-12))


def _dimensional_overlap_sim(obs: Tuple[float, ...], anchor: Tuple[float, ...]) -> float:
    """Sparse-friendly similarity: fraction of anchor's nonzero dims
    that obs also has nonzero on. In [0, 1].

    For sparse high-dim observations (like SAT clauses with k literals
    out of N variables), this captures "does this obs touch my
    variable" without being washed out by zero-padding distance.
    """
    if not anchor:
        return 0.0
    anchor_nonzero = [i for i, a in enumerate(anchor) if abs(a) > 1e-6]
    if not anchor_nonzero:
        return 0.0
    overlap = sum(1 for i in anchor_nonzero if i < len(obs) and abs(obs[i]) > 1e-6)
    return overlap / len(anchor_nonzero)


def _project(x: Tuple[float, ...], anchor: Tuple[float, ...],
             axis: Tuple[float, ...]) -> float:
    """Signed projection of (x - anchor) onto axis. Returns scalar in
    units of axis length (axis assumed unit-length).
    """
    if not axis:
        return 0.0
    return sum((xi - ai) * ui for xi, ai, ui in zip(x, anchor, axis))


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


@dataclass
class CoordinateFrame(ReferenceFrame):
    """Reference frame over a coordinate space.

    Holds claims (anchored, with polarity axes) and an integrated set
    of observations. Domain-agnostic: works wherever observations and
    claim anchors live in the same ℝ^d.
    """

    claims: List[CoordinateClaim] = field(default_factory=list)
    integrated: dict[str, Observation] = field(default_factory=dict)
    bandwidth: float = 1.0
    topic_threshold: float = 0.3   # min sim to be on-topic
    pro_threshold: float = 0.05    # min |projection| to take a stance
    contain_distance: float = 0.05 # ||x - integrated|| < this -> contained
    use_dim_overlap: bool = True   # use dimensional-overlap gate instead of pure Gaussian
    _total_stake: float = 0.0

    # ----- ReferenceFrame interface -----

    def contains(self, content: Any) -> Tuple[bool, float]:
        coords = _coords_of(content)
        if coords is None:
            return False, 0.0
        # Check against integrated observations - exact-coord match only
        best_sim = 0.0
        for obs in self.integrated.values():
            if _euclid_sq(coords, obs.coords) < self.contain_distance ** 2:
                return True, 1.0
            if not self.use_dim_overlap:
                sim = _gaussian_sim(coords, obs.coords, self.bandwidth)
                if sim > best_sim:
                    best_sim = sim
        return False, best_sim

    def find_claims(self, content: Any) -> List[Tuple[Claim, float]]:
        """Return claims related to content, ordered by similarity.

        Only top-level claims (depth 0) are returned. Sub-claims exist
        for refinement of the tree under contention, but stance
        detection happens at the thesis level. Without this filter, an
        observation slightly off a sibling sub-claim's anchor gets
        classified against that sub-claim instead of the thesis.

        Sparse-friendly mode (use_dim_overlap=True): a claim is
        on-topic if the observation touches any of the claim's anchor
        dimensions. This handles SAT-style sparse observations where
        zero-padding makes Euclidean similarity collapse.
        """
        coords = _coords_of(content)
        if coords is None:
            return []
        results: List[Tuple[Claim, float]] = []
        for claim in self.claims:   # top-level only
            if self.use_dim_overlap:
                sim = _dimensional_overlap_sim(coords, claim.anchor)
                if sim > 0:
                    results.append((claim, sim))
            else:
                sim = _gaussian_sim(coords, claim.anchor, self.bandwidth)
                if sim >= self.topic_threshold:
                    results.append((claim, sim))
        results.sort(key=lambda x: -x[1])
        return results

    def detect_stance(self, content: Any, claim: Claim) -> Tuple[Stance, float]:
        coords = _coords_of(content)
        if coords is None or not isinstance(claim, CoordinateClaim):
            return Stance.NEUTRAL, 0.0

        if self.use_dim_overlap:
            # Topical guard: does obs touch any of claim's anchor dims?
            overlap = _dimensional_overlap_sim(coords, claim.anchor)
            if overlap == 0:
                return Stance.NEUTRAL, 0.05
            # Stance: for each dim where claim's polarity_axis is
            # nonzero, check sign agreement with obs.
            agree = 0
            disagree = 0
            for i, u in enumerate(claim.polarity_axis):
                if abs(u) < 1e-6 or i >= len(coords) or abs(coords[i]) < 1e-6:
                    continue
                if (u > 0) == (coords[i] > 0):
                    agree += 1
                else:
                    disagree += 1
            total = agree + disagree
            if total == 0:
                return Stance.NEUTRAL, 0.05
            net = (agree - disagree) / total   # in [-1, 1]
            if net > self.pro_threshold:
                return Stance.PRO, abs(net)
            elif net < -self.pro_threshold:
                return Stance.CON, abs(net)
            else:
                return Stance.NEUTRAL, 1.0 - abs(net)

        # Original Gaussian-distance mode
        sim = _gaussian_sim(coords, claim.anchor, self.bandwidth)
        if sim < self.topic_threshold:
            return Stance.NEUTRAL, max(sim, 0.05)
        proj = _project(coords, claim.anchor, claim.polarity_axis)
        dist = math.sqrt(_euclid_sq(coords, claim.anchor))
        if dist < 1e-9:
            return Stance.NEUTRAL, 0.0
        normalized = max(-1.0, min(1.0, proj / dist))
        if normalized > self.pro_threshold:
            return Stance.PRO, normalized
        elif normalized < -self.pro_threshold:
            return Stance.CON, -normalized
        else:
            return Stance.NEUTRAL, 1.0 - abs(normalized)

    def absorb(self, content: Any) -> "CoordinateFrame":
        """Return a new frame with content integrated.

        Frames in the abstract API are immutable; absorption produces
        a new frame. We share the claim list (claims are not mutated
        by absorb) but build a new integrated dict.
        """
        if not isinstance(content, Observation):
            return self
        new_integrated = dict(self.integrated)
        new_integrated[content.id] = content
        return CoordinateFrame(
            claims=self.claims,
            integrated=new_integrated,
            bandwidth=self.bandwidth,
            topic_threshold=self.topic_threshold,
            pro_threshold=self.pro_threshold,
            contain_distance=self.contain_distance,
            _total_stake=self._total_stake,
        )

    def get_adjacent(self, content: Any) -> List[Any]:
        """Return claims adjacent to content, ordered by similarity.

        Adjacency in coordinate space = nearest claims by Gaussian
        similarity, excluding any whose anchor matches content exactly.
        """
        coords = _coords_of(content)
        if coords is None:
            return []
        scored: List[Tuple[float, CoordinateClaim]] = []
        for claim in self._all_claims():
            if _euclid_sq(coords, claim.anchor) < 1e-12:
                continue
            sim = _gaussian_sim(coords, claim.anchor, self.bandwidth)
            scored.append((sim, claim))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    @property
    def total_stake(self) -> float:
        if self._total_stake > 0:
            return self._total_stake
        # Recompute from claims
        total = sum(c.stake for c in self._all_claims())
        return max(total, 0.01)

    # ----- Helpers -----

    def _all_claims(self) -> List[CoordinateClaim]:
        out: List[CoordinateClaim] = []
        stack = list(self.claims)
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(c.children)
        return out


def _coords_of(content: Any) -> Optional[Tuple[float, ...]]:
    """Best-effort extraction of coordinates from content."""
    if isinstance(content, Observation):
        return content.coords
    if isinstance(content, CoordinateClaim):
        return content.anchor
    if isinstance(content, tuple) and all(isinstance(x, (int, float)) for x in content):
        return tuple(float(x) for x in content)
    return None


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class CoordinateProbe(NoveltyProbe):
    """Novelty probe over a CoordinateFrame.

    Walks the claim hierarchy by adjacency in coordinate space.
    Terminates with:
      - INTEGRATED       if frame.contains(content) at high similarity
      - CONTRADICTS_ROOT if a CON stance is detected on a high-stake claim
      - DISRUPTS         if a CON stance affects more than a stake threshold
      - ORTHOGONAL       if no claim is on-topic at any visited point
      - MAX_ITERATIONS   if the loop runs out
    """

    def __init__(self, max_iterations: int = 10,
                 disruption_threshold: float = 0.5):
        super().__init__(max_iterations=max_iterations)
        self.disruption_threshold = disruption_threshold

    def fetch(self, focus: Focus, frame: ReferenceFrame) -> Any:
        # No external lookup -- the focus content carries everything.
        return focus.content

    def parse(self, data: Any, focus: Focus, frame: ReferenceFrame) -> ParseResult:
        # Containment check
        contained, sim = frame.contains(data)
        if contained:
            return ParseResult.terminate(
                Termination.INTEGRATED,
                similarity_to_frame=sim,
            )

        # Find related claims
        related = frame.find_claims(data)
        if not related:
            # Try expanding to adjacent
            adj = frame.get_adjacent(data)
            if not adj:
                return ParseResult.terminate(
                    Termination.ORTHOGONAL,
                    similarity_to_frame=0.0,
                )
            # Move to nearest adjacent
            return ParseResult.continue_to(
                focus.expand_to(adj[0], via="adjacent"),
                similarity_to_frame=0.0,
            )

        # Check stances on top related claims
        worst_con: Optional[Tuple[Stance, float, CoordinateClaim]] = None
        total_con_stake = 0.0
        for claim, _ in related[:3]:
            stance, conf = frame.detect_stance(data, claim)
            if stance == Stance.CON and conf >= 0.5:
                if worst_con is None or claim.depth < worst_con[2].depth:
                    worst_con = (stance, conf, claim)
                total_con_stake += claim.stake

        if worst_con is not None:
            _, _, conflict_claim = worst_con
            # Disrupts if affects significant stake
            if total_con_stake / frame.total_stake >= self.disruption_threshold:
                return ParseResult.terminate(
                    Termination.DISRUPTS,
                    stake_affected=total_con_stake,
                    contradiction_depth=conflict_claim.depth,
                )
            # Otherwise contradicts at the depth of the conflicting claim
            return ParseResult.terminate(
                Termination.CONTRADICTS_ROOT,
                stake_affected=total_con_stake,
                contradiction_depth=conflict_claim.depth,
            )

        # No contradiction; check if any PRO -> integrate cleanly.
        # PRO match = obs fits a known claim's positive direction.
        # Terminate as INTEGRATED.
        for claim, _ in related[:3]:
            stance, conf = frame.detect_stance(data, claim)
            if stance == Stance.PRO and conf >= 0.3:
                return ParseResult.terminate(
                    Termination.INTEGRATED,
                    similarity_to_frame=conf,
                )

        # Neutral relations only -- terminate as INTEGRATED at low resistance
        # (we found relevant claims but no conflict; the observation just sits).
        return ParseResult.terminate(
            Termination.INTEGRATED,
            similarity_to_frame=related[0][1] if related else 0.0,
        )

    def _get_max_depth(self, frame: ReferenceFrame) -> int:
        if isinstance(frame, CoordinateFrame):
            depths = [c.depth for c in frame._all_claims()]
            return max(depths, default=1)
        return super()._get_max_depth(frame)
