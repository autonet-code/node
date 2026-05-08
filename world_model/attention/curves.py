"""
Novelty-attention curves: how surprise reshapes allocation.

A curve maps a novelty score in [0, 1] to a shift in tendency allocations.
Surprising input pulls allocation toward the curiosity-shaped tendencies
(those whose role is exploration) and away from the comfort-shaped ones
(those whose role is conservation).

This module is intentionally minimal. The shape of the curve is a
sigmoid by default; presets differ in steepness and asymmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional


def sigmoid(x: float, steepness: float = 8.0, midpoint: float = 0.5) -> float:
    """Standard logistic sigmoid centered at ``midpoint``."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


@dataclass
class NoveltyAttentionCurve:
    """Maps novelty in [0,1] to a fraction-of-shift toward exploratory tendencies.

    The curve is a sigmoid by default. ``shift_at(novelty)`` returns a
    number in [0, max_shift] indicating how much of the allocation mass
    should rebalance toward the tendency tagged "exploratory" when the
    novelty score is what it is.

    The tagging is by tendency *id* (string) rather than a fixed enum.
    Callers configure which tendencies count as exploratory or
    conservative for the world they're modelling.
    """

    name: str = "balanced"
    steepness: float = 8.0
    midpoint: float = 0.5
    max_shift: float = 0.30
    exploratory_ids: tuple[str, ...] = ("curiosity",)
    conservative_ids: tuple[str, ...] = ("comfort", "survival")

    def shift_at(self, novelty: float) -> float:
        """Allocation mass to redistribute toward exploratory tendencies."""
        novelty = max(0.0, min(1.0, novelty))
        return self.max_shift * sigmoid(novelty, self.steepness, self.midpoint)


# Presets keyed by personality archetype. Domain-agnostic in shape; the
# tendency-id tags are defaults that callers may override.
EXPLORER_CURVE = NoveltyAttentionCurve(
    name="explorer",
    steepness=10.0,
    midpoint=0.35,    # surprises trigger sooner
    max_shift=0.45,
)

BALANCED_CURVE = NoveltyAttentionCurve(
    name="balanced",
    steepness=8.0,
    midpoint=0.5,
    max_shift=0.30,
)

CONSERVATIVE_CURVE = NoveltyAttentionCurve(
    name="conservative",
    steepness=6.0,
    midpoint=0.65,    # only big surprises move the needle
    max_shift=0.15,
)


@dataclass
class AttentionState:
    """Live attention state: a tendency set viewed through a novelty curve.

    The "effective" allocations are the base allocations plus the curve's
    response to the current novelty score. Effective allocations always
    sum to 1.0 (we redistribute, not inflate).
    """

    agent_set: "object"          # Forward-reference to TendencySet (avoid cyclic import)
    curve: NoveltyAttentionCurve = field(default_factory=lambda: BALANCED_CURVE)
    current_novelty: float = 0.0
    effective_allocations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._recompute()

    def update_novelty(self, novelty: float) -> None:
        """Set the current novelty score and recompute effective allocations."""
        self.current_novelty = max(0.0, min(1.0, novelty))
        self._recompute()

    @property
    def dominant_tendency(self) -> Optional[str]:
        """Id of the tendency with the highest effective allocation."""
        if not self.effective_allocations:
            return None
        return max(self.effective_allocations, key=self.effective_allocations.get)

    def describe(self) -> str:
        if not self.effective_allocations:
            return "AttentionState(empty)"
        top = sorted(self.effective_allocations.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{k}={v:.0%}" for k, v in top)
        return (
            f"AttentionState(novelty={self.current_novelty:.2f}, curve={self.curve.name}, "
            f"top={top_str})"
        )

    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        base = self._base_allocations()
        if not base:
            self.effective_allocations = {}
            return

        shift = self.curve.shift_at(self.current_novelty)
        explor = [tid for tid in base if tid in self.curve.exploratory_ids]
        conserv = [tid for tid in base if tid in self.curve.conservative_ids]

        if not explor or not conserv:
            self.effective_allocations = dict(base)
            return

        # Pull `shift` mass from conservative tendencies, distribute across exploratory.
        eff = dict(base)
        pulled = 0.0
        conserv_total = sum(base[t] for t in conserv)
        if conserv_total > 0:
            for t in conserv:
                taken = base[t] * (shift if conserv_total >= shift else 1.0)
                # Cap so we don't pull more than the tendency holds
                taken = min(taken, eff[t])
                eff[t] -= taken
                pulled += taken
        if pulled > 0 and explor:
            per_explor = pulled / len(explor)
            for t in explor:
                eff[t] += per_explor

        # Renormalize for safety
        total = sum(eff.values())
        if total > 0:
            eff = {k: v / total for k, v in eff.items()}
        self.effective_allocations = eff

    def _base_allocations(self) -> dict[str, float]:
        """Get base allocations from the agent_set (id -> fraction).

        Tolerant of either the new TendencySet shape (``.all()`` returns
        objects with ``.id`` and ``.allocation``) or older AgentSet shape.
        """
        result: dict[str, float] = {}
        try:
            members: Iterable = self.agent_set.all()
        except AttributeError:
            return result
        for m in members:
            tid = getattr(m, "id", None)
            if tid is None and hasattr(m, "tendency"):
                tend = m.tendency
                tid = getattr(tend, "value", str(tend))
            alloc = float(getattr(m, "allocation", 0.0))
            if tid is not None:
                result[tid] = alloc
        return result
