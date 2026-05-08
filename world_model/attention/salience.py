"""
Salience - Value Assignment and Reinforcement

Salience is the mechanism by which symbols acquire and maintain
attention-worthiness. It answers: "Why should I pay attention to this?"

From the original insight:
    "Every time an input stream produces a pattern, it is assigned
    a set of features and, if deemed relevant, it is appended to
    this sequence."

Salience can come from:
- Survival/utility value (hardcoded objectives)
- Novelty (deviation from expectations)
- Repetition (reinforcement through recurrence)
- Convergence (multiple sources agreeing)
- Association (connection to high-salience items)

This module provides pluggable salience functions that can integrate
with external systems (novelty scores, world model allocations, etc.)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from datetime import datetime
import math

from .sequence import Symbol

# Try to import world_model for integration
try:
    from world_model import AgentSet, Tendency
    _HAS_WORLD_MODEL = True
except ImportError:
    _HAS_WORLD_MODEL = False


class SalienceFunction(Protocol):
    """Protocol for salience computation."""
    def __call__(self, symbol: Symbol, context: Optional[Dict] = None) -> float:
        """Compute salience score for a symbol. Returns 0-1."""
        ...


# -----------------------------------------------------------------------------
# Built-in Salience Functions
# -----------------------------------------------------------------------------

def constant_salience(value: float = 0.5) -> SalienceFunction:
    """Returns a constant salience for all symbols."""
    def fn(symbol: Symbol, context: Optional[Dict] = None) -> float:
        return value
    return fn


def recency_salience(half_life_seconds: float = 60.0) -> SalienceFunction:
    """
    Salience decays exponentially with age.

    Recent symbols are more salient than old ones.
    """
    def fn(symbol: Symbol, context: Optional[Dict] = None) -> float:
        age = (datetime.now() - symbol.timestamp).total_seconds()
        return math.exp(-age * math.log(2) / half_life_seconds)
    return fn


def length_salience(optimal_length: int = 50, falloff: float = 0.02) -> SalienceFunction:
    """
    Salience based on content length.

    Very short or very long content is less salient.
    Optimal around a typical sentence length.
    """
    def fn(symbol: Symbol, context: Optional[Dict] = None) -> float:
        length = len(str(symbol.data))
        deviation = abs(length - optimal_length)
        return math.exp(-deviation * falloff)
    return fn


def keyword_salience(keywords: Dict[str, float]) -> SalienceFunction:
    """
    Salience boosted by presence of keywords.

    Args:
        keywords: Dict mapping keyword -> boost value
    """
    def fn(symbol: Symbol, context: Optional[Dict] = None) -> float:
        text = str(symbol.data).lower()
        boost = 0.0
        for keyword, weight in keywords.items():
            if keyword.lower() in text:
                boost += weight
        return min(1.0, 0.3 + boost)  # Base of 0.3
    return fn


# -----------------------------------------------------------------------------
# Composite Salience
# -----------------------------------------------------------------------------

class CompositeSalience:
    """
    Combines multiple salience functions.

    Supports different aggregation strategies:
    - max: Take the highest salience
    - mean: Average all saliences
    - product: Multiply (all must be high)
    - weighted: Weighted average
    """

    def __init__(
        self,
        functions: List[SalienceFunction],
        weights: Optional[List[float]] = None,
        aggregation: str = "mean"
    ):
        self.functions = functions
        self.weights = weights or [1.0] * len(functions)
        self.aggregation = aggregation

        assert len(self.weights) == len(self.functions)

    def __call__(self, symbol: Symbol, context: Optional[Dict] = None) -> float:
        scores = [fn(symbol, context) for fn in self.functions]

        if self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "mean":
            return sum(scores) / len(scores)
        elif self.aggregation == "product":
            result = 1.0
            for s in scores:
                result *= s
            return result
        elif self.aggregation == "weighted":
            total_weight = sum(self.weights)
            return sum(s * w for s, w in zip(scores, self.weights)) / total_weight
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


# -----------------------------------------------------------------------------
# Pluggable External Salience
# -----------------------------------------------------------------------------

class NoveltyAdapter:
    """
    Adapter to use novelty scores as salience.

    Plugs into the c:\\code\\novelty system. High novelty = high salience.
    """

    def __init__(self, novelty_fn: Callable[[Any], float], scale: float = 1.0):
        """
        Args:
            novelty_fn: Function that computes novelty score (0-1)
            scale: Multiplier for novelty contribution
        """
        self.novelty_fn = novelty_fn
        self.scale = scale

    def __call__(self, symbol: Symbol, context: Optional[Dict] = None) -> float:
        try:
            novelty = self.novelty_fn(symbol.data)
            return min(1.0, novelty * self.scale)
        except Exception:
            return 0.5  # Fallback to neutral


class AllocationAdapter:
    """
    Adapter to use world model allocations as salience.

    Plugs into the c:\\code\\life system. Symbols related to
    high-allocation tendencies get higher salience.
    """

    def __init__(
        self,
        get_allocations: Callable[[], Dict[str, float]],
        classify_fn: Callable[[Any], Optional[str]],
    ):
        """
        Args:
            get_allocations: Returns current tendency allocations
            classify_fn: Classifies a symbol's data into a tendency (or None)
        """
        self.get_allocations = get_allocations
        self.classify_fn = classify_fn

    @classmethod
    def from_agent_set(cls, agent_set, classify_fn=None):
        """
        Create an AllocationAdapter from a world_model AgentSet.

        Args:
            agent_set: An AgentSet instance from world_model
            classify_fn: Optional function to classify symbol data into a tendency name.
                         If not provided, uses a default that returns None.

        Returns:
            AllocationAdapter instance
        """
        if not _HAS_WORLD_MODEL:
            raise ImportError("world_model package not available")

        def get_allocations():
            return {t.value: a.allocation for t, a in agent_set.agents.items()}

        return cls(
            get_allocations=get_allocations,
            classify_fn=classify_fn or (lambda data: None),
        )

    def __call__(self, symbol: Symbol, context: Optional[Dict] = None) -> float:
        try:
            tendency = self.classify_fn(symbol.data)
            if tendency is None:
                return 0.3  # Unclassified gets low salience

            allocations = self.get_allocations()
            return allocations.get(tendency, 0.3)
        except Exception:
            return 0.3


# -----------------------------------------------------------------------------
# Salience Tracker
# -----------------------------------------------------------------------------

@dataclass
class SalienceRecord:
    """Record of salience computation for a symbol."""
    symbol_hash: str
    computed_salience: float
    component_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class SalienceTracker:
    """
    Tracks salience computations over time.

    Useful for:
    - Debugging why certain symbols got high/low salience
    - Computing salience trends
    - Identifying which components contribute most
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[SalienceRecord] = []
        self._by_hash: Dict[str, List[SalienceRecord]] = {}

    def record(
        self,
        symbol: Symbol,
        salience: float,
        components: Optional[Dict[str, float]] = None
    ):
        """Record a salience computation."""
        rec = SalienceRecord(
            symbol_hash=symbol.hash,
            computed_salience=salience,
            component_scores=components or {}
        )

        self._history.append(rec)
        if len(self._history) > self.max_history:
            old = self._history.pop(0)
            if old.symbol_hash in self._by_hash:
                self._by_hash[old.symbol_hash].remove(old)

        if symbol.hash not in self._by_hash:
            self._by_hash[symbol.hash] = []
        self._by_hash[symbol.hash].append(rec)

    def get_history(self, symbol_hash: str) -> List[SalienceRecord]:
        """Get salience history for a specific symbol."""
        return self._by_hash.get(symbol_hash, [])

    def average_salience(self) -> float:
        """Get average salience across all recorded symbols."""
        if not self._history:
            return 0.0
        return sum(r.computed_salience for r in self._history) / len(self._history)

    def top_components(self, n: int = 5) -> List[tuple[str, float]]:
        """Get the components that contribute most to salience."""
        component_totals: Dict[str, float] = {}
        component_counts: Dict[str, int] = {}

        for rec in self._history:
            for comp, score in rec.component_scores.items():
                component_totals[comp] = component_totals.get(comp, 0) + score
                component_counts[comp] = component_counts.get(comp, 0) + 1

        averages = [
            (comp, component_totals[comp] / component_counts[comp])
            for comp in component_totals
        ]
        averages.sort(key=lambda x: x[1], reverse=True)
        return averages[:n]
