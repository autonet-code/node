"""
Novelty Process - Integration with the novelty project.

This module provides a Process that uses novelty measurement as the
basis for attention routing. High-novelty symbols get boosted and
promoted to higher-tier sequences.

The key insight: what's in your attention sequences IS your reference
frame. Novelty measures against what you've already attended to.

Requires: The novelty project (c:\code\novelty) to be importable.
         Falls back gracefully if not available.

Usage:
    from attention.novelty_process import NoveltyProcess, SequenceFrame

    # Create a reference frame from sequence history
    frame = SequenceFrame(conscious_sequence)

    # Create process that routes by novelty
    proc = NoveltyProcess(
        id="novelty_router",
        inputs=[working_memory],
        outputs=[conscious],
        frame=frame,
    )
    proc.start()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import sys
import os

from .sequence import Sequence, Symbol
from .process import Process, Match

# World model is now a proper installed package
try:
    from world_model import (
        Observation, ObservationStore, AgentSet, Tendency,
        NoveltyAttentionCurve, AttentionState,
    )
    _HAS_WORLD_MODEL = True
except ImportError:
    _HAS_WORLD_MODEL = False

# Novelty probe comes from the novelty project (not a package)
_novelty_available = False
_NoveltyProbe = None
_ReferenceFrame = None
_NoveltyResult = None
_Termination = None
_Focus = None
_ParseResult = None

def _try_import_novelty():
    """Attempt to import from novelty project."""
    global _novelty_available, _NoveltyProbe, _ReferenceFrame, _NoveltyResult
    global _Termination, _Focus, _ParseResult

    if _novelty_available:
        return True

    # Try to find novelty in common locations
    novelty_paths = [
        os.path.join(os.environ.get("USERPROFILE", ""), "code", "novelty"),
        r"C:\code\novelty",
        os.path.expanduser("~/code/novelty"),
        os.path.join(os.path.dirname(__file__), "..", "..", "novelty"),
    ]

    for path in novelty_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            break

    try:
        from core import (
            NoveltyProbe,
            ReferenceFrame,
            NoveltyResult,
            Termination,
            Focus,
            ParseResult,
        )
        _NoveltyProbe = NoveltyProbe
        _ReferenceFrame = ReferenceFrame
        _NoveltyResult = NoveltyResult
        _Termination = Termination
        _Focus = Focus
        _ParseResult = ParseResult
        _novelty_available = True
        return True
    except ImportError:
        return False


def is_novelty_available() -> bool:
    """Check if novelty project is available."""
    return _try_import_novelty()


# =============================================================================
# Sequence as Reference Frame
# =============================================================================

class SequenceFrame:
    """
    Wraps an attention Sequence as a novelty ReferenceFrame.

    The key insight: what's in your attention sequence IS your reference
    frame. Novelty is measured against what you've already attended to.

    This creates a dynamic reference that evolves as symbols enter
    and leave the sequence.
    """

    def __init__(
        self,
        sequence: Sequence,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.sequence = sequence
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(str(a).lower().split())
        words_b = set(str(b).lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def contains(self, content: Any) -> tuple[bool, float]:
        """
        Check if content is already in the sequence.

        Returns (is_contained, max_similarity).
        """
        content_str = str(content)
        max_sim = 0.0

        for symbol in self.sequence:
            sim = self.similarity_fn(content_str, str(symbol.data))
            if sim > max_sim:
                max_sim = sim
            if sim > 0.95:  # Near-exact match
                return True, sim

        return max_sim > 0.7, max_sim

    def get_observations(self) -> List[str]:
        """Get all content in the sequence as strings."""
        return [str(s.data) for s in self.sequence]

    @property
    def total_stake(self) -> float:
        """Total value in the sequence (sum of symbol values)."""
        return sum(s.value for s in self.sequence)


# =============================================================================
# Simple Novelty Probe (standalone, no external dependency)
# =============================================================================

@dataclass
class SimpleNoveltyResult:
    """Result from simple novelty measurement."""
    is_novel: bool
    score: float  # 0-1, higher = more novel
    reason: str  # Why this score
    iterations: int = 1
    max_similarity: float = 0.0

    @property
    def composite(self) -> float:
        return self.score


class SimpleNoveltyProbe:
    """
    Simple novelty probe that doesn't require the full novelty project.

    Measures novelty as inverse of maximum similarity to sequence contents.
    Quick termination = familiar, many comparisons = novel.
    """

    def __init__(
        self,
        frame: SequenceFrame,
        similarity_threshold: float = 0.7,
    ):
        self.frame = frame
        self.similarity_threshold = similarity_threshold

    def measure(self, content: Any) -> SimpleNoveltyResult:
        """Measure novelty of content against the frame."""
        content_str = str(content)
        observations = self.frame.get_observations()

        if not observations:
            # Empty frame - everything is maximally novel
            return SimpleNoveltyResult(
                is_novel=True,
                score=1.0,
                reason="empty_frame",
                iterations=1,
            )

        # Find maximum similarity to any existing content
        max_sim = 0.0
        iterations = 0

        for obs in observations:
            iterations += 1
            sim = self.frame.similarity_fn(content_str, obs)
            if sim > max_sim:
                max_sim = sim
            if sim > 0.95:
                # Early termination - near exact match
                return SimpleNoveltyResult(
                    is_novel=False,
                    score=0.05,
                    reason="exact_match",
                    iterations=iterations,
                    max_similarity=sim,
                )

        # Novelty is inverse of similarity
        novelty_score = 1.0 - max_sim

        if max_sim > self.similarity_threshold:
            return SimpleNoveltyResult(
                is_novel=False,
                score=novelty_score,
                reason="similar_exists",
                iterations=iterations,
                max_similarity=max_sim,
            )
        elif max_sim < 0.2:
            return SimpleNoveltyResult(
                is_novel=True,
                score=min(1.0, novelty_score * 1.2),  # Boost orthogonal
                reason="orthogonal",
                iterations=iterations,
                max_similarity=max_sim,
            )
        else:
            return SimpleNoveltyResult(
                is_novel=True,
                score=novelty_score,
                reason="novel",
                iterations=iterations,
                max_similarity=max_sim,
            )


# =============================================================================
# Novelty Process
# =============================================================================

class NoveltyProcess(Process):
    """
    A Process that routes symbols based on novelty.

    Subscribes to input sequences, measures novelty of each symbol,
    and publishes high-novelty items to output sequences with boosted value.

    Can use either:
    - The full novelty project (if available)
    - SimpleNoveltyProbe (standalone fallback)

    The reference frame is built from sequence history, creating
    a dynamic measure of "what have I already attended to?"
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
        frame: Optional[SequenceFrame] = None,
        novelty_threshold: float = 0.5,
        boost_factor: float = 1.5,
        use_full_novelty: bool = True,
    ):
        """
        Args:
            id: Process identifier
            inputs: Sequences to subscribe to
            outputs: Sequences to publish high-novelty items to
            frame: Reference frame (sequence to compare against)
            novelty_threshold: Minimum novelty to publish (0-1)
            boost_factor: How much to boost novel symbol values
            use_full_novelty: Try to use full novelty project if available
        """
        super().__init__(id, inputs, outputs)
        self.novelty_threshold = novelty_threshold
        self.boost_factor = boost_factor

        # Set up frame - default to first output sequence
        if frame is not None:
            self.frame = frame
        elif outputs:
            self.frame = SequenceFrame(outputs[0])
        else:
            self.frame = None

        # Set up probe
        self._full_novelty = False
        if use_full_novelty and is_novelty_available():
            self._full_novelty = True
            # Would use HybridProbe here if we had full integration
            # For now, fall back to simple probe
            self._probe = SimpleNoveltyProbe(self.frame) if self.frame else None
        else:
            self._probe = SimpleNoveltyProbe(self.frame) if self.frame else None

    def match(self, symbol: Symbol) -> Optional[Match]:
        """
        Match symbols based on novelty score.

        High novelty -> high confidence match -> gets published
        Low novelty -> no match -> stays in input sequence
        """
        if self._probe is None:
            return None

        result = self._probe.measure(symbol.data)

        if result.score >= self.novelty_threshold:
            # Novel enough to promote
            boosted_value = symbol.value * self.boost_factor * result.score
            return Match(
                pattern_id=f"novelty:{result.reason}",
                symbol=symbol,
                confidence=result.score,
                response=Symbol(
                    data=symbol.data,
                    value=min(1.0, boosted_value),
                    source=self.id,
                    metadata={
                        **symbol.metadata,
                        "novelty_score": result.score,
                        "novelty_reason": result.reason,
                        "novelty_iterations": result.iterations,
                    }
                ),
                metadata={
                    "novelty_score": result.score,
                    "reason": result.reason,
                }
            )

        # Not novel enough
        return None

    def stats(self) -> dict:
        base = super().stats()
        base["novelty_threshold"] = self.novelty_threshold
        base["full_novelty_available"] = self._full_novelty
        base["frame_size"] = len(self.frame.sequence) if self.frame else 0
        return base


# =============================================================================
# Convenience: Wire up a complete novelty-aware attention pipeline
# =============================================================================

def create_novelty_pipeline(
    working_capacity: int = 20,
    conscious_capacity: int = 7,
    working_min_value: float = 0.3,
    conscious_min_value: float = 0.5,
    novelty_threshold: float = 0.4,
) -> tuple[Sequence, Sequence, NoveltyProcess]:
    """
    Create a two-tier attention pipeline with novelty routing.

    Returns (working_memory, conscious, novelty_process).

    Usage:
        working, conscious, proc = create_novelty_pipeline()
        proc.start()

        # Publish to working memory
        working.publish(Symbol(data="hello", value=0.5))

        # Novel items automatically promoted to conscious
        for symbol in conscious:
            print(f"Attended to: {symbol.data}")
    """
    working = Sequence(
        name="working_memory",
        capacity=working_capacity,
        min_value=working_min_value,
    )

    conscious = Sequence(
        name="conscious",
        capacity=conscious_capacity,
        min_value=conscious_min_value,
    )

    # Reference frame is the conscious sequence
    frame = SequenceFrame(conscious)

    proc = NoveltyProcess(
        id="novelty_router",
        inputs=[working],
        outputs=[conscious],
        frame=frame,
        novelty_threshold=novelty_threshold,
    )

    return working, conscious, proc


# =============================================================================
# Integrated World Model Pipeline
# =============================================================================

def create_integrated_pipeline(
    agent_set=None,
    attention_curve=None,
    novelty_threshold: float = 0.5,
    working_capacity: int = 20,
    conscious_capacity: int = 7,
):
    """
    Create a full attention pipeline integrated with world_model.

    This wires together:
    - World model agent allocations -> salience scoring
    - Novelty measurement -> attention routing
    - Attention curves -> allocation modulation

    Args:
        agent_set: Optional AgentSet from world_model. If None, uses defaults.
        attention_curve: Optional NoveltyAttentionCurve. If None, uses BALANCED_CURVE.
        novelty_threshold: Novelty score above which symbols are promoted.
        working_capacity: Working memory sequence capacity.
        conscious_capacity: Conscious attention sequence capacity.

    Returns:
        Tuple of (working_sequence, conscious_sequence, novelty_process, attention_state)

    Raises:
        ImportError: If world_model is not available
    """
    if not _HAS_WORLD_MODEL:
        raise ImportError(
            "world_model package is required for integrated pipeline. "
            "Install it or use create_novelty_pipeline() for standalone usage."
        )

    from world_model import AgentSet as WMAgentSet, BALANCED_CURVE

    if agent_set is None:
        agent_set = WMAgentSet()
    if attention_curve is None:
        attention_curve = BALANCED_CURVE

    attention_state = AttentionState(
        agent_set=agent_set,
        curve=attention_curve,
    )

    working, conscious, proc = create_novelty_pipeline(
        novelty_threshold=novelty_threshold,
        working_capacity=working_capacity,
        conscious_capacity=conscious_capacity,
    )

    return working, conscious, proc, attention_state
