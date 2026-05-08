"""
Novelty - Theoretical Definition

Novelty is deviation from a reference frame.

The reference frame is itself accumulated novelty - the residue of all prior
deviations that have been integrated into the observer's model.

THE LOOP
========

Novelty measurement is not a one-shot computation. It is a LOOP:

    while True:
        data = fetch(focus)
        verdict = parse(data, reference_frame)

        if verdict.terminates:
            break

        focus = verdict.next_focus
        reference_frame = reference_frame.absorb(verdict.partial)

The loop continues until a TERMINATION CONDITION is met. The termination
condition IS the novelty measurement - not something computed afterward.

TERMINATION CONDITIONS (= Novelty Components)
=============================================

1. INTEGRATED
   The concept fits naturally into the reference frame.
   -> Low integration_resistance
   -> Loop terminates quickly, few iterations

2. CONTRADICTS_ROOT
   The concept opposes a foundational claim.
   -> High contradiction_depth
   -> Loop terminates when it hits the conflict

3. ORTHOGONAL
   The concept has no connection to anything in the frame.
   -> High coverage_gap
   -> Loop terminates when search exhausts without finding connection

4. DISRUPTS_ALLOCATION
   Integrating the concept would restructure the frame's priorities.
   -> High allocation_disruption
   -> Loop terminates when disruption threshold exceeded

THE CUTOFF IS THE FRAME
=======================

The recursive question "what is the novelty of the things I'm comparing to?"
doesn't bottom out naturally. The CUTOFF is provided by the reference frame
itself - you stop when you hit concepts already integrated, or when you've
exhausted the frame's scope.

Without a frame, there is no cutoff, and novelty is undefined.
With a frame, the cutoff emerges from what's already inside it.

ITERATION COUNT MATTERS
=======================

How many iterations before termination encodes information:
- Quick termination (1-2 iterations) -> familiar or clearly foreign
- Long search before integration -> hard to place but eventually fits
- Long search before contradiction -> deep structural conflict
- Max iterations reached -> truly orthogonal, frame has no opinion

The loop isn't overhead. It IS the measurement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict, Tuple


# =============================================================================
# Termination Conditions
# =============================================================================

class Termination(Enum):
    """
    Why the novelty loop terminated.

    This IS the novelty measurement. The termination reason directly
    maps to which novelty component dominates.
    """
    INTEGRATED = "integrated"           # Fits in frame -> low integration_resistance
    CONTRADICTS_ROOT = "contradicts"    # Opposes foundation -> high contradiction_depth
    ORTHOGONAL = "orthogonal"           # No connection -> high coverage_gap
    DISRUPTS = "disrupts"               # Would restructure -> high allocation_disruption
    MAX_ITERATIONS = "max_iterations"   # Cutoff reached -> ambiguous/undefined


class Stance(Enum):
    """How an observation positions relative to a claim."""
    PRO = "pro"       # Supports/entails
    CON = "con"       # Opposes/contradicts
    NEUTRAL = "neutral"   # Related but no position


# =============================================================================
# The Core Loop Types
# =============================================================================

@dataclass
class Focus:
    """
    Current focus of the novelty probe.

    The focus is what we're currently examining - it shifts as the
    loop iterates, expanding outward from the initial concept.
    """
    content: Any
    depth: int = 0  # How many iterations from initial concept
    path: List[str] = field(default_factory=list)  # Trace of how we got here

    def expand_to(self, new_content: Any, via: str = "") -> "Focus":
        """Create new focus expanded from this one."""
        new_path = self.path + [via] if via else self.path
        return Focus(
            content=new_content,
            depth=self.depth + 1,
            path=new_path,
        )


@dataclass
class ParseResult:
    """
    Result of parsing fetched data against the reference frame.

    This is the output of each loop iteration. It either terminates
    the loop (with a reason) or provides the next focus.
    """
    # Termination
    terminates: bool = False
    termination_reason: Optional[Termination] = None

    # If not terminating, where to look next
    next_focus: Optional[Focus] = None

    # Partial integration (absorbed before continuing)
    absorbed: Optional[Any] = None

    # Metrics from this iteration
    similarity_to_frame: float = 0.0  # 0-1, how much this matched
    contradiction_depth: int = 0      # If contradicting, at what depth
    stake_affected: float = 0.0       # How much stake is touched

    @staticmethod
    def terminate(reason: Termination, **metrics) -> "ParseResult":
        """Create a terminating result."""
        return ParseResult(
            terminates=True,
            termination_reason=reason,
            **metrics
        )

    @staticmethod
    def continue_to(focus: Focus, absorbed: Any = None, **metrics) -> "ParseResult":
        """Create a continuing result."""
        return ParseResult(
            terminates=False,
            next_focus=focus,
            absorbed=absorbed,
            **metrics
        )


# =============================================================================
# Reference Frame (The Cutoff)
# =============================================================================

class ReferenceFrame(ABC):
    """
    A reference frame against which novelty is measured.

    The frame provides the CUTOFF for the recursive novelty question.
    Without a frame, novelty loops forever. With a frame, the loop
    terminates when it hits the frame's boundary.

    The frame consists of:
    - Claims (structured beliefs, hierarchical)
    - Observations (raw integrated experience)
    - Stakes (what the agent cares about)
    """

    @abstractmethod
    def contains(self, content: Any) -> Tuple[bool, float]:
        """
        Is this content already in the frame?

        Returns (is_contained, similarity) where:
        - is_contained: True if this is already integrated
        - similarity: 0-1 how close to existing content

        This is the primary CUTOFF check - if contained, loop terminates.
        """
        pass

    @abstractmethod
    def find_claims(self, content: Any) -> List[Tuple["Claim", float]]:
        """
        Find claims related to this content.

        Returns list of (claim, relevance) pairs, sorted by relevance.
        Used to determine where in the hierarchy to probe.
        """
        pass

    @abstractmethod
    def detect_stance(self, content: Any, claim: "Claim") -> Tuple[Stance, float]:
        """
        What stance does content take toward claim?

        Returns (stance, confidence).
        This determines if we have contradiction.
        """
        pass

    @abstractmethod
    def absorb(self, content: Any) -> "ReferenceFrame":
        """
        Return new frame with content integrated.

        Frames are immutable - absorption creates a new frame.
        """
        pass

    @abstractmethod
    def get_adjacent(self, content: Any) -> List[Any]:
        """
        Get content adjacent to this in the knowledge structure.

        Used to expand the focus when current content doesn't terminate.
        """
        pass

    @property
    @abstractmethod
    def total_stake(self) -> float:
        """Total stake weight in the frame."""
        pass


@dataclass
class Claim:
    """
    A claim in the reference frame's belief structure.

    Claims are hierarchical - depth 0 is foundational.
    """
    content: Any
    depth: int
    stake: float  # How much the agent cares about this
    children: List["Claim"] = field(default_factory=list)


# =============================================================================
# The Novelty Loop
# =============================================================================

@dataclass
class NoveltyResult:
    """
    Result of novelty measurement.

    This encodes HOW the loop terminated, which IS the novelty.
    """
    # How it terminated
    termination: Termination
    iterations: int

    # The four components (derived from termination + iteration count)
    integration_resistance: float = 0.0
    contradiction_depth: float = 0.0
    coverage_gap: float = 0.0
    allocation_disruption: float = 0.0

    # Trace
    path: List[str] = field(default_factory=list)

    @property
    def composite(self) -> float:
        """Geometric mean of components."""
        epsilon = 0.01
        components = [
            self.integration_resistance + epsilon,
            self.contradiction_depth + epsilon,
            self.coverage_gap + epsilon,
            self.allocation_disruption + epsilon,
        ]
        product = 1.0
        for c in components:
            product *= c
        return product ** (1.0 / len(components))

    @classmethod
    def from_loop(
        cls,
        termination: Termination,
        iterations: int,
        max_iterations: int,
        deepest_contradiction: int = 0,
        max_depth: int = 1,
        stake_affected: float = 0.0,
        total_stake: float = 1.0,
        path: List[str] = None,
    ) -> "NoveltyResult":
        """
        Construct result from loop execution data.

        Maps termination reason + metrics to the four components.
        """
        # Integration resistance: based on iteration count
        # Quick integration = low resistance, many iterations = high
        integration_resistance = min(iterations / max_iterations, 1.0)

        # Contradiction depth: based on how deep the conflict was
        # Normalized to max depth of frame
        if termination == Termination.CONTRADICTS_ROOT:
            # Shallower depth in hierarchy = MORE fundamental = higher score
            contradiction_depth = 1.0 - (deepest_contradiction / max(max_depth, 1))
        else:
            contradiction_depth = 0.0

        # Coverage gap: based on whether we found ANY connection
        if termination == Termination.ORTHOGONAL:
            coverage_gap = 1.0
        elif termination == Termination.MAX_ITERATIONS:
            coverage_gap = 0.8  # Ambiguous, but likely disconnected
        else:
            # Found connection - gap is inverse of how quickly
            coverage_gap = min(iterations / max_iterations, 1.0) * 0.5

        # Allocation disruption: based on stake affected
        if termination == Termination.DISRUPTS:
            allocation_disruption = min(stake_affected / max(total_stake, 0.01), 1.0)
        elif termination == Termination.CONTRADICTS_ROOT and deepest_contradiction <= 1:
            # Contradicting root affects everything
            allocation_disruption = 0.8
        else:
            allocation_disruption = 0.1

        return cls(
            termination=termination,
            iterations=iterations,
            integration_resistance=integration_resistance,
            contradiction_depth=contradiction_depth,
            coverage_gap=coverage_gap,
            allocation_disruption=allocation_disruption,
            path=path or [],
        )


class NoveltyProbe(ABC):
    """
    The novelty measurement loop.

    This is the core abstraction. Implementations provide:
    - fetch(): Get data for current focus
    - parse(): Evaluate data against frame, decide termination

    The loop itself is fixed - only fetch/parse vary.
    """

    def __init__(self, max_iterations: int = 20):
        self.max_iterations = max_iterations

    @abstractmethod
    def fetch(self, focus: Focus, frame: ReferenceFrame) -> Any:
        """
        Fetch data for the current focus.

        This could query an external source (Wikidata) or
        just return the focus content for internal evaluation.
        """
        pass

    @abstractmethod
    def parse(self, data: Any, focus: Focus, frame: ReferenceFrame) -> ParseResult:
        """
        Parse fetched data against the reference frame.

        Determines:
        - Should we terminate? (if so, why?)
        - If not, where to look next?
        - Any partial integration to absorb?
        """
        pass

    def measure(self, content: Any, frame: ReferenceFrame) -> NoveltyResult:
        """
        Run the novelty loop.

        This is the core operation. It loops until termination,
        then constructs the result from how/why it terminated.
        """
        focus = Focus(content=content, depth=0)
        current_frame = frame
        iterations = 0
        deepest_contradiction = float('inf')
        stake_affected = 0.0
        path = []

        while iterations < self.max_iterations:
            iterations += 1

            # FETCH
            data = self.fetch(focus, current_frame)

            # PARSE
            result = self.parse(data, focus, current_frame)

            # Track metrics
            if result.contradiction_depth > 0:
                deepest_contradiction = min(deepest_contradiction, result.contradiction_depth)
            stake_affected = max(stake_affected, result.stake_affected)

            # Check termination
            if result.terminates:
                return NoveltyResult.from_loop(
                    termination=result.termination_reason,
                    iterations=iterations,
                    max_iterations=self.max_iterations,
                    deepest_contradiction=deepest_contradiction if deepest_contradiction != float('inf') else 0,
                    max_depth=self._get_max_depth(frame),
                    stake_affected=stake_affected,
                    total_stake=frame.total_stake,
                    path=path,
                )

            # Absorb partial if any
            if result.absorbed is not None:
                current_frame = current_frame.absorb(result.absorbed)

            # Move to next focus
            if result.next_focus:
                if result.next_focus.path:
                    path.extend(result.next_focus.path)
                focus = result.next_focus
            else:
                # No next focus provided - can't continue
                return NoveltyResult.from_loop(
                    termination=Termination.ORTHOGONAL,
                    iterations=iterations,
                    max_iterations=self.max_iterations,
                    path=path,
                )

        # Hit max iterations
        return NoveltyResult.from_loop(
            termination=Termination.MAX_ITERATIONS,
            iterations=iterations,
            max_iterations=self.max_iterations,
            deepest_contradiction=deepest_contradiction if deepest_contradiction != float('inf') else 0,
            max_depth=self._get_max_depth(frame),
            stake_affected=stake_affected,
            total_stake=frame.total_stake,
            path=path,
        )

    def _get_max_depth(self, frame: ReferenceFrame) -> int:
        """Get max claim depth in frame. Override if frame provides this."""
        return 10  # Default estimate


# =============================================================================
# Axioms
# =============================================================================

"""
AXIOMS FOR THE NOVELTY LOOP

1. TERMINATION REQUIRED
   The loop MUST terminate. Without termination, novelty is undefined.
   The frame provides the cutoff that guarantees termination.

2. TERMINATION = MEASUREMENT
   The novelty score is not computed after the loop. It IS how the
   loop terminates. Different termination reasons = different novelty types.

3. ITERATION COUNT MATTERS
   Two concepts that terminate the same way but at different iteration
   counts have different novelty. More iterations = harder to place.

4. FRAME DETERMINES CUTOFF
   The same concept measured against different frames will have different
   novelty because the cutoff (what's already integrated) differs.

5. ABSORPTION SHIFTS CUTOFF
   frame.absorb(X) produces a new frame where X is integrated.
   Measuring X against the new frame will terminate faster (INTEGRATED).

6. ADJACENCY DETERMINES EXPANSION
   When the loop continues, it expands to adjacent content. The
   structure of adjacency (in the frame or external source) shapes
   which termination will eventually occur.
"""
