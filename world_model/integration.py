"""
Integration - Bridges between life, novelty, and attention.

This module composes:
1. Novelty-modulated attention ALLOCATION (from attention.curves)
   - Maps novelty scores to shifts in tendency allocations
   - High novelty -> CURIOSITY gets more weight

2. Attention symbol ROUTING (from attention project)
   - Bounded sequences filter symbols by salience
   - Novel symbols get promoted to conscious attention

3. Feedback loop back to life's agent dynamics
   - Symbols that survive attention routing become observations
   - Observations feed into the adversarial debate arena
   - Debate outcomes shift allocations, which affect future attention

The full cycle:
    Input -> Attention Routing -> Novelty Measurement -> Allocation Shift
      ^                                                        |
      |                                                        v
      +---- Arena Debate <---- New Observations <---- Attended Symbols
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime

from .models.agent import Tendency, AgentSet
from .models.observation import Observation, ObservationStore
from .models.tree import Tree, TreeStore
from .attention.curves import (
    NoveltyAttentionCurve,
    AttentionState,
    BALANCED_CURVE,
)


@dataclass
class AttentionEvent:
    """Record of an attention routing decision."""
    content: Any
    novelty_score: float
    was_promoted: bool
    effective_allocations: dict[Tendency, float]
    dominant_tendency: Tendency
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FeedbackEvent:
    """Record of a feedback cycle completion."""
    observation: Observation
    source_novelty: float
    allocation_before: dict[Tendency, float]
    allocation_after: dict[Tendency, float]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def max_shift(self) -> float:
        """Largest allocation change in this feedback event."""
        return max(
            abs(self.allocation_after.get(t, 0) - self.allocation_before.get(t, 0))
            for t in set(self.allocation_before) | set(self.allocation_after)
        )


class AttentionBridge:
    """
    Composes novelty-modulated allocation with symbol routing.

    This is the central integration point. It:
    1. Accepts incoming content (symbols, text, etc.)
    2. Measures novelty against the current reference frame
    3. Modulates attention allocations based on novelty
    4. Routes content through bounded attention sequences
    5. Converts attended content into observations for the feedback loop

    Usage:
        from world_model import AgentSet, EXPLORER_CURVE
        from world_model.integration import AttentionBridge

        agents = AgentSet()
        bridge = AttentionBridge(agent_set=agents, curve=EXPLORER_CURVE)

        # Process incoming content
        event = bridge.process("quantum computing enables new cryptography")
        print(event.novelty_score)
        print(event.effective_allocations)
    """

    def __init__(
        self,
        agent_set: Optional[AgentSet] = None,
        curve: Optional[NoveltyAttentionCurve] = None,
        observation_store: Optional[ObservationStore] = None,
        novelty_fn: Optional[Callable[[str, ObservationStore], float]] = None,
        promotion_threshold: float = 0.4,
    ):
        """
        Args:
            agent_set: The agent set with tendency allocations. Defaults to population average.
            curve: Attention curve controlling how novelty affects allocations.
            observation_store: Existing observations (reference frame). Created empty if None.
            novelty_fn: Custom novelty measurement function(content, store) -> score.
                       If None, uses simple word-overlap similarity.
            promotion_threshold: Minimum novelty to "promote" content to conscious attention.
        """
        self.agent_set = agent_set or AgentSet()
        self.attention_state = AttentionState(
            agent_set=self.agent_set,
            curve=curve or BALANCED_CURVE,
        )
        self.observation_store = observation_store or ObservationStore()
        self.novelty_fn = novelty_fn or self._default_novelty
        self.promotion_threshold = promotion_threshold

        # Event log
        self._attention_events: list[AttentionEvent] = []
        self._feedback_events: list[FeedbackEvent] = []

    def _default_novelty(self, content: str, store: ObservationStore) -> float:
        """Simple novelty: inverse of max word-overlap similarity."""
        if len(store) == 0:
            return 1.0

        content_words = set(content.lower().split())
        if not content_words:
            return 0.0

        max_sim = 0.0
        for obs in store.all():
            obs_words = set(obs.content.lower().split())
            if not obs_words:
                continue
            intersection = content_words & obs_words
            union = content_words | obs_words
            sim = len(intersection) / len(union)
            if sim > max_sim:
                max_sim = sim
            if sim > 0.95:
                break

        return 1.0 - max_sim

    def process(self, content: str) -> AttentionEvent:
        """
        Process incoming content through the attention bridge.

        1. Measure novelty against observation store
        2. Update attention state with novelty score
        3. Record effective allocations
        4. Determine if content is "promoted" (novel enough)

        Args:
            content: Text content to process

        Returns:
            AttentionEvent with routing decision and allocation state
        """
        # Measure novelty
        novelty = self.novelty_fn(content, self.observation_store)

        # Update attention state
        self.attention_state.update_novelty(novelty)

        # Create event record
        event = AttentionEvent(
            content=content,
            novelty_score=novelty,
            was_promoted=novelty >= self.promotion_threshold,
            effective_allocations=dict(self.attention_state.effective_allocations),
            dominant_tendency=self.attention_state.dominant_tendency,
        )

        self._attention_events.append(event)
        return event

    def absorb(self, content: str, source_id: str = "") -> FeedbackEvent:
        """
        Absorb content into the observation store (feedback loop).

        This is called when content has been "attended to" and should
        become part of the agent's knowledge base. It:
        1. Creates an observation from the content
        2. Adds it to the store (modifying the reference frame)
        3. Records allocation state before/after

        Args:
            content: The attended content to absorb
            source_id: Optional source identifier

        Returns:
            FeedbackEvent recording the state change
        """
        # Record current novelty and allocations
        novelty = self.novelty_fn(content, self.observation_store)
        alloc_before = dict(self.attention_state.effective_allocations)

        # Create and store observation
        obs = Observation(content=content, source_id=source_id)
        obs, is_new = self.observation_store.add(obs)

        # After absorption, the same content is less novel
        # Update attention state to reflect this
        if is_new:
            new_novelty = self.novelty_fn(content, self.observation_store)
            self.attention_state.update_novelty(new_novelty)

        alloc_after = dict(self.attention_state.effective_allocations)

        event = FeedbackEvent(
            observation=obs,
            source_novelty=novelty,
            allocation_before=alloc_before,
            allocation_after=alloc_after,
        )

        self._feedback_events.append(event)
        return event

    def process_and_absorb(self, content: str, source_id: str = "") -> tuple[AttentionEvent, Optional[FeedbackEvent]]:
        """
        Process content and absorb it if promoted.

        Convenience method that combines process() and absorb().
        Only absorbs if the content passes the promotion threshold.

        Returns:
            (attention_event, feedback_event or None)
        """
        att_event = self.process(content)

        feedback = None
        if att_event.was_promoted:
            feedback = self.absorb(content, source_id=source_id)

        return att_event, feedback

    @property
    def attention_history(self) -> list[AttentionEvent]:
        """All attention routing events."""
        return list(self._attention_events)

    @property
    def feedback_history(self) -> list[FeedbackEvent]:
        """All feedback loop events."""
        return list(self._feedback_events)

    @property
    def promotion_rate(self) -> float:
        """Fraction of processed content that was promoted."""
        if not self._attention_events:
            return 0.0
        promoted = sum(1 for e in self._attention_events if e.was_promoted)
        return promoted / len(self._attention_events)

    @property
    def average_novelty(self) -> float:
        """Average novelty score across all processed content."""
        if not self._attention_events:
            return 0.0
        return sum(e.novelty_score for e in self._attention_events) / len(self._attention_events)

    def describe(self) -> str:
        """Human-readable summary of bridge state."""
        lines = [
            f"AttentionBridge:",
            f"  Observations: {len(self.observation_store)}",
            f"  Events processed: {len(self._attention_events)}",
            f"  Promotion rate: {self.promotion_rate:.1%}",
            f"  Average novelty: {self.average_novelty:.2f}",
            f"  Current state: {self.attention_state.describe()}",
            f"  Agent allocations:",
        ]
        for tendency in Tendency:
            alloc = self.agent_set.agents[tendency].allocation
            lines.append(f"    {tendency.value}: {alloc:.1%}")
        return "\n".join(lines)


class ArenaFeedback:
    """
    Feeds attention outputs back into life's adversarial debate.

    This closes the loop: attended observations are staked in the arena,
    debate outcomes shift allocations, and shifted allocations change
    future attention routing.

    Usage:
        from world_model.dynamics import Arena
        from world_model.integration import ArenaFeedback

        bridge = AttentionBridge(agent_set=agents)
        feedback = ArenaFeedback(bridge=bridge)

        # Process a batch of content
        for text in incoming_content:
            bridge.process_and_absorb(text)

        # Run debate on accumulated observations
        result = feedback.run_debate(rounds=1)
        # Allocations are now updated, affecting future attention routing
    """

    def __init__(
        self,
        bridge: AttentionBridge,
        arena_factory: Optional[Callable] = None,
        learning_rate: float = 0.1,
    ):
        """
        Args:
            bridge: The AttentionBridge whose observations feed the debate.
            arena_factory: Callable that returns an Arena instance.
                          If None, tries to import from world_model.dynamics.
            learning_rate: How fast allocations shift after debate.
        """
        self.bridge = bridge
        self.learning_rate = learning_rate
        self._arena_factory = arena_factory
        self._debate_history: list[dict] = []

    def _get_arena(self):
        """Get or create an Arena instance."""
        if self._arena_factory:
            return self._arena_factory()

        try:
            from world_model.dynamics import Arena
            return Arena()
        except ImportError:
            raise ImportError(
                "world_model.dynamics.Arena not available. "
                "Provide an arena_factory to ArenaFeedback."
            )

    def run_debate(
        self,
        rounds: int = 1,
        verbose: bool = False,
    ) -> Optional[Any]:
        """
        Run adversarial debate on accumulated observations.

        This feeds the bridge's observation store into life's Arena,
        runs the debate, and applies the resulting allocation changes
        back to the bridge's agent set.

        Args:
            rounds: Number of debate rounds
            verbose: Print debate progress

        Returns:
            DebateResult from the arena (or None if insufficient observations)
        """
        store = self.bridge.observation_store
        agents = self.bridge.agent_set

        if len(store) < 3:
            return None

        # Snapshot allocations before debate
        alloc_before = {t: a.allocation for t, a in agents.agents.items()}

        arena = self._get_arena()
        trees, result = arena.run_full_debate(
            observations=store,
            agents=agents,
            rounds=rounds,
            learning_rate=self.learning_rate,
            verbose=verbose,
        )

        # Snapshot allocations after debate
        alloc_after = {t: a.allocation for t, a in agents.agents.items()}

        self._debate_history.append({
            "observations_count": len(store),
            "rounds": rounds,
            "winner": result.winner.value if result.winner else None,
            "allocation_before": {t.value: v for t, v in alloc_before.items()},
            "allocation_after": {t.value: v for t, v in alloc_after.items()},
        })

        return result

    def run_cycle(
        self,
        content_batch: list[str],
        source_id: str = "",
        debate_rounds: int = 1,
        verbose: bool = False,
    ) -> dict:
        """
        Run a complete attention -> absorb -> debate cycle.

        This is the full feedback loop in one call:
        1. Process all content through attention bridge
        2. Absorb promoted content as observations
        3. Run adversarial debate on all observations
        4. Allocations shift, affecting future processing

        Args:
            content_batch: List of content strings to process
            source_id: Source identifier for observations
            debate_rounds: Number of arena debate rounds
            verbose: Print progress

        Returns:
            Summary dict with cycle statistics
        """
        # Phase 1: Process and absorb
        promoted = 0
        for content in content_batch:
            att, fb = self.bridge.process_and_absorb(content, source_id=source_id)
            if att.was_promoted:
                promoted += 1

        # Phase 2: Debate (if enough observations)
        result = self.run_debate(rounds=debate_rounds, verbose=verbose)

        return {
            "content_processed": len(content_batch),
            "promoted": promoted,
            "promotion_rate": promoted / len(content_batch) if content_batch else 0,
            "total_observations": len(self.bridge.observation_store),
            "debate_ran": result is not None,
            "winner": result.winner.value if result and result.winner else None,
            "current_allocations": {
                t.value: a.allocation
                for t, a in self.bridge.agent_set.agents.items()
            },
        }

    @property
    def debate_history(self) -> list[dict]:
        """History of all debates run through this feedback loop."""
        return list(self._debate_history)
