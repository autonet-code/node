"""
Attention Processes

A Process subscribes to sequences, matches patterns, and publishes
to other sequences. Processes are the active agents in the attention
system - they watch for meaningful patterns and propagate associations.

From the original insight:
    "A process subscribed to short-term memory is looking at symbols
    as they come along, and it's job is to hash the symbols that are
    published. It's got a lookup table and every now and then it
    stumbles upon an entry that is populated with a response."

Processes enable:
- Pattern recognition (matching symbols to known patterns)
- Association (linking related concepts across sequences)
- Reinforcement (boosting value of repeated patterns)
- Spawning (creating new processes in response to patterns)
"""

from dataclasses import dataclass, field
from typing import (
    Callable, Optional, Any, Dict, List, Set,
    Awaitable, Union
)
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
import hashlib

from .sequence import Sequence, Symbol, Subscription


# Pattern matching result
@dataclass
class Match:
    """Result of a pattern match."""
    pattern_id: str
    symbol: Symbol
    confidence: float  # 0-1
    response: Optional[Any] = None  # Associated response if any
    metadata: dict = field(default_factory=dict)


class Process(ABC):
    """
    Base class for attention processes.

    A Process subscribes to one or more input sequences, applies
    pattern matching logic, and publishes results to output sequences.

    Subclass and implement `match()` to define custom pattern logic.
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
    ):
        self.id = id
        self.inputs = inputs
        self.outputs = outputs or []
        self._subscriptions: List[Subscription] = []
        self._active = False

        # Stats
        self.symbols_seen = 0
        self.matches_found = 0
        self.symbols_published = 0

    def start(self):
        """Start processing - subscribe to all inputs."""
        if self._active:
            return
        self._active = True
        for seq in self.inputs:
            sub = seq.subscribe(
                id=f"{self.id}@{seq.name}",
                callback=self._on_symbol
            )
            self._subscriptions.append(sub)

    def stop(self):
        """Stop processing - unsubscribe from all inputs."""
        self._active = False
        for sub in self._subscriptions:
            for seq in self.inputs:
                seq.unsubscribe(sub.id)
        self._subscriptions.clear()

    def _on_symbol(self, symbol: Symbol):
        """Handle incoming symbol from subscribed sequence."""
        self.symbols_seen += 1

        match = self.match(symbol)
        if match is not None:
            self.matches_found += 1
            self._handle_match(match)

    def _handle_match(self, match: Match):
        """Process a successful match."""
        # Create output symbol from match
        output = Symbol(
            data=match.response if match.response else match.symbol.data,
            value=match.symbol.value * match.confidence,
            source=self.id,
            metadata={
                "pattern_id": match.pattern_id,
                "confidence": match.confidence,
                "original_hash": match.symbol.hash,
                **match.metadata
            }
        )

        # Publish to all output sequences
        for seq in self.outputs:
            if seq.publish(output):
                self.symbols_published += 1

    @abstractmethod
    def match(self, symbol: Symbol) -> Optional[Match]:
        """
        Attempt to match a symbol against known patterns.

        Returns Match if pattern found, None otherwise.
        Override this in subclasses.
        """
        pass

    def stats(self) -> dict:
        return {
            "id": self.id,
            "active": self._active,
            "symbols_seen": self.symbols_seen,
            "matches_found": self.matches_found,
            "symbols_published": self.symbols_published,
            "inputs": [s.name for s in self.inputs],
            "outputs": [s.name for s in self.outputs],
        }


class LookupProcess(Process):
    """
    A process with a hash-based lookup table.

    Maps symbol hashes (or features) to responses. When a symbol
    matches an entry, the associated response is published.

    Example:
        proc = LookupProcess("word_lookup", [input_seq], [output_seq])
        proc.register("quick brown fox", response="pangram")
        proc.start()
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
        hash_fn: Optional[Callable[[Symbol], str]] = None,
    ):
        super().__init__(id, inputs, outputs)
        self._table: Dict[str, Any] = {}
        self._hash_fn = hash_fn or (lambda s: s.hash)

    def register(self, key: Any, response: Any, hash_key: bool = True):
        """
        Register a pattern -> response mapping.

        Args:
            key: The pattern to match (will be hashed if hash_key=True)
            response: The response to emit on match
            hash_key: Whether to hash the key (False for pre-computed hashes)
        """
        if hash_key:
            h = hashlib.sha256(str(key).encode()).hexdigest()[:16]
        else:
            h = key
        self._table[h] = response

    def match(self, symbol: Symbol) -> Optional[Match]:
        h = self._hash_fn(symbol)
        if h in self._table:
            return Match(
                pattern_id=h,
                symbol=symbol,
                confidence=1.0,
                response=self._table[h]
            )
        return None


class RepetitionProcess(Process):
    """
    A process that detects repeated symbols.

    Watches for the same symbol appearing multiple times within
    a window. When repetition is detected, boosts the symbol's
    value and publishes it.

    This implements the insight: "a process that is only interested
    in repeated symbols, and has the job of reinforcing their value."
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
        window_size: int = 10,
        min_repetitions: int = 2,
        boost_factor: float = 1.5,
    ):
        super().__init__(id, inputs, outputs)
        self.window_size = window_size
        self.min_repetitions = min_repetitions
        self.boost_factor = boost_factor

        self._window: List[str] = []  # Recent hashes
        self._counts: Dict[str, int] = {}  # Hash -> count in window

    def match(self, symbol: Symbol) -> Optional[Match]:
        h = symbol.hash

        # Update window
        self._window.append(h)
        self._counts[h] = self._counts.get(h, 0) + 1

        # Trim window
        if len(self._window) > self.window_size:
            old_h = self._window.pop(0)
            self._counts[old_h] -= 1
            if self._counts[old_h] == 0:
                del self._counts[old_h]

        # Check for repetition
        count = self._counts.get(h, 0)
        if count >= self.min_repetitions:
            return Match(
                pattern_id=f"repeat:{h}",
                symbol=symbol,
                confidence=min(1.0, count / self.window_size),
                response=symbol.boost(symbol.value * (self.boost_factor - 1)),
                metadata={"repetition_count": count}
            )

        return None


class ConvergenceProcess(Process):
    """
    A process that detects convergent signals from multiple sources.

    When multiple input sequences publish similar symbols within
    a time window, this indicates consensus/salience. The converging
    symbol is boosted and published.

    This captures the idea that attention should focus where
    multiple independent processes agree.
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
        window_ms: int = 500,
        min_sources: int = 2,
        boost_factor: float = 2.0,
    ):
        super().__init__(id, inputs, outputs)
        self.window_ms = window_ms
        self.min_sources = min_sources
        self.boost_factor = boost_factor

        # Hash -> list of (timestamp, source, symbol)
        self._recent: Dict[str, List[tuple]] = {}

    def match(self, symbol: Symbol) -> Optional[Match]:
        h = symbol.hash
        now = datetime.now()

        # Add to recent
        if h not in self._recent:
            self._recent[h] = []
        self._recent[h].append((now, symbol.source, symbol))

        # Clean old entries
        cutoff = now.timestamp() - (self.window_ms / 1000)
        self._recent[h] = [
            (t, s, sym) for t, s, sym in self._recent[h]
            if t.timestamp() > cutoff
        ]

        # Check for convergence
        sources = set(s for _, s, _ in self._recent[h])
        if len(sources) >= self.min_sources:
            # Multiple sources agree - this is salient
            avg_value = sum(sym.value for _, _, sym in self._recent[h]) / len(self._recent[h])
            return Match(
                pattern_id=f"converge:{h}",
                symbol=symbol,
                confidence=min(1.0, len(sources) / len(self.inputs)),
                response=symbol.with_value(avg_value * self.boost_factor),
                metadata={
                    "converging_sources": list(sources),
                    "convergence_count": len(self._recent[h])
                }
            )

        return None


class LoopDetector(Process):
    """
    A process that detects and breaks repetitive loops.

    From the original insight: "Avoid getting stuck in loops by
    listening for pattern repetitions... find a coherent continuation
    and append it to the sequence so as to get out of the loop."

    When a loop is detected, emits a special break symbol that
    downstream processes can use to redirect attention.
    """

    def __init__(
        self,
        id: str,
        inputs: List[Sequence],
        outputs: Optional[List[Sequence]] = None,
        pattern_length: int = 3,
        max_repeats: int = 2,
    ):
        super().__init__(id, inputs, outputs)
        self.pattern_length = pattern_length
        self.max_repeats = max_repeats

        self._history: List[str] = []

    def match(self, symbol: Symbol) -> Optional[Match]:
        h = symbol.hash
        self._history.append(h)

        # Keep history bounded
        max_history = self.pattern_length * (self.max_repeats + 1)
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

        # Check for repeating pattern
        if len(self._history) >= self.pattern_length * 2:
            for plen in range(2, self.pattern_length + 1):
                pattern = self._history[-plen:]
                repeats = 0

                # Count how many times pattern repeats
                for i in range(len(self._history) - plen, -1, -plen):
                    if self._history[i:i+plen] == pattern:
                        repeats += 1
                    else:
                        break

                if repeats >= self.max_repeats:
                    return Match(
                        pattern_id=f"loop:{':'.join(pattern)}",
                        symbol=symbol,
                        confidence=1.0,
                        response=Symbol(
                            data={"type": "loop_break", "pattern": pattern},
                            value=1.0,  # High priority
                            source=self.id,
                            metadata={"loop_length": plen, "repeats": repeats}
                        ),
                        metadata={"loop_detected": True}
                    )

        return None
