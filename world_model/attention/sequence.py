"""
Attention Sequences

A Sequence is a bounded buffer of symbols that processes can subscribe to.
This is the fundamental data structure for attention - a flowing stream
with finite capacity that forces prioritization.

From first principles:
- Consciousness is a "very short sequence of symbols"
- Items enter based on relevance/survival value
- Processes subscribe and react to patterns
- Items graduate to longer-term sequences based on assigned value

The key insight: attention emerges from the interaction between
multiple sequences with different capacities and filtering criteria.
"""

from dataclasses import dataclass, field
from typing import (
    Generic, TypeVar, Callable, Optional, Any,
    Iterator, Awaitable, Union
)
from collections import deque
from datetime import datetime
from enum import Enum
import asyncio
import hashlib
import threading


T = TypeVar('T')


@dataclass
class Symbol:
    """
    An atomic unit of attention.

    Symbols are the fundamental currency flowing through sequences.
    They carry data, an assigned value (salience), and metadata
    for pattern matching and provenance tracking.
    """
    data: Any
    value: float = 0.0  # Salience/priority score
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None  # Which sequence/process produced this
    metadata: dict = field(default_factory=dict)

    # For deduplication and pattern matching
    _hash: Optional[str] = field(default=None, repr=False)

    @property
    def hash(self) -> str:
        """Content-based hash for deduplication and matching."""
        if self._hash is None:
            content = str(self.data).encode('utf-8')
            self._hash = hashlib.sha256(content).hexdigest()[:16]
        return self._hash

    def with_value(self, value: float) -> "Symbol":
        """Return a copy with updated value."""
        return Symbol(
            data=self.data,
            value=value,
            timestamp=self.timestamp,
            source=self.source,
            metadata=self.metadata.copy(),
            _hash=self._hash
        )

    def boost(self, delta: float) -> "Symbol":
        """Return a copy with boosted value."""
        return self.with_value(self.value + delta)


# Callback types
SyncCallback = Callable[[Symbol], None]
AsyncCallback = Callable[[Symbol], Awaitable[None]]
Callback = Union[SyncCallback, AsyncCallback]

# Filter type: returns new value (None = reject)
Filter = Callable[[Symbol], Optional[float]]


class EvictionPolicy(Enum):
    """How to handle overflow when sequence is full."""
    DROP_OLDEST = "drop_oldest"  # FIFO - default
    DROP_LOWEST = "drop_lowest"  # Priority queue behavior
    DROP_NEWEST = "drop_newest"  # Reject new items when full


@dataclass
class Subscription:
    """A process subscribed to a sequence."""
    id: str
    callback: Callback
    is_async: bool = False
    filter: Optional[Filter] = None  # Optional pre-filter
    active: bool = True


class Sequence:
    """
    A bounded buffer of symbols with pub/sub semantics.

    Sequences are the rivers through which attention flows.
    They have finite capacity (forcing prioritization),
    support multiple subscribers (processes watching for patterns),
    and can filter incoming symbols based on value functions.

    Example:
        # Create a short "conscious" sequence
        conscious = Sequence(name="conscious", capacity=7)

        # Subscribe a process
        conscious.subscribe("logger", lambda s: print(s.data))

        # Publish symbols (filtered by value)
        conscious.publish(Symbol(data="hello", value=0.8))
    """

    def __init__(
        self,
        name: str,
        capacity: int = 7,  # Miller's 7±2
        eviction: EvictionPolicy = EvictionPolicy.DROP_OLDEST,
        min_value: float = 0.0,  # Minimum value to enter
    ):
        self.name = name
        self.capacity = capacity
        self.eviction = eviction
        self.min_value = min_value

        self._buffer: deque[Symbol] = deque(maxlen=capacity)
        self._subscribers: dict[str, Subscription] = {}
        self._lock = threading.RLock()

        # Stats
        self.total_published = 0
        self.total_rejected = 0
        self.total_evicted = 0

    def publish(self, symbol: Symbol) -> bool:
        """
        Publish a symbol to the sequence.

        Returns True if accepted, False if rejected (below min_value
        or evicted immediately due to lower priority).
        """
        with self._lock:
            # Value gate
            if symbol.value < self.min_value:
                self.total_rejected += 1
                return False

            # Tag source if not set
            if symbol.source is None:
                symbol = Symbol(
                    data=symbol.data,
                    value=symbol.value,
                    timestamp=symbol.timestamp,
                    source=self.name,
                    metadata=symbol.metadata,
                    _hash=symbol._hash
                )

            # Handle capacity
            if len(self._buffer) >= self.capacity:
                if self.eviction == EvictionPolicy.DROP_OLDEST:
                    self._buffer.popleft()
                    self.total_evicted += 1
                elif self.eviction == EvictionPolicy.DROP_LOWEST:
                    # Find lowest value item
                    min_idx = min(range(len(self._buffer)),
                                  key=lambda i: self._buffer[i].value)
                    if self._buffer[min_idx].value < symbol.value:
                        del self._buffer[min_idx]
                        self.total_evicted += 1
                    else:
                        # New symbol is lowest, reject it
                        self.total_rejected += 1
                        return False
                elif self.eviction == EvictionPolicy.DROP_NEWEST:
                    self.total_rejected += 1
                    return False

            self._buffer.append(symbol)
            self.total_published += 1

        # Notify subscribers (outside lock)
        self._notify(symbol)
        return True

    def _notify(self, symbol: Symbol):
        """Notify all subscribers of a new symbol."""
        for sub in list(self._subscribers.values()):
            if not sub.active:
                continue

            # Apply subscriber's filter
            if sub.filter is not None:
                new_value = sub.filter(symbol)
                if new_value is None:
                    continue  # Filtered out
                symbol = symbol.with_value(new_value)

            try:
                if sub.is_async:
                    # Schedule async callback
                    asyncio.create_task(sub.callback(symbol))
                else:
                    sub.callback(symbol)
            except Exception as e:
                # Don't let subscriber errors break the sequence
                pass

    def subscribe(
        self,
        id: str,
        callback: Callback,
        filter: Optional[Filter] = None
    ) -> Subscription:
        """
        Subscribe a process to this sequence.

        The callback is invoked for each new symbol. If filter is provided,
        it can transform the value or return None to skip.
        """
        is_async = asyncio.iscoroutinefunction(callback)
        sub = Subscription(
            id=id,
            callback=callback,
            is_async=is_async,
            filter=filter
        )
        with self._lock:
            self._subscribers[id] = sub
        return sub

    def unsubscribe(self, id: str) -> bool:
        """Remove a subscription."""
        with self._lock:
            if id in self._subscribers:
                del self._subscribers[id]
                return True
            return False

    def peek(self, n: Optional[int] = None) -> list[Symbol]:
        """Get recent symbols without removing them."""
        with self._lock:
            if n is None:
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def __iter__(self) -> Iterator[Symbol]:
        """Iterate over current buffer contents."""
        with self._lock:
            return iter(list(self._buffer))

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def stats(self) -> dict:
        """Get sequence statistics."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "current_size": len(self._buffer),
            "total_published": self.total_published,
            "total_rejected": self.total_rejected,
            "total_evicted": self.total_evicted,
            "subscribers": len(self._subscribers),
        }


class SequenceChain:
    """
    A chain of sequences with graduated capacity.

    Models the hierarchy from immediate awareness to short-term memory
    to longer-term storage. Each level has larger capacity but higher
    entry threshold.

    Example:
        chain = SequenceChain([
            ("conscious", 7, 0.5),    # Small, low threshold
            ("working", 20, 0.7),     # Medium
            ("short_term", 100, 0.8), # Larger, higher threshold
        ])
    """

    def __init__(self, levels: list[tuple[str, int, float]]):
        """
        Args:
            levels: List of (name, capacity, min_value) tuples
        """
        self.sequences: dict[str, Sequence] = {}
        self._order: list[str] = []

        prev_seq = None
        for name, capacity, min_value in levels:
            seq = Sequence(name=name, capacity=capacity, min_value=min_value)
            self.sequences[name] = seq
            self._order.append(name)

            # Wire up: when item leaves one sequence, try to enter next
            if prev_seq is not None:
                self._connect(prev_seq, seq)
            prev_seq = seq

    def _connect(self, source: Sequence, target: Sequence):
        """Connect sequences so graduated items flow downstream."""
        # This is a simplified connection - in practice you'd want
        # to track items that get evicted and try to promote them
        pass  # TODO: implement graduation logic

    def entry(self) -> Sequence:
        """Get the entry-point sequence."""
        return self.sequences[self._order[0]]

    def publish(self, symbol: Symbol) -> bool:
        """Publish to the entry sequence."""
        return self.entry().publish(symbol)

    def __getitem__(self, name: str) -> Sequence:
        return self.sequences[name]
