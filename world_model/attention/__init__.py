"""
Attention - Dynamic Resource Allocation for Symbol Streams

A framework for modeling attention as the routing of symbols through
cascading sequences with filtering, pattern matching, and reinforcement.

Core Insight:
    Attention is not a single mechanism but an emergent property of
    multiple sequences with subscribing processes that hash patterns,
    publish associations, and reinforce convergent signals.

Architecture:
    Sequence -> Process -> Sequence -> Process -> ...

    - Sequences are bounded buffers forcing prioritization
    - Processes subscribe to sequences and match patterns
    - Matched patterns publish to other sequences
    - Value (salience) determines what persists

Novelty-Attention Curves:
    Maps novelty scores to attention allocation shifts across tendencies.
    High novelty captures attention and shifts allocation toward CURIOSITY.
"""

from .curves import (
    NoveltyAttentionCurve,
    AttentionState,
    EXPLORER_CURVE,
    BALANCED_CURVE,
    CONSERVATIVE_CURVE,
    sigmoid,
)

from .sequence import (
    Symbol,
    Sequence,
    SequenceChain,
    EvictionPolicy,
    Subscription,
    Filter,
)

from .process import (
    Process,
    Match,
    LookupProcess,
    RepetitionProcess,
    ConvergenceProcess,
    LoopDetector,
)

from .salience import (
    SalienceFunction,
    CompositeSalience,
    SalienceTracker,
    SalienceRecord,
    # Built-in functions
    constant_salience,
    recency_salience,
    length_salience,
    keyword_salience,
    # Adapters for external systems
    NoveltyAdapter,
    AllocationAdapter,
)

from .novelty_process import (
    NoveltyProcess,
    SequenceFrame,
    SimpleNoveltyProbe,
    SimpleNoveltyResult,
    create_novelty_pipeline,
    create_integrated_pipeline,
    is_novelty_available,
)

__all__ = [
    # Curves (novelty-attention mapping)
    "NoveltyAttentionCurve",
    "AttentionState",
    "EXPLORER_CURVE",
    "BALANCED_CURVE",
    "CONSERVATIVE_CURVE",
    "sigmoid",

    # Core sequences
    "Symbol",
    "Sequence",
    "SequenceChain",
    "EvictionPolicy",
    "Subscription",
    "Filter",

    # Processes
    "Process",
    "Match",
    "LookupProcess",
    "RepetitionProcess",
    "ConvergenceProcess",
    "LoopDetector",

    # Salience
    "SalienceFunction",
    "CompositeSalience",
    "SalienceTracker",
    "SalienceRecord",
    "constant_salience",
    "recency_salience",
    "length_salience",
    "keyword_salience",
    "NoveltyAdapter",
    "AllocationAdapter",

    # Novelty integration
    "NoveltyProcess",
    "SequenceFrame",
    "SimpleNoveltyProbe",
    "SimpleNoveltyResult",
    "create_novelty_pipeline",
    "create_integrated_pipeline",
    "is_novelty_available",
]
