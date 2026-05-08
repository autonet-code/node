"""
World Model - Agent-relative computation for an open-roster reseeding engine.

Models decision-making as competing internal frames ("tendencies") that
stake positions on observations, debate in adversarial arenas, and shift
allocations based on novelty-modulated attention. The roster is open
(no fixed enum); tendencies are instantiated by an external authority
via a factory.

Subpackages:
    world_model.models      - Core data structures (Tendency, Tree, Observation, ...)
    world_model.attention   - Attention curves, symbol streams, salience, routing
    world_model.novelty     - Novelty measurement against reference frames
    world_model.dynamics    - Adversarial arena and training loop
    world_model.staking     - Evidence staking mechanisms
    world_model.analysis    - Sparsity, fictional-world simulators (read-only)
    world_model.extraction  - Observation extraction from text/voice/tweets
    world_model.agents      - Autonomous agents driven by world model
    world_model.storage     - JSON and Firestore persistence

Usage:
    from world_model import Tendency, TendencySet, Observation, ObservationStore
    from world_model import Tree, TreeStore, Node, Position, Stake
    from world_model import DefaultTendencyFactory, TendencySpec
    from world_model.attention import NoveltyAttentionCurve, Sequence, Symbol
    from world_model.novelty import measure_against_claims, HybridProbe
    from world_model.dynamics import Arena, Claim, DebateResult
"""

# Models
from .models import (
    # Legacy (deviation-based)
    DeviationNode, DeviationType, Edge, EdgeType,
    EvidencePointer, Source,
    # Current (observation + tree-based)
    Observation, ObservationStore,
    LODLevel, LODProtocol, LODPayload, TrivialLODProtocol, to_level,
    EventType, Event, EngineClock, Lineage, StakeWeightGraph,
    LineageRecorder, attach_lineage_to_tendencies,
    RetentionPolicy, UnboundedPolicy, DropOldestPolicy,
    RefuseWhenFullPolicy, BoundedRingPlusCompactionPolicy,
    CompactionTier, OutboxFullError, policy_from_dict,
    Tendency, TendencySet,
    TendencySpec, TendencyFactory, DefaultTendencyFactory,
    LEGACY_PERSONALITY_SPECS, build_legacy_personality_set,
    Tree, TreeStore, Node, Position, Stake,
)

# Attention curves (top-level convenience)
from .attention import (
    NoveltyAttentionCurve,
    AttentionState,
    EXPLORER_CURVE,
    BALANCED_CURVE,
    CONSERVATIVE_CURVE,
    sigmoid,
)

# Dynamics (reseed-and-equilibrate is the canonical operation)
from .dynamics import (
    PresentState,
    Substitution,
    ReseedResult,
    reseed_and_equilibrate,
)

__all__ = [
    # Legacy
    "DeviationNode", "DeviationType", "Edge", "EdgeType",
    "EvidencePointer", "Source",
    # Observations
    "Observation", "ObservationStore",
    # LOD scaffolding
    "LODLevel", "LODProtocol", "LODPayload", "TrivialLODProtocol", "to_level",
    # Lineage
    "EventType", "Event", "EngineClock", "Lineage", "StakeWeightGraph",
    "LineageRecorder", "attach_lineage_to_tendencies",
    "RetentionPolicy", "UnboundedPolicy", "DropOldestPolicy",
    "RefuseWhenFullPolicy", "BoundedRingPlusCompactionPolicy",
    "CompactionTier", "OutboxFullError", "policy_from_dict",
    # Tendencies + factory
    "Tendency", "TendencySet",
    "TendencySpec", "TendencyFactory", "DefaultTendencyFactory",
    "LEGACY_PERSONALITY_SPECS", "build_legacy_personality_set",
    # Trees
    "Tree", "TreeStore", "Node", "Position", "Stake",
    # Attention curves
    "NoveltyAttentionCurve", "AttentionState",
    "EXPLORER_CURVE", "BALANCED_CURVE", "CONSERVATIVE_CURVE",
    "sigmoid",
    # Dynamics
    "PresentState", "Substitution", "ReseedResult", "reseed_and_equilibrate",
]

__version__ = "2.0.0"
