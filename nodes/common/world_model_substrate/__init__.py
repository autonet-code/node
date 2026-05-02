"""World-model substrate adapter.

Drop-in replacement for the VL-JEPA / TextJEPA training pipeline that
plugs into the same protocol slots used by autonet's solver,
aggregator, verifier, and inference paths.
"""

from .adapter import (
    train_world_model_on_task,
    serialize_world,
    deserialize_world,
    build_charter_world,
    turn_to_observation,
    CHARTER,
    N_DIMS,
)
from .aggregate import (
    aggregate_contributions,
    apply_events,
    aggregate_stake_deltas,   # backwards-compat alias
    apply_stake_delta,        # backwards-compat alias
)
from .verify import verify_world_model_solution
from .infer import infer_with_world_model
from .events import (
    Event,
    SubClaimSprouted,
    ObservationAdded,
    event_from_dict,
    snapshot_node_scores,
)
from .reconcile import (
    EpochSnapshots,
    reconcile_epoch,
)
from .outcomes import (
    Outcome,
    outcome_from_conversation,
    outcomes_for_agent_dir,
)
from .usefulness_coords import (
    HashingEmbedder,
    SentenceTransformersEmbedder,
    default_usefulness_embedder,
    coords_for_problem_resolution,
    coords_for_query,
)
from .usefulness_training import (
    build_usefulness_world,
    train_world_model_on_usefulness,
    work_unit_to_observation,
)
from .mint_gate import (
    DEFAULT_CHARTER_IDS,
    apply_mint_gate,
    charter_violation_score,
)

__all__ = [
    "train_world_model_on_task",
    "serialize_world",
    "deserialize_world",
    "build_charter_world",
    "turn_to_observation",
    "CHARTER",
    "N_DIMS",
    "aggregate_contributions",
    "apply_events",
    "aggregate_stake_deltas",
    "apply_stake_delta",
    "verify_world_model_solution",
    "infer_with_world_model",
    "Event",
    "SubClaimSprouted",
    "ObservationAdded",
    "event_from_dict",
    "snapshot_node_scores",
    "EpochSnapshots",
    "reconcile_epoch",
    "Outcome",
    "outcome_from_conversation",
    "outcomes_for_agent_dir",
    "HashingEmbedder",
    "SentenceTransformersEmbedder",
    "default_usefulness_embedder",
    "coords_for_problem_resolution",
    "coords_for_query",
    "build_usefulness_world",
    "train_world_model_on_usefulness",
    "work_unit_to_observation",
    "DEFAULT_CHARTER_IDS",
    "apply_mint_gate",
    "charter_violation_score",
]
