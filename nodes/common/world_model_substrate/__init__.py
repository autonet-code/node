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
]
