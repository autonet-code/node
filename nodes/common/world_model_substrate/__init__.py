"""World-model substrate adapter.

A drop-in replacement for the VL-JEPA / TextJEPA training pipeline that
plugs into the same protocol slots:

  - ``train_world_model_on_task`` produces a stake delta with the same
    shape as ``train_vljepa_on_task``'s weight delta. The aggregator's
    contract (a dict of named arrays) is preserved.
  - ``aggregate_stake_deltas`` is FedAvg-shaped: weighted sum,
    associative, commutative. Same input/output protocol as
    ``aggregate_weight_deltas``.
  - ``verify_world_model_solution`` returns a 0-100 numeric score, same
    shape as ``verify_jepa_solution``.
  - ``infer_with_world_model`` runs a single inference pass against the
    aggregated global model, returning predictions + probabilities in the
    same JSON schema the inference node already emits.

The engine itself lives in ``world_model.generalized``. This module is
the autonet-side adapter only.
"""

from .adapter import (
    train_world_model_on_task,
    serialize_world,
    deserialize_world,
)
from .aggregate import aggregate_stake_deltas, apply_stake_delta
from .verify import verify_world_model_solution
from .infer import infer_with_world_model

__all__ = [
    "train_world_model_on_task",
    "serialize_world",
    "deserialize_world",
    "aggregate_stake_deltas",
    "apply_stake_delta",
    "verify_world_model_solution",
    "infer_with_world_model",
]
