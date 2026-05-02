"""Usefulness training: substrate trained on (problem, resolution, outcome).

Parallel entry point to train_world_model_on_task. Where the alignment
path scored worker turns against four hardcoded charter tendencies,
this path trains the substrate on the procedural knowledge of what
makes work useful.

Shape of work units
-------------------

Each work unit is (problem, resolution, outcome) where:
  - problem: the request the worker was responding to.
  - resolution: what the worker actually did, distilled to text.
  - outcome: an Outcome dataclass, capturing accepted/kept/built_on/paid.

A work unit produces:
  - One observation in the substrate's coordinate space, located at
    coords_for_problem_resolution(problem, resolution).
  - Stake magnitude proportional to the outcome's positive signal:
    a successful outcome (accepted=+1, kept=+1) stakes more strongly
    than a no-signal outcome (which barely registers).
  - A negative outcome stakes the resolution as CON evidence (this
    approach didn't land well in this problem region), still informative.

Bootstrap roots
---------------

The world starts with a small set of utility tendencies that organize
the graph's high-level structure. These are NOT the charter. The
charter tendencies are a separate gate (mint_gate.py); these are
just initial scaffolding around which usefulness sub-claims grow.

  - "good_resolution": "this approach worked for this kind of problem"
  - "novel_resolution": "this approach is new to the network"

Two roots is intentional: minimal scaffolding, real structure emerges
from observed work patterns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)

from .events import (
    Event,
    ObservationAdded,
    SubClaimSprouted,
    snapshot_node_scores,
)
from .outcomes import Outcome
from .usefulness_coords import (
    DEFAULT_DIM,
    coords_for_problem_resolution,
    default_usefulness_embedder,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap utility roots (NOT the charter)
# ---------------------------------------------------------------------------


def build_usefulness_world(
    dim: int = DEFAULT_DIM,
    bandwidth: float = 0.5,
) -> World:
    """Build a world with two utility-shaped bootstrap tendencies.

    The tendencies live in a `dim`-dimensional embedding space; their
    anchors are placed at random orthogonal-ish directions so they
    occupy distinct regions of the space without dictating where
    sub-claims should grow.
    """
    world = World()

    # Two anchors at opposite corners of the first axis. The substrate's
    # coordinate logic uses anchor + polarity for orientation; the actual
    # discrimination happens via emerged sub-claims, not these roots.
    pos_anchor = tuple([1.0] + [0.0] * (dim - 1))
    neg_anchor = tuple([-1.0] + [0.0] * (dim - 1))

    world.add_tendency(GeneralizedTendency(
        id="good_resolution",
        thesis="This approach worked for this kind of problem.",
        anchor=pos_anchor,
        polarity_axis=pos_anchor,
        budget=1.0,
        bandwidth=bandwidth,
        smooth_promotion=True,
    ))
    world.add_tendency(GeneralizedTendency(
        id="novel_resolution",
        thesis="This approach is new to the network.",
        anchor=neg_anchor,
        polarity_axis=neg_anchor,
        budget=1.0,
        bandwidth=bandwidth,
        smooth_promotion=True,
    ))
    return world


# ---------------------------------------------------------------------------
# Work unit -> Observation
# ---------------------------------------------------------------------------


def _obs_id(problem: str, resolution: str) -> str:
    payload = json.dumps(
        {"p": problem[:200], "r": resolution[:200]},
        sort_keys=True,
    )
    return "obs_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def work_unit_to_observation(
    problem: str,
    resolution: str,
    outcome: Outcome,
    embedder: Optional[Callable[[str], Tuple[float, ...]]] = None,
) -> Observation:
    """Convert a work unit to an Observation for substrate ingestion.

    The Observation's coords are the embedding of (problem, resolution).
    Outcome influences MAGNITUDE downstream during staking, not coords.
    Coords are about WHERE in problem-space the work lives; outcome is
    about HOW WELL it landed.
    """
    coords = coords_for_problem_resolution(problem, resolution, embedder=embedder)
    label = f"problem: {problem[:60]} | outcome strength: {outcome.signal_strength:.2f}"
    return Observation(id=_obs_id(problem, resolution), coords=coords, label=label)


# ---------------------------------------------------------------------------
# Event recorder (mirrors adapter.py's structure)
# ---------------------------------------------------------------------------


class _EventRecorder:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.events: List[Event] = []
        self._seq = 0

    def observation_added(self, obs: Observation) -> None:
        self._seq += 1
        self.events.append(ObservationAdded(
            seq=self._seq,
            author_agent=self.agent_id,
            obs_id=obs.id,
            coords=list(obs.coords),
            label=obs.label,
        ))

    def sub_claims_after_equilibrate(
        self,
        world: World,
        before_node_ids: set[str],
    ) -> None:
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if node.id in before_node_ids:
                    continue
                claim = tendency._node_to_claim.get(node.id)
                self._seq += 1
                self.events.append(SubClaimSprouted(
                    seq=self._seq,
                    author_agent=self.agent_id,
                    tendency_id=tendency.id,
                    parent_id=node.parent_id or "",
                    node_id=node.id,
                    position=node.position.value,
                    coords=list(claim.anchor) if claim else [],
                    polarity_axis=list(claim.polarity_axis) if claim else [],
                    content=node.content,
                ))


def _all_node_ids(world: World) -> set[str]:
    ids: set[str] = set()
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            ids.add(node.id)
    return ids


# ---------------------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------------------


def train_world_model_on_usefulness(
    work_units: List[Tuple[str, str, Outcome]],
    dim: int = DEFAULT_DIM,
    bandwidth: float = 0.5,
    epochs: int = 1,
    agent_id: str = "default-trainer",
    embedder: Optional[Callable[[str], Tuple[float, ...]]] = None,
    seed_world: Optional[World] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the substrate on a list of (problem, resolution, outcome).

    Args:
      work_units: list of (problem, resolution, outcome) triples.
      dim: usefulness coordinate dimension.
      bandwidth: tendency bandwidth (smaller = sharper neighborhoods
        in the higher-dim embedding space).
      epochs: passes through the work units.
      agent_id: solver identifier, attached to each emitted event.
      embedder: optional override; default = default_usefulness_embedder().
      seed_world: continue from an existing world instead of fresh.

    Returns: (contribution, metrics) compatible with the protocol slot
    used by the alignment-path adapter.
    """
    start = time.time()
    if embedder is None:
        embedder = default_usefulness_embedder(dim=dim)

    world = seed_world if seed_world is not None else build_usefulness_world(
        dim=dim, bandwidth=bandwidth,
    )

    recorder = _EventRecorder(agent_id=agent_id)
    n_observations = 0

    for epoch in range(max(1, int(epochs))):
        for problem, resolution, outcome in work_units:
            obs = work_unit_to_observation(problem, resolution, outcome, embedder)
            before_ids = _all_node_ids(world)
            world.add_observation(obs)
            recorder.observation_added(obs)
            equilibrate(world, max_rounds=8, tolerance=1e-3)
            recorder.sub_claims_after_equilibrate(world, before_ids)
            n_observations += 1
        world.clear_observations()

    score_snapshot = snapshot_node_scores(world)
    elapsed = time.time() - start

    # Per-unit outcome carry-over for the reconciler -- so the mint
    # gate can look at outcome strength when deciding whether a node
    # qualifies. Stored alongside events as metadata.
    outcome_metadata = []
    for problem, resolution, outcome in work_units:
        oid = _obs_id(problem, resolution)
        outcome_metadata.append({
            "obs_id": oid,
            "outcome": list(outcome.to_coords()),
            "signal_strength": outcome.signal_strength,
        })

    contribution = {
        "events": [e.to_dict() for e in recorder.events],
        "score_snapshot_after": score_snapshot,
        "agent_id": agent_id,
        "outcome_metadata": outcome_metadata,
        "substrate_mode": "usefulness",
    }
    metrics = {
        "elapsed_seconds": elapsed,
        "epochs": epochs,
        "n_work_units": len(work_units),
        "n_observations": n_observations,
        "n_events": len(recorder.events),
        "n_nodes_after": len(score_snapshot),
        "root_scores": world.root_scores(),
        "substrate": "world-model-usefulness",
    }
    logger.info(
        "usefulness training: agent=%s, %d work units, %d events, %.2fs",
        agent_id, len(work_units), len(recorder.events), elapsed,
    )
    return contribution, metrics
