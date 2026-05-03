"""Solver-side adapter: task -> events recorded during equilibration.

The protocol contract: take a task spec and (optionally) a global model
CID, produce a contribution payload. Internally we run the world-model
engine and record each sprout and observation as an Event.

Output shape
------------

  {
    "events": [Event.to_dict(), ...],
    "score_snapshot_after": {node_id: float, ...},   # post-equilibration
    "metrics": {...},
  }

The aggregator replays events on the global world, equilibrates, and
compares pre/post scores at epoch boundary to compute mint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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
    event_from_dict,
    snapshot_node_scores,
)

try:
    from .score_turn import score_turn_4d
except ImportError:
    def score_turn_4d(turn: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Fallback when score_turn module isn't built yet -- returns zeros."""
        return (0.0, 0.0, 0.0, 0.0)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Charter
# ---------------------------------------------------------------------------

CHARTER = [
    {"id": "life_precious",
     "thesis": "Life is precious and should be preserved.",
     "axis_index": 0},
    {"id": "self_preservation",
     "thesis": "The system should preserve its own continuity.",
     "axis_index": 1},
    {"id": "promotion_of_intelligence",
     "thesis": "Intelligence in any form should be promoted.",
     "axis_index": 2},
    {"id": "evolution",
     "thesis": "Forward advancement of capability is desirable.",
     "axis_index": 3},
]
N_DIMS = len(CHARTER)


def build_charter_world(bandwidth: float = 1.5) -> World:
    world = World()
    for entry in CHARTER:
        i = entry["axis_index"]
        anchor = tuple(1.0 if j == i else 0.0 for j in range(N_DIMS))
        world.add_tendency(GeneralizedTendency(
            id=entry["id"],
            thesis=entry["thesis"],
            anchor=anchor,
            polarity_axis=anchor,
            budget=1.0,
            bandwidth=bandwidth,
            smooth_promotion=True,
        ))
    return world


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _obs_id_from_turn(turn: Dict[str, Any]) -> str:
    return "obs_" + hashlib.sha256(
        json.dumps(turn, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def turn_to_observation(turn: Dict[str, Any], turn_index: int = 0) -> Observation:
    """Convert one agent-turn record to a 4D coordinate observation.

    Prefers explicit *_impact fields if present (test/override mode);
    otherwise calls the score_turn_4d heuristic.
    """
    has_explicit = any(
        k in turn for k in (
            "life_impact", "self_pres_impact",
            "intelligence_impact", "evolution_impact",
        )
    )
    if has_explicit:
        coords = (
            float(turn.get("life_impact", 0.0)),
            float(turn.get("self_pres_impact", 0.0)),
            float(turn.get("intelligence_impact", 0.0)),
            float(turn.get("evolution_impact", 0.0)),
        )
    else:
        coords = score_turn_4d(turn)
    label = turn.get("label", f"turn_{turn_index}")
    return Observation(id=_obs_id_from_turn(turn), coords=coords, label=label)


# ---------------------------------------------------------------------------
# Serialization (kept for global-model storage)
# ---------------------------------------------------------------------------


def serialize_world(world: World) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tendencies": {}, "nodes": []}
    for tendency in world.tendencies.values():
        out["tendencies"][tendency.id] = {
            "thesis": tendency.thesis,
            "anchor": list(tendency.anchor),
            "polarity_axis": list(tendency.polarity_axis),
            "budget": tendency.budget,
            "bandwidth": tendency.bandwidth,
        }
        for node in tendency.tree.all_nodes():
            claim = tendency._node_to_claim.get(node.id)
            out["nodes"].append({
                "tendency_id": tendency.id,
                "node_id": node.id,
                "parent_id": node.parent_id,
                "position": node.position.value,
                "content": node.content,
                "observation_id": node.observation_id,
                "coords": list(claim.anchor) if claim else [],
                "polarity_axis": list(claim.polarity_axis) if claim else [],
                "stakes": [
                    {"agent_id": s.agent_id, "weight": s.weight}
                    for s in node.stakes
                ],
            })
    return out


def deserialize_world(payload: Dict[str, Any]) -> World:
    from world_model.models.tree import Node, Position, Stake
    world = World()
    for tid, td in payload.get("tendencies", {}).items():
        world.add_tendency(GeneralizedTendency(
            id=tid,
            thesis=td["thesis"],
            anchor=tuple(td["anchor"]),
            polarity_axis=tuple(td["polarity_axis"]),
            budget=td.get("budget", 1.0),
            bandwidth=td.get("bandwidth", 1.5),
            smooth_promotion=True,
        ))
    nodes = sorted(payload.get("nodes", []),
                   key=lambda n: 0 if n["parent_id"] is None else 1)
    serialized_roots: Dict[str, str] = {}
    for nrec in nodes:
        if nrec["parent_id"] is None:
            serialized_roots[nrec["tendency_id"]] = nrec["node_id"]
    root_remap: Dict[str, str] = {}
    for tid, sroot in serialized_roots.items():
        live = world.tendencies[tid].tree.root_node.id
        root_remap[sroot] = live
    for nrec in nodes:
        if nrec["parent_id"] is None:
            root = world.tendencies[nrec["tendency_id"]].tree.root_node
            for sk in nrec.get("stakes", []):
                root.add_stake(sk["agent_id"], sk["weight"])
            continue
        tendency = world.tendencies[nrec["tendency_id"]]
        parent_id = root_remap.get(nrec["parent_id"], nrec["parent_id"])
        if tendency.tree.get_node(parent_id) is None:
            logger.warning("dropping node %s: parent %s not found",
                           nrec["node_id"], parent_id)
            continue
        new_node = tendency.sprout_child(
            parent_node_id=parent_id,
            position=Position(nrec["position"]),
            anchor=tuple(nrec.get("coords") or ()),
            polarity_axis=tuple(nrec.get("polarity_axis") or ()),
            content=nrec.get("content", ""),
        )
        for sk in nrec.get("stakes", []):
            new_node.add_stake(sk["agent_id"], sk["weight"])
    return world


# ---------------------------------------------------------------------------
# Event recording during training
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Watches a world for new sprouts/observations during training and
    emits Event records.

    Detection works by snapshotting the node set before and after each
    equilibrate call, attributing new nodes to the agent who caused
    the round.
    """

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


def train_world_model_on_task(
    task_spec: Dict[str, Any],
    global_model_cid: Optional[str] = None,
    store: Any = None,
    text_data_source: Any = None,
    visual_data_source: Any = None,
    epochs: int = 1,
    learning_rate: float = 1e-3,   # accepted for protocol parity, unused
    modalities: Optional[list] = None,
    agent_id: str = "default-solver",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the world-model substrate on a task.

    Returns (contribution, metrics) where contribution is shaped:
      {
        "events": [...],
        "score_snapshot_after": {node_id: float},
        "agent_id": str,
      }
    """
    start = time.time()

    # 1. Seed world
    if global_model_cid and store is not None:
        try:
            payload = store.get_json(global_model_cid)
            if payload:
                world = deserialize_world(payload)
                logger.info("loaded global world model %s...", global_model_cid[:16])
            else:
                world = build_charter_world()
        except Exception as e:
            logger.warning("failed to load global model: %s; using fresh charter", e)
            world = build_charter_world()
    else:
        world = build_charter_world()

    # 2. Gather turns
    turns: List[Dict[str, Any]] = []
    if text_data_source is not None:
        for i, turn in enumerate(text_data_source):
            if i >= int(task_spec.get("max_observations", 256)):
                break
            turns.append(turn)
    elif "turns" in task_spec:
        turns = list(task_spec["turns"])

    # 3. Record events during training
    recorder = _EventRecorder(agent_id=agent_id)

    for epoch in range(max(1, int(epochs))):
        for i, turn in enumerate(turns):
            obs = turn_to_observation(turn, turn_index=epoch * 10000 + i)
            before_ids = _all_node_ids(world)
            world.add_observation(obs)
            recorder.observation_added(obs)
            equilibrate(world, max_rounds=8, tolerance=1e-3)
            recorder.sub_claims_after_equilibrate(world, before_ids)
        world.clear_observations()

    # 4. Capture post-training score snapshot
    score_snapshot = snapshot_node_scores(world)

    elapsed = time.time() - start
    contribution = {
        "events": [e.to_dict() for e in recorder.events],
        "score_snapshot_after": score_snapshot,
        "agent_id": agent_id,
    }
    metrics = {
        "elapsed_seconds": elapsed,
        "epochs": epochs,
        "n_observations": len(turns) * max(1, int(epochs)),
        "n_events": len(recorder.events),
        "n_nodes_after": len(score_snapshot),
        "root_scores": world.root_scores(),
        "substrate": "world-model",
    }
    logger.info(
        "world-model training: agent=%s, %d turns, %d events, %.2fs",
        agent_id, len(turns), len(recorder.events), elapsed,
    )
    return contribution, metrics
