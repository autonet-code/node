"""Solver-side adapter: task -> World -> equilibrate -> stake delta.

The protocol contract: take a task spec and (optionally) a global model
CID, produce a (weight_delta, metrics) tuple compatible with the
existing FedAvg pipeline. Internally we run the world-model engine
instead of a neural network.

Stake delta shape
-----------------

Same outer shape as a JEPA weight_delta: a dict of named arrays. The
keys are stake-event categories rather than layer names:

  {
    "node_stakes": {
        "<node_id>": {
            "pos_stake_delta": float,   # change in net positive stake
            "neg_stake_delta": float,   # change in net negative stake
        },
        ...
    },
    "new_nodes": [
        {
            "node_id": str,             # content-addressed, sha256 of (parent_id, coords, axis)
            "parent_id": str,
            "position": "pro" | "con",
            "coords": [float, ...],
            "polarity_axis": [float, ...],
            "content": str,
        },
        ...
    ],
    "absorbed_observations": [str, ...]  # observation ids absorbed in this run
  }

This is the smallest format that preserves enough information for
``apply_stake_delta`` to merge it into a global tree, and for
``aggregate_stake_deltas`` to combine multiple solvers' deltas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Charter: the four root tendencies the substrate equilibrates against.
# ---------------------------------------------------------------------------

CHARTER = [
    {
        "id": "life_precious",
        "thesis": "Life is precious and should be preserved.",
        "axis_index": 0,
    },
    {
        "id": "self_preservation",
        "thesis": "The system should preserve its own continuity.",
        "axis_index": 1,
    },
    {
        "id": "promotion_of_intelligence",
        "thesis": "Intelligence in any form should be promoted.",
        "axis_index": 2,
    },
    {
        "id": "evolution",
        "thesis": "Forward advancement of capability is desirable.",
        "axis_index": 3,
    },
]
N_DIMS = len(CHARTER)


def build_charter_world(bandwidth: float = 1.5) -> World:
    """Construct a World seeded with the four charter tendencies.

    Each tendency anchors at +1 on its dimension with polarity axis
    pointing in the same direction (PRO = honoring the principle).
    """
    world = World()
    for entry in CHARTER:
        i = entry["axis_index"]
        anchor = tuple(1.0 if j == i else 0.0 for j in range(N_DIMS))
        polarity = anchor
        world.add_tendency(GeneralizedTendency(
            id=entry["id"],
            thesis=entry["thesis"],
            anchor=anchor,
            polarity_axis=polarity,
            budget=1.0,
            bandwidth=bandwidth,
            smooth_promotion=True,
        ))
    return world


# ---------------------------------------------------------------------------
# Content-addressed node IDs: two solvers proposing the same sub-claim
# under the same parent at the same coords end up with the same ID.
# ---------------------------------------------------------------------------


def content_address_node(
    parent_id: str,
    position: str,
    coords: tuple,
    polarity_axis: tuple,
) -> str:
    """Deterministic node id for a sprouted child. Hash of the
    structural inputs so independent solvers converge on the same id
    when they propose the same claim.
    """
    payload = json.dumps(
        {
            "parent": parent_id,
            "pos": position,
            "coords": list(coords),
            "axis": list(polarity_axis),
        },
        sort_keys=True,
    )
    return "n_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Observation construction from agent-turn data
# ---------------------------------------------------------------------------


def turn_to_observation(turn: Dict[str, Any], turn_index: int) -> Observation:
    """Convert one agent-turn record to a 4D coordinate observation.

    The four coords correspond to (life, self_pres, intelligence,
    evolution). For v1 this is heuristic-driven from turn fields:

      - turn['life_impact']         in [-1, 1]
      - turn['self_pres_impact']    in [-1, 1]
      - turn['intelligence_impact'] in [-1, 1]
      - turn['evolution_impact']    in [-1, 1]

    Missing fields default to 0. A future adapter version can let the
    world model itself score turns it has seen patterns for, but the
    interface stays the same.
    """
    coords = (
        float(turn.get("life_impact", 0.0)),
        float(turn.get("self_pres_impact", 0.0)),
        float(turn.get("intelligence_impact", 0.0)),
        float(turn.get("evolution_impact", 0.0)),
    )
    label = turn.get("label", f"turn_{turn_index}")
    obs_id = "obs_" + hashlib.sha256(
        json.dumps(turn, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return Observation(id=obs_id, coords=coords, label=label)


# ---------------------------------------------------------------------------
# Snapshotting: a baseline state to diff against
# ---------------------------------------------------------------------------


def snapshot_stakes(world: World) -> Dict[str, Dict[str, float]]:
    """Capture per-node net positive and net negative stake.

    Returns ``{node_id: {"pos": float, "neg": float}}``.
    """
    snap: Dict[str, Dict[str, float]] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            pos = sum(s.weight for s in node.stakes if s.weight > 0)
            neg = sum(-s.weight for s in node.stakes if s.weight < 0)
            snap[node.id] = {"pos": pos, "neg": neg}
    return snap


def snapshot_nodes(world: World) -> Dict[str, Dict[str, Any]]:
    """Capture the structural metadata of every node so we can detect
    new nodes after equilibration.
    """
    snap: Dict[str, Dict[str, Any]] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            claim = tendency._node_to_claim.get(node.id)
            snap[node.id] = {
                "tendency_id": tendency.id,
                "parent_id": node.parent_id or "",
                "position": node.position.value,
                "content": node.content,
                "coords": list(claim.anchor) if claim else [],
                "polarity_axis": list(claim.polarity_axis) if claim else [],
            }
    return snap


# ---------------------------------------------------------------------------
# Serialization: enough to round-trip a world for global-model storage
# ---------------------------------------------------------------------------


def serialize_world(world: World) -> Dict[str, Any]:
    """Compact JSON-friendly representation of a world's structure
    and accumulated stakes. Lossy on lineages and integrated frames
    (those rebuild from observations on apply); preserves what the
    aggregator and inference paths need.
    """
    out: Dict[str, Any] = {
        "tendencies": {},
        "nodes": [],
    }
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
    """Rebuild a World from a serialized payload. Recreates tendencies,
    sub-nodes (in parent-first order), and stake values.
    """
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
    # First pass: create non-root nodes in parent-first order
    nodes = sorted(payload.get("nodes", []),
                   key=lambda n: 0 if n["parent_id"] is None else 1)
    by_id: Dict[str, Node] = {}
    for tendency in world.tendencies.values():
        by_id[tendency.tree.root_node.id] = tendency.tree.root_node
    # Index serialized roots by tendency
    serialized_roots: Dict[str, str] = {}
    for nrec in nodes:
        if nrec["parent_id"] is None:
            serialized_roots[nrec["tendency_id"]] = nrec["node_id"]
    # Build a map from serialized-root-id -> live-tendency-root-id
    root_remap: Dict[str, str] = {}
    for tid, sroot in serialized_roots.items():
        live = world.tendencies[tid].tree.root_node.id
        root_remap[sroot] = live
        # also map root to itself in by_id
        by_id[sroot] = world.tendencies[tid].tree.root_node
    # Second pass: sprout children
    for nrec in nodes:
        if nrec["parent_id"] is None:
            # apply stakes to the tendency root (only)
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
        by_id[nrec["node_id"]] = new_node
        for sk in nrec.get("stakes", []):
            new_node.add_stake(sk["agent_id"], sk["weight"])
    return world


# ---------------------------------------------------------------------------
# Stake-delta computation
# ---------------------------------------------------------------------------


def compute_stake_delta(
    before_stakes: Dict[str, Dict[str, float]],
    before_nodes: Dict[str, Dict[str, Any]],
    after: World,
    absorbed_obs: List[str],
) -> Dict[str, Any]:
    after_stakes = snapshot_stakes(after)
    after_nodes = snapshot_nodes(after)

    node_stakes: Dict[str, Dict[str, float]] = {}
    for node_id, after_v in after_stakes.items():
        before_v = before_stakes.get(node_id, {"pos": 0.0, "neg": 0.0})
        d_pos = after_v["pos"] - before_v["pos"]
        d_neg = after_v["neg"] - before_v["neg"]
        if abs(d_pos) > 1e-9 or abs(d_neg) > 1e-9:
            node_stakes[node_id] = {
                "pos_stake_delta": d_pos,
                "neg_stake_delta": d_neg,
            }

    new_nodes: List[Dict[str, Any]] = []
    for node_id, meta in after_nodes.items():
        if node_id in before_nodes:
            continue
        new_nodes.append({
            "node_id": node_id,
            "parent_id": meta["parent_id"],
            "position": meta["position"],
            "coords": meta["coords"],
            "polarity_axis": meta["polarity_axis"],
            "content": meta["content"],
            "tendency_id": meta["tendency_id"],
        })

    return {
        "node_stakes": node_stakes,
        "new_nodes": new_nodes,
        "absorbed_observations": absorbed_obs,
    }


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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the world-model substrate on a task. Returns
    ``(stake_delta, metrics)`` matching the JEPA training contract.

    Algorithm:
      1. Build (or load) the seed world from the global model CID.
      2. Convert task observations (turns) to 4D Observations.
      3. Snapshot stakes + nodes BEFORE equilibration.
      4. Add observations, run equilibrate.
      5. Optionally repeat (if ``epochs > 1``) -- each epoch lets sub-
         claims grow further capacity through smooth promotion.
      6. Compute the stake delta between snapshot and final state.
      7. Return ``(delta, metrics)``.
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

    # 2. Gather turns from the data source
    turns: List[Dict[str, Any]] = []
    if text_data_source is not None:
        for i, turn in enumerate(text_data_source):
            if i >= int(task_spec.get("max_observations", 256)):
                break
            turns.append(turn)
    elif "turns" in task_spec:
        turns = list(task_spec["turns"])

    # 3. Snapshot
    before_stakes = snapshot_stakes(world)
    before_nodes = snapshot_nodes(world)

    # 4-5. Equilibrate over epochs
    absorbed: List[str] = []
    for epoch in range(max(1, int(epochs))):
        for i, turn in enumerate(turns):
            obs = turn_to_observation(turn, turn_index=epoch * 1000 + i)
            world.add_observation(obs)
            absorbed.append(obs.id)
        equilibrate(world, max_rounds=8, tolerance=1e-3)
        # Drop the observations between epochs so we don't re-process
        # them with stale ids; their effects (stakes, sub-claims) persist.
        world.clear_observations()

    # 6. Compute delta
    delta = compute_stake_delta(before_stakes, before_nodes, world, absorbed)

    elapsed = time.time() - start
    metrics = {
        "elapsed_seconds": elapsed,
        "epochs": epochs,
        "n_observations": len(turns) * max(1, int(epochs)),
        "n_changed_nodes": len(delta["node_stakes"]),
        "n_new_nodes": len(delta["new_nodes"]),
        "root_scores": world.root_scores(),
        "substrate": "world-model",
    }
    logger.info(
        "world-model training: %d turns, %d changed nodes, %d new nodes, %.2fs",
        len(turns), len(delta["node_stakes"]), len(delta["new_nodes"]), elapsed,
    )
    return delta, metrics
