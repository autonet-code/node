"""Aggregator-side: merge multiple solvers' stake deltas into one.

Same protocol contract as ``aggregate_weight_deltas``: takes a list of
deltas plus optional weights, returns a single merged delta. Then
``apply_stake_delta`` writes the merged delta into a base world to
produce the next global model.

Merge semantics
---------------

Stake deltas are linear-in-contributions:

  - ``node_stakes[node_id].pos_stake_delta`` and ``.neg_stake_delta``
    sum, each weighted by the per-solver weight (alignment_score in
    autonet's protocol).
  - ``new_nodes`` are content-addressed; two solvers proposing the same
    sub-claim under the same parent will produce the same node_id,
    and the union dedupes naturally.
  - ``absorbed_observations`` is a union (the global model treats an
    observation as 'seen' if any solver absorbed it).

This is associative and commutative, like FedAvg.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World
from world_model.models.tree import Position, Stake


logger = logging.getLogger(__name__)


def aggregate_stake_deltas(
    deltas: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Merge a list of solvers' stake deltas. Returns a delta with the
    same shape as a single solver's submission.
    """
    if not deltas:
        return {"node_stakes": {}, "new_nodes": [], "absorbed_observations": []}

    if weights is None:
        weights = [1.0] * len(deltas)
    total_weight = sum(weights)
    if total_weight <= 0:
        normalized = [1.0 / len(deltas)] * len(deltas)
    else:
        normalized = [w / total_weight for w in weights]

    # Sum per-node stake changes weighted by normalized weight
    merged_node_stakes: Dict[str, Dict[str, float]] = {}
    for d, w in zip(deltas, normalized):
        for node_id, change in d.get("node_stakes", {}).items():
            if node_id not in merged_node_stakes:
                merged_node_stakes[node_id] = {
                    "pos_stake_delta": 0.0,
                    "neg_stake_delta": 0.0,
                }
            merged_node_stakes[node_id]["pos_stake_delta"] += w * change.get("pos_stake_delta", 0.0)
            merged_node_stakes[node_id]["neg_stake_delta"] += w * change.get("neg_stake_delta", 0.0)

    # Union of new nodes; same content-addressed id -> identical structure
    merged_new_nodes: Dict[str, Dict[str, Any]] = {}
    for d in deltas:
        for nrec in d.get("new_nodes", []):
            nid = nrec["node_id"]
            if nid not in merged_new_nodes:
                merged_new_nodes[nid] = nrec

    # Union of absorbed observations
    merged_obs = set()
    for d in deltas:
        for obs_id in d.get("absorbed_observations", []):
            merged_obs.add(obs_id)

    out = {
        "node_stakes": merged_node_stakes,
        "new_nodes": list(merged_new_nodes.values()),
        "absorbed_observations": sorted(merged_obs),
    }
    logger.info(
        "aggregated %d stake deltas: %d nodes touched, %d new nodes",
        len(deltas), len(merged_node_stakes), len(merged_new_nodes),
    )
    return out


def apply_stake_delta(world: World, delta: Dict[str, Any]) -> World:
    """Apply a (merged) stake delta to a base world in place.

    Solvers and the aggregator each have their own UUID-based node ids
    for the SAME charter tendencies' roots. To bridge them we maintain
    a remap from solver-side ids to live-world ids:

      - Each tendency's root is keyed by tendency_id (solvers know
        which tendency they're under).
      - New nodes' content-addressed identity (parent_id, coords,
        polarity, position) gives them a deterministic id once we
        know the live parent. We sprout the child and remember the
        mapping.

    This means two solvers proposing the same sub-claim under the same
    parent will, after merge, end up at the same node in the live
    world.
    """
    # Build initial remap: solver root id -> live root id, by tendency.
    # We learn solver root ids from any new_node whose parent_id maps
    # to a tendency's root.
    remap: Dict[str, str] = {}
    new_nodes_by_id: Dict[str, Dict[str, Any]] = {
        n["node_id"]: n for n in delta.get("new_nodes", [])
    }

    def resolve_parent(nrec: Dict[str, Any]) -> Optional[str]:
        parent_id = nrec["parent_id"]
        if parent_id in remap:
            return remap[parent_id]
        tendency = world.tendencies.get(nrec.get("tendency_id"))
        if tendency is None:
            return None
        # Is the solver's parent_id itself a top-level root we know?
        # If the parent isn't in new_nodes_by_id, then the solver
        # treated it as the tendency's root.
        if parent_id not in new_nodes_by_id:
            live_root_id = tendency.tree.root_node.id
            remap[parent_id] = live_root_id
            return live_root_id
        # Parent is itself a new node; check if we've already sprouted
        # it (then it's in remap).
        return remap.get(parent_id)

    sprouted: set[str] = set()
    pending = list(new_nodes_by_id.values())
    safety = 0
    while pending and safety < 10:
        safety += 1
        next_round: List[Dict[str, Any]] = []
        for nrec in pending:
            tendency = world.tendencies.get(nrec.get("tendency_id"))
            if tendency is None:
                logger.warning("new node %s targets unknown tendency %s",
                               nrec["node_id"], nrec.get("tendency_id"))
                continue
            live_parent = resolve_parent(nrec)
            if live_parent is None:
                next_round.append(nrec)
                continue
            try:
                new_node = tendency.sprout_child(
                    parent_node_id=live_parent,
                    position=Position(nrec["position"]),
                    anchor=tuple(nrec.get("coords") or ()),
                    polarity_axis=tuple(nrec.get("polarity_axis") or ()),
                    content=nrec.get("content", ""),
                )
                remap[nrec["node_id"]] = new_node.id
                sprouted.add(nrec["node_id"])
            except Exception as e:
                logger.warning("failed to sprout %s: %s", nrec["node_id"], e)
        if len(next_round) == len(pending):
            # No progress this pass -- bail to avoid infinite loop.
            for nrec in next_round:
                logger.warning("stranded new node %s (parent %s unresolved)",
                               nrec["node_id"], nrec["parent_id"])
            break
        pending = next_round

    # Apply stake adjustments, using remap when available.
    for node_id, change in delta.get("node_stakes", {}).items():
        d_pos = change.get("pos_stake_delta", 0.0)
        d_neg = change.get("neg_stake_delta", 0.0)
        live_id = remap.get(node_id, node_id)
        for tendency in world.tendencies.values():
            node = tendency.tree.get_node(live_id)
            if node is None:
                continue
            if abs(d_pos) > 1e-9:
                node.add_stake("aggregator_pos", d_pos)
            if abs(d_neg) > 1e-9:
                node.add_stake("aggregator_neg", -d_neg)
            break

    return world
