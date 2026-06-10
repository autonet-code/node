"""Exact world snapshot / restore.

``snapshot_world`` captures EVERY piece of state that influences
future evolution of a generalized world; ``restore_world`` writes it
back verbatim (no re-derivation, no sprouting, no frame refresh).
The contract is:

    snapshot_world(restore_world(snapshot_world(w))) == snapshot_world(w)

and, stronger: applying the same observation/equilibration sequence to
``w`` and to ``restore_world(snapshot_world(w))`` keeps them
bit-identical. That contract is what makes checkpoint-restore a safe
fast-boot path (vs. full event-log replay).

State inventory (why each field is here):

  World
    observations    pending set: ``act()`` iterates it
    history         novelty floors / audit
  Node (shared objects, multi-parented across trees)
    id, observation_id, tree_id, content, metadata
    parents         full ParentLink list — the legacy serializer kept
                    only parents[0], which dropped co-parent edges
    stakes          ordered; weights preserved (lindblad writes
                    non-unit weights)
    n               persistent novelty — drives update_novelty and
                    rho_0 coherence
    pro/con_children ordered: float summation order in net_score
  GeneralizedTendency
    ctor params     id/thesis/anchor/polarity/budget/bandwidth + knobs
    tree id + node-index order   iteration order feeds act()
    claims          per-tendency claim tree incl. children order
    frame           integrated set (absorb state) + thresholds
    node_capacity   smooth-promotion state
    _con_positioned re-post short-circuit in act()
    last_stakes     convergence baseline for the next equilibrate

The payload is plain JSON-compatible (lists/dicts/str/float/bool/None)
so it can be dumped with ``json`` and content-addressed for the
network state-sync path.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.tree import Node, ParentLink, Position, Stake
from .coordinate_frame import CoordinateClaim, CoordinateFrame
from .observation import Observation
from .tendency import GeneralizedTendency
from .world import World

SCHEMA = 1

_TENDENCY_KNOBS = (
    "capacity_decay",
    "capacity_reach",
    "capacity_threshold",
    "smooth_promotion",
    "novelty_gamma_pro",
    "novelty_gamma_con",
    "novelty_drift",
    "veto_shaped",
    "veto_score_floor",
)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_world(world: World) -> Dict[str, Any]:
    """Capture the complete evolution-relevant state of a world."""
    # Global observation registry: pending + history + every frame's
    # integrated set (history/integrated entries can outlive the
    # pending set after clear_observations()).
    obs_registry: Dict[str, Dict[str, Any]] = {}

    def _register(obs: Observation) -> None:
        if obs.id not in obs_registry:
            obs_registry[obs.id] = {
                "coords": list(obs.coords),
                "label": obs.label,
            }

    for obs in world.observations.values():
        _register(obs)
    for obs in world.history:
        _register(obs)
    for tendency in world.tendencies.values():
        for obs in tendency.frame.integrated.values():
            _register(obs)

    # Global node registry in first-seen order (tendency order, then
    # each tree's index order). Shared nodes appear once.
    node_order: List[str] = []
    node_by_id: Dict[str, Node] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            if node.id not in node_by_id:
                node_by_id[node.id] = node
                node_order.append(node.id)

    nodes: List[Dict[str, Any]] = []
    for nid in node_order:
        node = node_by_id[nid]
        nodes.append({
            "id": node.id,
            "observation_id": node.observation_id,
            "tree_id": node.tree_id,
            "content": node.content,
            "metadata": dict(node.metadata),
            "n": node.n,
            "parents": [
                [p.parent_id, p.position.value, p.tendency_id]
                for p in node.parents
            ],
            "stakes": [
                {"agent_id": s.agent_id, "weight": s.weight}
                for s in node.stakes
            ],
            "pro_children": [c.id for c in node.pro_children],
            "con_children": [c.id for c in node.con_children],
            # Score caches are observable state, not just an
            # optimization: in a co-parented graph with cycles,
            # net_score depends on which node's recursion broke the
            # cycle first, so a cold-cache world can READ different
            # scores than the warm-cache world it was restored from
            # even when all structure matches. Carrying the caches
            # makes restored reads exact.
            "score_cache": [node._direct_weight, node._net_score],
        })

    tendencies: Dict[str, Any] = {}
    for tid, tendency in world.tendencies.items():
        # Per-tendency claim tree. Claims are mapped through
        # _node_to_claim; children references are converted to node
        # ids via reverse lookup on claim object identity.
        claim_to_nid = {
            id(claim): nid for nid, claim in tendency._node_to_claim.items()
        }
        claims: Dict[str, Any] = {}
        for nid, claim in tendency._node_to_claim.items():
            child_ids = []
            for child in claim.children:
                child_nid = claim_to_nid.get(id(child))
                if child_nid is None:
                    raise ValueError(
                        f"claim child of node {nid} in tendency {tid} has "
                        "no node mapping; snapshot would be lossy"
                    )
                child_ids.append(child_nid)
            claims[nid] = {
                "content": claim.content,
                "depth": claim.depth,
                "stake": claim.stake,
                "anchor": list(claim.anchor),
                "polarity_axis": list(claim.polarity_axis),
                "children": child_ids,
                "integrated_observation_ids": sorted(
                    claim.integrated_observation_ids
                ),
            }

        frame = tendency.frame
        tendencies[tid] = {
            "thesis": tendency.thesis,
            "anchor": list(tendency.anchor),
            "polarity_axis": list(tendency.polarity_axis),
            "budget": tendency.budget,
            "bandwidth": tendency.bandwidth,
            **{k: getattr(tendency, k) for k in _TENDENCY_KNOBS},
            "tree_id": tendency.tree.id,
            "node_index": list(tendency.tree._node_index.keys()),
            "claims": claims,
            "frame": {
                "integrated": list(frame.integrated.keys()),
                "bandwidth": frame.bandwidth,
                "topic_threshold": frame.topic_threshold,
                "pro_threshold": frame.pro_threshold,
                "contain_distance": frame.contain_distance,
                "use_dim_overlap": frame.use_dim_overlap,
                "total_stake": frame._total_stake,
            },
            "node_capacity": dict(tendency.node_capacity),
            "con_positioned": sorted(tendency._con_positioned),
            "last_stakes": [
                [target_tid, node_id, value]
                for (target_tid, node_id), value in tendency.last_stakes.items()
            ],
        }

    return {
        "schema": SCHEMA,
        "observations": obs_registry,
        "pending": list(world.observations.keys()),
        "history": [obs.id for obs in world.history],
        "nodes": nodes,
        "tendencies": tendencies,
    }


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def restore_world(payload: Dict[str, Any]) -> World:
    """Reconstruct a world exactly from a ``snapshot_world`` payload."""
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported world snapshot schema: {schema!r}")

    # One Observation instance per id, shared everywhere it appears.
    obs_by_id: Dict[str, Observation] = {
        oid: Observation(id=oid, coords=tuple(rec["coords"]),
                         label=rec.get("label", ""))
        for oid, rec in payload["observations"].items()
    }

    world = World()
    for oid in payload["pending"]:
        world.observations[oid] = obs_by_id[oid]
    world.history = [obs_by_id[oid] for oid in payload["history"]]

    # Construct tendencies from ctor params. This creates the root
    # node (deterministic id), root claim, and an empty frame; all of
    # those get their serialized state written over below.
    for tid, trec in payload["tendencies"].items():
        kwargs: Dict[str, Any] = dict(
            id=tid,
            thesis=trec["thesis"],
            anchor=tuple(trec["anchor"]),
            polarity_axis=tuple(trec["polarity_axis"]),
            budget=trec["budget"],
            bandwidth=trec["bandwidth"],
        )
        for knob in _TENDENCY_KNOBS:
            if knob in trec:
                kwargs[knob] = trec[knob]
        world.add_tendency(GeneralizedTendency(**kwargs))

    root_ids = {
        t.tree.root_node.id: tid for tid, t in world.tendencies.items()
    }

    # Materialize every node exactly once. Roots reuse the instances
    # the tendency constructors made (their ids are deterministic, so
    # the snapshot's root records match them).
    node_by_id: Dict[str, Node] = {}
    for nrec in payload["nodes"]:
        nid = nrec["id"]
        if nid in root_ids:
            node = world.tendencies[root_ids[nid]].tree.root_node
        else:
            node = Node(id=nid)
        node.observation_id = nrec["observation_id"]
        node.tree_id = nrec["tree_id"]
        node.content = nrec["content"]
        node.metadata = dict(nrec.get("metadata", {}))
        node.n = nrec["n"]
        node.parents = [
            ParentLink(pid, Position(pos), link_tid)
            for pid, pos, link_tid in nrec["parents"]
        ]
        node.stakes = [
            Stake(agent_id=s["agent_id"], weight=s["weight"])
            for s in nrec["stakes"]
        ]
        node.invalidate_cache()
        dw, ns = nrec.get("score_cache", [None, None])
        node._direct_weight = dw
        node._net_score = ns
        node_by_id[nid] = node

    # Second pass: wire child lists (shared node objects, exact order).
    for nrec in payload["nodes"]:
        node = node_by_id[nrec["id"]]
        node.pro_children = [node_by_id[c] for c in nrec["pro_children"]]
        node.con_children = [node_by_id[c] for c in nrec["con_children"]]

    for tid, trec in payload["tendencies"].items():
        tendency = world.tendencies[tid]
        root = tendency.tree.root_node

        tendency.tree.id = trec["tree_id"]
        tendency.tree._node_index = {
            nid: node_by_id[nid] for nid in trec["node_index"]
        }

        # Claims: the root's claim object is the one frame.claims
        # points at, so mutate it in place; others are fresh.
        claims_rec = trec["claims"]
        claim_by_nid: Dict[str, CoordinateClaim] = {}
        for nid, crec in claims_rec.items():
            if nid == root.id:
                claim = tendency._root_claim
                claim.content = crec["content"]
                claim.depth = crec["depth"]
                claim.stake = crec["stake"]
                claim.anchor = tuple(crec["anchor"])
                claim.polarity_axis = tuple(crec["polarity_axis"])
            else:
                claim = CoordinateClaim(
                    content=crec["content"],
                    depth=crec["depth"],
                    stake=crec["stake"],
                    anchor=tuple(crec["anchor"]),
                    polarity_axis=tuple(crec["polarity_axis"]),
                )
            claim.integrated_observation_ids = set(
                crec.get("integrated_observation_ids", [])
            )
            claim_by_nid[nid] = claim
        for nid, crec in claims_rec.items():
            claim_by_nid[nid].children = [
                claim_by_nid[c] for c in crec["children"]
            ]
        tendency._node_to_claim = claim_by_nid

        frec = trec["frame"]
        tendency._replace_frame(CoordinateFrame(
            claims=[tendency._root_claim],
            integrated={oid: obs_by_id[oid] for oid in frec["integrated"]},
            bandwidth=frec["bandwidth"],
            topic_threshold=frec["topic_threshold"],
            pro_threshold=frec["pro_threshold"],
            contain_distance=frec["contain_distance"],
            use_dim_overlap=frec["use_dim_overlap"],
            _total_stake=frec.get("total_stake", 0.0),
        ))

        tendency.node_capacity = dict(trec["node_capacity"])
        tendency._con_positioned = set(trec["con_positioned"])
        tendency.last_stakes = {
            (target_tid, node_id): value
            for target_tid, node_id, value in trec["last_stakes"]
        }

    return world


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def worlds_equal(a: World, b: World) -> bool:
    """True iff two worlds snapshot to the same payload (after a JSON
    normalization pass, so tuple-vs-list never matters).
    """
    import json

    norm_a = json.loads(json.dumps(snapshot_world(a)))
    norm_b = json.loads(json.dumps(snapshot_world(b)))
    return norm_a == norm_b
