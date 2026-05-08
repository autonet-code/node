"""Epoch-close pruning pass.

After equilibration, many sub-claims will have settled at near-zero
net_score and stayed there across rounds. Those nodes are dead weight:
they don't carry standing, they don't move, and every future
equilibration round still walks them. Pruning sweeps them away so the
tendency's tree stays focused on what's actually contested or
supported.

The rule:

  A subtree (rooted at a non-root node N) may be pruned when, over the
  observed score history, BOTH:
    - max(abs(net_score)) < score_threshold     (no standing)
    - max(abs(score change between adjacent     (no surprise/activity)
       checkpoints)) < novelty_threshold

The whole subtree under N is removed -- if N's voice never carried,
neither did anything beneath it.

ROOTs are sacred: every tendency keeps its founding claim regardless.

Pruning is idempotent: calling it twice in a row with no new activity
between calls prunes nothing the second time, because the nodes are
already gone.

Determinism: the function visits nodes in a fixed traversal order
(parent before children, pro-children before con-children, in the
order they currently sit in the parent's lists). Given identical
inputs, two callers produce identical pruned-id lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models.tree import Node, Position
from .world import World


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot_scores(world: World) -> Dict[str, float]:
    """Return {node_id: net_score} across every node in every tendency's tree.

    Node ids are globally unique (uuid4), so a flat dict is safe.
    """
    snap: Dict[str, float] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            snap[node.id] = node.net_score
    return snap


@dataclass
class ScoreHistory:
    """Append-only record of (round_index, snapshot) pairs.

    The class accumulates a snapshot per call to :meth:`record` and
    exposes :meth:`history_for` for prune_settled_negatives to consume.

    Determinism: snapshots are stored in append order; history_for
    returns the score series in that same order, so two callers
    feeding the same world in the same sequence get identical
    histories.
    """

    snapshots: List[Tuple[int, Dict[str, float]]] = field(default_factory=list)

    def record(self, world: World) -> None:
        """Capture the world's current scores under the next round index."""
        idx = len(self.snapshots)
        self.snapshots.append((idx, snapshot_scores(world)))

    def history_for(self, node_id: str) -> List[float]:
        """Return the score series for `node_id`.

        Snapshots that didn't include the node (e.g., taken before the
        node was sprouted) are skipped rather than reported as 0 --
        we only want to judge a node by epochs in which it actually
        existed.
        """
        series: List[float] = []
        for _idx, snap in self.snapshots:
            if node_id in snap:
                series.append(snap[node_id])
        return series

    def as_dict(self) -> Dict[str, List[float]]:
        """Materialize the {node_id: [score, ...]} mapping that
        prune_settled_negatives accepts via its `score_history` arg.
        """
        out: Dict[str, List[float]] = {}
        for _idx, snap in self.snapshots:
            for node_id, score in snap.items():
                out.setdefault(node_id, []).append(score)
        return out


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def _max_abs(series: List[float]) -> float:
    if not series:
        return 0.0
    return max(abs(v) for v in series)


def _max_abs_delta(series: List[float]) -> float:
    if len(series) < 2:
        return 0.0
    return max(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))


def _settled_quiet(
    series: List[float],
    score_threshold: float,
    novelty_threshold: float,
) -> bool:
    """Decide whether a node's score history is settled-and-quiet."""
    return (
        _max_abs(series) < score_threshold
        and _max_abs_delta(series) < novelty_threshold
    )


def _collect_subtree_ids(node: Node, _seen: Optional[set] = None) -> List[str]:
    """All ids in the subtree rooted at `node`, including `node` itself.

    Visits PRO children before CON children, in list order. Cycle-
    safe via the `_seen` set: under co-parenting, the same node may
    appear in multiple parents' child lists, so a naive walk could
    revisit. Used both to enumerate ids for index cleanup and to
    ensure deterministic behaviour.
    """
    if _seen is None:
        _seen = set()
    if node.id in _seen:
        return []
    _seen.add(node.id)
    ids: List[str] = [node.id]
    for child in node.pro_children:
        ids.extend(_collect_subtree_ids(child, _seen))
    for child in node.con_children:
        ids.extend(_collect_subtree_ids(child, _seen))
    return ids


def _detach_from_parent(tendency, node: Node) -> bool:
    """Remove `node` from its parent's pro/con children lists.

    Returns True if detachment happened. The parent's net_score cache
    is invalidated up to the root.
    """
    parent_id = node.parent_id
    if parent_id is None:
        return False
    parent = tendency.tree.get_node(parent_id)
    if parent is None:
        return False
    if node in parent.pro_children:
        parent.pro_children.remove(node)
    elif node in parent.con_children:
        parent.con_children.remove(node)
    else:
        return False
    # Invalidate up the spine
    tendency.tree._invalidate_ancestors(parent)
    return True


def _purge_node_from_tendency(tendency, node_id: str) -> None:
    """Remove a node id from the tendency-side bookkeeping (claim
    map, capacity map, and the tree's node index).

    Doesn't touch the parent linkage -- the caller must have already
    detached the subtree root via _detach_from_parent.
    """
    tendency._node_to_claim.pop(node_id, None)
    tendency.node_capacity.pop(node_id, None)
    tendency.tree._node_index.pop(node_id, None)
    # The frame is rebuilt from the root claim's children chain, which
    # we've already mutated. Refresh so cached topic/threshold state
    # stays consistent.
    # (Done once after all purges by the caller for efficiency.)


def prune_settled_negatives(
    world: World,
    score_threshold: float = 0.05,
    novelty_threshold: float = 0.02,
    score_history: Optional[Dict[str, List[float]]] = None,
) -> List[str]:
    """Prune subtrees that are settled-and-quiet across history.

    A node N (non-root) is pruned when its score history satisfies
    both:
      - max(abs(net_score)) < score_threshold
      - max(abs(adjacent delta)) < novelty_threshold

    The full subtree under N is removed: if N's voice never carried,
    neither did anything beneath it. The ids of every removed node
    (subtree-root plus descendants) are returned in traversal order.

    If `score_history` is None we fall back to single-snapshot mode:
    we use the current net_score as a one-element history. The
    novelty test trivially passes (no adjacent pair), so pruning then
    reduces to "abs(net_score) < score_threshold" -- a degenerate but
    still useful filter when no checkpointing is in place.

    The function is idempotent: a second call with no new activity
    will find no further candidates because the nodes are already gone.

    The function is deterministic: traversal order is parent-first,
    PRO-children before CON-children, in the order children currently
    sit in the parent's lists.
    """
    pruned: List[str] = []

    # Snapshot of current scores, used as a fallback per-node history
    # for nodes the caller didn't pass history for and for the
    # single-snapshot mode.
    current = snapshot_scores(world)

    for tendency in world.tendencies.values():
        root_id = tendency.tree.root_node.id

        # Walk the tree in deterministic order, collect candidate
        # subtree-roots first, then prune them. Decoupling the walk
        # from the mutation prevents iterator invalidation when we
        # remove children mid-walk.
        candidates: List[Node] = []

        def visit(node: Node) -> None:
            if node.position != Position.ROOT and node.id != root_id:
                if score_history is not None:
                    series = score_history.get(node.id, [current.get(node.id, node.net_score)])
                else:
                    series = [current.get(node.id, node.net_score)]
                if _settled_quiet(series, score_threshold, novelty_threshold):
                    candidates.append(node)
                    # Don't descend: the whole subtree goes with it,
                    # and we don't want to add already-doomed
                    # descendants to the candidate list.
                    return
            for child in list(node.pro_children):
                visit(child)
            for child in list(node.con_children):
                visit(child)

        visit(tendency.tree.root_node)

        if not candidates:
            continue

        # Prune each candidate subtree.
        for subtree_root in candidates:
            ids = _collect_subtree_ids(subtree_root)
            if not _detach_from_parent(tendency, subtree_root):
                # Subtree root is parentless -- shouldn't happen for
                # non-root nodes, but skip rather than crash.
                continue
            for nid in ids:
                _purge_node_from_tendency(tendency, nid)
                # Drop the matching claim from the parent claim's
                # children list as well so the frame doesn't
                # rediscover ghosts.
            pruned.extend(ids)

        # Rebuild the frame so its claim list reflects the surviving
        # claim hierarchy. The root claim's `children` list still
        # holds references to claims for nodes we just pruned, so we
        # need to walk and clean those up.
        _scrub_dead_claims(tendency)
        tendency._refresh_frame()

    return pruned


def prune_veto_negatives(
    world: World,
    floor: Optional[float] = None,
) -> List[str]:
    """Asymmetric pruning under veto-shaped tendency roots.

    A tendency tagged `veto_shaped=True` carries hard-veto semantics.
    For each direct child of its root, the child's *signed*
    contribution to the root's score is computed:

      - PRO children: contribution = +intrinsic_score(child)
      - CON children: contribution = -intrinsic_score(child)

    The child (and its subtree) is pruned outright when its signed
    contribution falls below `veto_score_floor` (a negative number).
    This catches both modes of failure:
      - A PRO child that should have supported correctness but had
        its intrinsic_score dragged deeply negative by accumulated
        CON descendants.
      - A CON child that has accumulated enough evidence that it
        actively contradicts correctness (its negative contribution
        clears the floor).

    Pruning ignores n and settled-quiet history; the threshold is
    structural. The floor defaults to each tendency's
    `veto_score_floor`.

    Returns the list of pruned node ids in traversal order.
    """
    from .tendency import _intrinsic_score_in_tendency
    from ..models.tree import Position

    pruned: List[str] = []

    for tendency in world.tendencies.values():
        if not getattr(tendency, "veto_shaped", False):
            continue
        active_floor = floor if floor is not None else tendency.veto_score_floor
        root = tendency.tree.root_node

        # For each direct child, compute its tendency-tree-aware
        # intrinsic score (only edges in THIS tendency contribute,
        # avoiding over-subtraction from co-parented children whose
        # edges in OTHER tendencies have a different polarity).
        candidates: List[Node] = []
        for child in list(root.pro_children):
            intr = _intrinsic_score_in_tendency(child, tendency.id)
            contribution = +intr
            if contribution < active_floor:
                candidates.append(child)
        for child in list(root.con_children):
            intr = _intrinsic_score_in_tendency(child, tendency.id)
            contribution = -intr
            if contribution < active_floor:
                candidates.append(child)

        for subtree_root in candidates:
            ids = _collect_subtree_ids(subtree_root)
            if not _detach_from_parent(tendency, subtree_root):
                continue
            for nid in ids:
                _purge_node_from_tendency(tendency, nid)
            pruned.extend(ids)

        if candidates:
            _scrub_dead_claims(tendency)
            tendency._refresh_frame()

    return pruned


def _scrub_dead_claims(tendency) -> None:
    """Remove dangling CoordinateClaim references whose backing nodes
    have been pruned.

    Walks the claim hierarchy (root_claim and recursive children) and
    drops any child claim that no longer has a corresponding node id
    in the tendency's _node_to_claim map. CoordinateClaim isn't
    hashable (it's a non-frozen dataclass with mutable fields), so we
    compare by identity.
    """
    surviving_ids = {id(c) for c in tendency._node_to_claim.values()}

    def scrub(claim) -> None:
        claim.children = [c for c in claim.children if id(c) in surviving_ids]
        for c in claim.children:
            scrub(c)

    scrub(tendency._root_claim)
