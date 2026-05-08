"""
Fractal-first classification: cascade equilibrium across a class hierarchy.

The flat classifier_wiring puts every class on equal footing in a single
graph and lets graph-walk averaging discriminate between them. That works
when classes are well-separated in feature space; it fails when many
classes occupy overlapping regions and the engine has to discriminate all
N pairs simultaneously.

The fractal-first approach: organize classes as a hierarchy, where
internal nodes are super-classes (clusters of similar leaf classes) and
leaves are individual classes. The engine classifies by *cascading down
the hierarchy*: at each level, equilibrate among the candidates at that
level, pick the winner (or a soft distribution), descend to its children.

This is the LOD machinery doing real work: a super-class at the top of
the tree is the LOD-0 view of a leaf class deep down. ``at_lod(K)``
walks between them. Each level of the cascade is its own equilibrium
problem, with fewer competitors and clearer distinctions than the
flat single-shot classification.

The hierarchy is built automatically from training-set centroids via
agglomerative clustering. Domain-agnostic.

Public API:

  build_hierarchy(profile) -> ClassHierarchy
  build_hierarchical_state(profile, hierarchy) -> dict
      {level_key -> PresentState} for each level of the cascade
  classify_cascade(case_features, profile, hierarchy, states) -> winning_class
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..models.tendency import Tendency, TendencySet
from ..models.factory import DefaultTendencyFactory, TendencySpec
from ..models.lineage import Lineage, StakeWeightGraph
from ..dynamics.reseed import PresentState, Substitution, reseed_and_equilibrate
from .classifier_wiring import (
    ClassifierProfile,
    contrast_to_edge_weight,
    case_feature_strength,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@dataclass
class HierarchyNode:
    """One node in the class hierarchy.

    Leaves correspond to individual classes; internal nodes correspond
    to super-classes (clusters). All nodes have a centroid in feature
    space (mean of member-class centroids).
    """

    node_id: str                      # unique within hierarchy: "leaf_K", "node_42"
    is_leaf: bool
    class_label: Optional[object]     # set on leaves
    centroid: list[float]
    children: list["HierarchyNode"] = field(default_factory=list)
    leaves: list[object] = field(default_factory=list)   # all leaf class labels under this node
    depth: int = 0                    # root = 0


@dataclass
class ClassHierarchy:
    """The full hierarchy: nodes by id, root, level groupings."""
    nodes: dict[str, HierarchyNode]
    root: HierarchyNode
    levels: list[list[HierarchyNode]]   # levels[0] = [root], levels[1] = root.children, etc.
    n_features: int

    def parent_of(self, node_id: str) -> Optional[HierarchyNode]:
        for n in self.nodes.values():
            for c in n.children:
                if c.node_id == node_id:
                    return n
        return None


def build_hierarchy(profile: ClassifierProfile) -> ClassHierarchy:
    """Run agglomerative clustering on per-class centroids; return a binary
    tree from the merge sequence. The tree's leaves are the individual
    classes; internal nodes are clusters.

    With N classes, agglomerative clustering produces N-1 merges. We
    reify each merge as a HierarchyNode whose children are the two
    things it merged.
    """
    classes = profile.classes
    n_classes = len(classes)
    centroids = np.array([profile.class_centroids[c] for c in classes])

    if n_classes < 2:
        # Degenerate: one class. Hierarchy is a single leaf-as-root.
        only_class = classes[0]
        leaf = HierarchyNode(
            node_id=f"leaf_{only_class}",
            is_leaf=True,
            class_label=only_class,
            centroid=list(profile.class_centroids[only_class]),
            leaves=[only_class],
            depth=0,
        )
        return ClassHierarchy(
            nodes={leaf.node_id: leaf},
            root=leaf,
            levels=[[leaf]],
            n_features=profile.n_features,
        )

    ac = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0, compute_full_tree=True,
        linkage='ward',
    )
    ac.fit(centroids)

    # Build leaf nodes
    nodes: dict[str, HierarchyNode] = {}
    leaf_node_ids: list[str] = []   # indexed by class index (0..n_classes-1)
    for i, c in enumerate(classes):
        nid = f"leaf_{c}"
        nodes[nid] = HierarchyNode(
            node_id=nid, is_leaf=True, class_label=c,
            centroid=list(centroids[i]),
            leaves=[c],
        )
        leaf_node_ids.append(nid)

    # Build internal nodes from the merge sequence. ac.children_ has
    # shape (n_merges, 2); merge i creates node n_classes + i, whose
    # children are the two items at indices ac.children_[i].
    internal_node_ids: list[Optional[str]] = []
    for i, (left_idx, right_idx) in enumerate(ac.children_):
        left_id = (
            leaf_node_ids[left_idx] if left_idx < n_classes
            else internal_node_ids[left_idx - n_classes]
        )
        right_id = (
            leaf_node_ids[right_idx] if right_idx < n_classes
            else internal_node_ids[right_idx - n_classes]
        )
        left_n = nodes[left_id]
        right_n = nodes[right_id]
        merged_leaves = list(left_n.leaves) + list(right_n.leaves)
        # Centroid = mean of constituent leaf centroids (NOT centroid of
        # children, which would weight one branch more than the other if
        # branches have different leaf counts -- we want
        # equal weighting per ground-truth class)
        merged_centroid = list(np.mean(
            [profile.class_centroids[c] for c in merged_leaves], axis=0
        ))
        new_id = f"node_{n_classes + i}"
        nodes[new_id] = HierarchyNode(
            node_id=new_id, is_leaf=False, class_label=None,
            centroid=merged_centroid,
            children=[left_n, right_n],
            leaves=merged_leaves,
        )
        internal_node_ids.append(new_id)

    root = nodes[internal_node_ids[-1]]

    # Compute depths
    def assign_depth(node: HierarchyNode, d: int) -> None:
        node.depth = d
        for c in node.children:
            assign_depth(c, d + 1)
    assign_depth(root, 0)

    # Group by depth
    max_depth = max(n.depth for n in nodes.values())
    levels = [[] for _ in range(max_depth + 1)]
    for n in nodes.values():
        levels[n.depth].append(n)

    return ClassHierarchy(
        nodes=nodes, root=root, levels=levels,
        n_features=profile.n_features,
    )


# ---------------------------------------------------------------------------
# Cascade classification
# ---------------------------------------------------------------------------


def _build_state_for_candidates(
    candidates: list[HierarchyNode],
    profile: ClassifierProfile,
    feature_prefix: str = "f",
    hierarchy: Optional[ClassHierarchy] = None,
    sibling_edge_weight: float = 0.4,
    parent_edge_weight: float = 0.3,
) -> PresentState:
    """Build a fresh PresentState containing one tendency per candidate node
    plus the standard 2 * n_features feature-evidence tendencies.

    Edge weights between candidate nodes and feature-evidence tendencies
    use the candidate's centroid (leaf or super-class) z-scored against
    the population.

    If hierarchy is provided, also adds intra-hierarchy edges:
      - Sibling edges: classes sharing a parent get connected (shared
        cluster membership = shared structure).
      - Parent edges: leaves connect to ancestor cluster nodes that
        are also in the candidates set.

    These edges let the engine propagate evidence within clusters
    without crossing into distant clusters, which is the actual fractal
    work the hierarchy is supposed to do for inference.
    """
    factory = DefaultTendencyFactory()
    n_features = profile.n_features
    n_candidates = len(candidates)

    candidate_alloc = 0.5 / n_candidates
    feature_alloc = 0.5 / (2 * n_features)

    specs: list[TendencySpec] = []
    for cand in candidates:
        specs.append(TendencySpec(
            id=f"cand_{cand.node_id}",
            initial_allocation=candidate_alloc,
        ))
    for f in range(n_features):
        specs.append(TendencySpec(
            id=f"{feature_prefix}{f:02d}_low",
            initial_allocation=feature_alloc,
        ))
        specs.append(TendencySpec(
            id=f"{feature_prefix}{f:02d}_high",
            initial_allocation=feature_alloc,
        ))

    tendencies = factory.build_set(specs)

    graph = StakeWeightGraph()
    for cand in candidates:
        # z-score the candidate's centroid against the *population*
        # statistics (not the local cluster's). This keeps every
        # candidate measured on the same scale across all cascade levels.
        for f in range(n_features):
            z = (cand.centroid[f] - profile.feature_pop_mean[f]) / max(
                profile.feature_pop_std[f], 1e-9
            )
            low_edge, high_edge = contrast_to_edge_weight(z)
            graph.add_edge(
                f"cand_{cand.node_id}",
                f"{feature_prefix}{f:02d}_high",
                high_edge,
            )
            graph.add_edge(
                f"cand_{cand.node_id}",
                f"{feature_prefix}{f:02d}_low",
                low_edge,
            )

    # Add intra-hierarchy edges if a hierarchy is provided
    if hierarchy is not None:
        candidate_ids = {c.node_id for c in candidates}
        for parent in candidates:
            if parent.is_leaf:
                continue
            # Sibling edges: every pair of children gets connected
            for i, c1 in enumerate(parent.children):
                if c1.node_id not in candidate_ids:
                    continue
                for c2 in parent.children[i + 1:]:
                    if c2.node_id not in candidate_ids:
                        continue
                    graph.add_edge(
                        f"cand_{c1.node_id}",
                        f"cand_{c2.node_id}",
                        sibling_edge_weight,
                    )
                # Parent edges: each child connects to this parent
                graph.add_edge(
                    f"cand_{c1.node_id}",
                    f"cand_{parent.node_id}",
                    parent_edge_weight,
                )

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


def _equilibrate_among_candidates(
    candidates: list[HierarchyNode],
    case_features,
    profile: ClassifierProfile,
    feature_prefix: str = "f",
    learning_rate: float = 0.3,
    tolerance: float = 1e-5,
    max_iterations: int = 300,
) -> dict[str, float]:
    """Run the engine's classification at one level. Returns
    {candidate_node_id -> post-equilibrium allocation}."""
    if len(candidates) == 1:
        # Only one candidate -- it wins by default. No work to do.
        return {candidates[0].node_id: 1.0}

    state = _build_state_for_candidates(candidates, profile, feature_prefix)

    n_features = profile.n_features
    feature_budget = 0.5 / n_features

    substitutions = []
    for f in range(n_features):
        low_strength, high_strength = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_high",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_high",
                allocation=feature_budget * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_low",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_low",
                allocation=feature_budget * low_strength,
            ),
        ))

    result = reseed_and_equilibrate(
        state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    return {
        c.node_id: result.state.tendencies.get(f"cand_{c.node_id}").allocation
        for c in candidates
    }


def classify_cascade_hard(
    case_features,
    profile: ClassifierProfile,
    hierarchy: ClassHierarchy,
    feature_prefix: str = "f",
    **kwargs,
):
    """Hard cascade: at each level, equilibrate among candidates, descend
    into the winner's children, recurse. Stops at a leaf.

    Failure mode: an early misclassification cuts off the correct
    branch and is unrecoverable. Soft cascade addresses this; this
    version is the simplest baseline.
    """
    current_node = hierarchy.root
    while not current_node.is_leaf:
        children = current_node.children
        allocs = _equilibrate_among_candidates(
            children, case_features, profile, feature_prefix, **kwargs
        )
        winner_id = max(allocs, key=allocs.get)
        current_node = next(c for c in children if c.node_id == winner_id)
    return current_node.class_label


def classify_unified_with_hierarchy(
    case_features,
    profile: ClassifierProfile,
    hierarchy: ClassHierarchy,
    feature_prefix: str = "f",
    **kwargs,
):
    """Single-pass equilibrium with both leaves AND internal hierarchy
    nodes as candidates. The internal nodes act as "concept tendencies"
    that capture super-class structure; leaves are individual classes.

    This is the fractal version proper: tendencies live at multiple
    scales simultaneously in one graph. The leaf with the highest
    allocation at equilibrium is the prediction. Internal-node
    allocations don't directly select; they just propagate evidence
    through their cluster's edges, helping their member leaves.

    Compared to flat: this adds candidate tendencies that aren't
    leaves -- they represent shared structure -- and lets the engine
    integrate evidence at multiple scales.
    """
    all_candidates = list(hierarchy.nodes.values())
    state = _build_state_for_candidates(
        all_candidates, profile, feature_prefix,
        hierarchy=hierarchy,
    )

    n_features = profile.n_features
    feature_budget = 0.5 / n_features

    substitutions = []
    for f in range(n_features):
        low_strength, high_strength = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_high",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_high",
                allocation=feature_budget * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_low",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_low",
                allocation=feature_budget * low_strength,
            ),
        ))

    result = reseed_and_equilibrate(
        state,
        substitutions=substitutions,
        propagate_via_graph=True,
        **kwargs,
    )

    # Read out: only consider leaves. Pick the leaf with the highest allocation.
    leaf_allocs: dict[object, float] = {}
    for n in all_candidates:
        if n.is_leaf:
            leaf_allocs[n.class_label] = result.state.tendencies.get(
                f"cand_{n.node_id}"
            ).allocation
    return max(leaf_allocs, key=leaf_allocs.get)


def classify_cascade_soft(
    case_features,
    profile: ClassifierProfile,
    hierarchy: ClassHierarchy,
    feature_prefix: str = "f",
    **kwargs,
):
    """Soft cascade: each leaf scored by the *geometric mean* of its
    path's per-level probabilities -- i.e. log-probability divided by
    path length -- rather than the raw product.

    Why geometric mean and not product: the hierarchy is generally
    unbalanced (agglomerative clustering produces binary trees where
    some leaves are 2 levels deep and others are 8+ levels deep). Raw
    product probability favors shallow leaves disproportionately just
    because they have fewer multiplications, regardless of how strong
    the allocation evidence was at each level. Geometric mean
    normalizes for path length.
    """
    # path_log_prob_sum[node_id] = sum of log probs along path from root
    path_log_sum: dict[str, float] = {hierarchy.root.node_id: 0.0}
    path_depth: dict[str, int] = {hierarchy.root.node_id: 0}
    leaf_geometric_mean: dict[object, float] = {}

    queue = [hierarchy.root]
    while queue:
        node = queue.pop(0)
        if node.is_leaf:
            depth = path_depth[node.node_id]
            log_sum = path_log_sum[node.node_id]
            # Avoid div-by-zero for the degenerate single-leaf case
            avg_log_prob = log_sum / max(depth, 1)
            leaf_geometric_mean[node.class_label] = avg_log_prob
            continue

        children = node.children
        allocs = _equilibrate_among_candidates(
            children, case_features, profile, feature_prefix, **kwargs
        )

        total = sum(allocs.values())
        if total <= 0:
            n_children = len(children)
            child_probs = {c.node_id: 1.0 / n_children for c in children}
        else:
            child_probs = {nid: a / total for nid, a in allocs.items()}

        parent_log_sum = path_log_sum[node.node_id]
        parent_depth = path_depth[node.node_id]
        for c in children:
            # Floor probability so log doesn't go to -inf
            p = max(child_probs[c.node_id], 1e-12)
            path_log_sum[c.node_id] = parent_log_sum + math.log(p)
            path_depth[c.node_id] = parent_depth + 1
            queue.append(c)

    return max(leaf_geometric_mean, key=leaf_geometric_mean.get)
