"""
Sparsity analysis of stake-weight graphs.

The engine supports localized inference (queries that touch only a small
subset of the agent graph) iff most stake-weight is concentrated on a
small fraction of edges. This module measures that property.

The hypothesis under test:

    The stake-weight distribution across (observation, agent, tree, node)
    edges is heavy-tailed -- the top 10% of edges carry >50% of total
    weight, and the distribution is better fit by a power law (or
    lognormal) than by a uniform / exponential alternative.

If the hypothesis holds: localized inference is architecturally viable.
If it fails: either the architecture or the data is wrong for local
inference, and the framework cannot scale via caching.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Edge representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StakeEdge:
    """A single agent-stake edge in the equilibrium graph.

    One StakeEdge per Stake instance attached to a Node. The weight is
    the stake's contribution; the other fields locate it in the graph.
    """

    tree_id: str
    node_id: str
    observation_id: Optional[str]
    position: str            # "pro" | "con" | "root"
    agent_id: str
    weight: float

    @property
    def signed_weight(self) -> float:
        """Weight with sign from position (con stakes count as negative)."""
        return -self.weight if self.position == "con" else self.weight


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_stake_edges(tree_store) -> list[StakeEdge]:
    """Extract every stake-edge from a TreeStore (live in-memory object)."""
    edges: list[StakeEdge] = []
    for tree in tree_store.all():
        if tree.root_node is not None:
            _walk_node(tree.root_node, edges)
    return edges


def _walk_node(node, edges: list[StakeEdge]) -> None:
    for stake in node.stakes:
        edges.append(StakeEdge(
            tree_id=node.tree_id,
            node_id=node.id,
            observation_id=node.observation_id,
            position=node.position.value if hasattr(node.position, "value") else str(node.position),
            agent_id=stake.agent_id,
            weight=stake.weight,
        ))
    for child in node.all_children:
        _walk_node(child, edges)


def extract_stake_edges_from_dict(world_model_dict: dict) -> list[StakeEdge]:
    """Extract stake-edges from a serialized world-model JSON document.

    Works on the format produced by WorldModelStore -- the structure under
    ``trees.items[*].root_node`` with recursive ``pro_children`` / ``con_children``.
    """
    edges: list[StakeEdge] = []
    trees = world_model_dict.get("trees", {}).get("items", {})
    if isinstance(trees, dict):
        tree_iter = trees.values()
    else:
        tree_iter = trees
    for tree in tree_iter:
        root = tree.get("root_node")
        if root is not None:
            _walk_dict_node(root, edges)
    return edges


def _walk_dict_node(node: dict, edges: list[StakeEdge]) -> None:
    for stake in node.get("stakes", []):
        edges.append(StakeEdge(
            tree_id=node.get("tree_id", ""),
            node_id=node.get("id", ""),
            observation_id=node.get("observation_id"),
            position=node.get("position", "root"),
            agent_id=stake["agent_id"],
            weight=float(stake["weight"]),
        ))
    for child in node.get("pro_children", []):
        _walk_dict_node(child, edges)
    for child in node.get("con_children", []):
        _walk_dict_node(child, edges)


# ---------------------------------------------------------------------------
# Distribution math
# ---------------------------------------------------------------------------


@dataclass
class PowerLawFit:
    """Result of a power-law MLE fit on a continuous tail.

    For a power-law density p(x) = (alpha-1) / x_min * (x / x_min)^(-alpha)
    over x >= x_min, the MLE estimator (Clauset et al. 2009) is:

        alpha_hat = 1 + n / sum(ln(x_i / x_min))

    The KS distance is the max absolute difference between the empirical
    CDF and the fitted CDF over the tail x >= x_min.
    """

    x_min: float
    alpha: float
    n_tail: int
    ks_distance: float

    @property
    def is_plausible(self) -> bool:
        """Heuristic: alpha in (1, 4] and KS < 0.15 with non-trivial tail size."""
        return (
            1.0 < self.alpha <= 4.0
            and self.ks_distance < 0.15
            and self.n_tail >= 10
        )


@dataclass
class SparsityReport:
    """Result of sparsity analysis on a set of stake-edges."""

    n_edges: int
    total_weight: float

    # Concentration: fraction of weight in top-k% of edges
    top_1pct_weight_share: float
    top_5pct_weight_share: float
    top_10pct_weight_share: float
    top_25pct_weight_share: float

    # Gini coefficient (0 = uniform, 1 = all weight on one edge)
    gini: float

    # Power-law fit on the tail
    power_law: Optional[PowerLawFit]

    # Per-agent breakdown: agent_id -> fraction of total weight
    agent_share: dict[str, float] = field(default_factory=dict)

    # Per-tree breakdown
    tree_share: dict[str, float] = field(default_factory=dict)

    @property
    def hypothesis_supported(self) -> bool:
        """Coarse binary verdict: heavy-tailed enough to support local inference.

        A distribution counts as heavy-tailed here if EITHER:
          - top 10% of edges hold >40% of total weight, OR
          - gini > 0.5 (substantial inequality)
        AND a power-law fit on the tail is statistically plausible.

        This is intentionally lenient compared to a strict >50% threshold.
        The continuous metrics (gini, top-k% shares, fitted alpha) are the
        primary signal; this property is a quick yes/no for sweeps.
        """
        heavy_share = self.top_10pct_weight_share > 0.40
        heavy_gini = self.gini > 0.50
        pl_ok = self.power_law is not None and self.power_law.is_plausible
        return (heavy_share or heavy_gini) and pl_ok

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        pl = (
            f"alpha={self.power_law.alpha:.2f} ks={self.power_law.ks_distance:.3f} "
            f"n_tail={self.power_law.n_tail} plausible={self.power_law.is_plausible}"
            if self.power_law is not None else "n/a (fit skipped)"
        )
        verdict = "SUPPORTED" if self.hypothesis_supported else "NOT supported"
        lines = [
            f"edges:           {self.n_edges}",
            f"total weight:    {self.total_weight:.4f}",
            f"gini:            {self.gini:.3f}",
            f"top 1%  share:   {self.top_1pct_weight_share:.1%}",
            f"top 5%  share:   {self.top_5pct_weight_share:.1%}",
            f"top 10% share:   {self.top_10pct_weight_share:.1%}  (>40% suggests heavy tail)",
            f"top 25% share:   {self.top_25pct_weight_share:.1%}",
            f"power-law fit:   {pl}",
            f"hypothesis:      {verdict}",
        ]
        return "\n".join(lines)


def compute_sparsity_metrics(
    edges: list[StakeEdge],
    fit_power_law: bool = True,
) -> SparsityReport:
    """Compute concentration metrics and (optionally) a power-law fit."""
    if not edges:
        return SparsityReport(
            n_edges=0,
            total_weight=0.0,
            top_1pct_weight_share=0.0,
            top_5pct_weight_share=0.0,
            top_10pct_weight_share=0.0,
            top_25pct_weight_share=0.0,
            gini=0.0,
            power_law=None,
        )

    weights = sorted((e.weight for e in edges), reverse=True)
    n = len(weights)
    total = sum(weights)

    def top_share(pct: float) -> float:
        k = max(1, int(round(n * pct)))
        return sum(weights[:k]) / total if total > 0 else 0.0

    report = SparsityReport(
        n_edges=n,
        total_weight=total,
        top_1pct_weight_share=top_share(0.01),
        top_5pct_weight_share=top_share(0.05),
        top_10pct_weight_share=top_share(0.10),
        top_25pct_weight_share=top_share(0.25),
        gini=_gini(weights),
        power_law=_fit_power_law(weights) if fit_power_law and n >= 20 else None,
    )

    # Per-agent share
    agent_totals: dict[str, float] = {}
    for e in edges:
        agent_totals[e.agent_id] = agent_totals.get(e.agent_id, 0.0) + e.weight
    report.agent_share = {a: w / total for a, w in agent_totals.items()} if total > 0 else {}

    # Per-tree share
    tree_totals: dict[str, float] = {}
    for e in edges:
        tree_totals[e.tree_id] = tree_totals.get(e.tree_id, 0.0) + e.weight
    report.tree_share = {t: w / total for t, w in tree_totals.items()} if total > 0 else {}

    return report


def _gini(sorted_desc_weights: list[float]) -> float:
    """Gini coefficient. Input must be sorted descending."""
    n = len(sorted_desc_weights)
    if n == 0:
        return 0.0
    s = sum(sorted_desc_weights)
    if s <= 0:
        return 0.0
    # Convert to ascending for the standard formula
    asc = sorted(sorted_desc_weights)
    cum = 0.0
    for i, w in enumerate(asc, start=1):
        cum += i * w
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def _fit_power_law(sorted_desc_weights: list[float]) -> Optional[PowerLawFit]:
    """Fit a continuous power-law to the upper tail via MLE.

    We sweep candidate x_min values and pick the one minimizing KS distance,
    following Clauset, Shalizi, Newman (2009) in spirit (simplified for
    small samples). Returns None if no plausible tail exists.
    """
    if len(sorted_desc_weights) < 20:
        return None

    # Candidate x_min values: try every weight value as a cutoff (excluding the
    # last few so we always have a non-trivial tail).
    asc = sorted(w for w in sorted_desc_weights if w > 0)
    if len(asc) < 20:
        return None

    candidates = sorted(set(asc[:-5]))  # skip the very top so n_tail >= 5
    best: Optional[PowerLawFit] = None

    for x_min in candidates:
        tail = [w for w in asc if w >= x_min]
        n_tail = len(tail)
        if n_tail < 10:
            continue

        # MLE for alpha
        try:
            log_sum = sum(math.log(w / x_min) for w in tail)
        except ValueError:
            continue
        if log_sum <= 0:
            continue
        alpha = 1.0 + n_tail / log_sum
        if alpha <= 1.0 or alpha > 10.0:
            continue

        # KS distance against fitted CDF
        ks = _ks_distance(tail, x_min, alpha)

        if best is None or ks < best.ks_distance:
            best = PowerLawFit(
                x_min=x_min,
                alpha=alpha,
                n_tail=n_tail,
                ks_distance=ks,
            )

    return best


def _ks_distance(tail: list[float], x_min: float, alpha: float) -> float:
    """Max distance between empirical CDF and fitted power-law CDF on tail."""
    asc = sorted(tail)
    n = len(asc)
    max_d = 0.0
    for i, x in enumerate(asc, start=1):
        empirical = i / n
        fitted = 1.0 - (x / x_min) ** (1.0 - alpha)
        d = abs(empirical - fitted)
        if d > max_d:
            max_d = d
    return max_d


# ---------------------------------------------------------------------------
# Synthetic data (for piping validation)
# ---------------------------------------------------------------------------


def synthetic_uniform_edges(
    n_edges: int = 500,
    n_agents: int = 7,
    n_trees: int = 7,
    weight: float = 0.1,
    seed: int = 42,
) -> list[StakeEdge]:
    """Negative control: uniform weights. Sparsity hypothesis should FAIL.

    Every edge has the same weight, so top 10% holds exactly 10% of total.
    """
    rng = random.Random(seed)
    edges: list[StakeEdge] = []
    agents = [f"agent_{i}" for i in range(n_agents)]
    trees = [f"tree_{i}" for i in range(n_trees)]
    for i in range(n_edges):
        edges.append(StakeEdge(
            tree_id=rng.choice(trees),
            node_id=f"node_{i}",
            observation_id=f"obs_{rng.randint(0, n_edges)}",
            position=rng.choice(["pro", "con"]),
            agent_id=rng.choice(agents),
            weight=weight,
        ))
    return edges


def synthetic_powerlaw_edges(
    n_edges: int = 500,
    n_agents: int = 7,
    n_trees: int = 7,
    alpha: float = 2.5,
    x_min: float = 0.01,
    seed: int = 42,
) -> list[StakeEdge]:
    """Positive control: power-law weights. Sparsity hypothesis should PASS.

    Weights drawn from a Pareto distribution with the given parameters.
    """
    rng = random.Random(seed)
    edges: list[StakeEdge] = []
    agents = [f"agent_{i}" for i in range(n_agents)]
    trees = [f"tree_{i}" for i in range(n_trees)]
    for i in range(n_edges):
        # Inverse-CDF sampling for continuous power law
        u = rng.random()
        weight = x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        edges.append(StakeEdge(
            tree_id=rng.choice(trees),
            node_id=f"node_{i}",
            observation_id=f"obs_{rng.randint(0, n_edges)}",
            position=rng.choice(["pro", "con"]),
            agent_id=rng.choice(agents),
            weight=weight,
        ))
    return edges
