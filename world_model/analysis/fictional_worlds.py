"""
Generate equilibrium-shaped stake graphs for fictional worlds.

A fictional world is a parametric spec: a roster of agents, a set of
observation kinds, and a coherence parameter that controls how strongly
agents' interests are differentiated.

The simulator produces stakes by drawing from agent-observation
affinities. Agents do not reason -- they stake by parameter. The output
shape matches what a real Arena run would produce, so the sparsity
analyzer consumes it without modification.

The hypothesis under test is no longer about a specific dataset. It is:

    Sparsity in stake-graphs is a function of world-coherence.
    Coherent worlds produce heavy-tailed stake distributions;
    incoherent worlds produce flat ones.

Run a sweep over the coherence parameter to characterize the curve.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .sparsity import StakeEdge


# ---------------------------------------------------------------------------
# World spec
# ---------------------------------------------------------------------------


@dataclass
class WorldSpec:
    """Parametric description of a fictional world.

    Parameters
    ----------
    name:
        Identifier for the world (used in logs / reports).
    agents:
        Names of agents in this world. Order does not matter.
    n_observations:
        How many distinct observations the world contains.
    n_trees:
        How many root claims (trees) exist. Each tree corresponds to one
        agent's flagship claim plus shared observation pool.
    coherence:
        In [0, 1]. Controls how strongly agent interests differentiate.

        - 0.0: every agent has identical affinity for every observation
               (perfectly incoherent world -- no structure)
        - 0.5: moderate differentiation
        - 1.0: each agent has affinity for a disjoint subset of
               observations (perfectly coherent world -- maximal structure)
    stake_density:
        Fraction of (agent, observation) pairs that produce a stake.
        At 1.0 every agent stakes every observation; at 0.1 only 10% do.
    seed:
        RNG seed for reproducibility.
    """

    name: str
    agents: list[str]
    n_observations: int = 200
    n_trees: int = 5
    coherence: float = 0.5
    stake_density: float = 0.4
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.coherence <= 1.0:
            raise ValueError(f"coherence must be in [0,1], got {self.coherence}")
        if not 0.0 < self.stake_density <= 1.0:
            raise ValueError(f"stake_density must be in (0,1], got {self.stake_density}")
        if len(self.agents) < 2:
            raise ValueError("need at least 2 agents")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def simulate_world(spec: WorldSpec) -> list[StakeEdge]:
    """Generate stake-edges for a fictional world per its spec.

    The procedure:

    1. Build an affinity matrix A[agent, observation] from coherence.
       At coherence=0, every entry is 0.5 (uniform).
       At coherence=1, each observation has affinity 1.0 with exactly
       one agent and 0.0 with the rest (perfect partition).
       Intermediate values interpolate.
    2. For each (agent, observation) pair, draw a uniform random number;
       if it falls below stake_density, generate a stake.
    3. Stake weight is drawn from a distribution shaped by affinity:
       high affinity -> heavy weight, low affinity -> light weight.
    4. Each stake is attached to a tree -- the tree the agent's flagship
       claim lives in -- with a random pro/con position.
    """
    rng = random.Random(spec.seed)
    n_agents = len(spec.agents)

    # 1. Affinity matrix: shape (n_agents, n_observations)
    affinity = _build_affinity_matrix(
        n_agents=n_agents,
        n_observations=spec.n_observations,
        coherence=spec.coherence,
        rng=rng,
    )

    # 2. Tree assignment: each agent owns one or more trees in rotation
    # so that with n_trees < n_agents, some agents share a tree, and with
    # n_trees > n_agents, some agents own multiple. We just assign tree_id
    # by hashing (agent, slot) deterministically.
    tree_ids = [f"{spec.name}_tree_{i}" for i in range(spec.n_trees)]

    # 3. Generate stakes
    edges: list[StakeEdge] = []
    for ai, agent in enumerate(spec.agents):
        for oi in range(spec.n_observations):
            if rng.random() >= spec.stake_density:
                continue
            aff = affinity[ai][oi]
            # Weight distribution: a Beta-shaped sample biased by affinity.
            # We use a simple parametric form: weight ~ aff^k * uniform(0,1)
            # with k controlling how much affinity stretches the tail.
            k = 2.0
            base = rng.random()
            weight = (aff ** k) * base + 1e-4   # tiny floor so no zeros
            tree_id = tree_ids[(ai + oi) % spec.n_trees]
            edges.append(StakeEdge(
                tree_id=tree_id,
                node_id=f"{spec.name}_node_{ai}_{oi}",
                observation_id=f"{spec.name}_obs_{oi}",
                position="pro" if rng.random() < 0.6 else "con",
                agent_id=agent,
                weight=weight,
            ))
    return edges


def _build_affinity_matrix(
    n_agents: int,
    n_observations: int,
    coherence: float,
    rng: random.Random,
) -> list[list[float]]:
    """Build agent-observation affinity matrix per the coherence parameter.

    Construction:
      - Each observation has a "true owner" agent assigned uniformly at random.
      - Affinity is 1.0 between an observation and its owner, 0.0 otherwise.
      - Then we soften that perfect partition by mixing it with a uniform
        baseline (every agent has 0.5 affinity for every observation).
        coherence=1 -> all weight on the partition.
        coherence=0 -> all weight on the uniform baseline.
    """
    matrix = [[0.0] * n_observations for _ in range(n_agents)]

    for oi in range(n_observations):
        owner = rng.randrange(n_agents)
        for ai in range(n_agents):
            partition = 1.0 if ai == owner else 0.0
            uniform = 0.5
            matrix[ai][oi] = coherence * partition + (1.0 - coherence) * uniform

    return matrix


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """One point in a coherence sweep."""

    coherence: float
    n_edges: int
    total_weight: float
    gini: float
    top_10pct_share: float
    top_25pct_share: float
    power_law_alpha: Optional[float]
    power_law_ks: Optional[float]
    power_law_plausible: bool


def coherence_sweep(
    coherences: Optional[list[float]] = None,
    agents: Optional[list[str]] = None,
    n_observations: int = 200,
    n_trees: int = 5,
    stake_density: float = 0.4,
    seeds: Optional[list[int]] = None,
) -> list[SweepPoint]:
    """Sweep the coherence parameter and report sparsity at each point.

    For each coherence value we simulate ``len(seeds)`` worlds and average
    the resulting metrics, so we don't read seed noise as signal.
    """
    from .sparsity import compute_sparsity_metrics

    if coherences is None:
        coherences = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    if agents is None:
        agents = [f"a{i}" for i in range(7)]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    points: list[SweepPoint] = []
    for c in coherences:
        per_seed_metrics = []
        for seed in seeds:
            spec = WorldSpec(
                name=f"world_c{c:.2f}_s{seed}",
                agents=agents,
                n_observations=n_observations,
                n_trees=n_trees,
                coherence=c,
                stake_density=stake_density,
                seed=seed,
            )
            edges = simulate_world(spec)
            report = compute_sparsity_metrics(edges)
            per_seed_metrics.append(report)

        # Average across seeds
        n_edges_avg = sum(m.n_edges for m in per_seed_metrics) / len(per_seed_metrics)
        total_weight_avg = sum(m.total_weight for m in per_seed_metrics) / len(per_seed_metrics)
        gini_avg = sum(m.gini for m in per_seed_metrics) / len(per_seed_metrics)
        top10_avg = sum(m.top_10pct_weight_share for m in per_seed_metrics) / len(per_seed_metrics)
        top25_avg = sum(m.top_25pct_weight_share for m in per_seed_metrics) / len(per_seed_metrics)

        with_pl = [m for m in per_seed_metrics if m.power_law is not None]
        if with_pl:
            alpha_avg = sum(m.power_law.alpha for m in with_pl) / len(with_pl)
            ks_avg = sum(m.power_law.ks_distance for m in with_pl) / len(with_pl)
            plausible = sum(1 for m in with_pl if m.power_law.is_plausible) >= len(with_pl) / 2
        else:
            alpha_avg = None
            ks_avg = None
            plausible = False

        points.append(SweepPoint(
            coherence=c,
            n_edges=int(round(n_edges_avg)),
            total_weight=total_weight_avg,
            gini=gini_avg,
            top_10pct_share=top10_avg,
            top_25pct_share=top25_avg,
            power_law_alpha=alpha_avg,
            power_law_ks=ks_avg,
            power_law_plausible=plausible,
        ))

    return points


def format_sweep_table(points: list[SweepPoint]) -> str:
    """Render a sweep result as a fixed-width table."""
    header = (
        f"{'coherence':>10} {'edges':>7} {'gini':>6} "
        f"{'top10%':>7} {'top25%':>7} {'alpha':>7} {'ks':>6} {'pl_ok':>5}"
    )
    lines = [header, "-" * len(header)]
    for p in points:
        alpha_str = f"{p.power_law_alpha:.2f}" if p.power_law_alpha is not None else "  -- "
        ks_str = f"{p.power_law_ks:.3f}" if p.power_law_ks is not None else " -- "
        lines.append(
            f"{p.coherence:>10.2f} {p.n_edges:>7d} {p.gini:>6.3f} "
            f"{p.top_10pct_share:>7.1%} {p.top_25pct_share:>7.1%} "
            f"{alpha_str:>7} {ks_str:>6} {str(p.power_law_plausible):>5}"
        )
    return "\n".join(lines)
