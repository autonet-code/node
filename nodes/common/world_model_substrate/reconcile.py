"""Epoch reconciliation: novelty, mint, attribution.

At an epoch boundary, the network looks at how each node's score moved
during the epoch and decides:

  - **Novelty** per node: descriptive measure of surprise (magnitude
    of score movement). Recorded for diagnostics.
  - **Mint** per node: rewarded subset of novelty. Only positive
    movement that ends with positive score qualifies. CON-contributors
    never mint.
  - **Attribution** per agent: which agent caused which score change,
    derived from the events they emitted during the epoch.

Per-agent mint amounts are what the network reports back through
RPB.recordTraining. Novelty stays internal as a diagnostic signal.

Mint formula (per node, per epoch)
----------------------------------

  score_change = score_at_close - score_at_start
  mint(node) = max(0, score_change) * survival_factor * I(score_close > 0)

  survival_factor = mean(score_during_epoch) / score_close
                    if score_close > 0 else 0
  (clipped to [0, 1])

  Negative score-changes don't mint at all -- their "credit" is not
  awarded. The network rewards constructive truth (PRO movement
  landing positive), not contention (CON movement, even if correct).

Attribution
-----------

For each node with positive mint, we look at the events that landed
on/under it during the epoch. Each PRO sub-claim sprouted by an agent
contributes to upward score movement; each CON sub-claim contributes
to downward. We attribute the *positive* portion of the score change
to the agents whose events were PRO-position at this node or under it.

Distribution: each contributing agent's share of the mint at this
node = (number of their PRO events that landed under this node) /
(total PRO events from any agent under this node).

Aggregate per-agent mint = sum over all nodes of (this agent's share).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from world_model.generalized import World

from .events import snapshot_node_scores


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshots over an epoch
# ---------------------------------------------------------------------------


class EpochSnapshots:
    """Records per-node score snapshots at epoch open, optional
    intermediate checkpoints, and epoch close.

    Used by the reconciler to compute per-node novelty, mint, and
    survival weighting.
    """

    def __init__(self):
        self.start: Dict[str, float] = {}
        self.checkpoints: List[Dict[str, float]] = []
        self.close: Dict[str, float] = {}

    def record_start(self, world: World) -> None:
        self.start = snapshot_node_scores(world)

    def record_checkpoint(self, world: World) -> None:
        self.checkpoints.append(snapshot_node_scores(world))

    def record_close(self, world: World) -> None:
        self.close = snapshot_node_scores(world)


# ---------------------------------------------------------------------------
# Per-node novelty and mint
# ---------------------------------------------------------------------------


def _node_novelty(node_id: str, snapshots: EpochSnapshots) -> float:
    """Magnitude of score movement at this node during the epoch.

    Computed as max change between adjacent snapshots, capturing both
    "score moved a lot and stayed there" (one big delta) and "score
    moved and reverted" (two big deltas, large but transient novelty).
    """
    series: List[float] = []
    if node_id in snapshots.start:
        series.append(snapshots.start[node_id])
    for cp in snapshots.checkpoints:
        if node_id in cp:
            series.append(cp[node_id])
    if node_id in snapshots.close:
        series.append(snapshots.close[node_id])
    if len(series) < 2:
        return 0.0
    return max(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))


def _node_survival_factor(node_id: str, snapshots: EpochSnapshots) -> float:
    """How well the score-change persisted through the epoch.

    Returns mean(score during epoch) / score_close, clipped to [0, 1].
    A change that landed early and held = high survival.
    A change that landed at the last second = low survival.
    A change that reverted = near zero.
    """
    score_close = snapshots.close.get(node_id, 0.0)
    if score_close <= 0:
        return 0.0
    series: List[float] = []
    if node_id in snapshots.start:
        series.append(snapshots.start[node_id])
    for cp in snapshots.checkpoints:
        if node_id in cp:
            series.append(cp[node_id])
    if node_id in snapshots.close:
        series.append(snapshots.close[node_id])
    if not series:
        return 0.0
    mean_score = sum(series) / len(series)
    if score_close == 0:
        return 0.0
    factor = mean_score / score_close
    return max(0.0, min(1.0, factor))


def _node_mint(node_id: str, snapshots: EpochSnapshots) -> float:
    """mint = max(0, score_close - score_start) * survival_factor *
    I(score_close > 0).

    Negative or unchanged: zero. Positive movement landing positive:
    proportional to the rise.
    """
    score_start = snapshots.start.get(node_id, 0.0)
    score_close = snapshots.close.get(node_id, 0.0)
    if score_close <= 0:
        return 0.0
    rise = score_close - score_start
    if rise <= 0:
        return 0.0
    survival = _node_survival_factor(node_id, snapshots)
    return rise * survival


def _node_novelty_alone(node_id: str, snapshots: EpochSnapshots) -> float:
    """Diagnostic: how much novelty was generated, regardless of mint."""
    return _node_novelty(node_id, snapshots)


# ---------------------------------------------------------------------------
# Attribution: events to agents
# ---------------------------------------------------------------------------


def _events_under_node(
    target_node_id: str,
    events: List[Dict[str, Any]],
    world: World,
) -> List[Dict[str, Any]]:
    """Return the subset of events whose nodes fall under (or equal)
    target_node_id in the live world's tree. Walks the world to
    determine descent.
    """
    # Build the descendant set of target_node_id
    descendant_ids: set[str] = set()
    for tendency in world.tendencies.values():
        target = tendency.tree.get_node(target_node_id)
        if target is None:
            continue
        # BFS down
        queue = [target]
        while queue:
            n = queue.pop()
            descendant_ids.add(n.id)
            queue.extend(n.pro_children)
            queue.extend(n.con_children)
        break
    if not descendant_ids:
        return []
    out: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("kind") != "sub_claim_sprouted":
            continue
        # The event's node_id may or may not equal a live-world id
        # (depending on remap). Match by both raw and remapped if
        # possible. For now, just match raw against descendant ids.
        if ev.get("node_id") in descendant_ids:
            out.append(ev)
    return out


def _attribute_node_mint(
    node_id: str,
    node_mint_amount: float,
    events: List[Dict[str, Any]],
    world: World,
) -> Dict[str, float]:
    """Distribute a node's mint across agents whose events caused
    upward score movement at or under this node.

    Heuristic: PRO sub-claims sprouted under the node are the events
    that drove score up. Each agent's share = their PRO-event count
    under this node / total PRO-event count under this node.

    If no PRO events found, the mint accrues to "_unattributed" (this
    happens when score moved due to organic propagation from elsewhere
    rather than a direct sprout under this node).
    """
    pro_events_by_agent: Dict[str, int] = {}
    relevant = _events_under_node(node_id, events, world)
    for ev in relevant:
        if ev.get("position") != "pro":
            continue
        agent = ev.get("author_agent", "_unattributed")
        pro_events_by_agent[agent] = pro_events_by_agent.get(agent, 0) + 1
    total = sum(pro_events_by_agent.values())
    if total == 0:
        return {"_unattributed": node_mint_amount}
    return {
        agent: node_mint_amount * (count / total)
        for agent, count in pro_events_by_agent.items()
    }


# ---------------------------------------------------------------------------
# Top-level reconciliation
# ---------------------------------------------------------------------------


def reconcile_epoch(
    world: World,
    snapshots: EpochSnapshots,
    events: List[Dict[str, Any]],
    agent_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute per-agent mint and per-agent novelty for the epoch.

    Args:
      world: the post-replay global world (with all events applied
        and equilibrated).
      snapshots: epoch-start / checkpoints / epoch-close score
        snapshots from EpochSnapshots.
      events: the merged event list from aggregator.aggregate_contributions.
      agent_weights: optional per-agent multipliers (e.g., from
        alignment scoring). Defaults to 1.0 for all.

    Returns:
      {
        "agent_mint": {agent_id: float, ...},      # what goes on chain
        "agent_novelty": {agent_id: float, ...},   # diagnostic
        "node_mint": {node_id: float, ...},        # per-node breakdown
        "node_novelty": {node_id: float, ...},     # per-node breakdown
        "total_mint": float,
        "total_novelty": float,
      }
    """
    agent_weights = agent_weights or {}

    # Discover all node ids that appear in any snapshot
    all_node_ids: set[str] = set()
    all_node_ids.update(snapshots.start.keys())
    for cp in snapshots.checkpoints:
        all_node_ids.update(cp.keys())
    all_node_ids.update(snapshots.close.keys())

    node_mint: Dict[str, float] = {}
    node_novelty: Dict[str, float] = {}
    agent_mint: Dict[str, float] = {}
    agent_novelty: Dict[str, float] = {}

    for node_id in all_node_ids:
        nov = _node_novelty(node_id, snapshots)
        mint = _node_mint(node_id, snapshots)
        if nov > 0:
            node_novelty[node_id] = nov
        if mint > 0:
            node_mint[node_id] = mint

        # Distribute novelty to whoever generated events under this node
        # (regardless of whether mint > 0 -- novelty is descriptive).
        attribution = _attribute_node_mint(node_id, 1.0, events, world)
        for agent, share in attribution.items():
            agent_novelty[agent] = agent_novelty.get(agent, 0.0) + nov * share

        # Distribute mint similarly, weighted by agent_weight if provided
        if mint > 0:
            mint_attribution = _attribute_node_mint(node_id, mint, events, world)
            for agent, amount in mint_attribution.items():
                weight = agent_weights.get(agent, 1.0)
                agent_mint[agent] = agent_mint.get(agent, 0.0) + amount * weight

    total_mint = sum(agent_mint.values())
    total_novelty = sum(agent_novelty.values())

    logger.info(
        "epoch reconciliation: %d nodes, %d agents, total mint %.4f, total novelty %.4f",
        len(node_mint), len(agent_mint), total_mint, total_novelty,
    )
    return {
        "agent_mint": agent_mint,
        "agent_novelty": agent_novelty,
        "node_mint": node_mint,
        "node_novelty": node_novelty,
        "total_mint": total_mint,
        "total_novelty": total_novelty,
    }
