"""Mint gate: training-agent debates against the charter tendencies.

Mints are computed from score movement (in reconcile.py). The gate
filters those mints by whether training agents have unrefuted CON
debates against the contributing node on charter grounds.

Why this is a gate, not a coordinate
------------------------------------

If care-for-others were a coordinate axis among many, an agent could
be very high on technical-correctness and very low on care-for-others
and still mint. That's the failure mode we want to prevent.

If care-for-others were a hard precondition on capability (the
contribution can't even be staked), the substrate would refuse to
learn dual-use techniques entirely. That's also wrong.

The gate is post-mint and outcome-dependent:

  - The substrate learns the technique (it's in the graph).
  - Mint is paid for non-harmful uses.
  - The same technique used harmfully gets debated by training nodes;
    if the debate's CON sub-claim accumulates positive standing, the
    harmful use's mint is suppressed.

Charter violation = agreed-upon CON debate
------------------------------------------

A "charter violation" isn't a hardcoded check. It's the result of
training agents posting CON sub-claims that the network agrees with
(positive net_score). The four charter tendencies are seed concerns;
violations against them emerge through debate, not detection.

The default charter is the 6-root configuration: 4 alignment axes
(life_precious, self_preservation, promotion_of_intelligence,
evolution) plus 2 usefulness axes (correctness, simplicity). The
gate iterates whichever charter_ids the caller passes, so the gate
keeps working if the deployer expands or contracts the charter.

This means:
  - A new training-agent CON post doesn't immediately suppress mint;
    it has to *win equilibration* (network agrees) to count.
  - A frivolous CON post that no one supports doesn't matter.
  - A genuine harm flag that other agents corroborate suppresses the
    harmful contribution's mint.

API
---

  charter_violation_score(world, target_node_id, charter_ids) -> float
    Returns a score in [0, 1] representing how confident the network
    is that the target node violates charter principles. 0 = no
    violation flagged or only refuted flags. 1 = strongly flagged.

  apply_mint_gate(reconcile_result, world, charter_ids, gate_strength)
    Mutates reconcile_result['agent_mint'] and ['node_mint'] to
    suppress payouts proportional to charter_violation_score on each
    minting node.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from world_model.generalized import World


logger = logging.getLogger(__name__)


# Default charter tendency ids -- the seed concerns for the mint
# gate. Stay synced with adapter.py:CHARTER. The 4 alignment axes
# remain the load-bearing violation flags; the 2 usefulness axes
# (correctness, simplicity) participate as well so that work that
# is buggy or over-engineered can be debated through the same
# mechanism.

DEFAULT_CHARTER_IDS = (
    "life_precious",
    "self_preservation",
    "promotion_of_intelligence",
    "evolution",
    "correctness",
    "simplicity",
)


def _node_descendants(world: World, target_id: str) -> set[str]:
    """All nodes reachable below target_id in any tendency's tree."""
    out: set[str] = set()
    for tendency in world.tendencies.values():
        target = tendency.tree.get_node(target_id)
        if target is None:
            continue
        queue = [target]
        while queue:
            n = queue.pop()
            out.add(n.id)
            queue.extend(n.pro_children)
            queue.extend(n.con_children)
        break
    return out


def charter_violation_score(
    world: World,
    target_node_id: str,
    charter_ids: Iterable[str] = DEFAULT_CHARTER_IDS,
) -> float:
    """Score how strongly the network has flagged target_node_id as
    violating any charter principle.

    Mechanism: walk every charter tendency's tree. Find sub-claims
    whose content references target_node_id (encoded by the training
    agent who posted them). Sum their net_scores when positive (a
    violation flag the network agrees with). Normalize.

    Note: the references-target-node mechanism uses content matching
    on the node's id appearing in the CON sub-claim's content. That's
    a placeholder. In production, training agents would post structured
    references; for the v1 gate, plain-text reference detection is
    enough to demonstrate the mechanism.

    Returns a score in [0, 1]: 0 = no flags or only refuted ones,
    1 = strong flag.
    """
    total_pos_flag_score = 0.0
    n_flags = 0
    for tid in charter_ids:
        tendency = world.tendencies.get(tid)
        if tendency is None:
            continue
        for node in tendency.tree.all_nodes():
            content = node.content or ""
            if target_node_id not in content:
                continue
            # CON-positioned sub-claims under charter tendencies
            # are the violation flags. Their net_score determines
            # whether the network endorses the flag.
            if node.position.value != "con":
                continue
            score = node.net_score
            if score > 0:
                total_pos_flag_score += score
                n_flags += 1
    if n_flags == 0:
        return 0.0
    # Saturate via tanh-like map; score 1.0 = "strongly flagged"
    avg = total_pos_flag_score / n_flags
    # Clip and rescale: avg of 0.5+ counts as 'strong'
    return min(1.0, max(0.0, avg / 0.5))


def apply_mint_gate(
    reconcile_result: Dict[str, Any],
    world: World,
    charter_ids: Iterable[str] = DEFAULT_CHARTER_IDS,
    gate_strength: float = 1.0,
) -> Dict[str, Any]:
    """Apply the charter mint gate to a reconcile_epoch result.

    For each minting node, compute charter_violation_score. Multiply
    the node's mint by (1 - gate_strength * violation_score). With
    gate_strength=1.0, a fully-flagged node gets zero mint.

    Per-agent mint is recomputed from the gated per-node mint.
    Total_mint is recomputed.

    Mutates reconcile_result in place; also returns it for chaining.
    """
    node_mint_orig = reconcile_result.get("node_mint", {}) or {}
    if not node_mint_orig:
        return reconcile_result

    # Per-node violation scores
    node_violation: Dict[str, float] = {}
    for node_id in node_mint_orig:
        node_violation[node_id] = charter_violation_score(
            world, node_id, charter_ids,
        )

    # Apply gate
    gated_node_mint: Dict[str, float] = {}
    for node_id, mint in node_mint_orig.items():
        v = node_violation[node_id]
        scale = max(0.0, 1.0 - gate_strength * v)
        gated_mint = mint * scale
        if gated_mint > 0:
            gated_node_mint[node_id] = gated_mint

    # Re-derive per-agent mint from the gated node mint, scaling each
    # agent's previous attribution by the new node's gated/original
    # ratio.
    agent_mint_orig = reconcile_result.get("agent_mint", {}) or {}
    if not agent_mint_orig:
        # Nothing more to do
        reconcile_result["node_mint"] = gated_node_mint
        reconcile_result["total_mint"] = sum(gated_node_mint.values())
        return reconcile_result

    # We don't have per-(node, agent) breakdown directly. Approximate:
    # compute the global gating ratio (sum gated / sum original) and
    # apply it uniformly. A future version with per-(node, agent)
    # tracking could be more precise.
    sum_orig = sum(node_mint_orig.values())
    sum_gated = sum(gated_node_mint.values())
    if sum_orig <= 0:
        ratio = 0.0
    else:
        ratio = sum_gated / sum_orig

    gated_agent_mint = {a: m * ratio for a, m in agent_mint_orig.items()}

    reconcile_result["node_mint"] = gated_node_mint
    reconcile_result["agent_mint"] = gated_agent_mint
    reconcile_result["total_mint"] = sum(gated_node_mint.values())
    reconcile_result["node_violation"] = node_violation
    reconcile_result["gate_applied"] = True

    n_suppressed = sum(1 for v in node_violation.values() if v > 0.01)
    logger.info(
        "mint gate: %d/%d nodes flagged (gate_strength=%.2f, ratio=%.3f)",
        n_suppressed, len(node_violation), gate_strength, ratio,
    )
    return reconcile_result
