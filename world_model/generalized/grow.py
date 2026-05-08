"""Growth rule: sprout sub-arguments under contested nodes.

A node sprouts children when:
  1. The owning tendency has staked PRO on it strongly (defense), AND
  2. At least one other tendency has staked CON on it strongly (attack), AND
  3. The dispute is significant enough -- |PRO_total| + |CON_total| > threshold.

When it fires, the node splits in two PRO/CON children at offsets along
the parent's polarity axis. The PRO child takes a finer-grained
stake-favorable cut; the CON child takes the opposing cut. Each child
inherits the parent's polarity axis but with its anchor displaced.

This implements depth-as-resolution: contested regions get sharper
distinctions; uncontested regions stay shallow.
"""

from __future__ import annotations

from typing import List, Tuple

from ..models.tree import Node, Position
from .observation import Observation
from .world import World


def propose_growth(
    world: World,
    contention_threshold: float = 0.15,
    offset: float = 0.5,
) -> int:
    """Walk every tendency's tree; sprout children under contested nodes.

    Returns the number of new nodes added. Mutates each tendency's
    tree and frame in place.
    """
    growths = 0
    for tendency in world.tendencies.values():
        # Walk a snapshot of the current nodes so we don't iterate while mutating
        for node in list(tendency.tree.all_nodes()):
            if not _is_contested(node, contention_threshold):
                continue
            if not node.is_leaf:
                continue   # only grow at leaves; contested non-leaves already split

            parent_claim = tendency._node_to_claim.get(node.id)
            if parent_claim is None:
                continue

            # Compute child anchors by displacing along parent polarity axis
            anchor = parent_claim.anchor
            axis = parent_claim.polarity_axis
            if not anchor or not axis:
                continue

            pro_anchor = tuple(a + offset * u for a, u in zip(anchor, axis))
            con_anchor = tuple(a - offset * u for a, u in zip(anchor, axis))

            # PRO child
            tendency.sprout_child(
                parent_node_id=node.id,
                position=Position.PRO,
                anchor=pro_anchor,
                polarity_axis=axis,
                content=f"{parent_claim.content}+",
            )
            # CON child
            tendency.sprout_child(
                parent_node_id=node.id,
                position=Position.CON,
                anchor=con_anchor,
                polarity_axis=tuple(-u for u in axis),
                content=f"{parent_claim.content}-",
            )
            growths += 2
    return growths


def _is_contested(node: Node, threshold: float) -> bool:
    """A node is contested if it has both a positive and a negative
    stake of magnitude >= threshold.
    """
    has_pos = any(s.weight >= threshold for s in node.stakes)
    has_neg = any(s.weight <= -threshold for s in node.stakes)
    return has_pos and has_neg
