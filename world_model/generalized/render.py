"""Render: surface a graph region as structured output.

The decoder side of the inference path. Takes a Region (from locate)
and walks its claims to produce a structure callers can hand to an
LLM at the I/O boundary, present to a human, or hand to another
agent.

The render output is a dict, not a string. Language synthesis is the
LLM-adapter's job, not the engine's. Render's job is to surface the
graph's structure faithfully so the adapter has good material to
work with.

Output shape
------------

  {
    "region_size": int,
    "nodes": [
      {
        "tendency_id": str,
        "node_id": str,
        "content": str,
        "depth": int,
        "position": "root" | "pro" | "con",
        "score": float,
        "distance_from_query": float,
        "ancestors": [
          # path back to the root tendency, root-first
          {"node_id": str, "content": str, "position": str, "score": float},
          ...
        ],
        "descendants": [
          # immediate children (one level deep), most relevant first
          {"node_id": str, "content": str, "position": str, "score": float},
          ...
        ],
      },
      ...
    ],
    "by_tendency": {
      "<tendency_id>": {"thesis": str, "root_score": float, "n_nodes": int},
      ...
    },
  }

This is enough for an LLM decoder to render a coherent answer
("the network's view on X is ... because ... and ..."), enough for
a human to inspect, and enough for a downstream agent to walk the
structure further.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .locate import Region
from .world import World


def render(
    world: World,
    region: Region,
    descendants_per_node: int = 3,
) -> Dict[str, Any]:
    """Surface a graph region's structure for downstream consumption.

    Args:
      world: the world the region was located in.
      region: list of (tendency_id, node_id, distance) tuples.
      descendants_per_node: how many immediate children to surface
        per node (most-positive-scored first).

    Returns: structured dict (see module docstring).
    """
    nodes_out: List[Dict[str, Any]] = []
    by_tendency: Dict[str, Dict[str, Any]] = {}

    for tendency_id, node_id, distance in region:
        tendency = world.tendencies.get(tendency_id)
        if tendency is None:
            continue
        node = tendency.tree.get_node(node_id)
        if node is None:
            continue

        # Walk ancestors back to the root, root-first
        ancestors: List[Dict[str, Any]] = []
        cursor = node
        chain: List[Any] = []
        while cursor is not None and cursor.parent_id is not None:
            parent = tendency.tree.get_node(cursor.parent_id)
            if parent is None:
                break
            chain.append(parent)
            cursor = parent
        for anc in reversed(chain):
            ancestors.append({
                "node_id": anc.id,
                "content": anc.content,
                "position": anc.position.value,
                "score": anc.net_score,
            })

        # Immediate children, most-positive-scored first
        children = list(node.pro_children) + list(node.con_children)
        children.sort(key=lambda n: -n.net_score)
        descendants: List[Dict[str, Any]] = []
        for child in children[:descendants_per_node]:
            descendants.append({
                "node_id": child.id,
                "content": child.content,
                "position": child.position.value,
                "score": child.net_score,
            })

        nodes_out.append({
            "tendency_id": tendency_id,
            "node_id": node_id,
            "content": node.content,
            "depth": _depth(node, tendency),
            "position": node.position.value,
            "score": node.net_score,
            "distance_from_query": distance,
            "ancestors": ancestors,
            "descendants": descendants,
        })

        if tendency_id not in by_tendency:
            by_tendency[tendency_id] = {
                "thesis": tendency.thesis,
                "root_score": tendency.tree.root_node.net_score,
                "n_nodes": 0,
            }
        by_tendency[tendency_id]["n_nodes"] += 1

    return {
        "region_size": len(nodes_out),
        "nodes": nodes_out,
        "by_tendency": by_tendency,
    }


def _depth(node: Any, tendency: Any) -> int:
    d = 0
    cursor = node
    while cursor is not None and cursor.parent_id is not None:
        d += 1
        cursor = tendency.tree.get_node(cursor.parent_id)
    return d
