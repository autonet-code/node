"""
Tree model - value hierarchies where agents stake on nodes.

Each tree is rooted in a core value. Nodes represent observations positioned
as pro or con relative to their parent. Weight propagation follows:

    net_score = direct_weight + Σ(pro_children.net_score) - Σ(con_children.net_score)

The same observation can appear in multiple trees with different positions.

Under the post-and-coparent refactor, a node carries a *list* of
parent links: each link is (parent_node_id, position, tendency_id).
A node with parents in multiple tendencies is a "work item" that
emerged structurally from cross-tendency relevance. Backward-compat
properties `Node.parent_id` and `Node.position` expose the first
parent link so single-parent callers keep working.

Stakes are unit-weight ("posts"): each `Stake.weight` is always 1.0.
The weight field is preserved in the schema for forward-compat, but
the dynamics no longer modulate it.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from enum import Enum
import uuid


class Position(Enum):
    """Position of a node relative to its parent."""
    PRO = "pro"   # Supports the parent claim
    CON = "con"   # Contradicts the parent claim
    ROOT = "root" # Root node, no parent


class ParentLink(NamedTuple):
    """A single parent edge on a node.

    A node can have multiple parent links: one per tendency in which
    it participates. Each link records the parent node id, the
    position (PRO/CON) at this edge, and which tendency owns this
    link.
    """
    parent_id: str
    position: Position
    tendency_id: str


@dataclass
class Stake:
    """A unit-weight post placed by an agent on a node.

    Under the post-and-coparent refactor, every stake has weight=1.0.
    The weight field is preserved for forward-compatibility but is
    not currently varied.
    """
    agent_id: str       # Which tendency (from Tendency enum value)
    weight: float = 1.0 # Unit-weight post; field reserved for future use

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: dict) -> "Stake":
        return cls(agent_id=data["agent_id"], weight=data.get("weight", 1.0))


@dataclass
class Node:
    """
    A node in a value tree.

    Each node links to an observation and carries one or more parent
    links (one per tendency in which it participates). The first
    parent link is exposed as `node.parent_id` / `node.position` for
    backward compatibility; multi-parent nodes (work items) read from
    `node.parents`. Agents post unit-weight stakes on nodes; scores
    propagate up the tree.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    observation_id: Optional[str] = None  # Links to atomic observation (None for abstract nodes)
    tree_id: str = ""

    # Parent links: one entry per (parent_id, position, tendency_id).
    # Empty list = root node. Length-1 list = standard single-parent
    # node. Length-2+ list = multi-parented (work item) node bridging
    # multiple tendencies.
    parents: list[ParentLink] = field(default_factory=list)

    # Content for root/abstract nodes that don't link to observations
    content: str = ""

    # Posts (unit-weight stakes) from agents
    stakes: list[Stake] = field(default_factory=list)

    # Metadata (e.g., reasoning behind stakes)
    metadata: dict = field(default_factory=dict)

    # Children
    pro_children: list["Node"] = field(default_factory=list)
    con_children: list["Node"] = field(default_factory=list)

    # Persistent novelty (continuous "surprise" state).
    # Decays under PRO confirmation, regrows under CON contradiction,
    # drifts upward without observations. See lindblad/NOVELTY_REFACTOR.md.
    # 1.0 = maximally surprising (fresh, untested); 0.0 = fully settled.
    n: float = 1.0

    # Cached scores (recomputed on demand)
    _direct_weight: Optional[float] = field(default=None, repr=False)
    _net_score: Optional[float] = field(default=None, repr=False)
    _in_progress: bool = field(default=False, repr=False)

    @property
    def parent_id(self) -> Optional[str]:
        """Backward-compat: first parent link's parent id, or None for root."""
        return self.parents[0].parent_id if self.parents else None

    @parent_id.setter
    def parent_id(self, value: Optional[str]) -> None:
        """Backward-compat setter: replace first parent's parent_id.

        Used by callers that mutate parent_id directly (e.g.
        `child.parent_id = parent.id` in legacy code paths). If the
        node has no parents yet, creates a length-1 list with ROOT
        position; if it has parents, replaces the first link's
        parent_id while preserving its position and tendency_id.
        """
        if value is None:
            self.parents = []
            return
        if not self.parents:
            self.parents = [ParentLink(value, Position.ROOT, "")]
        else:
            old = self.parents[0]
            self.parents[0] = ParentLink(value, old.position, old.tendency_id)

    @property
    def position(self) -> Position:
        """Backward-compat: first parent link's position. ROOT if no parents."""
        return self.parents[0].position if self.parents else Position.ROOT

    @position.setter
    def position(self, value: Position) -> None:
        """Backward-compat setter for callers that assign position
        directly. If the node has no parents, stores a placeholder
        link; if it has parents, replaces the first link's position.
        """
        if not self.parents:
            self.parents = [ParentLink("", value, "")]
        else:
            old = self.parents[0]
            self.parents[0] = ParentLink(old.parent_id, value, old.tendency_id)

    @property
    def direct_weight(self) -> float:
        """Number of posts on this node (each weight=1)."""
        if self._direct_weight is None:
            self._direct_weight = sum(s.weight for s in self.stakes)
        return self._direct_weight

    @property
    def net_score(self) -> float:
        """
        Score after weight propagation from children.

        net_score = direct_weight + Σ(pro_children.net_score) - Σ(con_children.net_score)

        Cycle protection: under co-parenting a node may transitively
        appear in its own subtree (rare but possible if cross-tendency
        edge discovery joins two regions). A sentinel `_in_progress`
        flag short-circuits reentrant access to 0.0 so the recursion
        terminates cleanly; the cache then completes with the
        consistent (cycle-broken) value.
        """
        if self._net_score is None:
            if self._in_progress:
                return 0.0
            self._in_progress = True
            try:
                pro_sum = sum(child.net_score for child in self.pro_children)
                con_sum = sum(child.net_score for child in self.con_children)
                self._net_score = self.direct_weight + pro_sum - con_sum
            finally:
                self._in_progress = False
        return self._net_score

    def invalidate_cache(self):
        """Clear cached scores - call after modifying stakes or children."""
        self._direct_weight = None
        self._net_score = None
        self._in_progress = False

    def add_stake(self, agent_id: str, weight: float = 1.0):
        """Add a unit-weight post from an agent.

        The weight argument is accepted for backward compatibility but
        is normalized to 1.0 under the post-only refactor. Negative
        weights remain meaningful only at the World.apply_stakes layer
        (signed intents); on the node itself, every stored Stake is
        weight=1.
        """
        self.stakes.append(Stake(agent_id=agent_id, weight=weight))
        self.invalidate_cache()

    def add_post(self, agent_id: str) -> None:
        """Convenience: append a unit-weight post from `agent_id`."""
        self.add_stake(agent_id=agent_id, weight=1.0)

    def add_parent_link(
        self,
        parent_id: str,
        position: Position,
        tendency_id: str,
    ) -> None:
        """Append a parent link if not already present.

        Idempotent on (parent_id, tendency_id): re-calling with the
        same parent in the same tendency is a no-op. Self-references
        (parent_id == self.id) are silently refused -- a node can't
        be its own parent.
        """
        if parent_id == self.id:
            return
        for link in self.parents:
            if link.parent_id == parent_id and link.tendency_id == tendency_id:
                return
        self.parents.append(ParentLink(parent_id, position, tendency_id))
        self.invalidate_cache()

    def add_child(self, child: "Node", position: Position, tendency_id: str = ""):
        """Add a child node with the given position.

        Records the parent link on the child rather than overwriting a
        single parent_id field. The tendency_id argument identifies
        which tree this edge belongs to; it defaults to empty for
        backward compatibility but should be set explicitly by
        callers that own a tendency context.
        """
        if position not in (Position.PRO, Position.CON):
            raise ValueError(f"Child position must be PRO or CON, got {position}")

        child.add_parent_link(self.id, position, tendency_id or self.tree_id)
        if not child.tree_id:
            child.tree_id = self.tree_id

        if position == Position.PRO:
            if child not in self.pro_children:
                self.pro_children.append(child)
        else:
            if child not in self.con_children:
                self.con_children.append(child)

        self.invalidate_cache()

    @property
    def all_children(self) -> list["Node"]:
        """All children (pro and con)."""
        return self.pro_children + self.con_children

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.pro_children) == 0 and len(self.con_children) == 0

    @property
    def is_root(self) -> bool:
        """Whether this is the root node."""
        return len(self.parents) == 0

    def stakes_by_agent(self) -> dict[str, float]:
        """Total stake count per agent on this node."""
        by_agent: dict[str, float] = {}
        for stake in self.stakes:
            by_agent[stake.agent_id] = by_agent.get(stake.agent_id, 0) + stake.weight
        return by_agent

    def __repr__(self):
        content = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"Node({self.position.value}, score={self.net_score:.2f}, '{content}')"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "observation_id": self.observation_id,
            "tree_id": self.tree_id,
            "parents": [
                {
                    "parent_id": p.parent_id,
                    "position": p.position.value,
                    "tendency_id": p.tendency_id,
                }
                for p in self.parents
            ],
            # Backward-compat fields (first parent link). Older readers
            # that read parent_id/position directly stay working.
            "parent_id": self.parent_id,
            "position": self.position.value,
            "content": self.content,
            "stakes": [s.to_dict() for s in self.stakes],
            "metadata": self.metadata,
            "n": self.n,
            "pro_children": [c.to_dict() for c in self.pro_children],
            "con_children": [c.to_dict() for c in self.con_children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        # Prefer the new `parents` list when present; fall back to the
        # old (parent_id, position) fields for legacy serializations.
        if "parents" in data and data["parents"]:
            parents = [
                ParentLink(
                    parent_id=p["parent_id"],
                    position=Position(p["position"]),
                    tendency_id=p.get("tendency_id", ""),
                )
                for p in data["parents"]
            ]
        elif data.get("parent_id") is not None:
            parents = [
                ParentLink(
                    parent_id=data["parent_id"],
                    position=Position(data["position"]),
                    tendency_id=data.get("tree_id", ""),
                )
            ]
        else:
            parents = []
        node = cls(
            id=data["id"],
            observation_id=data.get("observation_id"),
            tree_id=data.get("tree_id", ""),
            parents=parents,
            content=data.get("content", ""),
            stakes=[Stake.from_dict(s) for s in data.get("stakes", [])],
            metadata=data.get("metadata", {}),
            n=data.get("n", 1.0),
        )
        node.pro_children = [cls.from_dict(c) for c in data.get("pro_children", [])]
        node.con_children = [cls.from_dict(c) for c in data.get("con_children", [])]
        return node


@dataclass
class Tree:
    """
    A value hierarchy rooted in a core concern.

    Trees represent different "lenses" through which observations are viewed.
    The same observation can appear in multiple trees with different positions
    and scores.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_value: str = ""          # The core concern (e.g., "decentralized coordination")
    description: str = ""
    root_node: Optional[Node] = None

    # Index for fast lookup
    _node_index: dict[str, Node] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.root_node is None:
            # Create root node from root_value. No parents = root.
            self.root_node = Node(
                tree_id=self.id,
                content=self.root_value,
            )
        self._index_nodes(self.root_node)

    def _index_nodes(self, node: Node):
        """Recursively index all nodes for fast lookup."""
        self._node_index[node.id] = node
        for child in node.all_children:
            self._index_nodes(child)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self._node_index.get(node_id)

    def add_node(
        self,
        parent_id: str,
        node: Node,
        position: Position,
        tendency_id: str = "",
    ) -> Node:
        """Add a node as a child of the specified parent.

        The tendency_id argument identifies which tree the new edge
        belongs to. It defaults to the tree's own id when not given,
        matching the common case of one tendency per tree.
        """
        parent = self.get_node(parent_id)
        if parent is None:
            raise ValueError(f"Parent node not found: {parent_id}")

        node.tree_id = self.id
        parent.add_child(node, position, tendency_id=tendency_id or self.id)
        self._node_index[node.id] = node

        # Invalidate ancestors' caches up to root
        self._invalidate_ancestors(parent)

        return node

    def _invalidate_ancestors(self, node: Node):
        """Invalidate caches of this node and all ancestors."""
        node.invalidate_cache()
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent:
                self._invalidate_ancestors(parent)

    @property
    def score(self) -> float:
        """Overall tree score (root node's net score)."""
        return self.root_node.net_score if self.root_node else 0.0

    def all_nodes(self) -> list[Node]:
        """All nodes in the tree."""
        return list(self._node_index.values())

    def nodes_by_observation(self, observation_id: str) -> list[Node]:
        """Find all nodes linked to a specific observation."""
        return [n for n in self._node_index.values() if n.observation_id == observation_id]

    def contested_nodes(self, min_stakes: int = 2) -> list[Node]:
        """
        Find nodes with stakes from multiple agents.

        These are the points of internal tension.
        """
        contested = []
        for node in self._node_index.values():
            unique_agents = len(set(s.agent_id for s in node.stakes))
            if unique_agents >= min_stakes:
                contested.append(node)
        return contested

    def depth(self, node: Optional[Node] = None) -> int:
        """Maximum depth from node (default: root) to leaves."""
        if node is None:
            node = self.root_node
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self.depth(c) for c in node.all_children)

    def __repr__(self):
        return f"Tree('{self.root_value}', score={self.score:.2f}, nodes={len(self._node_index)})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "root_value": self.root_value,
            "description": self.description,
            "root_node": self.root_node.to_dict() if self.root_node else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tree":
        tree = cls(
            id=data["id"],
            root_value=data["root_value"],
            description=data.get("description", ""),
            root_node=Node.from_dict(data["root_node"]) if data.get("root_node") else None,
        )
        return tree


@dataclass
class TreeStore:
    """Collection of trees for a person's worldview."""

    trees: dict[str, Tree] = field(default_factory=dict)  # id -> Tree

    def add(self, tree: Tree) -> Tree:
        """Add a tree."""
        self.trees[tree.id] = tree
        return tree

    def get(self, tree_id: str) -> Optional[Tree]:
        """Get tree by ID."""
        return self.trees.get(tree_id)

    def find_by_value(self, root_value: str) -> Optional[Tree]:
        """Find tree by root value."""
        for tree in self.trees.values():
            if tree.root_value.lower() == root_value.lower():
                return tree
        return None

    def all(self) -> list[Tree]:
        """All trees."""
        return list(self.trees.values())

    def observation_appearances(self, observation_id: str) -> list[tuple[Tree, Node]]:
        """
        Find all appearances of an observation across trees.

        Returns list of (tree, node) tuples.
        """
        appearances = []
        for tree in self.trees.values():
            for node in tree.nodes_by_observation(observation_id):
                appearances.append((tree, node))
        return appearances

    def meaning_vector(self, observation_id: str) -> dict[str, float]:
        """
        Get an observation's score across all trees.

        Returns {tree_id: score} representing the observation's meaning
        in this person's value system.
        """
        vector = {}
        for tree_id, tree in self.trees.items():
            nodes = tree.nodes_by_observation(observation_id)
            if nodes:
                # Average score if observation appears multiple times in tree
                vector[tree_id] = sum(n.net_score for n in nodes) / len(nodes)
        return vector

    def __len__(self) -> int:
        return len(self.trees)

    def to_dict(self) -> dict:
        return {"trees": {tid: t.to_dict() for tid, t in self.trees.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "TreeStore":
        store = cls()
        for tree_data in data.get("trees", {}).values():
            store.add(Tree.from_dict(tree_data))
        return store
