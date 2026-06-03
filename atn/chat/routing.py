"""Agent-id hierarchy helpers.

Autonet agent ids are dotted lineage: the orchestrator is ``"orchestrator"``
(depth 0), its first delegate is ``"orchestrator.1"`` (depth 1), that
delegate's first child is ``"orchestrator.1.1"`` (depth 2), and so on
(``atn/runtime/agent_registry.py`` generate_child_id). Depth is the number of
dot-separated segments after the root.

These helpers turn an agent id into the routing decisions the ChatService
needs:

  - depth 0 (orchestrator)        -> rendered in the bound CHANNEL
  - depth 1 (top-level delegate)  -> gets its own THREAD
  - depth >= 2 (nested sub-agent) -> rendered as a pinned TILE inside its
                                     top-level delegate's thread

Discord threads are one level deep only, which is why every transitive
descendant of a top-level delegate collapses into that delegate's thread.
"""
from __future__ import annotations

from typing import Callable, Optional

AgentId = str

# A resolver: agent_id -> its parent's id (or None/"" for the root). Lets the
# routing helpers work for agents whose id is NOT dotted lineage -- e.g. the
# create_agent path gives human-readable ids like "test-simple" with the
# parent carried out-of-band (agent.registered event / snapshot parent_id).
ParentOf = Callable[[AgentId], Optional[AgentId]]

# The root agent id. Autonet's primary agent is always "orchestrator".
ORCHESTRATOR = "orchestrator"


# How many ancestors to walk before giving up (cycle / bad-data guard).
_MAX_LINEAGE = 64


def _is_root(pid: AgentId | None, of: AgentId) -> bool:
    """True if `pid` denotes the root (no enclosing agent above `of`)."""
    return pid is None or pid == "" or pid == of or pid == ORCHESTRATOR


def _is_root_id(node: AgentId, parent_of: ParentOf) -> bool:
    """A node is a root if it has no parent (empty/None) in the map, or is the
    orchestrator. Roots own the channel; their children are thread owners."""
    if node == ORCHESTRATOR:
        return True
    p = parent_of(node)
    return p == "" or p is None


def agent_depth(agent_id: AgentId, parent_of: ParentOf | None = None) -> int:
    """Depth in the delegate tree. orchestrator=0, its children=1, ...

    With `parent_of` supplied AND the agent present in it, depth is computed by
    walking the lineage to the root (works for non-dotted create_agent ids).
    Otherwise falls back to dotted-id parsing (id.count(".")), so existing
    dotted delegates behave exactly as before.
    """
    if not agent_id:
        return 0
    if agent_id == ORCHESTRATOR:
        return 0
    if parent_of is not None and parent_of(agent_id) is not None:
        # Count edges from the agent up until we reach a node whose parent
        # denotes root. A root node (parent == "" / None) is depth 0; its
        # children depth 1; etc. "Root" is defined by an empty parent, which is
        # how a channel's bound agent is rooted — so a subtree under any bound
        # agent gets the same 0/1/2 depths as the orchestrator's subtree.
        if _is_root_id(agent_id, parent_of):
            return 0
        depth = 0
        cur = agent_id
        for _ in range(_MAX_LINEAGE):
            if _is_root_id(cur, parent_of):
                return depth
            depth += 1
            cur = parent_of(cur)
            if cur is None:
                return depth
        return depth
    return agent_id.count(".")


def is_orchestrator(agent_id: AgentId, parent_of: ParentOf | None = None) -> bool:
    return agent_depth(agent_id, parent_of) == 0


def is_top_level_delegate(agent_id: AgentId, parent_of: ParentOf | None = None) -> bool:
    """A depth-1 agent (child of the orchestrator) -> own thread."""
    return agent_depth(agent_id, parent_of) == 1


def top_level_delegate(agent_id: AgentId, parent_of: ParentOf | None = None) -> AgentId | None:
    """The depth-1 ancestor whose thread an agent renders into.

    - orchestrator            -> None (renders in the channel, not a thread)
    - a child of orchestrator -> itself
    - a deeper descendant     -> its depth-1 ancestor

    With `parent_of`, walks the real lineage (handles non-dotted ids); else
    falls back to dotted parsing. Returns None for the orchestrator.
    """
    if parent_of is not None and parent_of(agent_id) is not None:
        # The agent is itself a root (bound agent / orchestrator) -> no thread.
        if _is_root_id(agent_id, parent_of):
            return None
        # Walk up until cur's parent is a root: cur is then the depth-1 node
        # (the thread owner).
        cur = agent_id
        for _ in range(_MAX_LINEAGE):
            p = parent_of(cur)
            if p is None or _is_root_id(p, parent_of):
                return cur
            cur = p
        return cur
    depth = agent_id.count(".")
    if depth == 0:
        return None
    parts = agent_id.split(".")
    return ".".join(parts[:2])  # "orchestrator" + first index


def parent_id(agent_id: AgentId) -> AgentId | None:
    """The immediate parent's id (dotted ids only), or None for the root."""
    if agent_id.count(".") == 0:
        return None
    return agent_id.rsplit(".", 1)[0]


def is_descendant_of(agent_id: AgentId, ancestor_id: AgentId) -> bool:
    """True if agent_id is ancestor_id or a (transitive) child of it."""
    return agent_id == ancestor_id or agent_id.startswith(ancestor_id + ".")
