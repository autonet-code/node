"""Delegate registry — tracks the hierarchy of spawned sub-agents.

When the orchestrator (or any sub-agent) calls the ``delegate`` tool, a new
sub-agent is created.  This module tracks the tree of delegates with
hierarchical IDs (``orch.1``, ``orch.1.2``), status, and metadata.

The registry is used for:
- Observability: the UI can show the full delegation tree
- Interrupt cascade: killing a parent kills all descendants
- Persistence: the tree survives daemon restart (JSON file)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DelegateStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class DelegateNode:
    """A single delegate sub-agent in the hierarchy."""
    agent_id: str               # Hierarchical: "orch.1", "orch.1.2"
    parent_id: str | None       # None for root orchestrator
    agent_type: str             # explore, implement, research, debug, review
    prompt: str                 # Task description sent to the sub-agent
    title: str                  # Short human-readable label
    status: DelegateStatus = DelegateStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    result_preview: str = ""    # First 500 chars of the result
    error: str | None = None
    tokens_used: int = 0
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "agent_type": self.agent_type,
            "prompt": self.prompt[:500],
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_preview": self.result_preview,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DelegateNode:
        return cls(
            agent_id=d["agent_id"],
            parent_id=d.get("parent_id"),
            agent_type=d.get("agent_type", "implement"),
            prompt=d.get("prompt", ""),
            title=d.get("title", ""),
            status=DelegateStatus(d.get("status", "pending")),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(timezone.utc),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            result_preview=d.get("result_preview", ""),
            error=d.get("error"),
            tokens_used=d.get("tokens_used", 0),
            tool_calls=d.get("tool_calls", 0),
        )


class DelegateRegistry:
    """Tracks the tree of delegate sub-agents.

    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self._nodes: dict[str, DelegateNode] = {}
        self._child_counters: dict[str, int] = {}  # parent_id -> next child number
        self._store_path = store_path

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def generate_child_id(self, parent_id: str) -> str:
        """Generate the next hierarchical child ID for a parent.

        Example: "orch" -> "orch.1", then "orch.2", etc.
                 "orch.1" -> "orch.1.1", "orch.1.2", etc.
        """
        count = self._child_counters.get(parent_id, 0) + 1
        self._child_counters[parent_id] = count
        return f"{parent_id}.{count}"

    def register(
        self,
        agent_id: str,
        parent_id: str | None,
        agent_type: str,
        prompt: str,
        title: str = "",
    ) -> DelegateNode:
        """Register a new delegate sub-agent."""
        node = DelegateNode(
            agent_id=agent_id,
            parent_id=parent_id,
            agent_type=agent_type,
            prompt=prompt,
            title=title or agent_id,
        )
        self._nodes[agent_id] = node
        log.info("Delegate registered: %s (parent=%s, type=%s)", agent_id, parent_id, agent_type)
        return node

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def update_status(
        self,
        agent_id: str,
        status: DelegateStatus,
        *,
        result_preview: str = "",
        error: str | None = None,
        tokens_used: int = 0,
        tool_calls: int = 0,
    ) -> DelegateNode | None:
        """Update a delegate's status and metadata."""
        node = self._nodes.get(agent_id)
        if node is None:
            log.warning("update_status: unknown delegate %s", agent_id)
            return None

        node.status = status
        if result_preview:
            node.result_preview = result_preview[:500]
        if error is not None:
            node.error = error
        if tokens_used:
            node.tokens_used = tokens_used
        if tool_calls:
            node.tool_calls = tool_calls
        if status in (DelegateStatus.COMPLETED, DelegateStatus.FAILED, DelegateStatus.KILLED):
            node.completed_at = datetime.now(timezone.utc)

        return node

    def remove(self, agent_id: str) -> bool:
        """Remove a delegate and all its descendants from the registry."""
        if agent_id not in self._nodes:
            return False
        # Remove descendants first (children, grandchildren, ...)
        for desc in self.get_descendants(agent_id):
            self._nodes.pop(desc.agent_id, None)
        self._nodes.pop(agent_id, None)
        log.info("Delegate removed: %s", agent_id)
        return True

    # ------------------------------------------------------------------
    # Tree queries
    # ------------------------------------------------------------------

    def get_node(self, agent_id: str) -> DelegateNode | None:
        return self._nodes.get(agent_id)

    def get_children(self, parent_id: str) -> list[DelegateNode]:
        """Direct children of a parent."""
        return [n for n in self._nodes.values() if n.parent_id == parent_id]

    def get_descendants(self, agent_id: str) -> list[DelegateNode]:
        """All descendants (children, grandchildren, ...) via BFS."""
        descendants: list[DelegateNode] = []
        queue = [agent_id]
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            for child in children:
                descendants.append(child)
                queue.append(child.agent_id)
        return descendants

    def get_active(self) -> list[DelegateNode]:
        """All currently running delegates."""
        return [
            n for n in self._nodes.values()
            if n.status in (DelegateStatus.PENDING, DelegateStatus.RUNNING)
        ]

    def get_tree(self) -> dict[str, Any]:
        """Full hierarchy as a dict for the UI/snapshot."""
        nodes = [n.to_dict() for n in self._nodes.values()]
        active_count = len(self.get_active())
        return {
            "nodes": nodes,
            "active_count": active_count,
            "total_count": len(nodes),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_orphans(self) -> int:
        """Mark any PENDING/RUNNING delegates as KILLED (e.g. after restart)."""
        count = 0
        for node in self._nodes.values():
            if node.status in (DelegateStatus.PENDING, DelegateStatus.RUNNING):
                node.status = DelegateStatus.KILLED
                node.error = "Orphaned after restart"
                node.completed_at = datetime.now(timezone.utc)
                count += 1
        if count:
            log.info("Cleaned up %d orphaned delegates", count)
        return count

    def clear(self) -> None:
        """Remove all delegate records."""
        self._nodes.clear()
        self._child_counters.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save delegate tree to JSON."""
        if not self._store_path:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }
        self._store_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def load(self) -> None:
        """Load delegate tree from JSON."""
        if not self._store_path or not self._store_path.exists():
            return
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
            for nd in raw.get("nodes", []):
                node = DelegateNode.from_dict(nd)
                self._nodes[node.agent_id] = node
            # Rebuild child counters from loaded data
            for node in self._nodes.values():
                if node.parent_id:
                    # Parse the child number from the agent_id
                    parts = node.agent_id.rsplit(".", 1)
                    if len(parts) == 2:
                        try:
                            child_num = int(parts[1])
                            current = self._child_counters.get(node.parent_id, 0)
                            if child_num > current:
                                self._child_counters[node.parent_id] = child_num
                        except ValueError:
                            pass
            log.info("Loaded %d delegate records", len(self._nodes))
        except Exception:
            log.exception("Failed to load delegate registry from %s", self._store_path)
