"""Per-agent inbox — the unified communication and trigger surface.

Every agent has one inbox.  Messages arrive from other agents, the scheduler,
or the user.  The Runtime watches inboxes and decides when
to trigger runs.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from .models import InboxMessage, MessagePriority, MessageType


class InboxManager:
    """Manages all agent inboxes centrally."""

    def __init__(self) -> None:
        self._queues: dict[str, list[InboxMessage]] = defaultdict(list)
        # Per-agent asyncio.Event — set when a new message arrives.
        self._notify: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def post(self, message: InboxMessage) -> None:
        """Post a message to an agent's inbox."""
        self._queues[message.target].append(message)
        # Notify anyone waiting on this inbox.
        if message.target in self._notify:
            self._notify[message.target].set()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def drain(self, agent_id: str) -> list[InboxMessage]:
        """Remove and return all messages, oldest first."""
        messages = self._queues.pop(agent_id, [])
        # Reset the notification flag.
        if agent_id in self._notify:
            self._notify[agent_id].clear()
        return messages

    def peek(self, agent_id: str) -> list[InboxMessage]:
        """Return a copy of all messages without removing them."""
        return list(self._queues.get(agent_id, []))

    def count(self, agent_id: str) -> int:
        return len(self._queues.get(agent_id, []))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def has_trigger(self, agent_id: str) -> bool:
        return any(
            m.type == MessageType.TRIGGER
            for m in self._queues.get(agent_id, [])
        )

    def has_wake_priority(self, agent_id: str) -> bool:
        """True if there's a message urgent enough to wake an idle agent."""
        return any(
            m.priority in (MessagePriority.HIGH, MessagePriority.URGENT)
            for m in self._queues.get(agent_id, [])
        )

    # ------------------------------------------------------------------
    # Async waiting
    # ------------------------------------------------------------------

    def get_notify_event(self, agent_id: str) -> asyncio.Event:
        """Return an Event that is set whenever a message arrives."""
        if agent_id not in self._notify:
            self._notify[agent_id] = asyncio.Event()
        return self._notify[agent_id]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_agent(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)
        self._notify.pop(agent_id, None)
