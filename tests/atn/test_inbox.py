"""Tests for the InboxManager — message posting, routing, queries, and draining."""
from __future__ import annotations

import asyncio

import pytest

from atn.inbox import InboxManager
from atn.models import InboxMessage, MessagePriority, MessageType


def _msg(
    target: str,
    *,
    source: str = "user",
    msg_type: MessageType = MessageType.WORK,
    priority: MessagePriority = MessagePriority.NORMAL,
    data: dict | None = None,
) -> InboxMessage:
    """Helper to create test messages."""
    return InboxMessage(
        id=InboxMessage.generate_id(),
        source=source,
        target=target,
        type=msg_type,
        priority=priority,
        data=data or {},
    )


class TestInboxPost:
    """Tests for posting messages."""

    def test_post_single_message(self):
        inbox = InboxManager()
        msg = _msg("agent-1")
        inbox.post(msg)
        assert inbox.count("agent-1") == 1

    def test_post_multiple_messages(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1"))
        inbox.post(_msg("agent-1"))
        inbox.post(_msg("agent-1"))
        assert inbox.count("agent-1") == 3

    def test_post_to_different_agents(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1"))
        inbox.post(_msg("agent-2"))
        assert inbox.count("agent-1") == 1
        assert inbox.count("agent-2") == 1

    def test_count_empty_inbox(self):
        inbox = InboxManager()
        assert inbox.count("nonexistent") == 0


class TestInboxDrain:
    """Tests for draining and peeking at messages."""

    def test_drain_returns_all_messages(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", source="a"))
        inbox.post(_msg("agent-1", source="b"))
        messages = inbox.drain("agent-1")
        assert len(messages) == 2
        assert messages[0].source == "a"
        assert messages[1].source == "b"

    def test_drain_empties_inbox(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1"))
        inbox.drain("agent-1")
        assert inbox.count("agent-1") == 0

    def test_drain_empty_inbox_returns_empty_list(self):
        inbox = InboxManager()
        assert inbox.drain("nonexistent") == []

    def test_peek_returns_copy(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1"))
        peeked = inbox.peek("agent-1")
        assert len(peeked) == 1
        # Inbox is not drained
        assert inbox.count("agent-1") == 1

    def test_peek_empty_returns_empty(self):
        inbox = InboxManager()
        assert inbox.peek("nonexistent") == []


class TestInboxQueries:
    """Tests for has_trigger and has_wake_priority."""

    def test_has_trigger_true(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", msg_type=MessageType.TRIGGER))
        assert inbox.has_trigger("agent-1") is True

    def test_has_trigger_false_with_other_types(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", msg_type=MessageType.WORK))
        inbox.post(_msg("agent-1", msg_type=MessageType.INFO))
        assert inbox.has_trigger("agent-1") is False

    def test_has_trigger_false_empty(self):
        inbox = InboxManager()
        assert inbox.has_trigger("nonexistent") is False

    def test_has_wake_priority_high(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", priority=MessagePriority.HIGH))
        assert inbox.has_wake_priority("agent-1") is True

    def test_has_wake_priority_urgent(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", priority=MessagePriority.URGENT))
        assert inbox.has_wake_priority("agent-1") is True

    def test_has_wake_priority_false_normal(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", priority=MessagePriority.NORMAL))
        assert inbox.has_wake_priority("agent-1") is False

    def test_has_wake_priority_false_low(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1", priority=MessagePriority.LOW))
        assert inbox.has_wake_priority("agent-1") is False


class TestInboxNotify:
    """Tests for the async notification mechanism."""

    def test_get_notify_event_creates_event(self):
        inbox = InboxManager()
        event = inbox.get_notify_event("agent-1")
        assert isinstance(event, asyncio.Event)
        assert not event.is_set()

    def test_get_notify_event_same_agent_returns_same(self):
        inbox = InboxManager()
        e1 = inbox.get_notify_event("agent-1")
        e2 = inbox.get_notify_event("agent-1")
        assert e1 is e2

    @pytest.mark.asyncio
    async def test_post_sets_notify_event(self):
        inbox = InboxManager()
        event = inbox.get_notify_event("agent-1")
        assert not event.is_set()
        inbox.post(_msg("agent-1"))
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_drain_clears_notify_event(self):
        inbox = InboxManager()
        event = inbox.get_notify_event("agent-1")
        inbox.post(_msg("agent-1"))
        assert event.is_set()
        inbox.drain("agent-1")
        assert not event.is_set()


class TestInboxCleanup:
    """Tests for removing agent inboxes."""

    def test_remove_agent_clears_messages(self):
        inbox = InboxManager()
        inbox.post(_msg("agent-1"))
        inbox.remove_agent("agent-1")
        assert inbox.count("agent-1") == 0
        assert inbox.peek("agent-1") == []

    def test_remove_agent_clears_notify(self):
        inbox = InboxManager()
        inbox.get_notify_event("agent-1")
        inbox.remove_agent("agent-1")
        # Getting a new event should create a fresh one
        event = inbox.get_notify_event("agent-1")
        assert not event.is_set()

    def test_remove_nonexistent_is_safe(self):
        inbox = InboxManager()
        inbox.remove_agent("nonexistent")  # Should not raise
