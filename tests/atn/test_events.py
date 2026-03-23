"""Tests for the EventBus — subscribe, emit, unsubscribe, and history."""
from __future__ import annotations

import pytest

from atn.events import Event, EventBus, EventType


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def _event(etype: EventType = EventType.AGENT_REGISTERED, source: str = "test", **data) -> Event:
    return Event(type=etype, source=source, data=data)


class TestEventBusSubscribe:
    """Tests for subscribing and emitting events."""

    @pytest.mark.asyncio
    async def test_subscribe_receives_matching_event(self, bus: EventBus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(EventType.AGENT_REGISTERED, handler)
        event = _event(EventType.AGENT_REGISTERED)
        await bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_subscribe_ignores_non_matching_event(self, bus: EventBus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(EventType.AGENT_REGISTERED, handler)
        await bus.emit(_event(EventType.AGENT_UNREGISTERED))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_wildcard_receives_all_events(self, bus: EventBus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(None, handler)
        await bus.emit(_event(EventType.AGENT_REGISTERED))
        await bus.emit(_event(EventType.EXECUTION_STARTED))
        await bus.emit(_event(EventType.RUNTIME_STARTED))

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, bus: EventBus):
        received_a = []
        received_b = []

        async def handler_a(e: Event):
            received_a.append(e)

        async def handler_b(e: Event):
            received_b.append(e)

        bus.subscribe(EventType.STEP_STARTED, handler_a)
        bus.subscribe(EventType.STEP_STARTED, handler_b)
        await bus.emit(_event(EventType.STEP_STARTED))

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_specific_and_wildcard_both_fire(self, bus: EventBus):
        specific = []
        wildcard = []

        async def specific_h(e: Event):
            specific.append(e)

        async def wildcard_h(e: Event):
            wildcard.append(e)

        bus.subscribe(EventType.AGENT_ACTIVATED, specific_h)
        bus.subscribe(None, wildcard_h)
        await bus.emit(_event(EventType.AGENT_ACTIVATED))

        assert len(specific) == 1
        assert len(wildcard) == 1


class TestEventBusUnsubscribe:
    """Tests for removing handlers."""

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self, bus: EventBus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(EventType.AGENT_REGISTERED, handler)
        await bus.emit(_event(EventType.AGENT_REGISTERED))
        assert len(received) == 1

        bus.unsubscribe(EventType.AGENT_REGISTERED, handler)
        await bus.emit(_event(EventType.AGENT_REGISTERED))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_handler_is_safe(self, bus: EventBus):
        async def handler(e: Event):
            pass

        # Should not raise
        bus.unsubscribe(EventType.AGENT_REGISTERED, handler)

    @pytest.mark.asyncio
    async def test_unsubscribe_wildcard(self, bus: EventBus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(None, handler)
        await bus.emit(_event(EventType.RUNTIME_STARTED))
        assert len(received) == 1

        bus.unsubscribe(None, handler)
        await bus.emit(_event(EventType.RUNTIME_STARTED))
        assert len(received) == 1


class TestEventBusErrorHandling:
    """Tests for handler error isolation."""

    @pytest.mark.asyncio
    async def test_handler_error_does_not_break_other_handlers(self, bus: EventBus):
        received = []

        async def bad_handler(e: Event):
            raise ValueError("boom")

        async def good_handler(e: Event):
            received.append(e)

        bus.subscribe(EventType.STEP_FAILED, bad_handler)
        bus.subscribe(EventType.STEP_FAILED, good_handler)
        await bus.emit(_event(EventType.STEP_FAILED))

        # good_handler should still receive the event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_wildcard_error_does_not_break_emission(self, bus: EventBus):
        received = []

        async def bad_wildcard(e: Event):
            raise RuntimeError("fail")

        async def good_specific(e: Event):
            received.append(e)

        bus.subscribe(None, bad_wildcard)
        bus.subscribe(EventType.STEP_COMPLETED, good_specific)
        await bus.emit(_event(EventType.STEP_COMPLETED))

        assert len(received) == 1


class TestEventBusHistory:
    """Tests for event history tracking."""

    @pytest.mark.asyncio
    async def test_history_records_events(self, bus: EventBus):
        await bus.emit(_event(EventType.RUNTIME_STARTED))
        await bus.emit(_event(EventType.AGENT_REGISTERED))
        assert len(bus.history) == 2

    @pytest.mark.asyncio
    async def test_history_is_capped(self, bus: EventBus):
        bus._max_history = 5
        for i in range(10):
            await bus.emit(_event(EventType.STEP_OUTPUT, source=f"s{i}"))
        assert len(bus.history) == 5
        # Most recent events are kept
        assert bus.history[-1].source == "s9"

    @pytest.mark.asyncio
    async def test_history_returns_copy(self, bus: EventBus):
        await bus.emit(_event(EventType.RUNTIME_STARTED))
        h = bus.history
        h.clear()
        assert len(bus.history) == 1  # Original not affected


class TestEventDataclass:
    """Tests for the Event frozen dataclass."""

    def test_event_has_auto_fields(self):
        e = Event(type=EventType.RUNTIME_STARTED, source="test")
        assert e.timestamp is not None
        assert e.event_id  # non-empty
        assert e.data == {}

    def test_event_is_frozen(self):
        e = Event(type=EventType.RUNTIME_STARTED, source="test")
        with pytest.raises(AttributeError):
            e.source = "other"

    def test_event_with_data(self):
        e = Event(type=EventType.AGENT_REGISTERED, source="rt", data={"agent_id": "x"})
        assert e.data["agent_id"] == "x"
