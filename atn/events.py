"""Event system — the backbone of observability.

Every component in ATN emits structured events through the EventBus.
The CLI dashboard, execution log, and (later) WebSocket API all subscribe here.
This is the single source of truth for what is happening in the system.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

log = logging.getLogger(__name__)


class EventType(Enum):
    # Runtime lifecycle
    RUNTIME_STARTED = "runtime.started"
    RUNTIME_STOPPED = "runtime.stopped"

    # Agent lifecycle
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_ACTIVATED = "agent.activated"
    AGENT_DEACTIVATED = "agent.deactivated"

    # Execution lifecycle
    EXECUTION_QUEUED = "execution.queued"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_KILLED = "execution.killed"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_KILLED = "step.killed"
    STEP_OUTPUT = "step.output"

    # Inbox
    MESSAGE_POSTED = "message.posted"
    MESSAGE_DRAINED = "message.drained"

    # Scheduler
    SCHEDULE_TRIGGERED = "schedule.triggered"

    # Delegate sub-agents
    DELEGATE_SPAWNED = "delegate.spawned"
    DELEGATE_RUNNING = "delegate.running"
    DELEGATE_COMPLETED = "delegate.completed"
    DELEGATE_FAILED = "delegate.failed"
    DELEGATE_KILLED = "delegate.killed"

    # Agent tool use (from bridge subprocess)
    AGENT_TOOL_USE_START = "agent.tool_use_start"
    AGENT_TOOL_USE_RESULT = "agent.tool_use_result"

    # Voice service
    VOICE_STARTED = "voice.started"
    VOICE_STOPPED = "voice.stopped"
    VOICE_SPEAKING = "voice.speaking"
    VOICE_RECORDING = "voice.recording"
    VOICE_TRANSCRIBED = "voice.transcribed"

    # Autonet network service
    AUTONET_STARTED = "autonet.started"
    AUTONET_STOPPED = "autonet.stopped"
    AUTONET_STATUS = "autonet.status"
    AUTONET_WALLET = "autonet.wallet"

    # Generic / custom events
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class Event:
    """A structured event emitted by any component in the system."""
    type: EventType
    source: str                 # who emitted: "runtime", agent_id, "scheduler", etc.
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])


# Type alias for event handler coroutines.
# None key = wildcard (subscribe to all events).
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Central pub-sub event bus.

    All observability flows through here.  Handlers are async coroutines.
    Subscribe with event_type=None to receive every event.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}
        self._history: list[Event] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType | None, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    async def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Specific-type handlers
        for handler in self._handlers.get(event.type, []):
            try:
                await handler(event)
            except Exception:
                log.exception("Event handler error for %s", event.type.value)

        # Wildcard handlers
        for handler in self._handlers.get(None, []):
            try:
                await handler(event)
            except Exception:
                log.exception("Wildcard handler error for %s", event.type.value)

    @property
    def history(self) -> list[Event]:
        return list(self._history)
