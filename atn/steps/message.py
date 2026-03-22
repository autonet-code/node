"""Message step executor — post to another agent's inbox.

Posts a message to the target agent's inbox and moves on.  If you need
the target's result, add a collect step later in the pipeline.

Config keys:
  target      (str, required)  Agent ID to message.
  msg_type    (str)  MessageType value.  Default: "trigger".
  priority    (str)  MessagePriority value.  Default: "normal".
  data        (dict) Static payload.  Merged with previous step output if present.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..events import Event, EventType
from ..models import (
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    StepDefinition,
    StepResult,
    StepType,
)
from .base import StepContext, StepExecutor


class MessageStepExecutor(StepExecutor):

    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        target: str = step.config["target"]
        msg_type = MessageType(step.config.get("msg_type", "trigger"))
        priority = MessagePriority(step.config.get("priority", "normal"))

        result = StepResult(
            step_index=step_index,
            step_name=step.name or f"message_{step_index}",
            step_type=StepType.MESSAGE,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        if context.cancel_event.is_set():
            result.status = ExecutionStatus.KILLED
            result.completed_at = datetime.now(timezone.utc)
            return result

        if context.inbox_manager is None:
            result.status = ExecutionStatus.FAILED
            result.error = "No inbox_manager in context"
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Build message payload — only explicit static data from config.
        # The message step is a signal, not a data transport.  Data flows
        # through the output store, not through inboxes.
        payload: dict[str, Any] = {}
        if "data" in step.config:
            payload.update(step.config["data"])

        posted_at = datetime.now(timezone.utc)
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=context.agent_id,
            target=target,
            type=msg_type,
            priority=priority,
            data=payload,
        )
        context.inbox_manager.post(msg)

        # Emit event
        if context.event_bus:
            await context.event_bus.emit(Event(
                type=EventType.MESSAGE_POSTED,
                source=context.agent_id,
                data={
                    "agent_id": context.agent_id,
                    "target": target,
                    "message_id": msg.id,
                    "msg_type": msg_type.value,
                    "priority": priority.value,
                },
            ))

        result.status = ExecutionStatus.COMPLETED
        result.output = {
            "target": target,
            "message_id": msg.id,
            "posted_at": posted_at.isoformat(),
        }
        result.completed_at = datetime.now(timezone.utc)
        return result
