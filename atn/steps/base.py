"""Base step interface and execution context."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import StepDefinition, StepResult

if TYPE_CHECKING:
    from ..connectors_manager import ConnectorManager
    from ..events import EventBus
    from ..inbox import InboxManager
    from ..runtime import Runtime
    from ..store import OutputStore


@dataclass
class StepContext:
    """Everything a step needs to execute."""
    agent_id: str
    execution_id: str
    cancel_event: asyncio.Event             # kill switch — checked before and during
    previous_outputs: list[Any]             # outputs from earlier steps in this pipeline
    inbox_messages: list[Any]               # messages drained from the inbox at pipeline start
    work_dir: Path = field(default_factory=lambda: Path("."))
    env: dict[str, str] = field(default_factory=dict)
    # Services — available for inter-agent steps (message, pull, collect)
    inbox_manager: InboxManager | None = None
    output_store: OutputStore | None = None
    event_bus: EventBus | None = None
    runtime: Runtime | None = None
    connectors: ConnectorManager | None = None
    connector_ids: list[str] = field(default_factory=list)


class StepExecutor(ABC):
    """Base class for step executors.  One subclass per StepType."""

    @abstractmethod
    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        ...
