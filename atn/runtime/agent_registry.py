"""Agent lifecycle & hierarchy management.

Owns: _agents, _status, _running_count, _schedule_table, _heartbeat_table,
      _last_idle, _child_counters
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..events import Event, EventBus, EventType
from ..inbox import InboxManager
from ..models import (
    AgentDefinition,
    AgentStatus,
    ExecutionRecord,
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
)
from ..store import ExecutionLog, OutputStore

log = logging.getLogger(__name__)


class AgentRegistry:
    """Agent registration, activation, hierarchy, and idle tracking."""

    def __init__(
        self,
        events: EventBus,
        execution_log: ExecutionLog,
        inbox: InboxManager,
        output_store: OutputStore,
        config: Any,
    ) -> None:
        self.events = events
        self.execution_log = execution_log
        self.inbox = inbox
        self.output_store = output_store
        self._config = config

        self._agents: dict[str, AgentDefinition] = {}
        self._status: dict[str, AgentStatus] = {}
        self._running_count: dict[str, int] = {}
        self._schedule_table: dict[str, float] = {}
        self._heartbeat_table: dict[str, float] = {}
        self._last_idle: dict[str, datetime] = {}
        self._child_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_agent(self, defn: AgentDefinition) -> str:
        self._agents[defn.id] = defn
        self._status[defn.id] = AgentStatus.REGISTERED
        self._running_count[defn.id] = 0
        # Heartbeat and legacy schedule are mutually exclusive.
        if defn.heartbeat:
            self._heartbeat_table[defn.id] = parse_interval(defn.heartbeat.interval)
            self._schedule_table.pop(defn.id, None)
            self._last_idle.setdefault(defn.id, datetime.now(timezone.utc))
        elif defn.schedule:
            self._schedule_table[defn.id] = parse_interval(defn.schedule)
            self._heartbeat_table.pop(defn.id, None)
            self._last_idle[defn.id] = datetime.now(timezone.utc)
        # Hydrate execution history
        n = self.execution_log.hydrate(defn.id)
        if n:
            log.debug("Hydrated %d execution record(s) for %s", n, defn.id)
        await self.events.emit(Event(
            type=EventType.AGENT_REGISTERED,
            source="runtime",
            data={"agent_id": defn.id, "name": defn.name,
                  "mode": defn.mode.value,
                  "steps": len(defn.steps), "schedule": defn.schedule,
                  "description": defn.description,
                  "model": defn.model,
                  "parent_id": defn.parent_id,
                  "concurrency": defn.concurrency},
        ))
        return defn.id

    async def unregister_agent(self, agent_id: str, *, _force: bool = False) -> None:
        """Remove an agent from the registry.

        NOTE: Callers must handle killing executions, cleaning up providers,
        conversations, and files before calling this.
        """
        from ..orchestrator import ORCHESTRATOR_ID
        if agent_id == ORCHESTRATOR_ID and not _force:
            raise ValueError("The orchestrator cannot be unregistered")
        self._agents.pop(agent_id, None)
        self._status.pop(agent_id, None)
        self._running_count.pop(agent_id, None)
        self._schedule_table.pop(agent_id, None)
        self._heartbeat_table.pop(agent_id, None)
        self._last_idle.pop(agent_id, None)
        self.inbox.remove_agent(agent_id)
        self.output_store.remove(agent_id)
        self.execution_log.remove_agent(agent_id)
        await self.events.emit(Event(
            type=EventType.AGENT_UNREGISTERED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    async def activate_agent(self, agent_id: str) -> None:
        self._require_agent(agent_id)
        self._status[agent_id] = AgentStatus.ACTIVE
        await self.events.emit(Event(
            type=EventType.AGENT_ACTIVATED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    async def deactivate_agent(self, agent_id: str) -> None:
        self._require_agent(agent_id)
        self._status[agent_id] = AgentStatus.STOPPED
        await self.events.emit(Event(
            type=EventType.AGENT_DEACTIVATED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self._agents.get(agent_id)

    def get_status(self, agent_id: str) -> AgentStatus | None:
        return self._status.get(agent_id)

    def list_agents(self) -> list[tuple[AgentDefinition, AgentStatus]]:
        return [
            (defn, self._status[aid])
            for aid, defn in self._agents.items()
        ]

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def _resolve_parent_agent_id(self, parent_id: str) -> str:
        if parent_id in self._agents:
            return parent_id
        from ..orchestrator import ORCHESTRATOR_ID
        if parent_id == "orch" and ORCHESTRATOR_ID in self._agents:
            return ORCHESTRATOR_ID
        return parent_id

    def get_children(self, agent_id: str) -> list[AgentDefinition]:
        from ..orchestrator import ORCHESTRATOR_ID
        children = []
        for defn in self._agents.values():
            if defn.parent_id == agent_id:
                children.append(defn)
            elif (agent_id == ORCHESTRATOR_ID
                  and defn.parent_id == "orch"):
                children.append(defn)
        return children

    def get_descendants(self, agent_id: str) -> list[AgentDefinition]:
        descendants: list[AgentDefinition] = []
        queue = [agent_id]
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            for child in children:
                descendants.append(child)
                queue.append(child.id)
        return descendants

    def generate_child_id(self, parent_id: str) -> str:
        count = self._child_counters.get(parent_id, 0) + 1
        self._child_counters[parent_id] = count
        return f"{parent_id}.{count}"

    # ------------------------------------------------------------------
    # Failure propagation
    # ------------------------------------------------------------------

    async def notify_parent_of_failure(
        self, agent_id: str, record: ExecutionRecord,
        active_providers: dict,
    ) -> None:
        if record.status != ExecutionStatus.FAILED:
            return
        defn = self._agents.get(agent_id)
        if not defn or not defn.parent_id:
            return
        resolved_parent = self._resolve_parent_agent_id(defn.parent_id)
        if resolved_parent not in self._agents:
            return

        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=agent_id,
            target=resolved_parent,
            type=MessageType.ALERT,
            priority=MessagePriority.HIGH,
            data={
                "type": "child_error",
                "child_agent": agent_id,
                "child_name": defn.name,
                "error": record.error or "Unknown error",
                "execution_id": record.execution_id,
                "instruction": (
                    f"Your child agent '{defn.name}' ({agent_id}) failed: "
                    f"{record.error or 'Unknown error'}. "
                    f"Investigate and decide whether to retry, fix, or escalate."
                ),
            },
        )
        self.inbox.post(msg)
        log.info(
            "Failure propagation: posted child_error ALERT for %s -> parent %s",
            agent_id, resolved_parent,
        )

        parent_provider = active_providers.get(resolved_parent)
        if parent_provider is None:
            parent_provider = active_providers.get(defn.parent_id)
        if parent_provider is not None:
            inject_text = (
                f"[CHILD FAILED] Agent '{defn.name}' ({agent_id}) "
                f"failed with error: {record.error or 'Unknown error'}"
            )
            try:
                await parent_provider.send_user_message(inject_text)
                log.info("Injected failure alert into parent %s bridge session", resolved_parent)
            except Exception:
                log.debug("Could not inject failure into parent bridge (may not support send_user_message)")

    async def on_agent_completed(
        self, agent_id: str, record: ExecutionRecord,
        active_providers: dict,
    ) -> None:
        defn = self._agents.get(agent_id)
        if not defn:
            return
        if record.status == ExecutionStatus.COMPLETED:
            if defn.schedule or defn.heartbeat:
                self._status[agent_id] = AgentStatus.ACTIVE
            else:
                self._status[agent_id] = AgentStatus.COMPLETED
        self._last_idle[agent_id] = datetime.now(timezone.utc)

        parent_id = defn.parent_id
        if not parent_id:
            return
        resolved_parent = self._resolve_parent_agent_id(parent_id)
        if resolved_parent not in self._agents:
            return

        result_preview = ""
        if isinstance(record.output, dict):
            result_preview = str(record.output.get("result", ""))[:2000]
        elif record.output is not None:
            result_preview = str(record.output)[:2000]

        status_str = record.status.value
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=agent_id,
            target=resolved_parent,
            type=MessageType.WORK,
            priority=MessagePriority.HIGH,
            data={
                "type": "child_completed",
                "child_id": agent_id,
                "child_name": defn.name,
                "status": status_str,
                "output_preview": result_preview[:2000],
                "result_preview": result_preview[:2000],
                "error": record.error,
                "instruction": (
                    f"Your child agent '{defn.name}' has {status_str}. "
                    f"Check its output with get_output('{agent_id}')."
                ),
            },
        )
        self.inbox.post(msg)
        log.info("Innate wake-up: posted child_completed for %s -> parent %s", agent_id, resolved_parent)

        parent_provider = active_providers.get(resolved_parent)
        if parent_provider is None:
            parent_provider = active_providers.get(parent_id)
        if parent_provider is not None:
            inject_text = (
                f"[CHILD COMPLETED] Agent '{defn.name}' ({agent_id}) "
                f"finished with status: {record.status.value}.\n"
                f"Result preview: {result_preview[:2000]}"
            )
            if record.error:
                inject_text += f"\nError: {record.error}"
            try:
                await parent_provider.send_user_message(inject_text)
                log.info("Injected child_completed message into parent %s bridge session", resolved_parent)
            except Exception as exc:
                log.warning("Failed to inject into parent %s session: %s", resolved_parent, exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_interval(schedule: str) -> float:
    m = re.match(r"^(\d+)\s*([smh])$", schedule.strip().lower())
    if not m:
        raise ValueError(f"Invalid schedule: {schedule!r}  (use e.g. '30s', '5m', '1h')")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600}[unit]
