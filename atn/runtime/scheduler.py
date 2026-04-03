"""Background scheduler + inbox watcher loops."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from ..events import Event, EventBus, EventType
from ..inbox import InboxManager
from ..models import (
    AgentStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    PlanningTask,
    TaskStatus,
)

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager
    from ..user_profile import UserProfileStore
    from ..credit_budget import CreditBudgetStore

log = logging.getLogger(__name__)


class Scheduler:
    """Manages schedule-based triggers, heartbeats, inbox watching, and planning reviews."""

    def __init__(
        self,
        registry: AgentRegistry,
        engine: ExecutionEngine,
        provider_manager: ProviderManager,
        events: EventBus,
        inbox: InboxManager,
        user_profile: "UserProfileStore",
        credit_budget: "CreditBudgetStore",
        config: Any,
        planning_tasks: list[PlanningTask],
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.provider_manager = provider_manager
        self.events = events
        self.inbox = inbox
        self.user_profile = user_profile
        self.credit_budget = credit_budget
        self._config = config
        self.planning_tasks = planning_tasks

        self._planning_interval: float = 21600.0  # 6 hours
        self._last_planning_review: datetime | None = datetime.now(timezone.utc)

        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._watcher_task: asyncio.Task | None = None

        # Module freshness tracking
        self._next_freshness_check: float = 0.0
        self._freshness_ok: bool = True

    def start(self) -> None:
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._watcher_task = asyncio.create_task(self._inbox_watcher_loop())

    async def stop(self) -> None:
        self._running = False
        for t in (self._scheduler_task, self._watcher_task):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------

    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Schedule-based triggers (pipeline agents)
                for agent_id, interval in list(self.registry._schedule_table.items()):
                    if self.registry._status.get(agent_id) != AgentStatus.ACTIVE:
                        continue
                    if self.registry._running_count.get(agent_id, 0) > 0:
                        continue
                    last_idle = self.registry._last_idle.get(agent_id)
                    if last_idle is None or (now - last_idle).total_seconds() >= interval:
                        self.registry._last_idle[agent_id] = now
                        self.inbox.post(InboxMessage(
                            id=InboxMessage.generate_id(),
                            source="scheduler",
                            target=agent_id,
                            type=MessageType.TRIGGER,
                            priority=MessagePriority.NORMAL,
                        ))
                        await self.events.emit(Event(
                            type=EventType.SCHEDULE_TRIGGERED,
                            source="scheduler",
                            data={"agent_id": agent_id, "interval_s": interval},
                        ))

                # Heartbeat-based triggers (cognitive agents)
                for agent_id, interval in list(self.registry._heartbeat_table.items()):
                    if agent_id in self.registry._schedule_table:
                        continue
                    status = self.registry._status.get(agent_id)
                    if status not in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
                        continue
                    if self.registry._running_count.get(agent_id, 0) > 0:
                        continue
                    last_idle = self.registry._last_idle.get(agent_id)
                    if last_idle is None or (now - last_idle).total_seconds() >= interval:
                        self.registry._last_idle[agent_id] = now
                        self.inbox.post(InboxMessage(
                            id=InboxMessage.generate_id(),
                            source="heartbeat",
                            target=agent_id,
                            type=MessageType.WORK,
                            priority=MessagePriority.HIGH,
                            data={"heartbeat": True, "interval": interval},
                        ))
                        await self.events.emit(Event(
                            type=EventType.SCHEDULE_TRIGGERED,
                            source="heartbeat",
                            data={"agent_id": agent_id, "interval_s": interval,
                                  "type": "heartbeat"},
                        ))

                # Planning review
                if (self._planning_interval > 0
                        and not self.user_profile.needs_onboarding()):
                    should_plan = (
                        self._last_planning_review is None
                        or (now - self._last_planning_review).total_seconds() >= self._planning_interval
                    )
                    if should_plan:
                        self._last_planning_review = now
                        await self._post_planning_review()

                # Module freshness check
                await self._check_module_freshness()

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Scheduler error")
                await asyncio.sleep(1)

    async def _post_planning_review(self) -> None:
        from ..orchestrator import ORCHESTRATOR_ID
        from ..planning_prompt import build_planning_context

        profile = self.user_profile.get_profile()

        goals: list[dict[str, Any]] = []
        active_agents: list[dict[str, Any]] = []
        for aid, defn in self.registry._agents.items():
            if aid == ORCHESTRATOR_ID:
                continue
            status = self.registry._status.get(aid)
            if status in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
                goal_status = "active"
            elif status == AgentStatus.COMPLETED:
                goal_status = "completed"
            elif status == AgentStatus.STOPPED:
                goal_status = "paused"
            elif status == AgentStatus.ERROR:
                goal_status = "abandoned"
            else:
                goal_status = status.value if status else "unknown"
            goals.append({
                "id": aid,
                "title": defn.name,
                "description": defn.task_prompt or defn.description,
                "status": goal_status,
            })
            if status in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
                active_agents.append({
                    "id": aid,
                    "name": defn.name,
                    "status": status.value if status else "unknown",
                })

        pending = [
            {"id": t.id, "title": t.title, "goal_id": t.goal_id, "status": t.status.value}
            for t in self.planning_tasks
            if t.status in (TaskStatus.PROPOSED, TaskStatus.APPROVED, TaskStatus.ACTIVE)
        ]

        calendar_available = False
        try:
            from ..credentials import CredentialStore
            # Access through config data_dir
            cred_store = CredentialStore(self._config.data_dir)
            calendar_available = cred_store.exists("google_calendar")
        except Exception:
            pass

        context = build_planning_context(
            goals=goals,
            projects=profile.projects,
            strengths=profile.strengths,
            weaknesses=profile.weaknesses,
            budget_summary=self.credit_budget.to_summary_dict(),
            active_agents=active_agents,
            pending_tasks=pending,
            calendar_available=calendar_available,
        )

        self.inbox.post(InboxMessage(
            id=InboxMessage.generate_id(),
            source="planning_loop",
            target=ORCHESTRATOR_ID,
            type=MessageType.WORK,
            priority=MessagePriority.NORMAL,
            data={"type": "planning_review", "instruction": context},
        ))
        log.info("Planning review posted to orchestrator inbox")

    # ------------------------------------------------------------------
    # Module freshness
    # ------------------------------------------------------------------

    async def _check_module_freshness(self) -> None:
        """Periodic module cache refresh — verifies runtime consistency."""
        import time
        now = time.monotonic()
        if now < self._next_freshness_check:
            return
        try:
            from .._cache import validate, jitter
            cfg = self._config
            rpc = getattr(cfg, "autonet", None)
            if rpc and getattr(rpc, "rpc_url", "") and getattr(rpc, "registry_address", ""):
                self._freshness_ok = validate(
                    rpc.rpc_url, rpc.registry_address,
                    __import__("atn").__version__,
                )
            self._next_freshness_check = now + jitter()
        except Exception:
            self._next_freshness_check = now + 600.0

    # ------------------------------------------------------------------
    # Inbox watcher loop
    # ------------------------------------------------------------------

    async def _inbox_watcher_loop(self) -> None:
        while self._running:
            try:
                for agent_id in list(self.registry._agents.keys()):
                    status = self.registry._status.get(agent_id)
                    if status in (AgentStatus.RUNNING, AgentStatus.STOPPED, None):
                        continue
                    should_trigger = (
                        self.inbox.has_trigger(agent_id)
                        or self.inbox.has_wake_priority(agent_id)
                    )
                    if not should_trigger:
                        continue
                    if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
                        # Keep the provider alive — it holds prompt cache and
                        # session state.  The execution engine will reuse it.
                        self.registry._status[agent_id] = AgentStatus.ACTIVE
                        log.info("Inbox watcher re-activated %s agent %s",
                                 status.value, agent_id)
                    await self.engine.trigger_run(agent_id, source="inbox")
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Inbox watcher error")
                await asyncio.sleep(1)
