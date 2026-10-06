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
)

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager
    from ..user_profile import UserProfileStore
    from ..credit_budget import CreditBudgetStore

log = logging.getLogger(__name__)

# Crash-loop backoff (inbox watcher): an ERROR agent with pending messages
# waits base * 2^(fails-1) seconds before re-activation; after MAX
# consecutive failures it is parked until manual reactivation.
_CRASH_LOOP_BASE_COOLDOWN = 10.0
_CRASH_LOOP_MAX_FAILURES = 3


class Scheduler:
    """Manages schedule-based triggers, heartbeats, and inbox watching.

    The old periodic planning review (a 6h digest pushed to the
    fleet root's inbox) was removed 2026-08-30: everything it carried is
    pullable via the planning/profile tools, and a fleet that wants the
    impulse gives an agent a heartbeat instead.
    """

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

        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._watcher_task: asyncio.Task | None = None

        # Module freshness tracking
        self._next_freshness_check: float = 0.0
        self._freshness_ok: bool = True

        # Crash-loop backoff: agents parked after repeated instant failures
        # (see _inbox_watcher_loop). Cleared on any successful trigger.
        self._crash_loop_parked: set[str] = set()

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
                    # A non-positive interval would fire a full execution on
                    # every poll; the registry floors it, this is the backstop.
                    if interval <= 0:
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

                # Module freshness check
                await self._check_module_freshness()

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Scheduler error")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Module freshness
    # ------------------------------------------------------------------

    async def _check_module_freshness(self) -> None:
        """Periodic module cache refresh â€” verifies runtime consistency."""
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
                    if status == AgentStatus.ERROR:
                        # Crash-loop backoff: an agent that keeps failing
                        # instantly (e.g. unresolvable provider) would
                        # otherwise be re-triggered every poll â€” observed
                        # live at ~2 failures/sec. Exponential cool-down,
                        # then a hard stop pending manual reactivation.
                        fails = self.registry._consec_failures.get(agent_id, 0)
                        if fails >= _CRASH_LOOP_MAX_FAILURES:
                            if agent_id not in self._crash_loop_parked:
                                self._crash_loop_parked.add(agent_id)
                                log.warning(
                                    "Agent %s failed %d times in a row â€” "
                                    "parking (reactivate manually or "
                                    "post a new message after fixing it)",
                                    agent_id, fails,
                                )
                            continue
                        last_idle = self.registry._last_idle.get(agent_id)
                        if last_idle is not None:
                            cooldown = _CRASH_LOOP_BASE_COOLDOWN * (2 ** max(0, fails - 1))
                            elapsed = (datetime.now(timezone.utc) - last_idle).total_seconds()
                            if elapsed < cooldown:
                                continue
                    if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
                        # Keep the provider alive â€” it holds prompt cache and
                        # session state.  The execution engine will reuse it.
                        self.registry._status[agent_id] = AgentStatus.ACTIVE
                        log.info("Inbox watcher re-activated %s agent %s",
                                 status.value, agent_id)
                    self._crash_loop_parked.discard(agent_id)
                    await self.engine.trigger_run(agent_id, source="inbox")
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Inbox watcher error")
                await asyncio.sleep(1)
