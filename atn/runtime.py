"""The Runtime — central brain of ATN.

Responsibilities:
  - Agent registry (register / unregister / activate / deactivate)
  - Inbox watching — triggers runs when trigger or wake-priority messages arrive
  - Scheduler — posts trigger messages on a timer
  - Pipeline execution — runs the step pipeline, tracks progress
  - Concurrency enforcement — respects per-agent concurrency limits
  - Kill switches — per-execution, per-agent, and global
  - Snapshot — single observable state for the CLI dashboard
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .events import Event, EventBus, EventType
from .inbox import InboxManager
from .models import (
    AgentDefinition,
    AgentStatus,
    ExecutionRecord,
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    StepType,
    TokenUsage,
)
from .agent_registry import DelegateRegistry
from .config import ATNConfig, save_provider_to_config, remove_provider_from_config
from .connectors_manager import ConnectorManager, ConnectorSpec
from .providers.anthropic import AnthropicProvider
from .providers.base import ProviderError
from .providers.bridge import BridgeProvider
from .providers.ollama import OllamaProvider
from .providers.openai_compat import OpenAICompatibleProvider
from .steps.base import StepContext, StepExecutor
from .steps.cognitive import CognitiveStepExecutor
from .steps.collect import CollectStepExecutor
from .steps.message import MessageStepExecutor
from .steps.pull import PullStepExecutor
from .steps.script import ScriptStepExecutor
from .conversation import ConversationStore
from .credit_budget import CreditBudgetStore
from .credentials import CredentialStore
from .models import PlanningTask, TaskStatus, TaskType
from .store import AgentOutput, ExecutionLog, OutputStore
from .user_profile import UserProfileStore

log = logging.getLogger(__name__)


class Runtime:

    def __init__(self, event_bus: EventBus, data_dir: Path | None = None, config: ATNConfig | None = None) -> None:
        self.events = event_bus
        self.data_dir = data_dir
        self._config = config or ATNConfig()
        self.inbox = InboxManager()
        self.output_store = OutputStore()
        self.execution_log = ExecutionLog(agents_dir=self._config.agents_dir)
        self.conversation = ConversationStore(self._config.data_dir)
        self.credential_store = CredentialStore(self._config.data_dir)
        self.user_profile = UserProfileStore(self._config.data_dir)
        self.credit_budget = CreditBudgetStore(self._config.data_dir)

        # Planning tasks (in-memory, persisted to JSON)
        self.planning_tasks: list[PlanningTask] = []
        self._planning_tasks_path = self._config.data_dir / "planning_tasks.json"
        self._load_planning_tasks()

        # Planning loop interval (seconds).  0 = disabled.
        self._planning_interval: float = 21600.0  # 6 hours
        self._last_planning_review: datetime | None = None

        # MCP connectors — start with bundled specs, overlay user config
        from .connectors import get_bundled_specs
        connector_specs = get_bundled_specs()
        # User-defined connectors override bundled ones
        for name, cc in self._config.connectors.items():
            connector_specs[name] = ConnectorSpec(
                mode=cc.mode,
                package=cc.package,
                entry=cc.entry,
                command=cc.command or None,
                args=cc.args,
                env=cc.env,
                env_required=cc.env_required,
            )
        self.connectors = ConnectorManager(connector_specs)

        # Inject stored credentials into connectors that need them
        self._inject_connector_credentials()

        # Custom (user-defined) providers — IDs not in _KNOWN_PROVIDERS
        self._custom_providers: set[str] = set()

        # Delegate sub-agent registry
        self.delegate_registry = DelegateRegistry(
            store_path=self._config.data_dir / "delegates.json",
        )
        # Active delegate providers — for mid-session message injection
        self._delegate_providers: dict[str, BridgeProvider] = {}
        # Background tasks for running delegates
        self._delegate_tasks: dict[str, asyncio.Task] = {}
        # Completed delegate results — stored until collected
        self._delegate_results: dict[str, dict[str, Any]] = {}
        # Events for signalling delegate completion to delegate_collect
        self._delegate_done: dict[str, asyncio.Event] = {}

        # Voice service (lazy — started only if voice config is enabled)
        self.voice = None  # type: Any  # VoiceService | None

        # Autonet network service (optional — works without it)
        from .autonet_service import AutonetBridge
        self.autonet = AutonetBridge(self._config.autonet, event_bus=self.events)

        # Agent registry
        self._agents: dict[str, AgentDefinition] = {}
        self._status: dict[str, AgentStatus] = {}

        # Execution tracking
        self._executions: dict[str, ExecutionRecord] = {}       # eid -> record
        self._tasks: dict[str, asyncio.Task] = {}               # eid -> task
        self._cancels: dict[str, asyncio.Event] = {}            # eid -> cancel
        self._running_count: dict[str, int] = {}                # agent_id -> count
        self._interrupt_hooks: dict[str, Callable] = {}         # eid -> async callable

        # Step executors
        cognitive = CognitiveStepExecutor()
        self._setup_providers(cognitive)
        self._executors: dict[StepType, StepExecutor] = {
            StepType.SCRIPT: ScriptStepExecutor(),
            StepType.MESSAGE: MessageStepExecutor(),
            StepType.PULL: PullStepExecutor(),
            StepType.COLLECT: CollectStepExecutor(),
            StepType.COGNITIVE: cognitive,
        }

        # Scheduler
        self._schedule_table: dict[str, float] = {}             # agent_id -> seconds
        self._last_scheduled: dict[str, datetime] = {}

        # Background loops
        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._watcher_task: asyncio.Task | None = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        self._running = True

        # Recover any executions that crashed mid-pipeline last run
        recovered = self.execution_log.recover_running()
        if recovered:
            log.warning("Recovered %d crashed execution(s) from previous run", len(recovered))

        # Delegates are ephemeral — clear the registry on startup.
        # The execution log retains historical records if needed.
        self.delegate_registry.clear()
        self.delegate_registry.save()

        # Auto-detect bridge and Ollama (async probes)
        await self._auto_detect_providers()

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._watcher_task = asyncio.create_task(self._inbox_watcher_loop())

        # Start voice service if configured
        if self._config.voice.enabled:
            await self.start_voice()

        # Start autonet service if configured
        if self._config.autonet.enabled:
            await self.autonet.start()

        await self.events.emit(Event(
            type=EventType.RUNTIME_STARTED,
            source="runtime",
        ))

    async def stop(self) -> None:
        self._running = False
        await self.kill_all()
        # Stop voice service
        if self.voice is not None:
            await self.voice.stop()
            self.voice = None
        for t in (self._scheduler_task, self._watcher_task):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        # Stop autonet service
        if self.autonet.state.status.value in ("running", "paused"):
            await self.autonet.stop()
        # Stop all MCP connectors
        await self.connectors.stop_all()
        # Close providers that hold resources (e.g. bridge subprocess)
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            for provider in cognitive._providers.values():
                if hasattr(provider, "close"):
                    try:
                        await provider.close()
                    except Exception:
                        pass
        await self.events.emit(Event(
            type=EventType.RUNTIME_STOPPED,
            source="runtime",
        ))

    # ==================================================================
    # Voice Service
    # ==================================================================

    async def start_voice(self) -> dict:
        """Start the voice service (TTS/STT/mixer).

        Requires ``pip install atn[voice]``.  Returns a status dict.
        """
        if self.voice is not None:
            return {"status": "already_running"}
        try:
            from .voice_service import VOICE_AVAILABLE, VoiceService
            if not VOICE_AVAILABLE:
                return {
                    "status": "unavailable",
                    "error": "Voice extras not installed.  "
                             "Install with: pip install atn[voice]",
                }
            self.voice = VoiceService(self.events, self, self._config.voice)
            await self.voice.start()
            return {"status": "started", "backend": self._config.voice.backend}
        except Exception as exc:
            log.warning("Failed to start voice service: %s", exc)
            self.voice = None
            return {"status": "failed", "error": str(exc)}

    async def stop_voice(self) -> dict:
        """Stop the voice service."""
        if self.voice is None:
            return {"status": "not_running"}
        await self.voice.stop()
        self.voice = None
        return {"status": "stopped"}

    def _voice_snapshot(self) -> dict:
        """Voice status for the snapshot."""
        if self.voice:
            return self.voice.get_status()
        try:
            from .voice_service import VOICE_AVAILABLE
        except ImportError:
            VOICE_AVAILABLE = False
        return {"running": False, "available": VOICE_AVAILABLE}

    # ==================================================================
    # Agent Management
    # ==================================================================

    async def register_agent(self, defn: AgentDefinition) -> str:
        self._agents[defn.id] = defn
        self._status[defn.id] = AgentStatus.REGISTERED
        self._running_count[defn.id] = 0
        if defn.schedule:
            self._schedule_table[defn.id] = self._parse_interval(defn.schedule)
        await self.events.emit(Event(
            type=EventType.AGENT_REGISTERED,
            source="runtime",
            data={"agent_id": defn.id, "name": defn.name,
                  "steps": len(defn.steps), "schedule": defn.schedule},
        ))
        return defn.id

    async def unregister_agent(self, agent_id: str, *, _force: bool = False) -> None:
        from .orchestrator import ORCHESTRATOR_ID
        if agent_id == ORCHESTRATOR_ID and not _force:
            raise ValueError("The orchestrator cannot be unregistered")
        await self.kill_agent(agent_id)
        self._agents.pop(agent_id, None)
        self._status.pop(agent_id, None)
        self._running_count.pop(agent_id, None)
        self._schedule_table.pop(agent_id, None)
        self._last_scheduled.pop(agent_id, None)
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

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self._agents.get(agent_id)

    def get_status(self, agent_id: str) -> AgentStatus | None:
        return self._status.get(agent_id)

    def list_agents(self) -> list[tuple[AgentDefinition, AgentStatus]]:
        return [
            (defn, self._status[aid])
            for aid, defn in self._agents.items()
        ]

    # ==================================================================
    # Execution — trigger & pipeline
    # ==================================================================

    async def trigger_run(self, agent_id: str, source: str = "user") -> str | None:
        """Trigger an agent run.  Returns execution_id, or None if at concurrency limit."""
        self._require_agent(agent_id)
        defn = self._agents[agent_id]

        if self._running_count.get(agent_id, 0) >= defn.concurrency:
            await self.events.emit(Event(
                type=EventType.EXECUTION_QUEUED,
                source="runtime",
                data={"agent_id": agent_id, "reason": "concurrency_limit",
                      "limit": defn.concurrency},
            ))
            return None

        eid = ExecutionRecord.generate_id()
        cancel = asyncio.Event()

        record = ExecutionRecord(
            execution_id=eid,
            agent_id=agent_id,
            status=ExecutionStatus.RUNNING,
            trigger_source=source,
        )
        self._executions[eid] = record
        self._cancels[eid] = cancel
        self._running_count[agent_id] = self._running_count.get(agent_id, 0) + 1
        self._status[agent_id] = AgentStatus.RUNNING
        self.execution_log.record(record)

        task = asyncio.create_task(self._execute_pipeline(defn, record, cancel))
        self._tasks[eid] = task

        await self.events.emit(Event(
            type=EventType.EXECUTION_STARTED,
            source=agent_id,
            data={"agent_id": agent_id, "execution_id": eid,
                  "trigger_source": source, "total_steps": len(defn.steps)},
        ))
        return eid

    async def _execute_pipeline(
        self,
        defn: AgentDefinition,
        record: ExecutionRecord,
        cancel: asyncio.Event,
    ) -> None:
        """Run every step in sequence.  Updates the record in-place."""
        previous: list[Any] = []

        # Drain inbox — separate trigger messages from work messages.
        # Trigger messages that carry data (e.g. user instructions) are
        # included so the content reaches the cognitive step's {inbox}.
        all_messages = self.inbox.drain(defn.id)
        work_messages = [
            m for m in all_messages
            if m.type != MessageType.TRIGGER or m.data
        ]

        # Start any MCP connectors this agent needs
        if defn.connector_ids:
            try:
                await self.connectors.ensure_started(defn.connector_ids)
            except Exception as exc:
                record.status = ExecutionStatus.FAILED
                record.error = f"Failed to start connectors: {exc}"
                record.completed_at = datetime.now(timezone.utc)
                self.execution_log.record(record)
                self.execution_log.persist(record)
                self._running_count[defn.id] = max(0, self._running_count.get(defn.id, 1) - 1)
                self._executions.pop(record.execution_id, None)
                self._tasks.pop(record.execution_id, None)
                self._cancels.pop(record.execution_id, None)
                await self.events.emit(Event(
                    type=EventType.EXECUTION_FAILED,
                    source=defn.id,
                    data={"agent_id": defn.id, "execution_id": record.execution_id,
                          "status": "failed", "error": record.error},
                ))
                return

        try:
            for i, step_def in enumerate(defn.steps):
                if cancel.is_set():
                    record.status = ExecutionStatus.KILLED
                    break

                record.current_step = i
                executor = self._executors.get(step_def.type)
                if executor is None:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"No executor for step type: {step_def.type.value}"
                    break

                await self.events.emit(Event(
                    type=EventType.STEP_STARTED,
                    source=defn.id,
                    data={"agent_id": defn.id, "execution_id": record.execution_id,
                          "step_index": i, "step_name": step_def.name,
                          "step_type": step_def.type.value},
                ))

                agent_work_dir = self._config.agents_dir / defn.id
                agent_work_dir.mkdir(parents=True, exist_ok=True)
                ctx = StepContext(
                    agent_id=defn.id,
                    execution_id=record.execution_id,
                    cancel_event=cancel,
                    previous_outputs=previous,
                    inbox_messages=work_messages,
                    work_dir=agent_work_dir,
                    env={},
                    inbox_manager=self.inbox,
                    output_store=self.output_store,
                    event_bus=self.events,
                    runtime=self,
                    connectors=self.connectors if defn.connector_ids else None,
                    connector_ids=defn.connector_ids,
                )

                # Budget check before cognitive steps
                if step_def.type == StepType.COGNITIVE and defn.budgets:
                    provider_name = step_def.config.get("provider", "")
                    if isinstance(provider_name, list):
                        provider_name = provider_name[0] if provider_name else ""
                    limit = defn.budgets.get(provider_name, 0)
                    if limit > 0:
                        current = record.token_usage.get(provider_name)
                        used = current.total if current else 0
                        if used >= limit:
                            record.status = ExecutionStatus.FAILED
                            record.error = (
                                f"Token budget exceeded for '{provider_name}': "
                                f"used {used}, limit {limit}"
                            )
                            break

                step_result = await executor.execute(step_def, i, ctx)
                record.step_results.append(step_result)
                previous.append(step_result.output)

                # Extract token usage from cognitive steps
                if (step_def.type == StepType.COGNITIVE
                        and step_result.status == ExecutionStatus.COMPLETED
                        and isinstance(step_result.output, dict)):
                    provider_name = step_def.config.get("provider", "")
                    if isinstance(provider_name, list):
                        provider_name = provider_name[0] if provider_name else ""
                    _accumulate_usage(record, provider_name, step_result.output)

                # Accumulate child token usage from collect steps
                if (step_def.type == StepType.COLLECT
                        and step_result.status == ExecutionStatus.COMPLETED
                        and isinstance(step_result.output, dict)):
                    _accumulate_child_usage(record, step_result.output, self.execution_log)

                # Update execution log incrementally so in-progress
                # state is queryable (for UI) and survives crashes (disk).
                self.execution_log.record(record)
                self.execution_log.persist_running(record)

                # Emit step completion event
                etype = {
                    ExecutionStatus.COMPLETED: EventType.STEP_COMPLETED,
                    ExecutionStatus.FAILED: EventType.STEP_FAILED,
                    ExecutionStatus.KILLED: EventType.STEP_KILLED,
                }.get(step_result.status, EventType.STEP_FAILED)

                event_data: dict[str, Any] = {
                    "agent_id": defn.id,
                    "execution_id": record.execution_id,
                    "step_index": i,
                    "step_name": step_result.step_name,
                    "step_type": step_result.step_type.value,
                    "status": step_result.status.value,
                }
                if step_result.error:
                    event_data["error"] = step_result.error
                if step_result.output:
                    event_data["output_preview"] = _preview(step_result.output)
                if step_result.pid:
                    event_data["pid"] = step_result.pid

                await self.events.emit(Event(type=etype, source=defn.id, data=event_data))

                # Abort pipeline on failure
                if step_result.status in (ExecutionStatus.FAILED, ExecutionStatus.KILLED):
                    record.status = step_result.status
                    record.error = step_result.error
                    break
            else:
                # All steps succeeded — output is the last meaningful step's
                # result (skip message steps which only produce routing metadata).
                record.status = ExecutionStatus.COMPLETED
                record.output = None
                for sr in reversed(record.step_results):
                    if sr.step_type != StepType.MESSAGE and sr.output is not None:
                        record.output = sr.output
                        break

        except asyncio.CancelledError:
            record.status = ExecutionStatus.KILLED
            record.error = "Force-cancelled by kill switch"

        except Exception as exc:
            record.status = ExecutionStatus.FAILED
            record.error = str(exc)
            log.exception("Pipeline error for agent %s", defn.id)

        finally:
            record.completed_at = datetime.now(timezone.utc)
            self.execution_log.record(record)
            self.execution_log.persist(record)

            # Update output store on success
            if record.status == ExecutionStatus.COMPLETED:
                self.output_store.write(AgentOutput(
                    agent_id=defn.id,
                    data=record.output,
                    status=record.status,
                    execution_id=record.execution_id,
                ))

            # Bookkeeping
            self._running_count[defn.id] = max(0, self._running_count.get(defn.id, 1) - 1)
            self._executions.pop(record.execution_id, None)
            self._tasks.pop(record.execution_id, None)
            self._cancels.pop(record.execution_id, None)
            self._interrupt_hooks.pop(record.execution_id, None)

            if self._running_count.get(defn.id, 0) == 0:
                if record.status == ExecutionStatus.FAILED:
                    self._status[defn.id] = AgentStatus.ERROR
                elif self._status.get(defn.id) == AgentStatus.RUNNING:
                    self._status[defn.id] = AgentStatus.ACTIVE

            # Final execution event
            etype = {
                ExecutionStatus.COMPLETED: EventType.EXECUTION_COMPLETED,
                ExecutionStatus.FAILED: EventType.EXECUTION_FAILED,
                ExecutionStatus.KILLED: EventType.EXECUTION_KILLED,
            }.get(record.status, EventType.EXECUTION_FAILED)

            await self.events.emit(Event(
                type=etype,
                source=defn.id,
                data={
                    "agent_id": defn.id,
                    "execution_id": record.execution_id,
                    "status": record.status.value,
                    "steps_completed": len(record.step_results),
                    "output_preview": _preview(record.output) if record.output else None,
                    "error": record.error,
                },
            ))

    # ==================================================================
    # Kill switches
    # ==================================================================

    async def kill_execution(self, execution_id: str) -> bool:
        cancel = self._cancels.get(execution_id)
        if not cancel:
            return False

        # If there's an interrupt hook (e.g. bridge orchestration), call it
        # first so the running provider can wind down gracefully before we
        # set the cancel event.
        hook = self._interrupt_hooks.get(execution_id)
        if hook:
            try:
                await hook()
            except Exception:
                log.debug("Interrupt hook error for %s", execution_id, exc_info=True)

        cancel.set()
        task = self._tasks.get(execution_id)
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                # Cancel event didn't terminate in time — force-cancel the task.
                # The pipeline's finally block still runs, cleaning up bookkeeping.
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
        return True

    def register_interrupt_hook(self, execution_id: str, hook: Callable) -> None:
        """Register an async callable invoked before kill_execution sets cancel."""
        self._interrupt_hooks[execution_id] = hook

    def unregister_interrupt_hook(self, execution_id: str) -> None:
        """Remove an interrupt hook (called automatically on pipeline cleanup)."""
        self._interrupt_hooks.pop(execution_id, None)

    # ------------------------------------------------------------------
    # Delegate message injection & interrupt
    # ------------------------------------------------------------------

    async def send_delegate_message(self, agent_id: str, content: str) -> bool:
        """Inject a user message into a running delegate's session.

        Returns True if the message was delivered, False if the delegate
        isn't running or the agent_id is unknown.
        """
        provider = self._delegate_providers.get(agent_id)
        if provider is None:
            return False
        await provider.send_user_message(content)
        return True

    async def interrupt_delegate(self, agent_id: str) -> bool:
        """Interrupt a running delegate's session.

        Calls bridge.interrupt() which tells the Claude SDK to wind down
        gracefully.  Returns True if the interrupt was sent.
        """
        provider = self._delegate_providers.get(agent_id)
        if provider is None:
            return False
        await provider.interrupt()
        return True

    async def interrupt_orchestrator(self) -> bool:
        """Interrupt the running orchestrator session.

        Finds the orchestrator's bridge provider and calls interrupt().
        Returns True if the interrupt was sent.
        """
        from .providers.bridge import BridgeProvider
        from .steps.cognitive import CognitiveStepExecutor
        cognitive = self._executors.get(StepType.COGNITIVE)
        if not isinstance(cognitive, CognitiveStepExecutor):
            return False
        provider = cognitive._providers.get("claude_max")
        if not isinstance(provider, BridgeProvider):
            return False
        await provider.interrupt()
        return True

    # ------------------------------------------------------------------
    # Context inspection
    # ------------------------------------------------------------------

    def _get_bridge_provider(self, agent_id: str | None = None) -> Any:
        """Resolve a BridgeProvider for the given agent.

        If agent_id is None or "orchestrator", returns the orchestrator's
        bridge provider.  Otherwise looks up the delegate's provider.
        Returns None if not found or not a BridgeProvider.
        """
        from .providers.bridge import BridgeProvider
        if agent_id is None or agent_id == "orchestrator":
            from .steps.cognitive import CognitiveStepExecutor
            cognitive = self._executors.get(StepType.COGNITIVE)
            if not isinstance(cognitive, CognitiveStepExecutor):
                return None
            provider = cognitive._providers.get("claude_max")
            return provider if isinstance(provider, BridgeProvider) else None
        else:
            provider = self._delegate_providers.get(agent_id)
            from .providers.bridge import BridgeProvider as BP
            return provider if isinstance(provider, BP) else None

    def get_session_stats(self, agent_id: str | None = None) -> dict[str, Any]:
        """Return session stats for the orchestrator or a delegate.

        Lightweight — returns cached stats without asking the bridge.
        """
        provider = self._get_bridge_provider(agent_id)
        if provider is None:
            target = agent_id or "orchestrator"
            return {"error": f"No active bridge session for '{target}'"}
        return provider.session_stats

    async def get_session_context(self, agent_id: str | None = None) -> dict[str, Any]:
        """Fetch conversation history from the bridge for the orchestrator or a delegate.

        This is the source of truth — the actual messages the model is
        working with, as stored by the Claude Agent SDK.
        """
        provider = self._get_bridge_provider(agent_id)
        if provider is None:
            target = agent_id or "orchestrator"
            return {"error": f"No active bridge session for '{target}'"}
        return await provider.get_session_context()

    async def kill_agent(self, agent_id: str) -> int:
        """Kill all running executions for an agent.  Returns count killed."""
        eids = [eid for eid, rec in self._executions.items() if rec.agent_id == agent_id]
        for eid in eids:
            await self.kill_execution(eid)
        return len(eids)

    async def kill_all(self) -> int:
        """Nuclear option.  Returns count killed."""
        # Fire all interrupt hooks (including delegate sub-agents)
        for hook in list(self._interrupt_hooks.values()):
            try:
                await hook()
            except Exception:
                pass
        for cancel in self._cancels.values():
            cancel.set()
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    # ==================================================================
    # Scheduler loop
    # ==================================================================

    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                for agent_id, interval in list(self._schedule_table.items()):
                    if self._status.get(agent_id) != AgentStatus.ACTIVE:
                        continue
                    last = self._last_scheduled.get(agent_id)
                    if last is None or (now - last).total_seconds() >= interval:
                        self._last_scheduled[agent_id] = now
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

                # Planning review — post a work message to the orchestrator
                # when the interval has elapsed and the user has completed onboarding.
                if (self._planning_interval > 0
                        and not self.user_profile.needs_onboarding()):
                    should_plan = (
                        self._last_planning_review is None
                        or (now - self._last_planning_review).total_seconds() >= self._planning_interval
                    )
                    if should_plan:
                        self._last_planning_review = now
                        await self._post_planning_review()

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Scheduler error")
                await asyncio.sleep(1)

    async def _post_planning_review(self) -> None:
        """Build planning context and post a work message to the orchestrator."""
        from .orchestrator import ORCHESTRATOR_ID
        from .planning_prompt import build_planning_context

        profile = self.user_profile.get_profile()

        # Build active agents summary
        active_agents: list[dict[str, Any]] = []
        for aid, defn in self._agents.items():
            if aid == ORCHESTRATOR_ID:
                continue
            status = self._status.get(aid)
            if status in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
                active_agents.append({
                    "id": aid,
                    "name": defn.name,
                    "status": status.value if status else "unknown",
                })

        # Pending tasks
        pending = [
            {"id": t.id, "title": t.title, "goal_id": t.goal_id, "status": t.status.value}
            for t in self.planning_tasks
            if t.status in (TaskStatus.PROPOSED, TaskStatus.APPROVED, TaskStatus.ACTIVE)
        ]

        # Calendar availability
        calendar_available = self.credential_store.exists("google_calendar")

        context = build_planning_context(
            goals=profile.goals,
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

    # ==================================================================
    # Inbox watcher loop
    # ==================================================================

    async def _inbox_watcher_loop(self) -> None:
        while self._running:
            try:
                for agent_id in list(self._agents.keys()):
                    status = self._status.get(agent_id)
                    if status not in (AgentStatus.ACTIVE, AgentStatus.REGISTERED):
                        continue
                    should_trigger = (
                        self.inbox.has_trigger(agent_id)
                        or self.inbox.has_wake_priority(agent_id)
                    )
                    if should_trigger:
                        await self.trigger_run(agent_id, source="inbox")
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Inbox watcher error")
                await asyncio.sleep(1)

    # ==================================================================
    # Snapshot for dashboard
    # ==================================================================

    def snapshot(self) -> dict:
        agents = {}
        for aid, defn in self._agents.items():
            last_output = self.output_store.read(aid)
            agent_info: dict = {
                "name": defn.name,
                "description": defn.description,
                "model": defn.model,
                "status": self._status[aid].value,
                "schedule": defn.schedule,
                "concurrency": defn.concurrency,
                "running": self._running_count.get(aid, 0),
                "steps": len(defn.steps),
                "step_types": sorted({s.type.value for s in defn.steps}),
                "inbox": self.inbox.count(aid),
                "last_output": _preview(last_output.data) if last_output else None,
                "path": str(self._config.agents_dir / aid),
            }
            if defn.connector_ids:
                agent_info["connector_ids"] = defn.connector_ids
            agents[aid] = agent_info
        executions = {}
        for eid, rec in self._executions.items():
            defn = self._agents.get(rec.agent_id)
            step_label = ""
            if defn and rec.current_step < len(defn.steps):
                s = defn.steps[rec.current_step]
                step_label = f"[{rec.current_step}] {s.name} ({s.type.value})"
            executions[eid] = {
                "agent_id": rec.agent_id,
                "step": step_label,
                "trigger": rec.trigger_source,
                "started_at": rec.started_at.isoformat(),
            }
        # Orchestrator info
        from .orchestrator import ORCHESTRATOR_ID
        orch_defn = self._agents.get(ORCHESTRATOR_ID)
        orch_info = None
        if orch_defn:
            step_cfg = orch_defn.steps[0].config if orch_defn.steps else {}
            raw_provider = step_cfg.get("provider", "")
            # Provider may be a string or a fallback chain list
            if isinstance(raw_provider, list):
                primary_provider = raw_provider[0] if raw_provider else ""
                fallback_providers = raw_provider[1:] if len(raw_provider) > 1 else []
            else:
                primary_provider = raw_provider
                fallback_providers = []
            orch_info = {
                "provider": primary_provider,
                "model": step_cfg.get("model", ""),
                "available_models": self._get_available_models(primary_provider),
                "fallback_providers": fallback_providers,
            }

        # Connectors
        from .connectors import get_bundled_specs
        from .oauth import requires_oauth
        bundled_ids = set(get_bundled_specs().keys())
        running_ids = set(self.connectors.list_running())
        connectors = {}
        for cid in self.connectors.list_available():
            spec = self.connectors.get_spec(cid)
            c_info: dict[str, Any] = {
                "name": spec.name if spec else cid,
                "description": spec.description if spec else "",
                "mode": spec.mode if spec else "",
                "running": cid in running_ids,
                "bundled": cid in bundled_ids,
            }
            # Auth status for connectors that need OAuth
            if requires_oauth(cid):
                c_info["requires_oauth"] = True
                c_info["authenticated"] = self.credential_store.exists(cid)
            if cid in running_ids:
                session = self.connectors._sessions.get(cid)
                c_info["tool_count"] = len(session.tools) if session else 0
            # Which agents use this connector
            using_agents = [
                aid for aid, d in self._agents.items()
                if d.connector_ids and cid in d.connector_ids
            ]
            if using_agents:
                c_info["used_by"] = using_agents
            connectors[cid] = c_info

        # Providers summary (lightweight — no Ollama probe in snapshot)
        cognitive = self._executors.get(StepType.COGNITIVE)
        registered_providers: set[str] = set()
        if isinstance(cognitive, CognitiveStepExecutor):
            registered_providers = set(cognitive._providers.keys())
        providers_summary = {}
        for pid, info in self._KNOWN_PROVIDERS.items():
            is_active = pid in registered_providers
            if is_active:
                configured = True
            elif info["auth_type"] == "api_key":
                configured = bool(self._resolve_api_key(pid))
            else:
                configured = False
            providers_summary[pid] = {
                "name": info["name"],
                "auth_type": info["auth_type"],
                "configured": configured,
                "active": is_active,
                "orchestrator_capable": info.get("orchestrator_capable", False),
            }
        # Custom providers
        for pid in sorted(self._custom_providers):
            is_active = pid in registered_providers
            providers_summary[pid] = {
                "name": pid,
                "auth_type": "api_key",
                "configured": True,
                "active": is_active,
                "orchestrator_capable": True,
                "custom": True,
            }

        # User profile, budget, planning
        pending_task_count = sum(
            1 for t in self.planning_tasks
            if t.status in (TaskStatus.PROPOSED, TaskStatus.APPROVED, TaskStatus.ACTIVE)
        )

        return {
            "system": {
                "os": platform.system(),
                "version": platform.version(),
                "arch": platform.machine(),
                "python": platform.python_version(),
                "shell": "powershell" if platform.system() == "Windows" else "bash",
            },
            "orchestrator": orch_info,
            "providers": providers_summary,
            "agents": agents,
            "executions": executions,
            "connectors": connectors,
            "user": self.user_profile.to_summary_dict(),
            "budget": self.credit_budget.to_summary_dict(),
            "planning": {
                "last_review": self._last_planning_review.isoformat() if self._last_planning_review else None,
                "next_review": (
                    (self._last_planning_review + timedelta(seconds=self._planning_interval)).isoformat()
                    if self._last_planning_review and self._planning_interval > 0
                    else None
                ),
                "interval_hours": self._planning_interval / 3600 if self._planning_interval > 0 else 0,
                "pending_tasks": pending_task_count,
            },
            "delegates": self.delegate_registry.get_tree(),
            "voice": self._voice_snapshot(),
            "autonet": self.autonet.get_status(),
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

    # Known provider defaults (endpoints, models)
    _PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_model": "gemini-2.0-flash",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "",
        },
    }

    def _resolve_api_key(self, provider_name: str) -> str:
        """Resolve API key for a provider.

        Priority: credential store > config.yaml > empty string.
        """
        # Check credential store first
        creds = self.credential_store.load(f"provider_{provider_name}")
        if creds.get("api_key"):
            return creds["api_key"]
        # Fall back to config.yaml
        pconfig = self._config.providers.get(provider_name)
        if pconfig and pconfig.api_key:
            return pconfig.api_key
        return ""

    def _setup_providers(self, cognitive: CognitiveStepExecutor) -> None:
        """Initialise LLM providers from config and register them.

        API keys are resolved from the credential store first, then config.yaml.
        This handles providers that require explicit configuration (API keys).
        Bridge and Ollama are auto-detected in ``_auto_detect_providers()``.
        """
        for name, pconfig in self._config.providers.items():
            try:
                if name in ("claude_max", "ollama"):
                    # Auto-detected in _auto_detect_providers() — skip here
                    # unless explicitly configured with overrides.
                    continue
                elif name == "anthropic":
                    api_key = self._resolve_api_key("anthropic")
                    if not api_key:
                        log.warning("Provider 'anthropic' has no API key — skipping")
                        continue
                    provider = AnthropicProvider(
                        api_key=api_key,
                        default_model=pconfig.default_model,
                        base_url=pconfig.base_url,
                    )
                elif name in ("gemini", "openai") or pconfig.base_url:
                    # OpenAI-compatible providers
                    defaults = self._PROVIDER_DEFAULTS.get(name, {})
                    base_url = pconfig.base_url or defaults.get("base_url", "")
                    if not base_url:
                        log.warning("Provider '%s' has no base_url — skipping", name)
                        continue
                    api_key = self._resolve_api_key(name)
                    if not api_key:
                        log.warning("Provider '%s' has no API key — skipping", name)
                        continue
                    provider = OpenAICompatibleProvider(
                        name=name,
                        base_url=base_url,
                        api_key=api_key,
                        default_model=pconfig.default_model or defaults.get("default_model", ""),
                    )
                else:
                    # Custom provider — treat as OpenAI-compatible if it has
                    # base_url or type: openai_compat in extra config.
                    ptype = pconfig.extra.get("type", "")
                    base_url = pconfig.base_url
                    if ptype == "openai_compat" or base_url:
                        if not base_url:
                            log.warning("Custom provider '%s' has no base_url — skipping", name)
                            continue
                        api_key = self._resolve_api_key(name)
                        provider = OpenAICompatibleProvider(
                            name=name,
                            base_url=base_url,
                            api_key=api_key,
                            default_model=pconfig.default_model,
                        )
                        self._custom_providers.add(name)
                    else:
                        log.warning("Unknown provider '%s' — skipping", name)
                        continue

                cognitive.register_provider(provider)
                log.info("Registered provider: %s (model=%s)", name, pconfig.default_model or "(default)")

            except ProviderError as exc:
                log.warning("Failed to initialise provider '%s': %s", name, exc)

        # Also register API-key providers that are in the credential store
        # but not in config.yaml (configured via the UI).
        for pid in ("anthropic", "gemini", "openai"):
            if pid in self._config.providers:
                continue  # already handled above
            api_key = self._resolve_api_key(pid)
            if not api_key:
                continue
            try:
                self._hot_register_provider(pid, api_key)
            except Exception as exc:
                log.warning("Failed to register credential-store provider '%s': %s", pid, exc)

    async def _auto_detect_providers(self) -> None:
        """Auto-detect local providers at startup.

        Zero-config providers (Ollama, Claude Max bridge) are probed here.
        Called from ``start()`` after the event loop is running.
        """
        cognitive = self._executors.get(StepType.COGNITIVE)
        if not isinstance(cognitive, CognitiveStepExecutor):
            return

        # Claude Max bridge — probe if not already registered
        if "claude_max" not in cognitive._providers:
            try:
                ready = await self._probe_bridge()
                if ready:
                    self._hot_register_provider("claude_max", "")
                    log.info("Auto-detected Claude Max bridge")
                else:
                    log.info("Claude Max bridge not available")
            except Exception as exc:
                log.warning("Claude Max bridge auto-detect failed: %s", exc)

        # Ollama — probe HTTP endpoint (no auth, no install step)
        if "ollama" not in cognitive._providers:
            try:
                models = await self._probe_ollama()
                if models is not None:
                    defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
                    pconfig = self._config.providers.get("ollama")
                    provider = OllamaProvider(
                        base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                        default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                    )
                    cognitive.register_provider(provider)
                    log.info("Auto-detected Ollama (%d model(s))", len(models))
                else:
                    log.info("Ollama not available at localhost:11434")
            except Exception as exc:
                log.warning("Ollama auto-detect failed: %s", exc)

    def _get_available_models(self, provider_name: str, *, require_active: bool = False) -> list[dict[str, str]]:
        """Return the list of selectable models for a given provider.

        Each entry has 'id' (the model ID passed to the SDK) and 'name'
        (human-readable label for the UI).

        Args:
            require_active: If True, only return models when the provider is
                actually registered with the cognitive executor.
        """
        if require_active:
            cognitive = self._executors.get(StepType.COGNITIVE)
            if isinstance(cognitive, CognitiveStepExecutor):
                if provider_name not in cognitive._providers:
                    return []
            else:
                return []

        _PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
            "claude_max": [
                {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
                {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
                {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"},
            ],
        }
        return _PROVIDER_MODELS.get(provider_name, [])

    # ==================================================================
    # Provider management (for Config UI)
    # ==================================================================

    # Registry of all known provider slots the UI can configure.
    _KNOWN_PROVIDERS: dict[str, dict[str, Any]] = {
        "claude_max": {
            "name": "Claude Max",
            "description": "Claude via Claude Code bridge (requires Claude Max subscription)",
            "auth_type": "bridge",
            "orchestrator_capable": True,
        },
        "anthropic": {
            "name": "Anthropic API",
            "description": "Claude models via Anthropic API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Gemini models via Google AI API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT models via OpenAI API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Local models via Ollama — no API key needed",
            "auth_type": "local",
            "orchestrator_capable": False,
        },
    }

    async def provider_list(self) -> list[dict[str, Any]]:
        """Return status of all known providers for the Config UI.

        Each entry includes: id, name, description, auth_type, configured,
        active (registered with cognitive executor), models, and usage.

        ``configured`` means the provider has been set up (has credentials or
        has been probed successfully).  ``active`` means it's registered with
        the cognitive executor and ready for LLM calls.
        """
        cognitive = self._executors.get(StepType.COGNITIVE)
        registered: set[str] = set()
        if isinstance(cognitive, CognitiveStepExecutor):
            registered = set(cognitive._providers.keys())

        usage_by_provider = self._aggregate_provider_usage()

        providers = []
        for pid, info in self._KNOWN_PROVIDERS.items():
            is_active = pid in registered

            # ``configured`` = provider has what it needs to work.
            # For active providers this is always True.
            # For inactive ones, check if credentials exist.
            if is_active:
                configured = True
            elif info["auth_type"] == "api_key":
                configured = bool(self._resolve_api_key(pid))
            else:
                # bridge and local: configured only if active
                configured = False

            # Hint for the UI: what action is needed to activate this provider.
            if is_active:
                setup_hint = None
            elif info["auth_type"] == "bridge":
                setup_hint = "connect"  # show a "Connect" button
            elif info["auth_type"] == "api_key":
                setup_hint = "api_key"  # show API key input
            else:
                setup_hint = None       # local: auto-detected

            entry: dict[str, Any] = {
                "id": pid,
                "name": info["name"],
                "description": info["description"],
                "auth_type": info["auth_type"],
                "configured": configured,
                "active": is_active,
                "orchestrator_capable": info.get("orchestrator_capable", False),
                "models": self._get_available_models(pid),
                "setup_hint": setup_hint,
            }

            # Usage stats
            usage = usage_by_provider.get(pid)
            if usage:
                entry["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "total_tokens": usage.total,
                }

            # Ollama: probe for available models
            if pid == "ollama":
                entry["local_models"] = await self._probe_ollama()

            providers.append(entry)

        # Custom (user-defined) providers
        for pid in sorted(self._custom_providers):
            is_active = pid in registered
            pconfig = self._config.providers.get(pid)
            base_url = pconfig.base_url if pconfig else ""

            entry = {
                "id": pid,
                "name": pid,  # user chose the ID as the display name
                "description": f"Custom OpenAI-compatible provider ({base_url})",
                "auth_type": "api_key",
                "configured": True,
                "active": is_active,
                "orchestrator_capable": True,
                "custom": True,
                "base_url": base_url,
                "default_model": pconfig.default_model if pconfig else "",
                "models": [],
                "setup_hint": None,
            }

            usage = usage_by_provider.get(pid)
            if usage:
                entry["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "total_tokens": usage.total,
                }

            providers.append(entry)

        return providers

    async def configure_provider(self, provider_id: str, api_key: str = "") -> dict[str, Any]:
        """Configure a provider with an API key.  Validates with a test call.

        For 'ollama', no api_key is needed — just checks connectivity.
        For 'claude_max', no api_key — checks bridge availability.

        Returns {"status": "ok"} on success, raises ValueError on failure.
        """
        info = self._KNOWN_PROVIDERS.get(provider_id)
        if not info:
            raise ValueError(f"Unknown provider: {provider_id}")

        if info["auth_type"] == "api_key":
            if not api_key:
                raise ValueError(f"Provider '{provider_id}' requires an API key")

            # Validate with a lightweight test call
            await self._validate_api_key(provider_id, api_key)

            # Store in credential store
            self.credential_store.save(f"provider_{provider_id}", {"api_key": api_key})

            # Hot-register the provider so it's immediately available
            self._hot_register_provider(provider_id, api_key)

            return {"status": "ok", "provider": provider_id}

        elif info["auth_type"] == "local":
            # Ollama — probe connectivity and hot-register
            models = await self._probe_ollama()
            if models is None:
                raise ValueError("Cannot connect to Ollama at localhost:11434. Is it running?")
            self._hot_register_provider(provider_id, "")
            return {"status": "ok", "provider": provider_id, "models": models}

        elif info["auth_type"] == "bridge":
            # Claude Max — full three-state flow:
            #   1. Check SDK/CLI is available (installed)
            #   2. Check authentication status
            #   3. Trigger login if needed, then probe bridge

            auth = await self._check_claude_auth()

            if not auth["installed"]:
                raise ValueError(
                    "Claude Code SDK not found. Install Claude Code "
                    "(https://claude.ai/download) and run 'bun install' "
                    "in the bridge/ directory."
                )

            if not auth["logged_in"]:
                # Trigger browser-based OAuth login
                log.info("Claude Code not authenticated — launching browser login")
                login_ok = await self._trigger_claude_login()
                if not login_ok:
                    raise ValueError(
                        "Claude Code authentication failed or timed out. "
                        "Try running 'claude auth login' manually in a terminal."
                    )
                # Re-check auth after login
                auth = await self._check_claude_auth()
                if not auth["logged_in"]:
                    raise ValueError(
                        "Authentication completed but status check still shows "
                        "not logged in. Try again or check 'claude auth status'."
                    )

            # Authenticated — probe the bridge to verify it's fully functional
            ready = await self._probe_bridge()
            if not ready:
                raise ValueError(
                    "Claude Code is authenticated but the bridge failed to respond. "
                    "Check that 'bun' is installed and bridge dependencies are set up."
                )

            # Hot-register the bridge provider
            self._hot_register_provider(provider_id, "")
            return {
                "status": "ok",
                "provider": provider_id,
                "email": auth.get("email", ""),
                "subscription": auth.get("subscription", ""),
            }

        return {"status": "ok", "provider": provider_id}

    async def remove_provider(self, provider_id: str) -> dict[str, str]:
        """Remove stored credentials for a provider."""
        self.credential_store.delete(f"provider_{provider_id}")
        # Unregister from cognitive executor
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            cognitive._providers.pop(provider_id, None)
        log.info("Provider '%s' removed", provider_id)
        return {"status": "removed", "provider": provider_id}

    async def add_custom_provider(
        self,
        provider_id: str,
        name: str,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
    ) -> dict[str, Any]:
        """Add a user-defined OpenAI-compatible provider.

        Validates the endpoint, persists to config.yaml, and hot-registers.

        Returns {"status": "ok", "provider": provider_id} on success.
        Raises ValueError on failure.
        """
        if not provider_id:
            raise ValueError("Provider ID is required")
        if provider_id in self._KNOWN_PROVIDERS:
            raise ValueError(f"'{provider_id}' is a built-in provider — use configure_provider() instead")
        if not base_url:
            raise ValueError("Base URL is required for custom providers")

        # Validate the endpoint with a lightweight test call
        import httpx
        try:
            test_url = base_url.rstrip("/")
            if not test_url.endswith("/models"):
                models_url = test_url.rstrip("/")
                if models_url.endswith("/chat/completions"):
                    models_url = models_url.rsplit("/chat/completions", 1)[0]
                models_url += "/models"
            else:
                models_url = test_url

            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(models_url, headers=headers)
                # Accept 200 (models list) or 404 (some providers don't serve /models)
                # or 401 (key required but we might not have one yet)
                if resp.status_code in (401, 403) and api_key:
                    raise ValueError(f"API key rejected by {base_url} (HTTP {resp.status_code})")
        except httpx.ConnectError:
            raise ValueError(f"Cannot connect to {base_url} — check the URL")
        except httpx.TimeoutException:
            raise ValueError(f"Timeout connecting to {base_url}")
        except ValueError:
            raise  # re-raise our own ValueErrors
        except Exception:
            pass  # don't block on unexpected probe failures

        # Persist to config.yaml
        spec: dict[str, Any] = {
            "type": "openai_compat",
            "base_url": base_url,
            "default_model": default_model,
        }
        save_provider_to_config(provider_id, spec)

        # Store API key in credential store if provided
        if api_key:
            self.credential_store.save(f"provider_{provider_id}", {"api_key": api_key})

        # Hot-register
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            provider = OpenAICompatibleProvider(
                name=provider_id,
                base_url=base_url,
                api_key=api_key,
                default_model=default_model,
            )
            cognitive.register_provider(provider)

        self._custom_providers.add(provider_id)
        log.info("Custom provider added: %s (%s)", provider_id, base_url)
        return {"status": "ok", "provider": provider_id, "name": name}

    async def remove_custom_provider(self, provider_id: str) -> dict[str, str]:
        """Remove a user-defined custom provider.

        Unregisters from cognitive executor, removes from config.yaml and
        credential store.
        """
        if provider_id in self._KNOWN_PROVIDERS:
            raise ValueError(f"'{provider_id}' is a built-in provider — use remove_provider() instead")

        # Unregister from cognitive executor
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            cognitive._providers.pop(provider_id, None)

        # Remove from config.yaml
        remove_provider_from_config(provider_id)

        # Remove credentials
        self.credential_store.delete(f"provider_{provider_id}")

        self._custom_providers.discard(provider_id)
        log.info("Custom provider removed: %s", provider_id)
        return {"status": "removed", "provider": provider_id}

    async def _validate_api_key(self, provider_id: str, api_key: str) -> None:
        """Make a minimal test call to validate an API key."""
        import httpx
        try:
            if provider_id == "anthropic":
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code == 403:
                        raise ValueError("API key does not have access")
                    # 200 or 429 (rate limited) both mean the key is valid
                    if resp.status_code not in (200, 429, 529):
                        raise ValueError(f"Unexpected response: {resp.status_code}")

            elif provider_id == "gemini":
                defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
                base_url = defaults.get("base_url", "")
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code not in (200, 429):
                        raise ValueError(f"Unexpected response: {resp.status_code}")

            elif provider_id == "openai":
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code not in (200, 429):
                        raise ValueError(f"Unexpected response: {resp.status_code}")

        except httpx.ConnectError:
            raise ValueError(f"Cannot reach {provider_id} API — check your network")
        except httpx.TimeoutException:
            raise ValueError(f"Timeout connecting to {provider_id} API")

    def _hot_register_provider(self, provider_id: str, api_key: str) -> None:
        """Register a newly-configured provider with the cognitive executor."""
        cognitive = self._executors.get(StepType.COGNITIVE)
        if not isinstance(cognitive, CognitiveStepExecutor):
            return

        try:
            if provider_id == "claude_max":
                pconfig = self._config.providers.get("claude_max")
                bridge_script = pconfig.extra.get("bridge_script", "") if pconfig else ""
                provider = BridgeProvider(
                    model=pconfig.default_model if pconfig else "sonnet",
                    bridge_script=bridge_script if bridge_script else None,
                )
            elif provider_id == "anthropic":
                pconfig = self._config.providers.get("anthropic")
                provider = AnthropicProvider(
                    api_key=api_key,
                    default_model=pconfig.default_model if pconfig else "",
                    base_url=pconfig.base_url if pconfig else "",
                )
            elif provider_id in ("gemini", "openai"):
                defaults = self._PROVIDER_DEFAULTS.get(provider_id, {})
                pconfig = self._config.providers.get(provider_id)
                provider = OpenAICompatibleProvider(
                    name=provider_id,
                    base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                    api_key=api_key,
                    default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                )
            elif provider_id == "ollama":
                defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
                pconfig = self._config.providers.get("ollama")
                provider = OllamaProvider(
                    base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                    default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                )
            else:
                return

            cognitive.register_provider(provider)
            log.info("Hot-registered provider: %s", provider_id)

        except ProviderError as exc:
            log.warning("Failed to hot-register provider '%s': %s", provider_id, exc)

    async def _probe_ollama(self) -> list[dict[str, str]] | None:
        """Probe Ollama for available models.  Returns list of models or None if unreachable."""
        import httpx
        pconfig = self._config.providers.get("ollama")
        defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
        base_url = (pconfig.base_url if pconfig else "") or defaults.get("base_url", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                if resp.status_code != 200:
                    return None
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    size = m.get("size", 0)
                    size_gb = f"{size / 1e9:.1f}GB" if size else ""
                    models.append({"id": name, "name": name, "size": size_gb})
                return models
        except Exception:
            return None

    async def _probe_bridge(self) -> bool:
        """Probe whether the Claude Max bridge is functional.

        Spawns the bridge subprocess, sends a ping, and checks for a pong.
        Returns True if the bridge is ready (bun installed, SDK resolved),
        False otherwise.
        """
        try:
            provider = BridgeProvider(model="sonnet")
            resp = await asyncio.wait_for(provider._send_request({"type": "ping"}), timeout=15)
            ok = resp.get("ok", False)
            await provider.close()
            return bool(ok)
        except Exception as exc:
            log.debug("Bridge probe failed: %s", exc)
            return False

    def _resolve_sdk_cli(self) -> Path | None:
        """Find the Claude Agent SDK cli.js bundled with the bridge.

        Returns the path to cli.js or None if not found.
        """
        bridge_dir = Path(__file__).resolve().parent.parent / "bridge"
        cli_js = bridge_dir / "node_modules" / "@anthropic-ai" / "claude-agent-sdk" / "cli.js"
        if cli_js.exists():
            return cli_js
        return None

    async def _check_claude_auth(self) -> dict[str, Any]:
        """Check Claude Code authentication status via the SDK CLI.

        Returns a dict with:
          installed: bool  — whether the SDK CLI is available
          logged_in: bool  — whether the user is authenticated
          email: str       — authenticated user's email (if logged in)
          subscription: str — subscription type (if logged in)
        """
        cli_js = self._resolve_sdk_cli()
        if cli_js is None:
            return {"installed": False, "logged_in": False}

        try:
            proc = await asyncio.create_subprocess_exec(
                "bun", str(cli_js), "auth", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            data = json.loads(stdout.decode("utf-8", errors="replace"))
            return {
                "installed": True,
                "logged_in": bool(data.get("loggedIn", False)),
                "email": data.get("email", ""),
                "subscription": data.get("subscriptionType", ""),
            }
        except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError) as exc:
            log.debug("Claude auth status check failed: %s", exc)
            return {"installed": True, "logged_in": False}
        except Exception as exc:
            log.debug("Claude auth status check error: %s", exc)
            return {"installed": False, "logged_in": False}

    async def _trigger_claude_login(self) -> bool:
        """Trigger Claude Code browser-based OAuth login via the SDK CLI.

        Runs ``cli.js auth login`` which opens the default browser for
        authentication.  Blocks until the user completes the flow or the
        process exits.  Returns True on success, False on failure.
        """
        cli_js = self._resolve_sdk_cli()
        if cli_js is None:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "bun", str(cli_js), "auth", "login",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Give the user up to 5 minutes to complete OAuth in the browser
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
            return proc.returncode == 0
        except asyncio.TimeoutError:
            log.warning("Claude login timed out after 5 minutes")
            if proc.returncode is None:
                proc.terminate()
            return False
        except Exception as exc:
            log.debug("Claude login failed: %s", exc)
            return False

    def _aggregate_provider_usage(self) -> dict[str, TokenUsage]:
        """Aggregate token usage across all execution history, per provider.

        Scans in-memory execution logs (hydrated from JSONL at startup).
        """
        totals: dict[str, TokenUsage] = {}
        for agent_id in list(self.execution_log._logs.keys()):
            for rec in self.execution_log._logs[agent_id]:
                for provider, usage in rec.token_usage.items():
                    if provider not in totals:
                        totals[provider] = TokenUsage(provider=provider)
                    totals[provider].input_tokens += usage.input_tokens
                    totals[provider].output_tokens += usage.output_tokens
                    totals[provider].cache_read_tokens += usage.cache_read_tokens
                    totals[provider].cache_creation_tokens += usage.cache_creation_tokens
        return totals

    async def setup_orchestrator(self, **kwargs: Any) -> str:
        """Create and register the orchestrator agent from config.

        If the user hasn't completed onboarding, the orchestrator uses the
        onboarding system prompt.  Otherwise it uses the default fleet-management
        prompt with user context injected.

        Returns the orchestrator agent ID.
        """
        from .orchestrator import build_system_prompt_with_context, create_orchestrator_agent

        # Build system prompt based on onboarding state
        needs_onboarding = self.user_profile.needs_onboarding()
        if needs_onboarding:
            system_prompt = build_system_prompt_with_context(
                data_dir=self._config.data_dir, onboarding=True,
            )
        else:
            profile = self.user_profile.get_profile()
            # Build lightweight profile/goals summaries for prompt injection
            profile_parts: list[str] = []
            if profile.summary:
                profile_parts.append(profile.summary)
            if profile.strengths:
                profile_parts.append("Strengths: " + ", ".join(profile.strengths))
            if profile.weaknesses:
                profile_parts.append("Weaknesses: " + ", ".join(profile.weaknesses))
            profile_summary = "\n".join(profile_parts)

            goals_lines: list[str] = []
            for g in profile.goals:
                status = g.get("status", "active")
                tf = g.get("timeframe", "")
                goals_lines.append(f"- [{status}] {g.get('title', '?')} ({tf}): {g.get('description', '')}")
            goals_summary = "\n".join(goals_lines) if goals_lines else ""

            system_prompt = build_system_prompt_with_context(
                profile_summary=profile_summary,
                goals_summary=goals_summary,
                data_dir=self._config.data_dir,
            )

        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = system_prompt

        defn = create_orchestrator_agent(self._config.orchestrator, **kwargs)
        await self.register_agent(defn)
        await self.activate_agent(defn.id)
        log.info("Orchestrator registered and activated (provider=%s, onboarding=%s)",
                 defn.steps[0].config.get("provider", "?"), needs_onboarding)

        # Inject status briefing if this is a fresh conversation
        if self.conversation.turn_count() == 0:
            self._inject_status_briefing()

        return defn.id

    async def set_orchestrator_model(self, model: str) -> str:
        """Change the orchestrator model and start a new conversation.

        Re-registers the orchestrator agent with the new model and resets
        the conversation history.  Returns the new orchestrator agent ID.
        """
        from .orchestrator import ORCHESTRATOR_ID, create_orchestrator_agent

        # Validate
        orch_defn = self._agents.get(ORCHESTRATOR_ID)
        raw_provider = ""
        if orch_defn and orch_defn.steps:
            raw_provider = orch_defn.steps[0].config.get("provider", "")
        primary_provider = raw_provider[0] if isinstance(raw_provider, list) else raw_provider
        available = self._get_available_models(primary_provider)
        available_ids = [m["id"] for m in available]
        if available_ids and model not in available_ids:
            raise ValueError(
                f"Invalid model {model!r}. Available: {', '.join(available_ids)}"
            )

        # Kill any running orchestrator executions
        await self.kill_agent(ORCHESTRATOR_ID)

        # Update config in memory and persist to disk
        self._config.orchestrator.model = model
        from .config import save_orchestrator_model_to_config
        save_orchestrator_model_to_config(model)

        # Unregister old, re-register with new model
        if ORCHESTRATOR_ID in self._agents:
            await self.unregister_agent(ORCHESTRATOR_ID, _force=True)
        defn = create_orchestrator_agent(self._config.orchestrator)
        await self.register_agent(defn)
        await self.activate_agent(defn.id)

        # Clear session so the new model starts a fresh SDK session
        self._clear_bridge_session()

        # Also update the BridgeProvider's default model so it matches.
        # The step config passes the model explicitly, but this ensures
        # consistency if anything falls back to provider._model.
        from .providers.bridge import BridgeProvider
        from .steps.cognitive import CognitiveStepExecutor
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            provider = cognitive._providers.get("claude_max")
            if isinstance(provider, BridgeProvider):
                provider._model = model

        log.info("Orchestrator model changed to '%s'", model)
        return defn.id

    async def new_conversation(self) -> None:
        """Reset the orchestrator conversation without changing the model."""
        from .orchestrator import ORCHESTRATOR_ID
        await self.kill_agent(ORCHESTRATOR_ID)
        self.conversation.reset()
        self._clear_bridge_session()
        self._inject_status_briefing()
        log.info("Conversation reset")

    def _clear_bridge_session(self) -> None:
        """Clear the bridge provider's session ID so the next orchestration
        starts a fresh SDK session instead of resuming the old one."""
        provider = self._get_bridge_provider()
        if provider is not None:
            provider._session_id = ""
            provider._sdk_num_turns = 0
            provider._cumulative_turns = 0
            provider._total_cost_usd = 0.0
            provider._cumulative_input_tokens = 0
            provider._cumulative_output_tokens = 0
            provider._cumulative_cache_read = 0
            provider._cumulative_cache_creation = 0
            provider._last_input_tokens = 0

    def _inject_status_briefing(self) -> None:
        """Build a deterministic status briefing and inject it as the first
        system turn in the active conversation.  Gives the orchestrator (and
        the user) immediate context about the current state of the fleet."""
        from .orchestrator import ORCHESTRATOR_ID

        agents = []
        for aid, defn in self._agents.items():
            if aid == ORCHESTRATOR_ID:
                continue
            status = self._status.get(aid, AgentStatus.REGISTERED).value
            running = self._running_count.get(aid, 0)
            schedule = defn.schedule or None

            # Last execution info
            last = self.execution_log.get_latest(aid)
            last_info = ""
            if last:
                elapsed = ""
                if last.completed_at:
                    ago = (datetime.now(timezone.utc) - last.completed_at).total_seconds()
                    if ago < 60:
                        elapsed = f"{int(ago)}s ago"
                    elif ago < 3600:
                        elapsed = f"{int(ago / 60)}m ago"
                    else:
                        elapsed = f"{ago / 3600:.1f}h ago"
                last_info = f", last run: {last.status.value}"
                if elapsed:
                    last_info += f" ({elapsed})"
                if last.error:
                    last_info += f" — {last.error[:80]}"

            line = f"- {defn.name} [{aid}] ({status}"
            if schedule:
                line += f", every {schedule}"
            if running:
                line += f", {running} running"
            line += f"{last_info})"
            agents.append(line)

        # Build the briefing
        orch_defn = self._agents.get(ORCHESTRATOR_ID)
        model = ""
        if orch_defn and orch_defn.steps:
            model = orch_defn.steps[0].config.get("model", "")

        lines = []
        if model:
            lines.append(f"Model: {model}")
        if agents:
            lines.append(f"{len(agents)} agent(s) currently registered:")
            lines.extend(agents)
        else:
            lines.append("No agents registered. Clean slate.")

        briefing = "\n".join(lines)
        self.conversation.add_system_turn(briefing)

    def _inject_connector_credentials(self) -> None:
        """Load stored credentials and inject them into connectors as env vars.

        Called at init and after OAuth token exchange.  Connectors that need
        credentials (e.g. google_calendar) receive them as environment
        variables in their subprocess.
        """
        for cid in self.connectors.list_available():
            creds = self.credential_store.load(cid)
            if creds:
                self.connectors.set_runtime_env(cid, {
                    k: str(v) for k, v in creds.items()
                })
                log.debug("Injected credentials for connector '%s'", cid)

    # ==================================================================
    # Planning task persistence
    # ==================================================================

    def _load_planning_tasks(self) -> None:
        """Load planning tasks from JSON."""
        if not self._planning_tasks_path.exists():
            return
        try:
            import json
            raw = json.loads(self._planning_tasks_path.read_text(encoding="utf-8"))
            for d in raw:
                try:
                    self.planning_tasks.append(PlanningTask(
                        id=d["id"],
                        goal_id=d["goal_id"],
                        title=d["title"],
                        description=d.get("description", ""),
                        task_type=TaskType(d.get("task_type", "automation")),
                        status=TaskStatus(d.get("status", "proposed")),
                        agent_id=d.get("agent_id"),
                        calendar_event_id=d.get("calendar_event_id"),
                        created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(timezone.utc),
                        completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
                    ))
                except Exception:
                    log.warning("Skipping corrupt planning task entry")
        except Exception:
            log.warning("Failed to load planning tasks from %s", self._planning_tasks_path, exc_info=True)

    def _save_planning_tasks(self) -> None:
        """Persist planning tasks to JSON."""
        import json
        data = [
            {
                "id": t.id,
                "goal_id": t.goal_id,
                "title": t.title,
                "description": t.description,
                "task_type": t.task_type.value,
                "status": t.status.value,
                "agent_id": t.agent_id,
                "calendar_event_id": t.calendar_event_id,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in self.planning_tasks
        ]
        try:
            self._planning_tasks_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except Exception:
            log.exception("Failed to save planning tasks")

    @staticmethod
    def _parse_interval(schedule: str) -> float:
        m = re.match(r"^(\d+)\s*([smh])$", schedule.strip().lower())
        if not m:
            raise ValueError(f"Invalid schedule: {schedule!r}  (use e.g. '30s', '5m', '1h')")
        val, unit = int(m.group(1)), m.group(2)
        return val * {"s": 1, "m": 60, "h": 3600}[unit]


def _preview(obj: Any, max_len: int = 120) -> str:
    """Truncated string preview of an object for event data."""
    if obj is None:
        return ""
    s = str(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


def _accumulate_usage(record: ExecutionRecord, provider: str, output: dict) -> None:
    """Extract token usage from cognitive step output and add to record."""
    usage = output.get("usage")
    if not usage or not isinstance(usage, dict) or not provider:
        return
    existing = record.token_usage.get(provider)
    if existing is None:
        existing = TokenUsage(provider=provider)
        record.token_usage[provider] = existing
    existing.input_tokens += usage.get("input_tokens", 0)
    existing.output_tokens += usage.get("output_tokens", 0)
    existing.cache_read_tokens += usage.get("cache_read_tokens", 0)
    existing.cache_creation_tokens += usage.get("cache_creation_tokens", 0)


def _accumulate_child_usage(
    record: ExecutionRecord,
    collect_output: dict,
    execution_log: "ExecutionLog",
) -> None:
    """Pull token usage from a child execution (via collect step) into the parent."""
    # The collect step output comes from the output store, which has execution_id.
    child_eid = collect_output.get("execution_id")
    if not child_eid:
        return
    child_agent = collect_output.get("source", "")
    if not child_agent:
        return
    # Find the child execution record
    history = execution_log.get_history(child_agent, limit=20)
    child_rec = None
    for rec in reversed(history):
        if rec.execution_id == child_eid:
            child_rec = rec
            break
    if child_rec is None:
        return
    # Merge child's token usage into parent, per provider
    for provider, child_usage in child_rec.token_usage.items():
        existing = record.token_usage.get(provider)
        if existing is None:
            existing = TokenUsage(provider=provider)
            record.token_usage[provider] = existing
        existing.input_tokens += child_usage.input_tokens
        existing.output_tokens += child_usage.output_tokens
        existing.cache_read_tokens += child_usage.cache_read_tokens
        existing.cache_creation_tokens += child_usage.cache_creation_tokens
