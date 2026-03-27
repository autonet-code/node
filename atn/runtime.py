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
    AgentMode,
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

# ---------------------------------------------------------------------------
# Shell tools for non-bridge providers (Gemini, OpenAI, Ollama)
# BridgeProvider (Claude SDK) has these built-in; generic providers don't.
# ---------------------------------------------------------------------------

_SHELL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": "Execute a shell command and return stdout+stderr. Use for running scripts, git, npm, pip, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line number to start from (1-based)"},
                "limit": {"type": "integer", "description": "Max lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Search for a regex pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in"},
                "glob": {"type": "string", "description": "File glob pattern (e.g. '*.py')"},
            },
            "required": ["pattern"],
        },
    },
]


async def _exec_bash(inp: dict) -> dict:
    """Execute a shell command."""
    import subprocess
    cmd = inp.get("command", "")
    timeout = inp.get("timeout", 120)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(Path.cwd()),
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return {"output": output[:50000], "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


async def _exec_read_file(inp: dict) -> dict:
    """Read a file."""
    try:
        p = Path(inp["path"])
        if not p.exists():
            return {"error": f"File not found: {p}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        offset = max(0, inp.get("offset", 1) - 1)
        limit = inp.get("limit", len(lines))
        selected = lines[offset:offset + limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        return {"content": "".join(numbered)[:100000]}
    except Exception as e:
        return {"error": str(e)}


async def _exec_write_file(inp: dict) -> dict:
    """Write a file."""
    try:
        p = Path(inp["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"], encoding="utf-8")
        return {"status": "ok", "path": str(p), "bytes": len(inp["content"])}
    except Exception as e:
        return {"error": str(e)}


async def _exec_list_dir(inp: dict) -> dict:
    """List directory."""
    try:
        p = Path(inp["path"])
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        entries = []
        for item in sorted(p.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            entries.append({"name": item.name, "type": kind, "size": size})
        return {"entries": entries[:500]}
    except Exception as e:
        return {"error": str(e)}


async def _exec_search_files(inp: dict) -> dict:
    """Search files with grep."""
    import subprocess
    pattern = inp.get("pattern", "")
    path = inp.get("path", str(Path.cwd()))
    glob_pat = inp.get("glob", "")
    cmd = ["rg", "-n", "--max-count", "50", pattern, path]
    if glob_pat:
        cmd.extend(["--glob", glob_pat])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"matches": result.stdout[:50000]}
    except FileNotFoundError:
        # rg not available, fall back to grep
        cmd = ["grep", "-rn", pattern, path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {"matches": result.stdout[:50000]}
        except Exception as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


_SHELL_TOOL_EXECUTORS: dict[str, Any] = {
    "bash": _exec_bash,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "list_directory": _exec_list_dir,
    "search_files": _exec_search_files,
}


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

        # Shutdown event — signaled when stop() is called, allows input loop to exit
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._planning_tasks_path = self._config.data_dir / "planning_tasks.json"
        self._load_planning_tasks()

        # Planning loop interval (seconds).  0 = disabled.
        self._planning_interval: float = 21600.0  # 6 hours
        # Set to "now" so the first review waits for the full interval
        # instead of firing immediately at startup (before bridge is ready).
        self._last_planning_review: datetime | None = datetime.now(timezone.utc)

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
        # Delegate output directory — per-delegate text logs
        self._delegate_output_dir = self._config.data_dir / "delegates"
        self._delegate_output_dir.mkdir(parents=True, exist_ok=True)
        # Events for signalling delegate completion to delegate_collect
        self._delegate_done: dict[str, asyncio.Event] = {}
        # Per-agent conversation stores (agent_id -> ConversationStore)
        self._agent_conversations: dict[str, ConversationStore] = {}

        # --- Phase 2: Cognitive mode execution ---
        # Active bridge providers for cognitive agents (agent_id -> BridgeProvider)
        self._active_providers: dict[str, BridgeProvider] = {}
        # Completion callbacks: child_id -> parent_id
        self._completion_callbacks: dict[str, str] = {}
        # Child ID counters (mirrors DelegateRegistry logic at Runtime level)
        self._child_counters: dict[str, int] = {}

        # Unified tool registry (connectors + pipeline-agent-tools)
        from .tool_registry import ToolRegistry
        self.tool_registry = ToolRegistry(self)

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

        # Step executors — assign _executors BEFORE _setup_providers so that
        # any code path during provider registration can safely access it.
        cognitive = CognitiveStepExecutor()
        self._executors: dict[StepType, StepExecutor] = {
            StepType.SCRIPT: ScriptStepExecutor(),
            StepType.MESSAGE: MessageStepExecutor(),
            StepType.PULL: PullStepExecutor(),
            StepType.COLLECT: CollectStepExecutor(),
            StepType.COGNITIVE: cognitive,
        }
        self._setup_providers(cognitive)

        # Scheduler — idle-based: timer counts from when agent becomes idle
        self._schedule_table: dict[str, float] = {}             # agent_id -> seconds
        self._heartbeat_table: dict[str, float] = {}            # agent_id -> heartbeat seconds
        self._last_idle: dict[str, datetime] = {}               # agent_id -> when it became idle

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

        # Delegate registry tracks cognitive sub-agents for UI/snapshot.
        # On startup, mark orphaned (RUNNING/PENDING) entries as KILLED.
        # Delegate output logs are preserved so the Activity tab can show
        # the working thread after restart (both completed and crashed agents).
        self.delegate_registry.cleanup_orphans()
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
        self._shutdown_event.set()
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
        # Close active cognitive-mode providers (e.g. bridge subprocess)
        for provider in list(self._active_providers.values()):
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception:
                    pass
        self._active_providers.clear()
        # Close pipeline-registered providers (cognitive step executor)
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

    def _delegates_snapshot(self) -> dict:
        """Delegate tree enriched with accumulated output for active delegates."""
        tree = self.delegate_registry.get_tree()
        # Attach persisted output text so the UI can hydrate mid-execution
        for node in tree.get("nodes", []):
            if node.get("status") in ("pending", "running"):
                text = self.get_delegate_output(node["agent_id"])
                if text:
                    node["output"] = text
        return tree

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
        # Heartbeat and legacy schedule are mutually exclusive.
        # Heartbeat takes precedence when both are present.
        if defn.heartbeat:
            self._heartbeat_table[defn.id] = self._parse_interval(defn.heartbeat.interval)
            self._schedule_table.pop(defn.id, None)
            self._last_idle.setdefault(defn.id, datetime.now(timezone.utc))
        elif defn.schedule:
            self._schedule_table[defn.id] = self._parse_interval(defn.schedule)
            self._heartbeat_table.pop(defn.id, None)
            self._last_idle[defn.id] = datetime.now(timezone.utc)
        # Hydrate execution history from JSONL so get_latest/get_history
        # work immediately for ALL agents (pipeline and cognitive alike).
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
        from .orchestrator import ORCHESTRATOR_ID
        if agent_id == ORCHESTRATOR_ID and not _force:
            raise ValueError("The orchestrator cannot be unregistered")
        await self.kill_agent(agent_id)
        self._agents.pop(agent_id, None)
        self._status.pop(agent_id, None)
        self._running_count.pop(agent_id, None)
        self._schedule_table.pop(agent_id, None)
        self._heartbeat_table.pop(agent_id, None)
        self._last_idle.pop(agent_id, None)
        self.inbox.remove_agent(agent_id)
        self.output_store.remove(agent_id)
        self.execution_log.remove_agent(agent_id)
        # Clean up persistent files
        self._agent_conversations.pop(agent_id, None)
        import shutil
        agent_data_dir = self._config.data_dir / "agents" / agent_id
        if agent_data_dir.exists():
            shutil.rmtree(agent_data_dir, ignore_errors=True)
        delegate_log = self._delegate_output_dir / f"{agent_id}.log"
        if delegate_log.exists():
            delegate_log.unlink(missing_ok=True)
        # Remove agent YAML definition if saved by loader
        agent_yaml = self._config.agents_dir / f"{agent_id}.yaml"
        if agent_yaml.exists():
            agent_yaml.unlink(missing_ok=True)
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

        # Dispatch based on agent mode
        if defn.mode == AgentMode.COGNITIVE:
            task = asyncio.create_task(self._execute_cognitive_agent(defn, record, cancel))
        else:
            task = asyncio.create_task(self._execute_pipeline(defn, record, cancel))
        self._tasks[eid] = task

        await self.events.emit(Event(
            type=EventType.EXECUTION_STARTED,
            source=agent_id,
            data={"agent_id": agent_id, "execution_id": eid,
                  "trigger_source": source, "total_steps": len(defn.steps),
                  "mode": defn.mode.value},
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
                # Mark agent as idle for heartbeat scheduling
                self._last_idle[defn.id] = datetime.now(timezone.utc)
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

            # Propagate failure to parent agent
            if record.status == ExecutionStatus.FAILED:
                await self._notify_parent_of_failure(defn.id, record)

    # ==================================================================
    # Cognitive mode execution — ONE path for ALL cognitive agents
    # ==================================================================

    async def route_tool_call(
        self, name: str, tool_input: dict, agent_id: str,
    ) -> dict:
        """Universal tool router for all cognitive agents.

        Handles shell tools (non-bridge providers), MCP connector tools,
        and ATN framework tools.  Passes caller_id for fractality.
        """
        from .orchestrator.tools import execute_tool

        # Shell tools (for non-bridge providers)
        if name in _SHELL_TOOL_EXECUTORS:
            return await _SHELL_TOOL_EXECUTORS[name](tool_input)
        # Connector tools (mcp_{connector_id}_{tool_name})
        if name.startswith("mcp_") and self.connectors:
            parsed = self.connectors.parse_tool_name(name)
            if parsed:
                cid, tool_name = parsed
                return await self.connectors.call_tool(cid, tool_name, tool_input)
            return {"error": f"Unknown connector tool: {name}"}
        # Framework tools
        return await execute_tool(name, tool_input, self, caller_id=agent_id)

    async def _execute_cognitive_agent(
        self,
        defn: AgentDefinition,
        record: ExecutionRecord,
        cancel: asyncio.Event,
    ) -> None:
        """Run a cognitive-mode agent via provider.send_orchestrate().

        This is THE execution path for ALL cognitive agents — orchestrator
        and children alike.  Differences are driven by configuration:

        - Tool surface: defn.tools (["atn_full"] for orchestrator, ["atn_core"] for children)
        - Provider lifecycle: orchestrator reuses its provider across turns;
          children create ephemeral instances per execution.
        - Conversation: orchestrator records to runtime.conversation (the chat UI);
          children record to per-agent ConversationStore.
        - Session resume: orchestrator uses BridgeProvider._session_id;
          children append history to system prompt.
        """
        from .delegate_prompts import build_delegate_prompt
        from .orchestrator import ORCHESTRATOR_ID
        from .orchestrator.tools import _get_delegate_tools, get_tool_definitions_for_bridge

        is_orchestrator = defn.id == ORCHESTRATOR_ID
        sub_provider = None
        owns_provider = True  # whether we should close the provider in finally

        try:
            # --- Provider lifecycle ---
            if is_orchestrator:
                # Orchestrator: reuse existing provider if available (session resume)
                sub_provider = self._active_providers.get(defn.id)
                if sub_provider is None:
                    sub_provider = self._resolve_provider_with_fallback(defn)
                    owns_provider = True
                else:
                    owns_provider = False  # reusing existing, don't close
            else:
                # Child agents: clean up stale, create fresh
                stale_provider = self._active_providers.pop(defn.id, None)
                if stale_provider is not None:
                    log.info("Cleaning up stale provider for agent %s", defn.id)
                    try:
                        await stale_provider.close()
                    except Exception:
                        pass
                sub_provider = self._resolve_provider_with_fallback(defn)

            sub_provider.event_bus = self.events
            sub_provider.source_agent_id = defn.id

            # Track active provider for message injection / interrupt
            self._active_providers[defn.id] = sub_provider

            # Register interrupt hook so killing cascades
            self.register_interrupt_hook(record.execution_id, sub_provider.interrupt)

            # --- System prompt ---
            if defn.system_prompt:
                system_prompt = defn.system_prompt
            else:
                system_prompt = build_delegate_prompt(
                    defn.agent_type, defn.id, defn.parent_id,
                )

            # --- Tool surface (configurable, not hardcoded per path) ---
            if "atn_full" in (defn.tools or []):
                # Full orchestrator tool surface
                agent_tools = get_tool_definitions_for_bridge()
            else:
                # Scoped delegate tools
                agent_tools = _get_delegate_tools()

            # For non-bridge providers, add shell/file tools
            from .providers.bridge import BridgeProvider as _BP
            if not isinstance(sub_provider, _BP):
                agent_tools.extend(_SHELL_TOOLS)

            # Add connector tools
            if self.connectors:
                for cid, session in self.connectors._sessions.items():
                    if session and session.tools:
                        agent_tools.extend(
                            {"name": t["name"], "description": t.get("description", ""),
                             "input_schema": t.get("inputSchema", t.get("input_schema", {}))}
                            for t in session.tools
                        )

            # --- Drain inbox ---
            all_messages = self.inbox.drain(defn.id)
            work_messages = [
                m for m in all_messages
                if m.type != MessageType.TRIGGER or m.data
            ]

            # --- Build user message ---
            prompt_parts: list[str] = []
            for msg in work_messages:
                if msg.data:
                    instruction = msg.data.get("instruction", "")
                    if instruction:
                        # Tag voice-sourced messages
                        if msg.source == "voice":
                            instruction = f"🎤 [Voice Input] {instruction}"
                        prompt_parts.append(instruction)
                    else:
                        prompt_parts.append(str(msg.data))
            if not prompt_parts:
                prompt_parts.append(defn.description or defn.name)
            user_message = "\n\n".join(prompt_parts)

            # --- Session resume (orchestrator) or history injection (children) ---
            session_id = ""
            if is_orchestrator:
                # Bridge providers support SDK session resume via session_id
                session_id = getattr(sub_provider, '_session_id', "") or ""
                if not session_id:
                    # No active SDK session — prepend conversation history
                    # for non-bridge providers
                    history = self.conversation.get_history_for_prompt()
                    if history:
                        user_message = history + "\n\nUser: " + user_message
                # Record user turn in the global conversation store
                self.conversation.add_user_turn(user_message)
            else:
                # Child agents: record user turn in per-agent conversation store
                agent_convo = self.get_agent_conversation_store(defn.id)
                existing = agent_convo.get_turns()
                if not existing or existing[-1].role != "user" or existing[-1].content != user_message:
                    agent_convo.add_user_turn(user_message)

                # Append conversation history to system prompt for session continuity
                prior_turns = agent_convo.get_turns()
                if prior_turns and prior_turns[-1].role == "user" and prior_turns[-1].content == user_message:
                    prior_turns = prior_turns[:-1]
                if prior_turns:
                    system_prompt = self._append_history_to_prompt(system_prompt, prior_turns)

            # --- Inject UTC time ---
            from datetime import timezone as _tz
            now = datetime.now(_tz.utc)
            time_line = f"Current time: {now.strftime('%Y-%m-%dT%H:%M:%SZ')} ({now.strftime('%A, %B %d, %Y')})"
            user_message = f"[{time_line}]\n\n{user_message}"

            # --- Tool executor (unified for all agents) ---
            async def _tool_executor(name: str, tool_input: dict) -> dict:
                return await self.route_tool_call(name, tool_input, defn.id)

            # --- Streaming callback ---
            async def _on_chunk(text: str) -> None:
                self.append_delegate_output(defn.id, text)
                await self.events.emit(Event(
                    type=EventType.STEP_OUTPUT,
                    source=defn.id,
                    data={
                        "agent_id": defn.id,
                        "channel": "text",
                        "content": text,
                        "cognitive": True,
                    },
                ))

            # --- Run the agent ---
            send_kwargs: dict[str, Any] = {
                "message": user_message,
                "system": system_prompt,
                "tools": agent_tools,
                "max_turns": defn.max_turns,
                "tool_executor": _tool_executor,
                "on_chunk": _on_chunk,
            }
            if session_id:
                send_kwargs["session_id"] = session_id

            response = await sub_provider.send_orchestrate(**send_kwargs)

            # --- Process result ---
            result_text = response.text or ""
            total_tokens = (
                response.usage.input_tokens
                + response.usage.output_tokens
                + response.usage.cache_read_tokens
                + response.usage.cache_creation_tokens
            )

            if response.stop_reason == "interrupted" or cancel.is_set():
                record.status = ExecutionStatus.KILLED
                record.error = "Interrupted"
            else:
                record.status = ExecutionStatus.COMPLETED

            record.output = {
                "result": result_text,
                "tokens_used": total_tokens,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_tokens,
                    "cache_creation_tokens": response.usage.cache_creation_tokens,
                },
            }

            # Track token usage
            provider_key = getattr(sub_provider, 'name', 'claude_max')
            if provider_key not in record.token_usage:
                record.token_usage[provider_key] = TokenUsage(provider=provider_key)
            record.token_usage[provider_key].input_tokens += response.usage.input_tokens
            record.token_usage[provider_key].output_tokens += response.usage.output_tokens
            record.token_usage[provider_key].cache_read_tokens += response.usage.cache_read_tokens
            record.token_usage[provider_key].cache_creation_tokens += response.usage.cache_creation_tokens

            # Orchestrator: record assistant turn in global conversation store
            if is_orchestrator and result_text:
                self.conversation.add_assistant_turn(
                    result_text, execution_id=record.execution_id,
                )

        except asyncio.CancelledError:
            record.status = ExecutionStatus.KILLED
            record.error = "Force-cancelled by kill switch"

        except Exception as exc:
            record.status = ExecutionStatus.FAILED
            record.error = str(exc)
            log.exception("Cognitive agent error for %s", defn.id)

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

            # Record assistant turn in per-agent conversation store (children)
            if not is_orchestrator:
                _convo_text = ""
                if isinstance(record.output, dict):
                    _convo_text = record.output.get("result", "")
                elif record.output:
                    _convo_text = str(record.output)
                if record.error:
                    _convo_text = (
                        f"{_convo_text}\n\nError: {record.error}"
                        if _convo_text else f"Error: {record.error}"
                    )
                if _convo_text:
                    try:
                        _convo_store = self.get_agent_conversation_store(defn.id)
                        _convo_store.add_assistant_turn(_convo_text, execution_id=record.execution_id)
                    except Exception:
                        log.warning("Failed to record assistant turn for %s", defn.id)

            # Sync delegate registry (for UI observability)
            result_text = ""
            total_tokens = 0
            if isinstance(record.output, dict):
                result_text = record.output.get("result", "")
                total_tokens = record.output.get("tokens_used", 0)
            delegate_node = self.delegate_registry.get_node(defn.id)
            if delegate_node is not None:
                from .agent_registry import DelegateStatus
                if record.status == ExecutionStatus.COMPLETED:
                    self.delegate_registry.update_status(
                        defn.id, DelegateStatus.COMPLETED,
                        result_preview=result_text[:500],
                        tokens_used=total_tokens,
                    )
                elif record.status == ExecutionStatus.KILLED:
                    self.delegate_registry.update_status(
                        defn.id, DelegateStatus.KILLED,
                        result_preview=result_text[:500],
                        tokens_used=total_tokens,
                    )
                else:
                    self.delegate_registry.update_status(
                        defn.id, DelegateStatus.FAILED,
                        error=record.error,
                    )
                self.delegate_registry.save()

            # Bookkeeping
            self._running_count[defn.id] = max(0, self._running_count.get(defn.id, 1) - 1)
            self._executions.pop(record.execution_id, None)
            self._tasks.pop(record.execution_id, None)
            self._cancels.pop(record.execution_id, None)
            self._interrupt_hooks.pop(record.execution_id, None)

            # Provider lifecycle: orchestrator keeps its provider; children clean up
            if owns_provider and sub_provider is not None:
                self._active_providers.pop(defn.id, None)
                try:
                    await sub_provider.close()
                except Exception:
                    pass
            # Orchestrator provider stays in _active_providers for session resume

            if self._running_count.get(defn.id, 0) == 0:
                self._last_idle[defn.id] = datetime.now(timezone.utc)
                if record.status == ExecutionStatus.FAILED:
                    self._status[defn.id] = AgentStatus.ERROR
                elif self._status.get(defn.id) == AgentStatus.RUNNING:
                    self._status[defn.id] = AgentStatus.ACTIVE

            # Signal delegate_done for collect waiters
            done_event = self._delegate_done.get(defn.id)
            if done_event:
                done_event.set()

            # Innate wake-up: notify parent on completion
            await self._on_agent_completed(defn.id, record)

            # Propagate failure to parent agent
            if record.status == ExecutionStatus.FAILED:
                await self._notify_parent_of_failure(defn.id, record)

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
                    "mode": "cognitive",
                    "output_preview": _preview(record.output) if record.output else None,
                    "error": record.error,
                },
            ))

    def _resolve_provider_with_fallback(self, defn: AgentDefinition) -> Any:
        """Resolve a provider for a cognitive agent, trying fallback chain.

        If defn.provider is a list of provider names (e.g. ["claude_max", "anthropic"]),
        tries each in order using defn.cognitive_model.
        If defn.provider is a string, it may be a provider name OR a model name
        (for children that specify "gemini-2.5-flash" directly).
        """
        providers = defn.provider
        model = defn.cognitive_model or self._config.orchestrator.model or "claude-sonnet-4-6"

        if isinstance(providers, list):
            # Provider fallback chain — try each provider name in order
            for provider_name in providers:
                try:
                    return self._resolve_provider_by_name(provider_name, model, defn.id)
                except Exception:
                    log.info("Provider '%s' not available for %s, trying next", provider_name, defn.id)
            # All failed — try the first one and let it raise
            first = providers[0] if providers else "claude_max"
            return self._resolve_provider_by_name(first, model, defn.id)
        elif providers:
            # Single string — could be a provider name ("claude_max") or
            # a model name ("gemini-2.5-flash") for backward compat with children
            if providers in self._KNOWN_PROVIDERS or providers in self._custom_providers:
                return self._resolve_provider_by_name(providers, model, defn.id)
            # Treat as model name (legacy child agent path)
            return self._resolve_provider_for_model(providers, defn.id)
        else:
            return self._resolve_provider_for_model(model, defn.id)

    def _resolve_provider_by_name(self, provider_name: str, model: str, agent_id: str) -> Any:
        """Create a provider instance by provider name + model.

        Maps provider names (claude_max, anthropic, gemini, openai, ollama)
        to the correct provider class with the given model.
        """
        from .providers.bridge import BridgeProvider
        from .providers.openai_compat import OpenAICompatibleProvider

        if provider_name == "claude_max":
            return BridgeProvider(model=model)

        if provider_name == "anthropic":
            api_key = self._resolve_api_key("anthropic")
            if not api_key:
                raise ProviderError("No Anthropic API key configured")
            return AnthropicProvider(api_key=api_key, default_model=model, base_url="")

        if provider_name == "gemini":
            api_key = self._resolve_api_key("gemini")
            if not api_key:
                raise ProviderError("No Gemini API key configured")
            defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
            return OpenAICompatibleProvider(
                name=f"gemini-{agent_id}",
                base_url=defaults.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                api_key=api_key,
                default_model=model,
            )

        if provider_name == "openai":
            api_key = self._resolve_api_key("openai")
            if not api_key:
                raise ProviderError("No OpenAI API key configured")
            return OpenAICompatibleProvider(
                name=f"openai-{agent_id}",
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                default_model=model,
            )

        if provider_name == "ollama":
            defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
            pconfig = self._config.providers.get("ollama")
            return OllamaProvider(
                base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", "http://localhost:11434"),
                default_model=model,
            )

        # Custom provider
        if provider_name in self._custom_providers:
            pconfig = self._config.providers.get(provider_name)
            if pconfig and pconfig.base_url:
                api_key = self._resolve_api_key(provider_name)
                return OpenAICompatibleProvider(
                    name=provider_name,
                    base_url=pconfig.base_url,
                    api_key=api_key,
                    default_model=model,
                )

        raise ProviderError(f"Unknown provider: {provider_name}")

    @staticmethod
    def _append_history_to_prompt(system_prompt: str, prior_turns: list) -> str:
        """Append conversation history to system prompt for session continuity."""
        _ROLE_PREFIX = {"user": "User", "assistant": "Agent", "system": "System"}
        history_parts = []
        for turn in prior_turns:
            prefix = _ROLE_PREFIX.get(turn.role, turn.role.title())
            history_parts.append(f"{prefix}: {turn.content}")

        # Sliding window: keep under ~100k tokens (~400k chars)
        _HISTORY_CHAR_BUDGET = 400_000
        total_chars = sum(len(p) for p in history_parts)
        start = 0
        while start < len(history_parts) and total_chars > _HISTORY_CHAR_BUDGET:
            total_chars -= len(history_parts[start])
            start += 1
        if start > 0 and start < len(history_parts):
            history_parts = history_parts[start:]
            history_parts.insert(0, "[Earlier conversation trimmed]")
        elif start >= len(history_parts):
            history_parts = []

        if history_parts:
            conversation_history = "\n\n".join(history_parts)
            system_prompt += (
                "\n\n## Previous Conversation\n"
                "You have had previous interactions. Here is your conversation history. "
                "Continue from where you left off. You are the SAME agent — "
                "remember everything you did and were told.\n\n"
                + conversation_history
            )
        return system_prompt

    # ------------------------------------------------------------------
    # Hierarchy support
    # ------------------------------------------------------------------

    def _resolve_parent_agent_id(self, parent_id: str) -> str:
        """Resolve parent_id to actual agent registry ID.

        The delegate convention uses "orch" as shorthand for the orchestrator,
        but the Runtime registry uses the full ORCHESTRATOR_ID.
        """
        if parent_id in self._agents:
            return parent_id
        from .orchestrator import ORCHESTRATOR_ID
        if parent_id == "orch" and ORCHESTRATOR_ID in self._agents:
            return ORCHESTRATOR_ID
        return parent_id

    def get_children(self, agent_id: str) -> list[AgentDefinition]:
        """Return all agents whose parent_id matches agent_id.

        Also matches agents parented to "orch" when querying "orchestrator".
        """
        from .orchestrator import ORCHESTRATOR_ID
        children = []
        for defn in self._agents.values():
            if defn.parent_id == agent_id:
                children.append(defn)
            elif (agent_id == ORCHESTRATOR_ID
                  and defn.parent_id == "orch"):
                children.append(defn)
        return children

    def get_descendants(self, agent_id: str) -> list[AgentDefinition]:
        """All descendants (children, grandchildren, ...) via BFS."""
        descendants: list[AgentDefinition] = []
        queue = [agent_id]
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            for child in children:
                descendants.append(child)
                queue.append(child.id)
        return descendants

    # ------------------------------------------------------------------
    # Failure propagation: notify parent of child errors
    # ------------------------------------------------------------------

    async def _notify_parent_of_failure(
        self, agent_id: str, record: ExecutionRecord
    ) -> None:
        """Post an ALERT to the parent agent when a child execution fails.

        This ensures supervisor agents can react autonomously to errors
        in their child agents rather than having to poll for status.
        """
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

        # If parent has an active bridge session, inject directly
        parent_provider = self._active_providers.get(resolved_parent)
        if parent_provider is None:
            parent_provider = self._active_providers.get(defn.parent_id)
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

    # ------------------------------------------------------------------
    # Innate wake-up: child completion -> parent inbox message
    # ------------------------------------------------------------------

    async def _on_agent_completed(self, agent_id: str, record: ExecutionRecord) -> None:
        """Handle cognitive agent completion.

        - Sets agent status: COMPLETED for one-shots, ACTIVE for scheduled
          (only on success; errors are already set to ERROR by the caller).
        - Posts a HIGH-priority WORK message to the parent's inbox with the
          child's result preview.
        - If the parent has an active bridge session, injects via
          send_user_message() for immediate attention.
        """
        defn = self._agents.get(agent_id)
        if not defn:
            return

        # Set final status only on success — errors are already handled
        if record.status == ExecutionStatus.COMPLETED:
            if defn.schedule or defn.heartbeat:
                self._status[agent_id] = AgentStatus.ACTIVE
            else:
                self._status[agent_id] = AgentStatus.COMPLETED

        # Mark agent as idle (for heartbeat scheduling)
        self._last_idle[agent_id] = datetime.now(timezone.utc)

        # Clean up completion callback
        self._completion_callbacks.pop(agent_id, None)

        parent_id = defn.parent_id
        if not parent_id:
            return

        # Resolve "orch" -> "orchestrator" mapping
        resolved_parent = self._resolve_parent_agent_id(parent_id)

        # Only notify if parent is still registered
        if resolved_parent not in self._agents:
            return

        # Build result preview
        result_preview = ""
        if isinstance(record.output, dict):
            result_preview = str(record.output.get("result", ""))[:2000]
        elif record.output is not None:
            result_preview = str(record.output)[:2000]

        # Post HIGH-priority WORK message to parent's inbox
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
                "output_preview": result_preview[:500],
                "result_preview": result_preview[:500],
                "error": record.error,
                "instruction": (
                    f"Your child agent '{defn.name}' has {status_str}. "
                    f"Check its output with get_output('{agent_id}')."
                ),
            },
        )
        self.inbox.post(msg)
        log.info("Innate wake-up: posted child_completed for %s -> parent %s", agent_id, resolved_parent)

        # If parent has an active bridge session, inject directly
        # Check both parent_id (shorthand like "orch") and resolved_parent
        parent_provider = self._active_providers.get(resolved_parent)
        if parent_provider is None:
            parent_provider = self._active_providers.get(parent_id)
        if parent_provider is not None:
            inject_text = (
                f"[CHILD COMPLETED] Agent '{defn.name}' ({agent_id}) "
                f"finished with status: {record.status.value}.\n"
                f"Result preview: {result_preview[:500]}"
            )
            if record.error:
                inject_text += f"\nError: {record.error}"
            try:
                await parent_provider.send_user_message(inject_text)
                log.info("Injected child_completed message into parent %s bridge session", resolved_parent)
            except Exception as exc:
                log.warning("Failed to inject into parent %s session: %s", resolved_parent, exc)

    # ------------------------------------------------------------------
    # Child ID generation (unified with DelegateRegistry)
    # ------------------------------------------------------------------

    def generate_child_id(self, parent_id: str) -> str:
        """Generate the next hierarchical child ID for a parent.

        Example: "orch" -> "orch.1", then "orch.2", etc.
        """
        count = self._child_counters.get(parent_id, 0) + 1
        self._child_counters[parent_id] = count
        return f"{parent_id}.{count}"

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
        """Inject a user message into a running cognitive agent's session.

        Returns True if the message was delivered, False if the agent
        isn't running or has no active bridge session.
        """
        provider = self._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.send_user_message(content)
        return True

    async def interrupt_delegate(self, agent_id: str) -> bool:
        """Interrupt a running cognitive agent's session.

        Calls bridge.interrupt() which tells the Claude SDK to wind down
        gracefully.  Returns True if the interrupt was sent.
        """
        provider = self._active_providers.get(agent_id)
        if provider is None:
            return False
        await provider.interrupt()
        return True

    async def interrupt_orchestrator(self) -> bool:
        """Interrupt the running orchestrator session.

        Finds the orchestrator's active provider and calls interrupt().
        Returns True if the interrupt was sent.
        """
        from .orchestrator import ORCHESTRATOR_ID
        return await self.interrupt_delegate(ORCHESTRATOR_ID)

    # ------------------------------------------------------------------
    # Context inspection
    # ------------------------------------------------------------------

    def _resolve_provider_for_model(self, model_name: str, agent_id: str) -> "Provider":
        """Create the right provider instance for a given model name.

        - gemini-*  → OpenAICompatibleProvider via Gemini endpoint
        - gpt-*     → OpenAICompatibleProvider via OpenAI endpoint
        - Otherwise → BridgeProvider (Claude Agent SDK)
        """
        from .providers.bridge import BridgeProvider
        from .providers.openai_compat import OpenAICompatibleProvider

        model_lower = model_name.lower()

        if model_lower.startswith("gemini"):
            defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
            api_key = self._resolve_api_key("gemini")
            if not api_key:
                log.warning("No Gemini API key found, falling back to BridgeProvider")
                return BridgeProvider(model=model_name)
            return OpenAICompatibleProvider(
                name=f"gemini-{agent_id}",
                base_url=defaults.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                api_key=api_key,
                default_model=model_name,
            )

        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            defaults = self._PROVIDER_DEFAULTS.get("openai", {})
            api_key = self._resolve_api_key("openai")
            if not api_key:
                log.warning("No OpenAI API key found, falling back to BridgeProvider")
                return BridgeProvider(model=model_name)
            return OpenAICompatibleProvider(
                name=f"openai-{agent_id}",
                base_url=defaults.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
                default_model=model_name,
            )

        # Default: Claude via bridge
        return BridgeProvider(model=model_name)

    def _get_bridge_provider(self, agent_id: str | None = None) -> Any:
        """Resolve a BridgeProvider for the given agent.

        Looks up the agent's active provider in _active_providers.
        If agent_id is None, defaults to the orchestrator.
        Returns None if not found or not a BridgeProvider.
        """
        from .orchestrator import ORCHESTRATOR_ID
        from .providers.bridge import BridgeProvider
        target = agent_id or ORCHESTRATOR_ID
        provider = self._active_providers.get(target)
        return provider if isinstance(provider, BridgeProvider) else None

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

                # --- Schedule-based triggers (pipeline agents) ---
                for agent_id, interval in list(self._schedule_table.items()):
                    if self._status.get(agent_id) != AgentStatus.ACTIVE:
                        continue
                    if self._running_count.get(agent_id, 0) > 0:
                        continue
                    last_idle = self._last_idle.get(agent_id)
                    if last_idle is None or (now - last_idle).total_seconds() >= interval:
                        self._last_idle[agent_id] = now
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

                # --- Heartbeat-based triggers (cognitive agents) ---
                for agent_id, interval in list(self._heartbeat_table.items()):
                    # Skip if already handled by schedule table above
                    if agent_id in self._schedule_table:
                        continue
                    status = self._status.get(agent_id)
                    if status not in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
                        continue
                    # Never fire while agent has running executions
                    if self._running_count.get(agent_id, 0) > 0:
                        continue
                    last_idle = self._last_idle.get(agent_id)
                    if last_idle is None or (now - last_idle).total_seconds() >= interval:
                        self._last_idle[agent_id] = now
                        # Heartbeat posts a WORK message to the agent's own inbox
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

        # Build goals from agent registry — agents ARE goals
        goals: list[dict[str, Any]] = []
        active_agents: list[dict[str, Any]] = []
        for aid, defn in self._agents.items():
            if aid == ORCHESTRATOR_ID:
                continue
            status = self._status.get(aid)
            # Every non-orchestrator agent is a goal
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

        # Pending tasks
        pending = [
            {"id": t.id, "title": t.title, "goal_id": t.goal_id, "status": t.status.value}
            for t in self.planning_tasks
            if t.status in (TaskStatus.PROPOSED, TaskStatus.APPROVED, TaskStatus.ACTIVE)
        ]

        # Calendar availability
        calendar_available = self.credential_store.exists("google_calendar")

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
                "mode": defn.mode.value,
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
            # Hierarchy info for UI agent cards
            if defn.parent_id:
                agent_info["parent_id"] = self._resolve_parent_agent_id(defn.parent_id)
            children_count = sum(
                1 for d in self._agents.values()
                if d.parent_id and self._resolve_parent_agent_id(d.parent_id) == aid
            )
            if children_count:
                agent_info["children_count"] = children_count
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
            raw_provider = orch_defn.provider or ""
            # Provider may be a string or a fallback chain list
            if isinstance(raw_provider, list):
                primary_provider = raw_provider[0] if raw_provider else ""
                fallback_providers = raw_provider[1:] if len(raw_provider) > 1 else []
            else:
                primary_provider = raw_provider
                fallback_providers = []
            orch_model = orch_defn.cognitive_model or ""
            orch_info = {
                "provider": primary_provider,
                "model": orch_model,
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
            "delegates": self._delegates_snapshot(),
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
        # NOTE: Cannot use _hot_register_provider here because self._executors
        # hasn't been assigned yet.  Register directly on the cognitive instance.
        for pid in ("anthropic", "gemini", "openai"):
            if pid in self._config.providers:
                continue  # already handled above
            api_key = self._resolve_api_key(pid)
            if not api_key:
                continue
            try:
                if pid == "anthropic":
                    provider = AnthropicProvider(api_key=api_key, default_model="", base_url="")
                else:
                    defaults = self._PROVIDER_DEFAULTS.get(pid, {})
                    provider = OpenAICompatibleProvider(
                        name=pid,
                        base_url=defaults.get("base_url", ""),
                        api_key=api_key,
                        default_model=defaults.get("default_model", ""),
                    )
                cognitive.register_provider(provider)
                log.info("Registered credential-store provider: %s", pid)
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
                # Use orchestrator model if no provider-level default is set
                default_model = (
                    (pconfig.default_model if pconfig else "")
                    or self._config.orchestrator.model
                    or "claude-sonnet-4-6"
                )
                provider = BridgeProvider(
                    model=default_model,
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

        Always uses the fleet-management prompt, optionally enriched with
        user profile context if available.

        Returns the orchestrator agent ID.
        """
        from .orchestrator import build_system_prompt_with_context, create_orchestrator_agent

        # Build system prompt with whatever profile context is available
        profile = self.user_profile.get_profile()
        profile_parts: list[str] = []
        if profile.summary:
            profile_parts.append(profile.summary)
        if profile.strengths:
            profile_parts.append("Strengths: " + ", ".join(profile.strengths))
        if profile.weaknesses:
            profile_parts.append("Weaknesses: " + ", ".join(profile.weaknesses))
        profile_summary = "\n".join(profile_parts)

        # Goals are now agents — build summary from agent registry
        goals_lines: list[str] = []
        from .orchestrator import ORCHESTRATOR_ID as _ORCH_ID
        for aid, defn in self._agents.items():
            if aid == _ORCH_ID:
                continue
            status = self._status.get(aid)
            goal_status = status.value if status else "unknown"
            goals_lines.append(f"- [{goal_status}] {defn.name} (agent: {aid}): {defn.task_prompt or defn.description}")
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
        log.info("Orchestrator registered and activated (provider=%s)", defn.provider)

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
        raw_provider = orch_defn.provider if orch_defn else ""
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

    # ------------------------------------------------------------------
    # Per-agent conversation stores
    # ------------------------------------------------------------------

    def get_agent_conversation_store(self, agent_id: str) -> ConversationStore:
        """Get or create a ConversationStore for a cognitive agent."""
        if agent_id not in self._agent_conversations:
            store_dir = self._config.data_dir / "agents" / agent_id
            store_dir.mkdir(parents=True, exist_ok=True)
            self._agent_conversations[agent_id] = ConversationStore(store_dir)
        return self._agent_conversations[agent_id]

    async def send_agent_message(self, agent_id: str, text: str) -> dict:
        """Send a user message to a cognitive agent.

        If the agent is currently running, injects into the active session.
        Otherwise, posts a WORK message to trigger a new execution.
        Records the user turn in the agent's conversation store.
        """
        defn = self.get_agent(agent_id)
        if defn is None:
            return {"error": f"Agent '{agent_id}' not found"}
        if defn.mode != AgentMode.COGNITIVE:
            return {"error": f"Agent '{agent_id}' is not a cognitive agent"}

        store = self.get_agent_conversation_store(agent_id)

        # Try mid-session injection first
        if await self.send_delegate_message(agent_id, text):
            # Record in conversation store here — the execution engine
            # won't see this message (it goes directly to the provider).
            store.add_user_turn(text)
            return {"status": "injected", "agent_id": agent_id}

        # Not running — post to inbox.  Do NOT add_user_turn here;
        # the execution engine records it when it drains the inbox.
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            type=MessageType.WORK,
            source="user",
            target=agent_id,
            priority=MessagePriority.HIGH,
            data={"instruction": text},
        )
        self.inbox.post(msg)

        # Trigger execution — re-activate completed agents so they can resume
        status = self.get_status(agent_id)
        if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
            # Close any stale provider left over from the previous execution
            old_provider = self._active_providers.pop(agent_id, None)
            if old_provider is not None:
                try:
                    await old_provider.close()
                except Exception:
                    pass

            # Re-activate so trigger_run proceeds
            self._status[agent_id] = AgentStatus.ACTIVE
            log.info("Re-activated %s agent %s for follow-up message", status.value, agent_id)
            status = AgentStatus.ACTIVE
        if status == AgentStatus.ACTIVE:
            eid = await self.trigger_run(agent_id, source="user")
            return {"status": "triggered", "agent_id": agent_id, "execution_id": eid}

        return {"status": "queued", "agent_id": agent_id}

    # ------------------------------------------------------------------
    # Delegate output persistence
    # ------------------------------------------------------------------

    def append_delegate_output(self, agent_id: str, text: str) -> None:
        """Append a text chunk to the delegate's persistent output log."""
        path = self._delegate_output_dir / f"{agent_id}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    def get_delegate_output(self, agent_id: str) -> str:
        """Read the full accumulated output for a delegate."""
        path = self._delegate_output_dir / f"{agent_id}.log"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def clear_delegate_output(self, agent_id: str) -> None:
        """Remove a delegate's output log (cleanup after collection)."""
        path = self._delegate_output_dir / f"{agent_id}.log"
        if path.exists():
            path.unlink()

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
        model = orch_defn.cognitive_model if orch_defn else ""

        now = datetime.now(timezone.utc)
        lines = [f"Current time: {now.strftime('%Y-%m-%dT%H:%M:%SZ')} ({now.strftime('%A, %B %d, %Y')})"]
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
