"""Pipeline + cognitive execution paths.

Owns: _executions, _tasks, _cancels (shared with ExecutionControl)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from ..events import Event, EventBus, EventType
from ..inbox import InboxManager
from ..models import (
    AgentDefinition,
    AgentMode,
    AgentStatus,
    ExecutionRecord,
    ExecutionStatus,
    MessageType,
    StepType,
    TokenUsage,
)
from ..store import AgentOutput, ExecutionLog, OutputStore
from ..steps.base import StepContext, StepExecutor

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .provider_manager import ProviderManager
    from .session_manager import SessionManager
    from ..connectors_manager import ConnectorManager
    from ..agent_registry import DelegateRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shell tools for non-bridge providers
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
    try:
        p = Path(inp["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"], encoding="utf-8")
        return {"status": "ok", "path": str(p), "bytes": len(inp["content"])}
    except Exception as e:
        return {"error": str(e)}


async def _exec_list_dir(inp: dict) -> dict:
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


class ExecutionEngine:
    """Manages execution of pipeline and cognitive agents."""

    def __init__(
        self,
        registry: AgentRegistry,
        provider_manager: ProviderManager,
        session_manager: SessionManager,
        events: EventBus,
        execution_log: ExecutionLog,
        inbox: InboxManager,
        connectors: ConnectorManager,
        output_store: OutputStore,
        delegate_registry: DelegateRegistry,
        executors: dict[StepType, StepExecutor],
        config: Any,
    ) -> None:
        self.registry = registry
        self.provider_manager = provider_manager
        self.session_manager = session_manager
        self.events = events
        self.execution_log = execution_log
        self.inbox = inbox
        self.connectors = connectors
        self.output_store = output_store
        self.delegate_registry = delegate_registry
        self._executors = executors
        self._config = config

        # Shared execution state
        self._executions: dict[str, ExecutionRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancels: dict[str, asyncio.Event] = {}

        # Completion callbacks: child_id -> parent_id
        self._completion_callbacks: dict[str, str] = {}
        # Events for signalling delegate completion
        self._delegate_done: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    async def trigger_run(self, agent_id: str, source: str = "user") -> str | None:
        self.registry._require_agent(agent_id)
        defn = self.registry._agents[agent_id]

        if self.registry._running_count.get(agent_id, 0) >= defn.concurrency:
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
        self.registry._running_count[agent_id] = self.registry._running_count.get(agent_id, 0) + 1
        self.registry._status[agent_id] = AgentStatus.RUNNING
        self.execution_log.record(record)

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

    # ------------------------------------------------------------------
    # Tool routing
    # ------------------------------------------------------------------

    async def route_tool_call(
        self, name: str, tool_input: dict, agent_id: str,
    ) -> dict:
        from ..orchestrator.tools import execute_tool

        if name in _SHELL_TOOL_EXECUTORS:
            return await _SHELL_TOOL_EXECUTORS[name](tool_input)
        if name.startswith("mcp_") and self.connectors:
            parsed = self.connectors.parse_tool_name(name)
            if parsed:
                cid, tool_name = parsed
                return await self.connectors.call_tool(cid, tool_name, tool_input)
            return {"error": f"Unknown connector tool: {name}"}
        # Framework tools — pass the full Runtime reference
        return await execute_tool(name, tool_input, self._runtime_ref, caller_id=agent_id)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def _execute_pipeline(
        self,
        defn: AgentDefinition,
        record: ExecutionRecord,
        cancel: asyncio.Event,
    ) -> None:
        previous: list[Any] = []
        all_messages = self.inbox.drain(defn.id)
        work_messages = [
            m for m in all_messages
            if m.type != MessageType.TRIGGER or m.data
        ]

        if defn.connector_ids:
            try:
                await self.connectors.ensure_started(defn.connector_ids)
            except Exception as exc:
                record.status = ExecutionStatus.FAILED
                record.error = f"Failed to start connectors: {exc}"
                record.completed_at = datetime.now(timezone.utc)
                self.execution_log.record(record)
                self.execution_log.persist(record)
                self.registry._running_count[defn.id] = max(0, self.registry._running_count.get(defn.id, 1) - 1)
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
                    runtime=self._runtime_ref,
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

                # Extract token usage
                if (step_def.type == StepType.COGNITIVE
                        and step_result.status == ExecutionStatus.COMPLETED
                        and isinstance(step_result.output, dict)):
                    provider_name = step_def.config.get("provider", "")
                    if isinstance(provider_name, list):
                        provider_name = provider_name[0] if provider_name else ""
                    _accumulate_usage(record, provider_name, step_result.output)

                if (step_def.type == StepType.COLLECT
                        and step_result.status == ExecutionStatus.COMPLETED
                        and isinstance(step_result.output, dict)):
                    _accumulate_child_usage(record, step_result.output, self.execution_log)

                self.execution_log.record(record)
                self.execution_log.persist_running(record)

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

                if step_result.status in (ExecutionStatus.FAILED, ExecutionStatus.KILLED):
                    record.status = step_result.status
                    record.error = step_result.error
                    break
            else:
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

            if record.status == ExecutionStatus.COMPLETED:
                self.output_store.write(AgentOutput(
                    agent_id=defn.id,
                    data=record.output,
                    status=record.status,
                    execution_id=record.execution_id,
                ))

            self.registry._running_count[defn.id] = max(0, self.registry._running_count.get(defn.id, 1) - 1)
            self._executions.pop(record.execution_id, None)
            self._tasks.pop(record.execution_id, None)
            self._cancels.pop(record.execution_id, None)
            self._interrupt_hooks.pop(record.execution_id, None)

            if self.registry._running_count.get(defn.id, 0) == 0:
                self.registry._last_idle[defn.id] = datetime.now(timezone.utc)
                if record.status == ExecutionStatus.FAILED:
                    self.registry._status[defn.id] = AgentStatus.ERROR
                elif self.registry._status.get(defn.id) == AgentStatus.RUNNING:
                    self.registry._status[defn.id] = AgentStatus.ACTIVE

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

            if record.status == ExecutionStatus.FAILED:
                await self.registry.notify_parent_of_failure(
                    defn.id, record, self.provider_manager._active_providers,
                )

    # ------------------------------------------------------------------
    # Cognitive agent execution
    # ------------------------------------------------------------------

    async def _execute_cognitive_agent(
        self,
        defn: AgentDefinition,
        record: ExecutionRecord,
        cancel: asyncio.Event,
    ) -> None:
        from ..delegate_prompts import build_delegate_prompt
        from ..orchestrator import ORCHESTRATOR_ID
        from ..orchestrator.tools import _get_delegate_tools, get_tool_definitions_for_bridge

        is_orchestrator = defn.id == ORCHESTRATOR_ID
        sub_provider = None
        owns_provider = True

        try:
            # --- Provider lifecycle ---
            if is_orchestrator:
                sub_provider = self.provider_manager._active_providers.get(defn.id)
                if sub_provider is None:
                    sub_provider = self.provider_manager.resolve_provider_with_fallback(defn)
                    owns_provider = True
                else:
                    owns_provider = False
            else:
                stale_provider = self.provider_manager._active_providers.pop(defn.id, None)
                if stale_provider is not None:
                    log.info("Cleaning up stale provider for agent %s", defn.id)
                    try:
                        await stale_provider.close()
                    except Exception:
                        pass
                sub_provider = self.provider_manager.resolve_provider_with_fallback(defn)

            sub_provider.event_bus = self.events
            sub_provider.source_agent_id = defn.id
            self.provider_manager._active_providers[defn.id] = sub_provider

            # Register interrupt hook
            self.register_interrupt_hook(record.execution_id, sub_provider.interrupt)

            # --- System prompt ---
            if defn.system_prompt:
                system_prompt = defn.system_prompt
            else:
                system_prompt = build_delegate_prompt(
                    defn.agent_type, defn.id, defn.parent_id,
                )

            # --- Tool surface ---
            if "atn_full" in (defn.tools or []):
                agent_tools = get_tool_definitions_for_bridge()
            else:
                agent_tools = _get_delegate_tools()

            from ..providers.bridge import BridgeProvider as _BP
            if not isinstance(sub_provider, _BP):
                agent_tools.extend(_SHELL_TOOLS)

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
                    # Heartbeat trigger from the scheduler
                    if msg.source == "heartbeat" and msg.data.get("heartbeat"):
                        interval_s = msg.data.get("interval", 0)
                        prompt_parts.append(
                            f"[HEARTBEAT] This is an automatic heartbeat wake-up from the runtime "
                            f"(interval: {interval_s}s). This is NOT a user message. "
                            f"Check on active goals and delegates, take action if needed, "
                            f"or go back to idle if there is nothing to do."
                        )
                        continue
                    instruction = msg.data.get("instruction", "")
                    if instruction:
                        if msg.source == "voice":
                            instruction = f"\U0001f3a4 [Voice Input] {instruction}"
                        prompt_parts.append(instruction)
                    else:
                        prompt_parts.append(str(msg.data))
            if not prompt_parts:
                prompt_parts.append(defn.description or defn.name)
            user_message = "\n\n".join(prompt_parts)

            # --- Session resume / history ---
            session_id = ""
            if is_orchestrator:
                # Build history from PRIOR turns before recording this one.
                session_id = getattr(sub_provider, '_session_id', "") or ""
                if not session_id:
                    history = self.session_manager.conversation.get_history_for_prompt()
                else:
                    history = ""
                # Store the raw user message (skip if already recorded by
                # ws_server or send_agent_message to avoid duplicates).
                existing_turns = self.session_manager.conversation.get_turns()
                if (not existing_turns
                        or existing_turns[-1].role != "user"
                        or existing_turns[-1].content != user_message):
                    self.session_manager.conversation.add_user_turn(user_message)
                # Prepend conversation history for non-session providers
                if history:
                    user_message = history + "\n\nUser: " + user_message
            else:
                agent_convo = self.session_manager.get_agent_conversation_store(defn.id)
                existing = agent_convo.get_turns()
                if not existing or existing[-1].role != "user" or existing[-1].content != user_message:
                    agent_convo.add_user_turn(user_message)
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

            # --- Tool executor ---
            async def _tool_executor(name: str, tool_input: dict) -> dict:
                return await self.route_tool_call(name, tool_input, defn.id)

            # --- Streaming callback ---
            async def _on_chunk(text: str) -> None:
                self.session_manager.append_delegate_output(defn.id, text)
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

            provider_key = getattr(sub_provider, 'name', 'claude_max')
            if provider_key not in record.token_usage:
                record.token_usage[provider_key] = TokenUsage(provider=provider_key)
            record.token_usage[provider_key].input_tokens += response.usage.input_tokens
            record.token_usage[provider_key].output_tokens += response.usage.output_tokens
            record.token_usage[provider_key].cache_read_tokens += response.usage.cache_read_tokens
            record.token_usage[provider_key].cache_creation_tokens += response.usage.cache_creation_tokens

            if is_orchestrator and result_text:
                self.session_manager.conversation.add_assistant_turn(
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

            if record.status == ExecutionStatus.COMPLETED:
                self.output_store.write(AgentOutput(
                    agent_id=defn.id,
                    data=record.output,
                    status=record.status,
                    execution_id=record.execution_id,
                ))

            # Record assistant turn (children)
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
                        _convo_store = self.session_manager.get_agent_conversation_store(defn.id)
                        _convo_store.add_assistant_turn(_convo_text, execution_id=record.execution_id)
                    except Exception:
                        log.warning("Failed to record assistant turn for %s", defn.id)

            # Sync delegate registry
            result_text = ""
            total_tokens = 0
            if isinstance(record.output, dict):
                result_text = record.output.get("result", "")
                total_tokens = record.output.get("tokens_used", 0)
            delegate_node = self.delegate_registry.get_node(defn.id)
            if delegate_node is not None:
                from ..agent_registry import DelegateStatus
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
            self.registry._running_count[defn.id] = max(0, self.registry._running_count.get(defn.id, 1) - 1)
            self._executions.pop(record.execution_id, None)
            self._tasks.pop(record.execution_id, None)
            self._cancels.pop(record.execution_id, None)
            self._interrupt_hooks.pop(record.execution_id, None)

            # Provider lifecycle
            if owns_provider and sub_provider is not None:
                self.provider_manager._active_providers.pop(defn.id, None)
                try:
                    await sub_provider.close()
                except Exception:
                    pass

            if self.registry._running_count.get(defn.id, 0) == 0:
                self.registry._last_idle[defn.id] = datetime.now(timezone.utc)
                if record.status == ExecutionStatus.FAILED:
                    self.registry._status[defn.id] = AgentStatus.ERROR
                elif self.registry._status.get(defn.id) == AgentStatus.RUNNING:
                    self.registry._status[defn.id] = AgentStatus.ACTIVE

            # Signal delegate_done
            done_event = self._delegate_done.get(defn.id)
            if done_event:
                done_event.set()

            # Innate wake-up
            await self.registry.on_agent_completed(
                defn.id, record, self.provider_manager._active_providers,
            )

            if record.status == ExecutionStatus.FAILED:
                await self.registry.notify_parent_of_failure(
                    defn.id, record, self.provider_manager._active_providers,
                )

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

    # ------------------------------------------------------------------
    # Interrupt hooks (shared with ExecutionControl)
    # ------------------------------------------------------------------

    _interrupt_hooks: dict[str, Callable] = {}

    def register_interrupt_hook(self, execution_id: str, hook: Callable) -> None:
        self._interrupt_hooks[execution_id] = hook

    def unregister_interrupt_hook(self, execution_id: str) -> None:
        self._interrupt_hooks.pop(execution_id, None)

    # ------------------------------------------------------------------
    # History helper
    # ------------------------------------------------------------------

    @staticmethod
    def _append_history_to_prompt(system_prompt: str, prior_turns: list) -> str:
        _ROLE_PREFIX = {"user": "User", "assistant": "Agent", "system": "System"}
        history_parts = []
        for turn in prior_turns:
            prefix = _ROLE_PREFIX.get(turn.role, turn.role.title())
            history_parts.append(f"{prefix}: {turn.content}")

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
# Module-level helpers
# ------------------------------------------------------------------

def _preview(obj: Any, max_len: int = 120) -> str:
    if obj is None:
        return ""
    s = str(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


def _accumulate_usage(record: ExecutionRecord, provider: str, output: dict) -> None:
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
    execution_log: ExecutionLog,
) -> None:
    child_eid = collect_output.get("execution_id")
    if not child_eid:
        return
    child_agent = collect_output.get("source", "")
    if not child_agent:
        return
    history = execution_log.get_history(child_agent, limit=20)
    child_rec = None
    for rec in reversed(history):
        if rec.execution_id == child_eid:
            child_rec = rec
            break
    if child_rec is None:
        return
    for provider, child_usage in child_rec.token_usage.items():
        existing = record.token_usage.get(provider)
        if existing is None:
            existing = TokenUsage(provider=provider)
            record.token_usage[provider] = existing
        existing.input_tokens += child_usage.input_tokens
        existing.output_tokens += child_usage.output_tokens
        existing.cache_read_tokens += child_usage.cache_read_tokens
        existing.cache_creation_tokens += child_usage.cache_creation_tokens
