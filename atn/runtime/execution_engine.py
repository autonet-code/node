"""Pipeline + cognitive execution paths.

Owns: _executions, _tasks, _cancels (shared with ExecutionControl)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

from ..events import Event, EventBus, EventType
from ..providers.base import classify_model as _classify_model
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
from ..run_summary import extract_run_summary
from ..shell_tools import SHELL_TOOLS as _SHELL_TOOLS, SHELL_TOOL_EXECUTORS as _SHELL_TOOL_EXECUTORS

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .provider_manager import ProviderManager
    from .session_manager import SessionManager
    from ..connectors_manager import ConnectorManager
    from ..agent_registry import DelegateRegistry

log = logging.getLogger(__name__)


class _BudgetExceeded(Exception):
    """Flow control exception for budget exceeded condition."""
    pass


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

        # Trace logger (set by Runtime after construction; may be None)
        self.trace_logger: Any = None

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    async def trigger_run(self, agent_id: str, source: str = "user") -> str | None:
        self.registry._require_agent(agent_id)
        defn = self.registry._agents[agent_id]

        # Check if agent is budget-paused
        status = self.registry._status.get(agent_id)
        if status == AgentStatus.BUDGET_PAUSED:
            return None  # Can't run — budget exceeded

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

        # These tasks are fire-and-forget; without this, an exception escaping
        # the body (e.g. inside the finally, after the except clauses have run)
        # is silently swallowed by asyncio and the execution hangs with no
        # completion event. Surface it.
        def _log_task_exc(t: asyncio.Task, _aid: str = agent_id, _eid: str = eid) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                log.error("execution task for %s (%s) crashed: %r", _aid, _eid, exc,
                          exc_info=(type(exc), exc, exc.__traceback__))
        task.add_done_callback(_log_task_exc)

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
        # Surface-contributed tools (read channel history, etc.) route back to
        # whichever active surface offers them.
        if name.startswith("surface_"):
            rt_ref = getattr(self, "_runtime_ref", None)
            if rt_ref is not None and hasattr(rt_ref, "active_surfaces"):
                for surface in rt_ref.active_surfaces():
                    try:
                        if any(t.get("name") == name for t in (surface.agent_tools(agent_id) or [])):
                            return await surface.call_surface_tool(name, tool_input, agent_id)
                    except Exception as exc:
                        return {"error": f"surface tool {name} failed: {exc}"}
            return {"error": f"no surface offers tool: {name}"}
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
        from ..orchestrator.tools import resolve_tool_surface

        sub_provider = None
        owns_provider = True

        try:
            # --- Provider lifecycle (unified: all agents reuse existing provider) ---
            sub_provider = self.provider_manager._active_providers.get(defn.id)
            if sub_provider is None:
                sub_provider = self.provider_manager.resolve_provider_with_fallback(defn)
                owns_provider = True
            else:
                owns_provider = False

            sub_provider.event_bus = self.events
            sub_provider.source_agent_id = defn.id
            self.provider_manager._active_providers[defn.id] = sub_provider

            # Register interrupt hook
            self.register_interrupt_hook(record.execution_id, sub_provider.interrupt)

            # Pre-execution budget check
            provider_name = getattr(sub_provider, 'name', 'claude_max')
            ok, blocker = self.registry.check_budget(defn.id, provider_name)
            if not ok:
                record.status = ExecutionStatus.FAILED
                record.error = f"Budget exceeded (blocked by {blocker})"
                raise _BudgetExceeded()

            # --- System prompt ---
            # When we build the prompt from the delegate template, identity is
            # NOT in the (cacheable) system prompt — it's delivered in the first
            # user message so the prefix stays byte-identical across agents and
            # hits the prompt cache. Agents with a custom system_prompt own
            # their identity, so we don't inject for them.
            _inject_identity = False
            if defn.system_prompt:
                system_prompt = defn.system_prompt
            else:
                system_prompt = build_delegate_prompt(
                    defn.agent_type, defn.id, defn.parent_id,
                )
                _inject_identity = True

            # --- Constitutional preamble (registered agents only) ---
            # Injected by the runtime, not user-modifiable.  The constitution
            # text comes from the on-chain Registry and is cached by the
            # AutonetBridge.  Only agents with registered_on_chain=True get
            # the preamble — unregistered agents operate without it.
            if (defn.identity and defn.identity.registered_on_chain
                    and hasattr(self, '_autonet_bridge') and self._autonet_bridge):
                constitution = self._autonet_bridge.constitution_text
                if constitution:
                    from ..delegate_prompts import build_constitutional_preamble
                    preamble = build_constitutional_preamble(constitution)
                    system_prompt = preamble + system_prompt

            # --- Tool surface ---
            agent_tools = resolve_tool_surface(defn.tools or [])

            # Whether this agent should also get the SDK's native built-in
            # tools (Bash/Read/Write/Edit/Glob/Grep/WebSearch/...). These are
            # heavy in first-turn context, so they're OFF unless the agent's
            # definition explicitly asks via "sdk_builtin". A lean agent (no
            # sdk_builtin) runs with just the ATN MCP tools, much smaller prefix.
            native_tools = "sdk_builtin" in (defn.tools or [])

            from ..providers.bridge import BridgeProvider as _BP
            if not isinstance(sub_provider, _BP):
                agent_tools.extend(_SHELL_TOOLS)

            # Start this agent's declared connectors and add their tools. The
            # pipeline path does this too; cognitive agents need it for the
            # same reason — e.g. the support agent's dorg_members connector.
            # get_all_tools returns ToolDefinition objects; the bridge surface
            # is list[dict], so convert.
            if self.connectors and defn.connector_ids:
                try:
                    await self.connectors.ensure_started(defn.connector_ids)
                except Exception:
                    log.warning("connector start failed for %s (ids=%s)",
                                defn.id, defn.connector_ids, exc_info=True)
                agent_tools.extend(
                    {"name": td.name, "description": td.description,
                     "input_schema": td.input_schema}
                    for td in self.connectors.get_all_tools(defn.connector_ids)
                )

            # Surface-contributed tools — a Surface (e.g. the chat Surface) can
            # give an agent it binds the ability to act back on its channel
            # (e.g. read channel history). surface_-prefixed; routed back to the
            # surface in route_tool_call.
            rt_ref = getattr(self, "_runtime_ref", None)
            if rt_ref is not None and hasattr(rt_ref, "active_surfaces"):
                for surface in rt_ref.active_surfaces():
                    try:
                        agent_tools.extend(surface.agent_tools(defn.id) or [])
                    except Exception:
                        log.debug("surface.agent_tools failed", exc_info=True)

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

            # Prepend the identity header to the FIRST user message only (fresh
            # session, no prior turns) — keeps it out of the cached system
            # prefix while still orienting the agent. On resumed sessions the
            # agent already has it in history.
            if _inject_identity and not getattr(sub_provider, "_session_id", ""):
                from ..delegate_prompts import build_identity_header
                _ident = build_identity_header(
                    agent_id=defn.id,
                    agent_type=defn.agent_type,
                    parent_id=defn.parent_id,
                )
                user_message = f"{_ident}\n\n{user_message}"

            # --- Session resume / history (unified for all agents) ---
            agent_convo = self.session_manager.get_agent_conversation_store(defn.id)
            session_id = getattr(sub_provider, '_session_id', "") or ""
            # Record the user message (skip if already recorded by
            # ws_server or prior path to avoid duplicates).
            existing_turns = agent_convo.get_turns()
            if (not existing_turns
                    or existing_turns[-1].role != "user"
                    or existing_turns[-1].content != user_message):
                agent_convo.add_user_turn(user_message)

            # Non-session providers need history injected (session providers
            # have it in the SDK's conversation already).
            if not session_id:
                prior_turns = agent_convo.get_turns()
                # Exclude the current message so it doesn't appear twice
                if (prior_turns
                        and prior_turns[-1].role == "user"
                        and prior_turns[-1].content == user_message):
                    prior_turns = prior_turns[:-1]
                if prior_turns:
                    system_prompt = self._append_history_to_prompt(system_prompt, prior_turns)

            # --- Inject UTC time ---
            from datetime import timezone as _tz
            now = datetime.now(_tz.utc)
            time_line = f"Current time: {now.strftime('%Y-%m-%dT%H:%M:%SZ')} ({now.strftime('%A, %B %d, %Y')})"
            user_message = f"[{time_line}]\n\n{user_message}"

            # --- Tool executor (with call accumulation for run summary) ---
            _accumulated_tool_calls: list[dict[str, Any]] = []

            async def _tool_executor(name: str, tool_input: dict) -> dict:
                result = await self.route_tool_call(name, tool_input, defn.id)
                is_error = isinstance(result, dict) and "error" in result
                _accumulated_tool_calls.append({
                    "tool": name,
                    "args": tool_input,
                    "result": str(result)[:4000],
                    "success": not is_error,
                })
                return result

            # --- Streaming callback ---
            _streamed = {"text": False}

            async def _on_chunk(text: str) -> None:
                if text:
                    _streamed["text"] = True
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

            # --- Trace logging: record execution start ---
            if self.trace_logger is not None:
                self.trace_logger.begin_execution(
                    agent_id=defn.id,
                    execution_id=record.execution_id,
                    agent_type=defn.agent_type or "cognitive",
                    system_prompt=system_prompt,
                    # Store the raw user message before history prepending for cleaner traces
                    user_message="\n\n".join(prompt_parts) if prompt_parts else (defn.description or defn.name),
                )

            # --- Per-turn budget recorder (inner-loop enforcement) ---
            # Wired into send_orchestrate so each turn's tokens roll into the
            # cascading budget immediately. Returns (ok, blocker_id); on
            # ok=False the provider aborts the loop with stop_reason=budget_exceeded.
            _budget_provider_key = getattr(sub_provider, "name", "claude_max")
            # Track tokens recorded mid-loop so the post-loop reconciliation
            # only adds the *unrecorded* remainder. The base provider calls
            # this every turn; the bridge provider's stream events also call it
            # via _stream_events. Either way, the cascading counter sees the
            # right total (post-loop block subtracts already_recorded).
            _recorded_inflight: dict[str, int] = {}

            def _per_turn_recorder(
                turn_tokens: float,
                model: str = "",
            ) -> tuple[bool, str | None]:
                if turn_tokens <= 0:
                    return True, None
                _recorded_inflight[_budget_provider_key] = (
                    _recorded_inflight.get(_budget_provider_key, 0) + turn_tokens
                )
                from ..model_specs import resolve as _resolve_model
                spec = _resolve_model(model)
                # Per-model rate from the bridge estimator; all versions of a
                # class share a rate (Anthropic doesn't break out by version).
                rate_for_model = None
                if hasattr(sub_provider, "tokens_per_pct_for_model"):
                    rate_for_model = sub_provider.tokens_per_pct_for_model(spec.id)
                # Provide the rate under both class- and id-keys so the recorder
                # can resolve regardless of how it looks up.
                tpp = {spec.klass: rate_for_model} if rate_for_model else None
                exceeded = self.registry.record_token_usage(
                    defn.id, _budget_provider_key, turn_tokens,
                    model_class=spec.klass,
                    model_id=spec.id if spec.id != "default" else "",
                    tokens_per_pct=tpp,
                )
                if exceeded:
                    return False, exceeded
                return True, None

            # --- Run the agent ---
            send_kwargs: dict[str, Any] = {
                "message": user_message,
                "system": system_prompt,
                "tools": agent_tools,
                "native_tools": native_tools,
                "max_turns": defn.max_turns,
                "tool_executor": _tool_executor,
                "on_chunk": _on_chunk,
                "usage_recorder": _per_turn_recorder,
            }
            if session_id:
                send_kwargs["session_id"] = session_id
            # Long-horizon safeguards — providers that support them honour these;
            # bridge providers ignore unknown kwargs in their override.
            if defn.per_turn_input_max is not None:
                send_kwargs["per_turn_input_max"] = defn.per_turn_input_max
            if defn.repeat_call_limit is not None:
                send_kwargs["repeat_call_limit"] = defn.repeat_call_limit

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
            elif response.stop_reason == "budget_exceeded":
                # Budget was hit. Distinguish "produced an answer, then went
                # over" from "aborted mid-orchestration with nothing to show".
                # A run that actually answered is COMPLETED — don't bury the
                # answer in the error field and report it as FAILED. The budget
                # breach is surfaced separately (BUDGET_EXCEEDED event +
                # BUDGET_PAUSED status in the reconciliation block below).
                if result_text.strip():
                    record.status = ExecutionStatus.COMPLETED
                    record.error = None
                else:
                    record.status = ExecutionStatus.FAILED
                    record.error = "Budget exceeded before any output was produced"
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

            # If the answer never streamed as chunks (short single-block replies
            # often arrive only in the final response), emit it once as a text
            # event so live consumers (chat Surface embed, voice) get the full
            # answer instead of "(no text output)".
            if result_text and not _streamed["text"]:
                await self.events.emit(Event(
                    type=EventType.STEP_OUTPUT,
                    source=defn.id,
                    data={
                        "agent_id": defn.id,
                        "channel": "text",
                        "content": result_text,
                        "cognitive": True,
                        "final": True,
                    },
                ))

            provider_key = getattr(sub_provider, 'name', 'claude_max')
            if provider_key not in record.token_usage:
                record.token_usage[provider_key] = TokenUsage(provider=provider_key)
            record.token_usage[provider_key].input_tokens += response.usage.input_tokens
            record.token_usage[provider_key].output_tokens += response.usage.output_tokens
            record.token_usage[provider_key].cache_read_tokens += response.usage.cache_read_tokens
            record.token_usage[provider_key].cache_creation_tokens += response.usage.cache_creation_tokens

            # Record assistant turn to conversation store (unified for all agents).
            # For the root agent, get_agent_conversation_store returns the
            # central store visible in the UI.
            if result_text:
                agent_convo.add_assistant_turn(
                    result_text, execution_id=record.execution_id,
                )

            # Sync the provider's turn count with the conversation store
            # (the real source of truth that includes history from disk).
            all_turns = agent_convo.get_turns()
            sub_provider._cumulative_turns = len(all_turns)

            # Persist session stats to disk so they survive restarts
            if hasattr(sub_provider, 'session_stats'):
                agent_convo.save_session_stats(sub_provider.session_stats)

            # Reconciliation: subscription providers (bridge) compare predicted
            # vs actual subscription burn after each orchestration to keep the
            # tokens-per-pct estimator honest. Best-effort — failures here
            # never affect the user-visible execution result.
            if hasattr(sub_provider, "reconcile_after_orchestration"):
                try:
                    await sub_provider.reconcile_after_orchestration()
                except Exception:
                    log.exception("reconcile_after_orchestration failed; continuing")

        except asyncio.CancelledError:
            record.status = ExecutionStatus.KILLED
            record.error = "Force-cancelled by kill switch"
        except _BudgetExceeded:
            # Already handled — status and error set before raising
            pass
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

            # Record usage in cascading budget system. Subtract anything the
            # inner-loop recorder already booked so we don't double-count.
            # Meter REAL SPEND (Usage.budget_tokens), consistent with the
            # inner-loop recorder — NOT the raw 4-bucket total, which is context
            # size and would re-introduce the cache_read we deliberately
            # down-weight (and false-fail sane caps).
            for provider_key, usage in record.token_usage.items():
                total = usage.budget_tokens()
                already_recorded = _recorded_inflight.get(provider_key, 0)
                remainder = total - already_recorded
                if remainder > 0:
                    exceeded = self.registry.record_token_usage(defn.id, provider_key, remainder)
                    if exceeded:
                        # Auto-pause the exceeded agent
                        self.registry._status[exceeded] = AgentStatus.BUDGET_PAUSED
                        await self.events.emit(Event(
                            type=EventType.BUDGET_EXCEEDED,
                            source=exceeded,
                            data={
                                "agent_id": exceeded,
                                "triggered_by": defn.id,
                                "provider": provider_key,
                            },
                        ))

            # --- Trace logging: finalise trace ---
            if self.trace_logger is not None:
                _trace_result = ""
                if isinstance(record.output, dict):
                    _trace_result = record.output.get("result", "")
                elif record.output:
                    _trace_result = str(record.output)
                self.trace_logger.end_execution(
                    agent_id=defn.id,
                    execution_id=record.execution_id,
                    result_text=_trace_result,
                    status=record.status.value,
                    error=record.error,
                    completed_at=record.completed_at,
                )

            # Record error in conversation if the try block didn't get to
            # record the assistant turn (exception path).
            if record.error and record.status != ExecutionStatus.COMPLETED:
                try:
                    _err_store = self.session_manager.get_agent_conversation_store(defn.id)
                    _err_store.add_assistant_turn(
                        f"Error: {record.error}", execution_id=record.execution_id,
                    )
                except Exception:
                    log.warning("Failed to record error turn for %s", defn.id)

            # Sync delegate registry
            result_text = ""
            total_tokens = 0
            if isinstance(record.output, dict):
                result_text = record.output.get("result", "")
                total_tokens = record.output.get("tokens_used", 0)

            # Build deterministic run summary from accumulated tool calls
            run_summary = ""
            try:
                run_summary = extract_run_summary(
                    tool_calls=_accumulated_tool_calls,
                    max_turns=defn.max_turns,
                    actual_turns=len([
                        tc for tc in _accumulated_tool_calls
                    ]) if _accumulated_tool_calls else None,
                    status=record.status.value if record.status else "",
                    error=record.error,
                )
            except Exception:
                log.debug("Failed to extract run summary for %s", defn.id, exc_info=True)

            # Store run summary in output for downstream consumers
            if run_summary and isinstance(record.output, dict):
                record.output["run_summary"] = run_summary

            # Combine: deterministic summary + agent's last response
            if run_summary and result_text:
                combined_preview = f"{run_summary}\n\n---\n{result_text}"
            elif run_summary:
                combined_preview = run_summary
            else:
                combined_preview = result_text

            delegate_node = self.delegate_registry.get_node(defn.id)
            if delegate_node is not None:
                from ..agent_registry import DelegateStatus
                if record.status == ExecutionStatus.COMPLETED:
                    self.delegate_registry.update_status(
                        defn.id, DelegateStatus.COMPLETED,
                        result_preview=combined_preview[:2000],
                        tokens_used=total_tokens,
                    )
                elif record.status == ExecutionStatus.KILLED:
                    self.delegate_registry.update_status(
                        defn.id, DelegateStatus.KILLED,
                        result_preview=combined_preview[:2000],
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

            # Provider lifecycle: keep the provider alive in _active_providers
            # so it is reused across executions. This preserves session state,
            # prompt caching, and allows session_stats to be fetched after
            # execution completes. Cleanup happens on explicit reset
            # (new_conversation, agent unregistered) not after each execution.

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
