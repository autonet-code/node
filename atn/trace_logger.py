"""Structured trace logging for agent sessions.

Captures agent execution traces in the format required for VL-JEPA training.
Each completed cognitive execution produces a JSON trace file:

    {
        "session_id": "...",
        "agent_id": "...",
        "agent_type": "orchestrator | solver | probe | ...",
        "timestamp": "...",
        "turns": [
            {"role": "system", "content": "...", "tool_calls": [], "timestamp": "..."},
            {"role": "user",   "content": "...", "tool_calls": [], "timestamp": "..."},
            {"role": "assistant", "content": "...", "tool_calls": [...], "timestamp": "..."}
        ],
        "outcome": {
            "success": true,
            "task_completed": true,
            "errors": []
        }
    }

See C:\\code\\research\\AGENT_TRACE_TRAINING.md for the full specification.

Phase B Step 1 — Agent Trace Collection.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

log = logging.getLogger(__name__)

# Maximum bytes for any single turn's content (prevents traces from becoming huge)
_MAX_CONTENT_BYTES = 1_048_576  # 1 MB


@dataclass
class TraceLoggingConfig:
    """Configuration for structured trace logging.

    Add a ``trace_logging`` section to config.yaml to enable:

        trace_logging:
          enabled: true
          trace_dir: ~/.atn/traces   # default: {data_dir}/traces
          include_user_data: false   # consent gate; set true to include orchestrator sessions
          min_turns: 1               # quality filter: skip if fewer assistant turns
    """
    enabled: bool = False
    # Empty string means "use {data_dir}/traces/" (resolved at runtime)
    trace_dir: str = ""
    # Consent gate per AGENT_TRACE_TRAINING.md §Data Privacy and Epic 3 Story 3.1.
    # Orchestrator sessions contain user messages; only stored when this is True.
    include_user_data: bool = False
    # Quality filter: minimum number of assistant turns with content or tool calls.
    min_turns: int = 1


@dataclass
class _PendingTrace:
    """In-progress trace assembled during a single cognitive execution."""
    agent_id: str
    execution_id: str
    agent_type: str
    system_prompt: str
    user_message: str
    started_at: str  # ISO-8601 UTC
    # Tool calls keyed by tool_use_id while waiting for results
    _calls_in_flight: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Completed tool call records in arrival order
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class TraceLogger:
    """Records structured agent traces for VL-JEPA training data.

    Lifecycle:
        1. Created by Runtime.__init__ from config.
        2. attach(event_bus) subscribes to AGENT_TOOL_USE_* events.
        3. begin_execution(...) called at the start of each cognitive execution.
        4. end_execution(...) called in the finally block of each cognitive execution.
        5. Completed traces are quality-filtered and written to content-addressed storage.

    Tool calls are correlated per-agent: because an agent can only run one
    execution at a time, tool-use events from an agent map directly to its
    current pending trace.

    All methods run on the asyncio event loop — no locking required.
    """

    def __init__(self, config: TraceLoggingConfig, trace_dir: Path) -> None:
        self.config = config
        self._trace_dir = trace_dir
        # In-progress traces: agent_id -> _PendingTrace
        self._pending: dict[str, _PendingTrace] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def attach(self, event_bus: Any) -> None:
        """Subscribe to tool-use events on the given EventBus.

        Must be called before any executions start (typically in Runtime.start()).
        """
        from .events import EventType
        event_bus.subscribe(EventType.AGENT_TOOL_USE_START, self._on_tool_use_start)
        event_bus.subscribe(EventType.AGENT_TOOL_USE_RESULT, self._on_tool_use_result)
        log.info("TraceLogger attached (trace_dir=%s, enabled=%s)", self._trace_dir, self.config.enabled)

    # ------------------------------------------------------------------
    # Execution lifecycle hooks  (called from ExecutionEngine)
    # ------------------------------------------------------------------

    def begin_execution(
        self,
        agent_id: str,
        execution_id: str,
        agent_type: str,
        system_prompt: str,
        user_message: str,
    ) -> None:
        """Called at the start of a cognitive execution.

        Records the initial trace context.  Tool-use events that arrive
        between begin_execution() and end_execution() are accumulated under
        this agent's pending trace.

        No-op when trace logging is disabled.
        """
        if not self.config.enabled:
            return
        self._pending[agent_id] = _PendingTrace(
            agent_id=agent_id,
            execution_id=execution_id,
            agent_type=agent_type,
            system_prompt=system_prompt,
            user_message=user_message,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def end_execution(
        self,
        agent_id: str,
        execution_id: str,
        result_text: str,
        status: str,
        error: Optional[str],
        completed_at: Optional[datetime],
    ) -> None:
        """Called in the finally block of a cognitive execution.

        Builds the structured trace from accumulated data and writes it to
        the trace store if it passes quality and consent filters.

        No-op when trace logging is disabled or no pending trace exists.
        """
        if not self.config.enabled:
            return
        pending = self._pending.pop(agent_id, None)
        if pending is None or pending.execution_id != execution_id:
            return

        trace = self._build_trace(pending, result_text, status, error, completed_at)
        if trace is None:
            return

        if not self._passes_quality_filter(trace):
            log.debug(
                "TraceLogger: skipping low-quality trace for %s/%s",
                agent_id, execution_id,
            )
            return

        self._write_trace(trace)

    # ------------------------------------------------------------------
    # EventBus handlers
    # ------------------------------------------------------------------

    async def _on_tool_use_start(self, event: Any) -> None:
        """Accumulate a tool call start from the EventBus."""
        data = event.data
        agent_id = data.get("agent_id", "")
        tool_use_id = data.get("tool_use_id", "")
        if not agent_id or not tool_use_id:
            return
        pending = self._pending.get(agent_id)
        if pending is None:
            return
        pending._calls_in_flight[tool_use_id] = {
            "tool": data.get("tool_name", ""),
            "args": data.get("input", {}),
            "result": "",
            "success": True,
        }

    async def _on_tool_use_result(self, event: Any) -> None:
        """Update an in-flight tool call with its result."""
        data = event.data
        agent_id = data.get("agent_id", "")
        tool_use_id = data.get("tool_use_id", "")
        if not agent_id or not tool_use_id:
            return
        pending = self._pending.get(agent_id)
        if pending is None:
            return
        call = pending._calls_in_flight.pop(tool_use_id, None)
        if call is None:
            return
        is_error = data.get("is_error", False)
        result = data.get("result", data.get("content", ""))
        call["result"] = str(result)[:_MAX_CONTENT_BYTES]
        call["success"] = not is_error
        pending.tool_calls.append(call)

    # ------------------------------------------------------------------
    # Trace construction
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        pending: _PendingTrace,
        result_text: str,
        status: str,
        error: Optional[str],
        completed_at: Optional[datetime],
    ) -> Optional[dict[str, Any]]:
        """Assemble the structured trace dict from a completed execution."""
        turns: list[dict[str, Any]] = []

        # System turn — agent's constitutional prompt / instruction
        if pending.system_prompt:
            turns.append({
                "role": "system",
                "content": pending.system_prompt[:_MAX_CONTENT_BYTES],
                "tool_calls": [],
                "timestamp": pending.started_at,
            })

        # User turn — the task / user message
        if pending.user_message:
            turns.append({
                "role": "user",
                "content": pending.user_message[:_MAX_CONTENT_BYTES],
                "tool_calls": [],
                "timestamp": pending.started_at,
            })

        # Assistant turn — reasoning + tool calls + final response
        assistant_content = (result_text or "")[:_MAX_CONTENT_BYTES]
        end_ts = completed_at.isoformat() if completed_at else pending.started_at
        if assistant_content or pending.tool_calls:
            turns.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": pending.tool_calls,
                "timestamp": end_ts,
            })

        success = status == "completed"
        outcome: dict[str, Any] = {
            "success": success,
            "task_completed": success,
            "errors": [error] if error else [],
        }

        return {
            "session_id": uuid4().hex[:12],
            "agent_id": pending.agent_id,
            "agent_type": pending.agent_type or "cognitive",
            "timestamp": pending.started_at,
            "turns": turns,
            "outcome": outcome,
        }

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def _passes_quality_filter(self, trace: dict[str, Any]) -> bool:
        """Return True if the trace meets minimum quality standards.

        Checks:
        1. At least ``min_turns`` assistant turns with content or tool calls.
        2. Consent gate: orchestrator (user conversation) traces require
           ``include_user_data=True`` (per Epic 3 Story 3.1).
        """
        turns = trace.get("turns", [])

        # Count assistant turns that have actual content or tool calls
        meaningful_assistant_turns = sum(
            1 for t in turns
            if t.get("role") == "assistant"
            and (t.get("content") or t.get("tool_calls"))
        )
        if meaningful_assistant_turns < self.config.min_turns:
            return False

        # Consent gate: orchestrator sessions contain user data
        # Only store if the operator has opted in
        from .orchestrator import ORCHESTRATOR_ID
        if trace.get("agent_id") == ORCHESTRATOR_ID and not self.config.include_user_data:
            return False

        return True

    # ------------------------------------------------------------------
    # Content-addressed storage
    # ------------------------------------------------------------------

    def _write_trace(self, trace: dict[str, Any]) -> Optional[Path]:
        """Write a trace to the trace store.

        Uses content-addressed storage: the filename is derived from the
        SHA-256 of the JSON payload so identical traces are deduplicated
        automatically.

        Returns the path that was written, or None if skipped or failed.
        """
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.warning("TraceLogger: cannot create trace_dir %s", self._trace_dir)
            return None

        try:
            payload = json.dumps(trace, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            log.warning(
                "TraceLogger: cannot serialize trace for agent=%s",
                trace.get("agent_id", "?"),
            )
            return None

        # Content-addressed filename: first 24 hex chars of SHA-256
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        filename = f"{content_hash[:24]}.json"
        path = self._trace_dir / filename

        if path.exists():
            log.debug("TraceLogger: duplicate trace %s, skipping", filename)
            return None

        try:
            path.write_text(payload, encoding="utf-8")
        except Exception:
            log.warning("TraceLogger: failed to write %s", path, exc_info=True)
            return None

        tool_call_count = sum(
            len(t.get("tool_calls", [])) for t in trace.get("turns", [])
        )
        log.info(
            "TraceLogger: stored %s (agent=%s, %d tool calls, %d bytes)",
            filename,
            trace.get("agent_id", "?"),
            tool_call_count,
            len(payload),
        )
        return path
