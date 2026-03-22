"""Output store and execution log.

OutputStore  — per-agent *last-run result*.  This is the structured data other
               agents consume via pull steps.

ExecutionLog — per-agent *full trace history*.  Persisted to JSONL on disk.
               This is what the user (and the orchestrator, when debugging) inspects.

Disk layout (directory-per-agent model):
    agents_dir/<agent_id>/execution.jsonl            Completed execution records
    agents_dir/<agent_id>/<execution_id>.running.json  In-progress state (crash resilience)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ExecutionRecord, ExecutionStatus, StepResult, StepType, TokenUsage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output store
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """The meaningful result of an agent's last run."""
    agent_id: str
    data: Any
    status: ExecutionStatus
    execution_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OutputStore:
    """In-memory store of each agent's latest output."""

    def __init__(self) -> None:
        self._outputs: dict[str, AgentOutput] = {}

    def write(self, output: AgentOutput) -> None:
        self._outputs[output.agent_id] = output

    def read(self, agent_id: str) -> AgentOutput | None:
        return self._outputs.get(agent_id)

    def remove(self, agent_id: str) -> None:
        self._outputs.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Execution log — JSONL-backed with in-memory cache
# ---------------------------------------------------------------------------

class ExecutionLog:
    """Per-agent execution history.

    In-memory ring-buffer for fast access, backed by append-only JSONL files
    on disk for persistence and crash safety.

    Disk layout (directory-per-agent):
        agents_dir/<agent_id>/execution.jsonl
        agents_dir/<agent_id>/<execution_id>.running.json

    Each line in the JSONL is one completed ExecutionRecord serialised as JSON.
    """

    def __init__(self, agents_dir: Path | None = None, *, data_dir: Path | None = None, max_per_agent: int = 50) -> None:
        self._logs: dict[str, list[ExecutionRecord]] = {}
        self._max = max_per_agent
        self._agents_dir: Path | None = agents_dir

        # Legacy fallback: if only data_dir provided, use data_dir/execution/
        # (backward compat for callers that haven't been updated yet)
        if self._agents_dir is None and data_dir is not None:
            self._agents_dir = data_dir / "execution"
            self._agents_dir.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> Path | None:
        """Resolve the directory for an agent's execution files."""
        if self._agents_dir is None:
            return None
        return self._agents_dir / agent_id

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, execution: ExecutionRecord) -> None:
        """Insert or update an execution record in-memory."""
        logs = self._logs.setdefault(execution.agent_id, [])
        for i, existing in enumerate(logs):
            if existing.execution_id == execution.execution_id:
                logs[i] = execution
                return
        logs.append(execution)
        if len(logs) > self._max:
            del logs[0]

    def persist(self, execution: ExecutionRecord) -> None:
        """Append a completed execution to the JSONL file on disk and clean up any in-progress file."""
        agent_dir = self._agent_dir(execution.agent_id)
        if agent_dir is None:
            return
        agent_dir.mkdir(parents=True, exist_ok=True)
        path = agent_dir / "execution.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_record_to_dict(execution)) + "\n")
        except Exception:
            log.exception("Failed to persist execution %s", execution.execution_id)

        # Remove in-progress file now that the final record is in the JSONL
        running_path = agent_dir / f"{execution.execution_id}.running.json"
        try:
            running_path.unlink(missing_ok=True)
        except Exception:
            pass

    def persist_running(self, execution: ExecutionRecord) -> None:
        """Write current in-progress state to a running file (overwritten each step).

        This provides crash resilience: if the process dies mid-pipeline,
        the running file captures the last known state.  On final completion,
        persist() appends to JSONL and deletes this file.
        """
        agent_dir = self._agent_dir(execution.agent_id)
        if agent_dir is None:
            return
        agent_dir.mkdir(parents=True, exist_ok=True)
        running_path = agent_dir / f"{execution.execution_id}.running.json"
        try:
            data = json.dumps(_record_to_dict(execution), indent=2)
            running_path.write_text(data, encoding="utf-8")
        except Exception:
            log.exception("Failed to write running state for %s", execution.execution_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_latest(self, agent_id: str) -> ExecutionRecord | None:
        logs = self._logs.get(agent_id, [])
        return logs[-1] if logs else None

    def get_history(self, agent_id: str, limit: int = 10) -> list[ExecutionRecord]:
        return self._logs.get(agent_id, [])[-limit:]

    def get_all_running(self) -> list[ExecutionRecord]:
        return [
            rec
            for logs in self._logs.values()
            for rec in logs
            if rec.status == ExecutionStatus.RUNNING
        ]

    def get_by_id(self, execution_id: str) -> ExecutionRecord | None:
        """Look up an execution record by its execution ID (linear scan)."""
        for logs in self._logs.values():
            for rec in logs:
                if rec.execution_id == execution_id:
                    return rec
        return None

    def hydrate(self, agent_id: str) -> int:
        """Load history from JSONL into the in-memory cache.  Returns count loaded.

        Called at startup so that get_latest / get_history work immediately
        without waiting for new executions.  Respects the max_per_agent limit.
        """
        records = self.load_from_disk(agent_id)
        if not records:
            return 0
        self._logs[agent_id] = records[-self._max:]
        return len(records)

    def load_from_disk(self, agent_id: str) -> list[ExecutionRecord]:
        """Load full history from JSONL file.  Does NOT update in-memory cache."""
        agent_dir = self._agent_dir(agent_id)
        if agent_dir is None:
            return []
        path = agent_dir / "execution.jsonl"
        if not path.exists():
            return []
        records: list[ExecutionRecord] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_dict_to_record(json.loads(line)))
            except Exception:
                log.warning("Corrupt line %d in %s, skipping", line_no, path)
        return records

    def recover_running(self) -> list[ExecutionRecord]:
        """Load any orphaned in-progress executions from .running.json files.

        Called at startup to detect pipelines that crashed mid-execution.
        Scans all agent directories for .running.json files.
        Returns the recovered records (marked as FAILED) and deletes the files.
        """
        if self._agents_dir is None:
            return []
        recovered: list[ExecutionRecord] = []
        if not self._agents_dir.exists():
            return recovered
        for child in self._agents_dir.iterdir():
            if not child.is_dir():
                continue
            for path in child.glob("*.running.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rec = _dict_to_record(data)
                    rec.status = ExecutionStatus.FAILED
                    rec.error = rec.error or "Process crashed mid-execution (recovered from running file)"
                    rec.completed_at = rec.completed_at or datetime.now(timezone.utc)
                    recovered.append(rec)
                    # Persist as failed to JSONL, then remove running file
                    self.record(rec)
                    self.persist(rec)
                    log.warning(
                        "Recovered crashed execution %s for agent %s (%d steps completed)",
                        rec.execution_id, rec.agent_id, len(rec.step_results),
                    )
                except Exception:
                    log.warning("Could not recover running file %s, removing", path)
                    try:
                        path.unlink()
                    except Exception:
                        pass
        return recovered

    # ------------------------------------------------------------------
    # Clear / remove
    # ------------------------------------------------------------------

    def remove_agent(self, agent_id: str) -> None:
        """Remove in-memory records (does NOT delete on-disk files)."""
        self._logs.pop(agent_id, None)

    def clear_agent(self, agent_id: str) -> bool:
        """Clear both in-memory and on-disk history for an agent.  Returns True if file existed."""
        self._logs.pop(agent_id, None)
        agent_dir = self._agent_dir(agent_id)
        if agent_dir is None:
            return False
        path = agent_dir / "execution.jsonl"
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_all(self) -> int:
        """Clear all execution history.  Returns number of files deleted."""
        self._logs.clear()
        if self._agents_dir is None:
            return 0
        count = 0
        if not self._agents_dir.exists():
            return 0
        for child in self._agents_dir.iterdir():
            if not child.is_dir():
                continue
            path = child / "execution.jsonl"
            if path.exists():
                path.unlink()
                count += 1
        return count

    def disk_summary(self) -> dict[str, int]:
        """Return {agent_id: line_count} for all execution JSONL files on disk."""
        if self._agents_dir is None:
            return {}
        summary: dict[str, int] = {}
        if not self._agents_dir.exists():
            return summary
        for child in sorted(self._agents_dir.iterdir()):
            if not child.is_dir():
                continue
            path = child / "execution.jsonl"
            if path.exists():
                summary[child.name] = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        return summary


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _dt(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


def _record_to_dict(rec: ExecutionRecord) -> dict:
    d: dict[str, Any] = {
        "execution_id": rec.execution_id,
        "agent_id": rec.agent_id,
        "status": rec.status.value,
        "trigger_source": rec.trigger_source,
        "current_step": rec.current_step,
        "output": _safe_json(rec.output),
        "error": rec.error,
        "started_at": _dt(rec.started_at),
        "completed_at": _dt(rec.completed_at),
        "step_results": [_step_to_dict(s) for s in rec.step_results],
    }
    if rec.token_usage:
        d["token_usage"] = {
            provider: {
                "provider": tu.provider,
                "input_tokens": tu.input_tokens,
                "output_tokens": tu.output_tokens,
                "cache_read_tokens": tu.cache_read_tokens,
                "cache_creation_tokens": tu.cache_creation_tokens,
            }
            for provider, tu in rec.token_usage.items()
        }
    return d


def _step_to_dict(s: StepResult) -> dict:
    return {
        "step_index": s.step_index,
        "step_name": s.step_name,
        "step_type": s.step_type.value,
        "status": s.status.value,
        "output": _safe_json(s.output),
        "error": s.error,
        "started_at": _dt(s.started_at),
        "completed_at": _dt(s.completed_at),
        "pid": s.pid,
    }


def _safe_json(obj: Any) -> Any:
    """Ensure a value is JSON-serialisable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)


def _parse_dt(v: str | None) -> datetime | None:
    if v is None:
        return None
    return datetime.fromisoformat(v)


def _dict_to_record(d: dict) -> ExecutionRecord:
    # Deserialise token_usage if present
    token_usage: dict[str, TokenUsage] = {}
    for provider, tu_dict in d.get("token_usage", {}).items():
        token_usage[provider] = TokenUsage(
            provider=tu_dict.get("provider", provider),
            input_tokens=tu_dict.get("input_tokens", 0),
            output_tokens=tu_dict.get("output_tokens", 0),
            cache_read_tokens=tu_dict.get("cache_read_tokens", 0),
            cache_creation_tokens=tu_dict.get("cache_creation_tokens", 0),
        )

    return ExecutionRecord(
        execution_id=d["execution_id"],
        agent_id=d["agent_id"],
        status=ExecutionStatus(d["status"]),
        trigger_source=d["trigger_source"],
        current_step=d.get("current_step", 0),
        output=d.get("output"),
        error=d.get("error"),
        started_at=_parse_dt(d.get("started_at")),
        completed_at=_parse_dt(d.get("completed_at")),
        step_results=[_dict_to_step(s) for s in d.get("step_results", [])],
        token_usage=token_usage,
    )


def _dict_to_step(d: dict) -> StepResult:
    return StepResult(
        step_index=d["step_index"],
        step_name=d["step_name"],
        step_type=StepType(d["step_type"]),
        status=ExecutionStatus(d["status"]),
        output=d.get("output"),
        error=d.get("error"),
        started_at=_parse_dt(d.get("started_at")),
        completed_at=_parse_dt(d.get("completed_at")),
        pid=d.get("pid"),
    )
