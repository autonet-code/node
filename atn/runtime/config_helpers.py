"""Connector credential injection, planning task persistence, and config utilities."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import PlanningTask, TaskStatus, TaskType

log = logging.getLogger(__name__)


def inject_connector_credentials(connectors: Any, credential_store: Any) -> None:
    """Load stored credentials and inject them into connectors as env vars."""
    for cid in connectors.list_available():
        creds = credential_store.load(cid)
        if creds:
            connectors.set_runtime_env(cid, {
                k: str(v) for k, v in creds.items()
            })
            log.debug("Injected credentials for connector '%s'", cid)


def load_planning_tasks(path: Path) -> list[PlanningTask]:
    """Load planning tasks from JSON file."""
    tasks: list[PlanningTask] = []
    if not path.exists():
        return tasks
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for d in raw:
            try:
                tasks.append(PlanningTask(
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
        log.warning("Failed to load planning tasks from %s", path, exc_info=True)
    return tasks


def save_planning_tasks(tasks: list[PlanningTask], path: Path) -> None:
    """Persist planning tasks to JSON file."""
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
        for t in tasks
    ]
    try:
        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
    except Exception:
        log.exception("Failed to save planning tasks")
