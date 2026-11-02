"""What an external agent did through the daemon, for the owner to see.

An external agent's brain runs outside the daemon (Claude Code, Codex, any
MCP client), so the daemon has no chat or execution history for it. What it
does see is every call the agent makes through the MCP proxy. Each call is
kept here, per agent, in ``<data_dir>/activity/<agent_id>.jsonl`` (the newest
``KEEP`` entries), and the app shows them in the agent's Chat tab.

Previews are truncated, and nothing here is consensus state.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

KEEP = 500
PREVIEW_CHARS = 600
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def _preview(value: Any) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(
            value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "..."
    return text


class ActivityLog:
    def __init__(self, data_dir: Path):
        self._dir = Path(data_dir) / "activity"
        self._lock = threading.Lock()

    def _path(self, agent_id: str) -> Path:
        return self._dir / f"{_SAFE_ID.sub('_', agent_id)}.jsonl"

    def record(self, agent_id: str, tool: str, args: dict[str, Any],
               ok: bool, result: Any, started: float, ended: float) -> dict:
        args = {k: v for k, v in (args or {}).items() if k != "caller_id"}
        entry = {
            "ts": ended,
            "tool": tool,
            "ok": bool(ok),
            "duration_ms": int(max(0.0, ended - started) * 1000),
            "input": _preview(args),
            "result": _preview(result),
        }
        path = self._path(agent_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._trim(path)
        return entry

    def _trim(self, path: Path) -> None:
        # Rewrite only once the file holds twice what we keep.
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) > 2 * KEEP:
            path.write_text("\n".join(lines[-KEEP:]) + "\n", encoding="utf-8")

    def read(self, agent_id: str, limit: int = 200) -> list[dict]:
        """Newest first."""
        path = self._path(agent_id)
        with self._lock:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
        out = []
        for line in reversed(lines[-max(1, min(limit, KEEP)):]):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def now() -> float:
    return time.time()
