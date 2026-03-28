"""Orchestrator conversation persistence.

Stores the conversation history between the user and the orchestrator so that
context carries across invocations.  Each "session" is a linear sequence of
turns (user message + orchestrator response).

Disk layout:
    data_dir/conversations/
        active.jsonl          Current conversation (append-only)
        <session_id>.jsonl    Archived conversations

Each line in a JSONL file is one turn:
    {"role": "user", "content": "...", "timestamp": "..."}
    {"role": "assistant", "content": "...", "timestamp": "...", "execution_id": "..."}

The sliding window works by counting turns from the front. When the estimated
token count exceeds the budget, the oldest turns are dropped at read time —
the JSONL on disk keeps everything.  This means archived conversations are
always complete.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token for English text.
_CHARS_PER_TOKEN = 4

# Default context budget for conversation history (in tokens).
# Leave room for system prompt (~2k) + new user message + tool calls.
# Sonnet has 200k context.  We reserve 150k for history.
DEFAULT_HISTORY_TOKEN_BUDGET = 150_000


@dataclass
class ConversationTurn:
    """One turn in the conversation."""
    role: str                   # "user" or "assistant"
    content: str                # message text
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_id: str | None = None   # links assistant turns to execution records

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.execution_id:
            d["execution_id"] = self.execution_id
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ConversationTurn:
        return ConversationTurn(
            role=d["role"],
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else datetime.now(timezone.utc),
            execution_id=d.get("execution_id"),
        )


class ConversationStore:
    """Manages orchestrator conversation history with JSONL persistence.

    The "active" conversation is the current ongoing session.  When the user
    resets, the active conversation is archived (moved to <session_id>.jsonl)
    and a fresh active.jsonl is started.

    The in-memory buffer is the source of truth during runtime.  It's appended
    to disk after each turn for crash safety.
    """

    def __init__(self, data_dir: Path, *, history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET) -> None:
        self._dir = data_dir / "conversations"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._active_path = self._dir / "active.jsonl"
        self._budget = history_token_budget

        # In-memory buffer of the active conversation
        self._turns: list[ConversationTurn] = []

        # Load active conversation from disk if it exists
        self._hydrate()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_turns(self) -> list[ConversationTurn]:
        """Return all turns in the active conversation."""
        return list(self._turns)

    def get_turns_page(self, limit: int = 50, offset: int | None = None) -> tuple[list[ConversationTurn], int]:
        """Return a page of turns from the end (most recent first).

        Args:
            limit: Max number of turns to return.
            offset: Number of turns to skip from the end (0 = most recent).
                    None means start from the end.

        Returns:
            (turns_oldest_first, total_count)
        """
        total = len(self._turns)
        if offset is None:
            offset = 0
        # offset=0 means the latest `limit` turns
        end = total - offset
        start = max(0, end - limit)
        if end <= 0:
            return [], total
        return list(self._turns[start:end]), total

    def get_history_for_prompt(self, exclude_last: int = 0) -> str:
        """Build a conversation history string that fits within the token budget.

        Returns a formatted string of previous turns, newest last, trimmed
        from the front (oldest) if the total exceeds the budget.  Returns
        empty string if no history.

        Args:
            exclude_last: Number of turns to exclude from the end (e.g. 1 to
                skip the current user message that was just recorded).
        """
        turns = self._turns
        if exclude_last > 0:
            turns = turns[:-exclude_last] if len(turns) > exclude_last else []
        if not turns:
            return ""

        # Build formatted turns
        formatted: list[str] = []
        _ROLE_PREFIX = {"user": "User", "assistant": "Orchestrator", "system": "System"}
        for turn in turns:
            prefix = _ROLE_PREFIX.get(turn.role, turn.role.title())
            formatted.append(f"{prefix}: {turn.content}")

        # Estimate total tokens
        total_chars = sum(len(f) for f in formatted)
        budget_chars = self._budget * _CHARS_PER_TOKEN

        if total_chars <= budget_chars:
            return "\n\n".join(formatted)

        # Sliding window: drop oldest turns until we fit
        start = 0
        while start < len(formatted) and total_chars > budget_chars:
            total_chars -= len(formatted[start])
            start += 1

        if start >= len(formatted):
            return ""

        # Prepend a note that earlier context was trimmed
        trimmed = formatted[start:]
        return "[Earlier conversation trimmed]\n\n" + "\n\n".join(trimmed)

    def turn_count(self) -> int:
        """Number of turns in the active conversation."""
        return len(self._turns)

    def export_trace_turns(self) -> list[dict[str, Any]]:
        """Export the active conversation as structured trace turns.

        Converts the stored conversation history to the trace format used by
        TraceLogger and VL-JEPA training:

            [
                {"role": "user",      "content": "...", "tool_calls": [], "timestamp": "..."},
                {"role": "assistant", "content": "...", "tool_calls": [], "timestamp": "..."},
            ]

        Returns an empty list if there are no turns.
        """
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "tool_calls": [],
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in self._turns
        ]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_user_turn(self, content: str) -> ConversationTurn:
        """Record a user message."""
        turn = ConversationTurn(role="user", content=content)
        self._turns.append(turn)
        self._persist_turn(turn)
        return turn

    def add_assistant_turn(self, content: str, execution_id: str | None = None) -> ConversationTurn:
        """Record an orchestrator response."""
        turn = ConversationTurn(role="assistant", content=content, execution_id=execution_id)
        self._turns.append(turn)
        self._persist_turn(turn)
        return turn

    def add_system_turn(self, content: str) -> ConversationTurn:
        """Record a system-generated message (e.g. status briefing)."""
        turn = ConversationTurn(role="system", content=content)
        self._turns.append(turn)
        self._persist_turn(turn)
        return turn

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> str | None:
        """Archive the current conversation and start fresh.

        Returns the archived session ID, or None if there was nothing to archive.
        """
        if not self._turns:
            return None

        session_id = uuid4().hex[:12]
        archive_path = self._dir / f"{session_id}.jsonl"

        # Move active file to archive
        if self._active_path.exists():
            try:
                shutil.move(str(self._active_path), str(archive_path))
            except Exception:
                log.exception("Failed to archive conversation to %s", archive_path)
                # Fall back: copy then delete
                try:
                    shutil.copy2(str(self._active_path), str(archive_path))
                    self._active_path.unlink(missing_ok=True)
                except Exception:
                    log.exception("Failed to archive conversation (fallback)")

        self._turns.clear()
        log.info("Conversation reset (archived as %s, %d turns)", session_id, len(self._turns))
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        """List archived conversation sessions.

        Returns a list of {session_id, turn_count, started_at, ended_at} dicts,
        sorted newest first.
        """
        sessions: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.jsonl"), reverse=True):
            if path.name == "active.jsonl":
                continue
            session_id = path.stem
            turns = self._load_jsonl(path)
            if not turns:
                continue
            sessions.append({
                "session_id": session_id,
                "turn_count": len(turns),
                "started_at": turns[0].timestamp.isoformat(),
                "ended_at": turns[-1].timestamp.isoformat(),
                "first_message": turns[0].content[:200] if turns else "",
            })
        return sessions

    def get_session(self, session_id: str) -> list[ConversationTurn]:
        """Load an archived conversation by session ID."""
        path = self._dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return self._load_jsonl(path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_turn(self, turn: ConversationTurn) -> None:
        """Append one turn to the active JSONL file."""
        try:
            with open(self._active_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn.to_dict(), default=str) + "\n")
        except Exception:
            log.exception("Failed to persist conversation turn")

    def _hydrate(self) -> None:
        """Load the active conversation from disk into memory."""
        if not self._active_path.exists():
            return
        self._turns = self._load_jsonl(self._active_path)
        if self._turns:
            dirty = self._sanitize_turns()
            log.info("Hydrated active conversation: %d turns%s",
                     len(self._turns), " (sanitized)" if dirty else "")

    def _sanitize_turns(self) -> bool:
        """Fix user turns that were corrupted by history-prepending.

        Prior to the fix, the execution engine stored the full prompt
        (including ``System: ...``, ``User: ...``, ``Orchestrator: ...``
        history) as a single user turn.  This strips those back to just
        the actual user message and removes exact duplicates.
        """
        dirty = False
        cleaned: list[ConversationTurn] = []
        for turn in self._turns:
            content = turn.content
            # Detect history-embedded user turns: content has role prefixes
            # like "System: ...\nUser: ...\nOrchestrator: ..." baked in.
            if turn.role == "user" and "\nUser: " in content and (
                content.startswith("System: ")
                or content.startswith("User: ")
                or "\nOrchestrator: " in content
            ):
                # Extract the last "User: ..." segment as the real message.
                parts = content.split("\nUser: ")
                raw = parts[-1].strip()
                if raw:
                    turn = ConversationTurn(
                        role="user", content=raw,
                        timestamp=turn.timestamp,
                        execution_id=turn.execution_id,
                    )
                    dirty = True
            # Strip leading "[Current time: ...]\n\n" injected by the engine
            if turn.role == "user" and turn.content.startswith("[Current time:"):
                newline_idx = turn.content.find("\n\n")
                if newline_idx != -1:
                    turn = ConversationTurn(
                        role=turn.role,
                        content=turn.content[newline_idx + 2:],
                        timestamp=turn.timestamp,
                        execution_id=turn.execution_id,
                    )
                    dirty = True
            # Deduplicate consecutive identical turns
            if cleaned and cleaned[-1].role == turn.role and cleaned[-1].content == turn.content:
                dirty = True
                continue
            cleaned.append(turn)
        if dirty:
            self._turns = cleaned
            self._rewrite_active()
        return dirty

    def _rewrite_active(self) -> None:
        """Rewrite the active JSONL from the in-memory turns."""
        tmp = self._active_path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for turn in self._turns:
                    f.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self._active_path)
        except Exception:
            log.exception("Failed to rewrite sanitized conversation")
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _load_jsonl(path: Path) -> list[ConversationTurn]:
        """Load turns from a JSONL file."""
        turns: list[ConversationTurn] = []
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(ConversationTurn.from_dict(json.loads(line)))
                except Exception:
                    log.warning("Corrupt turn at line %d in %s, skipping", line_no, path)
        except Exception:
            log.exception("Failed to read conversation file %s", path)
        return turns
