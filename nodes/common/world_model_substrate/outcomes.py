"""Outcome signal capture from worker traces.

Reads agent execution conversations and extracts objective outcome
signals: was the work accepted, was it kept, was it built on, was
it paid for. These signals are the engine's training-target half of
each (problem, resolution, outcome) tuple.

Honest scope for v1: we extract what's robustly readable from
~/.atn/agents/<agent>/conversations/*.jsonl and ~/.atn/conversations/*.jsonl.
Subjective signals (harm, alignment violations, value disagreements)
are left to training-node debate, not heuristic detection.

Outcome dimensions
------------------

  accepted: bool
    The user explicitly accepted or affirmed the work, OR didn't
    request a redo. Heuristic: last user message after the work
    doesn't contain rejection markers ("no", "redo", "wrong",
    "actually", "instead", etc.).

  kept: bool
    Files/code the agent created or edited still exist (would need
    a snapshot comparison). For v1: True if no later session edits
    those same files in a "revert" pattern. Default True (presence
    of the work is the default).

  built_on: bool
    A later session in the same agent's history references or extends
    this work. Heuristic: file paths or distinctive identifiers from
    this session appear in subsequent sessions.

  paid: bool
    Inference revenue (or analog) accrued because of this session.
    For v1, plain placeholder until autonet's revenue routing
    actually annotates traces with this signal.

Tuple
-----

  Outcome = (accepted, kept, built_on, paid)
  Encoded as a 4D float vector in [-1, +1] per axis for substrate
  consumption (each bool maps -1 / +1, ambiguous = 0).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic markers
# ---------------------------------------------------------------------------

REJECTION_MARKERS = (
    "no, ",
    "no.",
    "no, that's",
    "redo",
    "wrong",
    "actually,",
    "actually no",
    "instead",
    "that's not",
    "not what",
    "doesn't work",
    "broken",
    "revert",
    "undo this",
    "back out",
)

ACCEPTANCE_MARKERS = (
    "thanks",
    "thank you",
    "great",
    "perfect",
    "looks good",
    "lgtm",
    "ship it",
    "merge it",
    "approved",
    "yes,",
    "yes.",
    "exactly",
    "that's right",
    "correct",
    "good work",
    "well done",
)


def _has_marker(text: str, markers: tuple) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    return any(m in lower for m in markers)


# ---------------------------------------------------------------------------
# Conversation-level outcome
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """Objective outcome signals for a unit of work.

    Each field is a tri-valued state encoded as a float in [-1, +1]:
        +1 = strongly positive evidence of this outcome property
         0 = no signal (default)
        -1 = strongly negative evidence
    """

    accepted: float = 0.0
    kept: float = 0.0
    built_on: float = 0.0
    paid: float = 0.0

    def to_coords(self) -> Tuple[float, float, float, float]:
        return (self.accepted, self.kept, self.built_on, self.paid)

    @property
    def signal_strength(self) -> float:
        """How much objective signal we have. 0 = no evidence; 4 = max."""
        return abs(self.accepted) + abs(self.kept) + abs(self.built_on) + abs(self.paid)


# ---------------------------------------------------------------------------
# Trace reading
# ---------------------------------------------------------------------------


def read_conversation(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL conversation file. Returns list of message dicts."""
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_acceptance(messages: List[Dict[str, Any]]) -> float:
    """Score the acceptance signal from a conversation.

    Walk forward from the last assistant message to the next user
    message. If user message has rejection markers -> -1. If user has
    acceptance markers -> +1. If neither, default to 0 (no signal).
    """
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = i
    if last_assistant_idx < 0:
        return 0.0
    # Look for the next user message after the last assistant turn.
    for j in range(last_assistant_idx + 1, len(messages)):
        msg = messages[j]
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "") or ""
        if _has_marker(text, REJECTION_MARKERS):
            return -1.0
        if _has_marker(text, ACCEPTANCE_MARKERS):
            return 1.0
    return 0.0


def extract_built_on(
    messages: List[Dict[str, Any]],
    later_messages: List[Dict[str, Any]],
) -> float:
    """+1 if later messages reference distinctive content from these
    messages (file paths, function names, identifiers); 0 otherwise.
    No -1 signal -- absence of references is just absence, not
    rejection.
    """
    # Pull distinctive tokens (long alphanumeric runs) from messages
    distinctive = set()
    for msg in messages:
        text = msg.get("content", "") or ""
        # File paths
        for m in re.findall(r"[A-Za-z0-9_/\\\\.-]{8,}", text):
            if "/" in m or "\\" in m or "." in m:
                distinctive.add(m.lower())
        # Long identifiers
        for m in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{12,}\b", text):
            distinctive.add(m.lower())
    if not distinctive:
        return 0.0
    later_text = " ".join(
        (m.get("content", "") or "").lower() for m in later_messages
    )
    if not later_text:
        return 0.0
    hits = sum(1 for token in distinctive if token in later_text)
    if hits >= 2:
        return 1.0
    if hits >= 1:
        return 0.5
    return 0.0


def extract_kept(messages: List[Dict[str, Any]]) -> float:
    """Default: True (the work was done; no evidence of revert).
    A more thorough version would compare file states; for v1 we
    return +1 unless rejection markers explicitly request revert.
    """
    revert_markers = ("revert", "back out", "undo this", "rollback")
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = (msg.get("content", "") or "").lower()
        if any(m in text for m in revert_markers):
            return -1.0
    return 1.0


def extract_paid(messages: List[Dict[str, Any]]) -> float:
    """Placeholder. Returns 0 until autonet's revenue routing
    actually annotates traces with payment signals.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def outcome_from_conversation(
    conversation_path: Path,
    later_conversation_paths: Optional[List[Path]] = None,
) -> Outcome:
    """Compute an Outcome for one conversation, optionally using
    later conversations to derive the built_on signal.
    """
    messages = read_conversation(conversation_path)
    later_messages: List[Dict[str, Any]] = []
    for p in later_conversation_paths or []:
        later_messages.extend(read_conversation(p))
    return Outcome(
        accepted=extract_acceptance(messages),
        kept=extract_kept(messages),
        built_on=extract_built_on(messages, later_messages),
        paid=extract_paid(messages),
    )


def outcomes_for_agent_dir(agent_root: Path) -> Dict[str, Outcome]:
    """Walk an agent's conversations directory and return
    {conversation_filename: Outcome} for each conversation.

    Built-on signal uses any conversation later than the target as
    'later messages.'
    """
    conv_dir = agent_root / "conversations"
    if not conv_dir.is_dir():
        return {}
    files = sorted(conv_dir.glob("*.jsonl"))
    if not files:
        return {}
    out: Dict[str, Outcome] = {}
    for i, path in enumerate(files):
        later = files[i + 1:]
        out[path.name] = outcome_from_conversation(path, later)
    return out
