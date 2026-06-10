"""Deterministic post-run summary extractor.

Scans accumulated tool-call data from a cognitive agent execution and produces
a structured, human-readable summary.  No LLM calls — pure parsing.

The summary becomes the ``result_preview`` sent to the parent agent in the
completion notification, giving it a factual overview of what the child did.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ToolCallRecord:
    """One tool invocation captured during execution."""
    tool: str
    args: dict[str, Any]
    result: str
    success: bool


def extract_run_summary(
    tool_calls: list[dict[str, Any]],
    max_turns: int | None = None,
    actual_turns: int | None = None,
    status: str = "",
    error: str | None = None,
) -> str:
    """Build a deterministic summary from accumulated tool-call data.

    Args:
        tool_calls: List of dicts, each with keys ``tool``, ``args``,
            ``result`` (str), ``success`` (bool).
        max_turns: The configured turn limit (None if unlimited).
        actual_turns: How many LLM turns actually occurred.
        status: Execution status string (e.g. "completed", "killed").
        error: Error message if the execution failed.

    Returns:
        A structured text summary, kept under 2000 chars.
    """
    records = [
        _ToolCallRecord(
            tool=tc.get("tool", ""),
            args=tc.get("args", {}),
            result=str(tc.get("result", "")),
            success=tc.get("success", True),
        )
        for tc in tool_calls
    ]

    lines: list[str] = []

    # --- Turn count ---
    lines.append(_turn_line(actual_turns, max_turns, status))

    # --- Files created / modified / read ---
    created, modified, read_count = _classify_files(records)
    if created:
        lines.append(f"Created: {', '.join(_short_paths(created))}")
    if modified:
        lines.append(f"Modified: {', '.join(_short_paths(modified))}")
    if read_count:
        lines.append(f"Files read: {read_count}")

    # --- Notable commands ---
    commands = _notable_commands(records)
    if commands:
        lines.append("Commands:")
        for cmd in commands:
            lines.append(f"  {cmd}")

    # --- Git commits ---
    commits = _git_commits(records)
    if commits:
        for c in commits:
            lines.append(f'Committed: "{c}"')

    # --- Errors ---
    errors = _errors(records)
    if errors:
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  - {e}")

    # --- Web searches ---
    searches = _web_searches(records)
    if searches:
        lines.append(f"Searched: {', '.join(searches)}")

    # --- Agents created ---
    agents = _agents_created(records)
    if agents:
        lines.append(f"Agents spawned: {', '.join(agents)}")

    # --- Last action ---
    last = _last_action(records)
    if last:
        lines.append(f"Last action: {last}")

    # --- Error from execution ---
    if error:
        lines.append(f"Error: {error[:200]}")

    summary = "\n".join(lines)
    # Hard cap at 1800 to leave room for the agent's own last response
    if len(summary) > 1800:
        summary = summary[:1797] + "..."
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn_line(actual: int | None, maximum: int | None, status: str) -> str:
    if actual is not None and maximum is not None:
        line = f"Completed {actual}/{maximum} turns."
        if actual >= maximum:
            line += " Hit turn limit — may be incomplete."
        return line
    if actual is not None:
        return f"Completed {actual} turns."
    return f"Status: {status}" if status else "Completed."


def _classify_files(
    records: list[_ToolCallRecord],
) -> tuple[list[str], list[str], int]:
    """Separate files into created, modified, and read-count."""
    created: list[str] = []
    modified: list[str] = []
    read_paths: set[str] = set()

    # Tool names that create files
    _WRITE_TOOLS = {"Write", "write_file", "write", "mcp__atn__write_file"}
    # Tool names that edit files
    _EDIT_TOOLS = {"Edit", "edit_file", "edit", "mcp__atn__edit_file"}
    # Tool names that read files
    _READ_TOOLS = {"Read", "read_file", "read", "cat", "mcp__atn__read_file"}

    written_paths: set[str] = set()

    for r in records:
        tool = r.tool
        path = _extract_path(r.args)

        if tool in _WRITE_TOOLS and path:
            norm = _norm_path(path)
            if norm not in written_paths:
                created.append(path)
                written_paths.add(norm)
        elif tool in _EDIT_TOOLS and path:
            norm = _norm_path(path)
            if norm not in written_paths:
                modified.append(path)
                written_paths.add(norm)
        elif tool in _READ_TOOLS and path:
            read_paths.add(_norm_path(path))
        elif tool in ("Glob", "Grep", "glob", "grep"):
            # Count as reads but don't track individual paths
            read_paths.add(f"search:{id(r)}")

    # Don't count files that were also created/modified
    read_only = read_paths - written_paths
    return created, modified, len(read_only)


def _extract_path(args: dict[str, Any]) -> str:
    """Pull a file path from tool args."""
    for key in ("file_path", "path", "filename", "file"):
        val = args.get(key)
        if val and isinstance(val, str):
            return val
    return ""


def _norm_path(path: str) -> str:
    """Normalize a path for deduplication."""
    return os.path.normpath(path).replace("\\", "/").lower()


def _short_paths(paths: list[str]) -> list[str]:
    """Return just the filename (or last 2 path components) for brevity."""
    result = []
    for p in paths:
        parts = p.replace("\\", "/").split("/")
        if len(parts) >= 2:
            result.append("/".join(parts[-2:]))
        else:
            result.append(parts[-1])
    return result


def _notable_commands(records: list[_ToolCallRecord]) -> list[str]:
    """Extract notable shell commands (build, test, git, etc.)."""
    _BASH_TOOLS = {"Bash", "bash", "shell", "run_command", "execute"}
    results: list[str] = []

    for r in records:
        if r.tool not in _BASH_TOOLS:
            continue
        cmd = r.args.get("command", "")
        if not cmd:
            continue
        # Skip trivial commands
        if _is_trivial_command(cmd):
            continue
        # Extract exit code if present in result
        exit_code = _extract_exit_code(r.result)
        short_cmd = _shorten_command(cmd)
        if exit_code is not None and exit_code != 0:
            results.append(f"{short_cmd} (exit {exit_code})")
        elif not r.success:
            results.append(f"{short_cmd} (failed)")
        else:
            results.append(short_cmd)

    return results[:10]  # Cap at 10 commands


def _is_trivial_command(cmd: str) -> bool:
    """Skip read-only or trivial commands."""
    cmd_stripped = cmd.strip()
    trivial_prefixes = ("cat ", "head ", "tail ", "ls", "pwd", "echo ", "which ", "type ")
    for prefix in trivial_prefixes:
        if cmd_stripped.startswith(prefix):
            return True
    return False


def _shorten_command(cmd: str) -> str:
    """Shorten a command for display."""
    cmd = cmd.strip()
    # For multi-line commands, take the first line
    first_line = cmd.split("\n")[0]
    if len(first_line) > 100:
        first_line = first_line[:97] + "..."
    return first_line


def _extract_exit_code(result: str) -> int | None:
    """Try to extract an exit code from tool result text."""
    # Common patterns: "exit code: 1", "Exit code: 0", "returned 1"
    m = re.search(r"exit[_ ]code[:\s]+(\d+)", result, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _git_commits(records: list[_ToolCallRecord]) -> list[str]:
    """Extract git commit messages."""
    _BASH_TOOLS = {"Bash", "bash", "shell", "run_command", "execute"}
    commits: list[str] = []

    for r in records:
        if r.tool not in _BASH_TOOLS:
            continue
        cmd = r.args.get("command", "")
        if "git commit" not in cmd:
            continue
        # Try to extract the commit message from the command
        msg = _extract_commit_message(cmd)
        if msg:
            commits.append(msg)
        elif r.success:
            # Try to find commit hash in result
            m = re.search(r"\[[\w/-]+\s+([a-f0-9]{7,})\]", r.result)
            if m:
                commits.append(f"(commit {m.group(1)})")

    return commits


def _extract_commit_message(cmd: str) -> str:
    """Extract commit message from a git commit command."""
    # HEREDOC style FIRST: git commit -m "$(cat <<'EOF'\nmessage\nEOF\n)".
    # The plain -m regex would otherwise match the heredoc's opening
    # `$(cat <<` as the message.
    m = re.search(r"cat\s+<<['\"]?EOF['\"]?\n(.*?)EOF", cmd, re.DOTALL)
    if m:
        msg = m.group(1).strip().split("\n")[0]
        return msg[:80] if len(msg) > 80 else msg
    # git commit -m "message"
    m = re.search(r'git\s+commit\s+.*-m\s+["\']([^"\']+)["\']', cmd)
    if m:
        msg = m.group(1)
        return msg[:80] if len(msg) > 80 else msg
    return ""


def _errors(records: list[_ToolCallRecord]) -> list[str]:
    """Extract notable errors from tool calls."""
    errors: list[str] = []
    for r in records:
        if r.success:
            continue
        error_text = r.result[:150] if r.result else "unknown error"
        errors.append(f"{r.tool}: {error_text}")

    return errors[:5]  # Cap at 5 errors


def _web_searches(records: list[_ToolCallRecord]) -> list[str]:
    """Extract web search queries."""
    _SEARCH_TOOLS = {"WebSearch", "web_search", "mcp__atn__web_search"}
    searches: list[str] = []
    for r in records:
        if r.tool not in _SEARCH_TOOLS:
            continue
        query = r.args.get("query", r.args.get("q", ""))
        if query:
            short_q = query[:60] if len(query) > 60 else query
            searches.append(f'"{short_q}"')
    return searches[:5]


def _agents_created(records: list[_ToolCallRecord]) -> list[str]:
    """Extract sub-agents that were spawned."""
    _CREATE_TOOLS = {
        "create_agent", "delegate", "mcp__atn__create_agent",
        "mcp__atn__delegate", "Agent",
    }
    agents: list[str] = []
    for r in records:
        if r.tool not in _CREATE_TOOLS:
            continue
        name = (
            r.args.get("name", "")
            or r.args.get("title", "")
            or r.args.get("description", "")
            or r.args.get("agent_type", "agent")
        )
        if name:
            agents.append(name[:40])
    return agents[:10]


def _last_action(records: list[_ToolCallRecord]) -> str:
    """Describe the last tool call."""
    if not records:
        return ""
    last = records[-1]
    tool = last.tool
    desc = ""

    if tool in ("Write", "write_file", "write"):
        path = _extract_path(last.args)
        desc = f"writing {os.path.basename(path)}" if path else "writing file"
    elif tool in ("Edit", "edit_file", "edit"):
        path = _extract_path(last.args)
        desc = f"editing {os.path.basename(path)}" if path else "editing file"
    elif tool in ("Read", "read_file", "read"):
        path = _extract_path(last.args)
        desc = f"reading {os.path.basename(path)}" if path else "reading file"
    elif tool in ("Bash", "bash", "shell", "run_command"):
        cmd = _shorten_command(last.args.get("command", ""))
        desc = f"running: {cmd[:60]}"
    elif tool in ("Grep", "grep"):
        pattern = last.args.get("pattern", "")
        desc = f'searching for "{pattern[:40]}"'
    elif tool in ("Glob", "glob"):
        pattern = last.args.get("pattern", "")
        desc = f'finding files "{pattern[:40]}"'
    else:
        desc = f"calling {tool}"

    return desc
