"""Claude Code sessions on this machine, for importing one as an ATN agent.

Claude Code keeps one transcript per session under
``~/.claude/projects/<folder slug>/<session id>.jsonl``; several sessions can
share a folder. An imported session becomes a claude_max agent whose bridge
run resumes that session in its own folder with the project's settings
(``AgentDefinition.claude_session``). A session must not be imported while a
Claude process may be writing it: two writers on one transcript fork it.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

# A transcript written this recently counts as in use even when no process
# is found for it (process scans can miss a CLI that hides its cwd).
RECENT_WRITE_SECS = 60
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def folder_slug(folder: str | os.PathLike) -> str:
    """Claude Code's project directory name: every non-alphanumeric
    character of the absolute path becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(folder).resolve()))


def project_dir(folder: str | os.PathLike) -> Path:
    return claude_home() / "projects" / folder_slug(folder)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
# Transcript lines the chat should not show: Claude Code's own bookkeeping
# around slash commands, and command output already summarised elsewhere.
_HIDDEN_PREFIXES = ("<local-command-caveat>", "<local-command-stdout>",
                    "<command-name>", "<command-message>", "<command-args>")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [str(p.get("text", "")) for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
    else:
        parts = []
    text = _REMINDER_RE.sub("", "\n\n".join(parts)).strip()
    return "" if text.startswith(_HIDDEN_PREFIXES) else text


def transcript_turns(path: Path, limit: int = 2000) -> list[tuple[str, str]]:
    """The conversation a Claude Code transcript shows, as (role, text):
    the user's own messages and the assistant's text, oldest first, the
    newest ``limit`` of them. Tool calls and results, subagent sidechains,
    meta lines and compaction summaries are left out; consecutive assistant
    lines (one per content block) merge into one reply."""
    turns: list[tuple[str, str]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                kind = d.get("type")
                if kind not in ("user", "assistant") or d.get("isSidechain") \
                        or d.get("isMeta") or d.get("isCompactSummary"):
                    continue
                text = _text_of((d.get("message") or {}).get("content"))
                if not text:
                    continue
                if kind == "assistant" and turns and turns[-1][0] == "assistant":
                    turns[-1] = ("assistant", turns[-1][1] + "\n\n" + text)
                else:
                    turns.append((kind, text))
    except OSError:
        return []
    return turns[-limit:]


def read_session(path: Path) -> dict[str, Any]:
    """Summary of one transcript: title, first prompt, model, cwd."""
    out: dict[str, Any] = {"session_id": path.stem, "title": "", "first_prompt": "",
                           "model": "", "cwd": "", "entrypoint": "", "messages": 0}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                kind = d.get("type")
                if kind in ("ai-title", "custom-title", "summary"):
                    title = d.get("customTitle") or d.get("aiTitle") or d.get("summary")
                    # The latest generated title wins; a user's own title
                    # beats any generated one.
                    if title and (kind == "custom-title" or not out.get("_custom")):
                        out["title"] = str(title)
                        out["_custom"] = out.get("_custom") or kind == "custom-title"
                elif kind == "user":
                    out["messages"] += 1
                    if not out["cwd"]:
                        out["cwd"] = d.get("cwd") or ""
                        out["entrypoint"] = d.get("entrypoint") or ""
                    if not out["first_prompt"]:
                        content = (d.get("message") or {}).get("content")
                        if isinstance(content, str):
                            out["first_prompt"] = content[:200]
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    out["first_prompt"] = str(part.get("text", ""))[:200]
                                    break
                elif kind == "assistant":
                    out["messages"] += 1
                    model = (d.get("message") or {}).get("model")
                    if model and not str(model).startswith("<"):
                        out["model"] = model
    except OSError:
        pass
    out.pop("_custom", None)
    return out


def claude_processes() -> list[dict[str, Any]]:
    """Running Claude Code processes: pid, cwd (when readable) and the
    session id when the command line names one (--resume <id>)."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = p.info.get("cmdline") or []
            joined = " ".join(cmd).lower()
            if not (name.startswith("claude") or "claude-code" in joined
                    or re.search(r"[\\/]claude(\.exe|\.cmd)?\b", joined)):
                continue
            if "claude-bridge" in joined:
                continue  # the daemon's own bridge, not a user session
            try:
                cwd = p.cwd()
            except Exception:
                cwd = ""
            sid = ""
            for flag in ("--resume", "-r", "--session-id"):
                if flag in cmd:
                    i = cmd.index(flag)
                    if i + 1 < len(cmd) and _SESSION_ID_RE.match(cmd[i + 1]):
                        sid = cmd[i + 1]
            found.append({"pid": p.info["pid"], "cwd": cwd, "session_id": sid})
        except Exception:
            continue
    return found


def list_sessions(folder: str | os.PathLike, *,
                  processes: list[dict[str, Any]] | None = None,
                  now: float | None = None) -> dict[str, Any]:
    """Every Claude Code session of ``folder``, newest first, each marked
    ``in_use`` (with why) when it must not be imported right now."""
    pdir = project_dir(folder)
    procs = claude_processes() if processes is None else processes
    now = time.time() if now is None else now
    here = [p for p in procs if p.get("cwd") and _norm(p["cwd"]) == _norm(Path(folder).resolve())]
    named = {p["session_id"] for p in here if p.get("session_id")}
    # A process in this folder whose session we can't tell could hold any of
    # its sessions.
    unknown = [p["pid"] for p in here if not p.get("session_id")]
    sessions = []
    for path in sorted(pdir.glob("*.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True):
        if not _SESSION_ID_RE.match(path.stem):
            continue
        info = read_session(path)
        mtime = path.stat().st_mtime
        info["last_active"] = int(mtime)
        reasons = []
        if path.stem in named:
            reasons.append("a Claude process has this session open")
        if unknown:
            reasons.append(f"Claude is running in this folder (pid {', '.join(map(str, unknown))})")
        if now - mtime < RECENT_WRITE_SECS:
            reasons.append("written in the last minute")
        info["in_use"] = bool(reasons)
        info["in_use_reason"] = "; ".join(reasons)
        sessions.append(info)
    return {"folder": str(Path(folder).resolve()), "project_dir": str(pdir),
            "exists": pdir.is_dir(), "sessions": sessions}


def import_check(folder: str, session_id: str, **kw: Any) -> tuple[dict[str, Any] | None, str]:
    """(session info, "") when ``session_id`` of ``folder`` can be imported
    now, else (None or info, why not)."""
    if not _SESSION_ID_RE.match(session_id or ""):
        return None, "session_id must be a Claude Code session id"
    listing = list_sessions(folder, **kw)
    for s in listing["sessions"]:
        if s["session_id"] == session_id:
            if s["in_use"]:
                return s, f"session is in use: {s['in_use_reason']}"
            return s, ""
    return None, f"no session {session_id} under {listing['project_dir']}"
