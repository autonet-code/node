"""Shell tools for non-bridge providers (Gemini, OpenAI, Ollama, Anthropic direct).

BridgeProvider (Claude SDK subprocess) has Bash/Read/Write/Glob/Grep built-in;
generic providers need these Python-implemented equivalents.

This module is the single source of truth for:
  - SHELL_TOOLS:          tool definition dicts (name, description, input_schema)
  - SHELL_TOOL_EXECUTORS: name -> async executor function mapping
  - dispatch():           the bundle envelope, {tool, args} -> result

DUAL NATURE (docs/tool_substrate.md — "Resident tools, loadouts, distros").
This file is simultaneously three things, and that is deliberate:

  1. An IMPORTABLE module. The daemon and the worker both import
     SHELL_TOOL_EXECUTORS by identity and await the executors in-process.
     That path is the default and stays untouched — a subprocess round-trip
     is ~470x slower than an in-process await (66ms vs 0.14ms measured), so
     the built-in must never pay it.
  2. A RUNNABLE pinned tool. The ``__main__`` block below speaks the sealed
     tool protocol (JSON envelope on stdin, JSON result on stdout), so this
     exact file can be executed as a tool subprocess.
  3. The CODE BLOB of the atn_shell module manifest. harness_distro hashes
     this file with inspect.getsource, so (2) makes an already-existing
     manifest honest for the first time: what the digest locks is now a
     program, not just a description of one.

The point of (2)+(3) is the reference implementation: it is the literal
contract a third-party shell provider must match. Whether such a provider may
REPLACE the built-in is a separate, currently-disabled question — see
atn/runtime/shell_provider.py for why that half is gated off.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool definitions (JSON-schema format expected by all providers)
# ---------------------------------------------------------------------------

SHELL_TOOLS: list[dict[str, Any]] = [
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
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. old_string must match the "
            "file exactly (including whitespace) and must be unique unless "
            "replace_all is set. Prefer this over write_file for changing "
            "existing files — it cannot clobber the rest of the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
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


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

async def exec_bash(inp: dict) -> dict:
    """Execute a shell command."""
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


async def exec_read_file(inp: dict) -> dict:
    """Read a file."""
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


async def exec_write_file(inp: dict) -> dict:
    """Write a file."""
    try:
        p = Path(inp["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"], encoding="utf-8")
        return {"status": "ok", "path": str(p), "bytes": len(inp["content"])}
    except Exception as e:
        return {"error": str(e)}


async def exec_edit_file(inp: dict) -> dict:
    """Exact-string replacement in a file."""
    try:
        p = Path(inp["path"])
        if not p.exists():
            return {"error": f"File not found: {p}"}
        old = inp["old_string"]
        new = inp["new_string"]
        if old == new:
            return {"error": "old_string and new_string are identical"}
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return {"error": "old_string not found in file — read the file and match it exactly"}
        if count > 1 and not inp.get("replace_all"):
            return {"error": f"old_string occurs {count} times — add surrounding context to make it unique, or set replace_all"}
        replaced = text.replace(old, new) if inp.get("replace_all") else text.replace(old, new, 1)
        p.write_text(replaced, encoding="utf-8")
        return {"status": "ok", "path": str(p), "replacements": count if inp.get("replace_all") else 1}
    except Exception as e:
        return {"error": str(e)}


async def exec_list_dir(inp: dict) -> dict:
    """List directory."""
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


async def exec_search_files(inp: dict) -> dict:
    """Search files with grep."""
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
        # rg not available, fall back to grep
        cmd = ["grep", "-rn", pattern, path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {"matches": result.stdout[:50000]}
        except Exception as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


SHELL_TOOL_EXECUTORS: dict[str, Any] = {
    "bash": exec_bash,
    "read_file": exec_read_file,
    "write_file": exec_write_file,
    "edit_file": exec_edit_file,
    "list_directory": exec_list_dir,
    "search_files": exec_search_files,
}


# ---------------------------------------------------------------------------
# Bundle envelope — the contract a shell PROVIDER implements
# ---------------------------------------------------------------------------

async def dispatch(envelope: dict) -> dict:
    """Route one ``{"tool": name, "args": {...}}`` envelope to an executor.

    Returns ``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": ...}``.

    The envelope exists because a BUNDLE provides several tools under ONE
    manifest digest — the digest is the unit of grant, review, and adoption,
    so five tools sharing one identity need an inner selector. Callers of the
    in-process fast path do NOT go through here; they index
    SHELL_TOOL_EXECUTORS directly.

    Never raises: an executor fault becomes an error envelope, because the
    caller is a subprocess boundary where an exception is just a non-zero
    exit with a traceback on stderr.
    """
    if not isinstance(envelope, dict):
        return {"ok": False, "error": "envelope must be an object"}
    name = envelope.get("tool")
    if not isinstance(name, str) or not name:
        return {"ok": False, "error": "envelope requires a 'tool' name"}
    args = envelope.get("args")
    if not isinstance(args, dict):
        args = {}
    fn = SHELL_TOOL_EXECUTORS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown shell tool: {name}"}
    try:
        return {"ok": True, "result": await fn(args)}
    except Exception as exc:  # noqa: BLE001 — a tool error is data, not a crash
        return {"ok": False, "error": f"{name} failed: {exc}"}


def _main() -> None:
    """Sealed-tool entrypoint: JSON envelope on stdin, JSON result on stdout.

    Exits 0 even on error — the protocol carries failure in the payload, and a
    non-zero exit would surface as "tool exited N" instead of the real reason.
    """
    import asyncio
    import json
    import sys

    try:
        envelope = json.loads(sys.stdin.read() or "{}")
    except (ValueError, UnicodeDecodeError) as exc:
        print(json.dumps({"ok": False, "error": f"bad envelope: {exc}"}))
        return
    try:
        out = asyncio.run(dispatch(envelope))
    except Exception as exc:  # noqa: BLE001 — absolute backstop
        out = {"ok": False, "error": f"dispatch failed: {exc}"}
    print(json.dumps(out))


if __name__ == "__main__":
    _main()
