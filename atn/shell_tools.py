"""Shell tools for non-bridge providers (Gemini, OpenAI, Ollama, Anthropic direct).

BridgeProvider (Claude SDK subprocess) has Bash/Read/Write/Glob/Grep built-in;
generic providers need these Python-implemented equivalents.

This module is the single source of truth for:
  - SHELL_TOOLS:          tool definition dicts (name, description, input_schema)
  - SHELL_TOOL_EXECUTORS: name -> async executor function mapping
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
    "list_directory": exec_list_dir,
    "search_files": exec_search_files,
}
