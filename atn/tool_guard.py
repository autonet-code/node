"""Deny-by-default capability guard for ADOPTED pinned tools.

docs/tool_substrate.md — Adoption rail, containment layer. Launched as

    python tool_guard.py <tool_script.py>

with the policy JSON in the ATN_TOOL_POLICY environment variable
(popped before the tool code runs). The parent (ToolStore) has already
scrubbed the environment and pointed cwd at a per-tool sandbox
directory; this runner adds the in-process floor: a sys.addaudithook
that hard-fails on any capability the manifest did not declare.

Policy shape (all deny by default):

    {"net": bool, "fs": bool, "spawn": bool}

  - net   — socket use ("socket.*" audit events)
  - fs    — file access outside the sandbox cwd + the Python runtime
            ("open" audit events; fd-based and in-tree access passes)
  - spawn — subprocess / exec ("subprocess.Popen", "os.system",
            "os.posix_spawn", "os.spawn", "os.exec")

An undeclared capability raises PermissionError: the tool process dies
with a traceback naming the capability, the ToolStore surfaces it as an
error frame, and the mismatch between manifest and behavior is exactly
the reproducible evidence a CON claim wants.

HONESTY, NOT HERMETICS: an audit hook is bypassable by sufficiently
determined native code (ctypes, extension modules). It makes
capability declarations honest for straightforward Python and turns
evasion attempts into deliberate, evidenced acts. The OS-level
isolated runner (vault track) is the wall; this is the tripwire in
front of it. Defense stays layered: consent gate, provenance
friction, and the CON evidence rail sit around this.
"""
from __future__ import annotations

import json
import os
import runpy
import sys


def _real(path: str) -> str:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def main() -> None:
    policy = json.loads(os.environ.pop("ATN_TOOL_POLICY", "{}"))
    script = sys.argv[1]

    allow_net = bool(policy.get("net"))
    allow_fs = bool(policy.get("fs"))
    allow_spawn = bool(policy.get("spawn"))

    allowed_prefixes = tuple({
        _real(os.getcwd()),
        _real(sys.prefix),
        _real(sys.base_prefix),
        _real(os.path.dirname(_real(script))),
    })

    spawn_events = ("subprocess.Popen", "os.system", "os.posix_spawn",
                    "os.spawn", "os.exec", "os.startfile")

    def hook(event: str, args: tuple) -> None:
        if not allow_net and event.startswith("socket."):
            raise PermissionError(f"undeclared capability: net ({event})")
        if not allow_spawn and event in spawn_events:
            raise PermissionError(f"undeclared capability: spawn ({event})")
        if not allow_fs and event == "open":
            path = args[0] if args else None
            if path is None or isinstance(path, int):
                return  # fd re-open: the fd was already policy-checked
            p = _real(os.fsdecode(path) if isinstance(path, bytes)
                      else str(path))
            if not p.startswith(allowed_prefixes):
                raise PermissionError(f"undeclared capability: fs ({p})")

    sys.addaudithook(hook)
    sys.argv = [script]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
