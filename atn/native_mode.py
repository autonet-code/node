"""Whether this daemon runs native agents (Decision 2026-11-02).

External agents (Claude Code, Codex, anything with a shell) are the main
journey; the daemon's own agent framework (its loop, providers, delegates,
goals, groups, voice, chat) is an opt-in mode. Off, the daemon still hosts
external agents and everything they use; it only refuses to create native
ones, and the app hides the surfaces that exist for them.

Saved as ``<data_dir>/native_agents.json`` ``{"enabled": bool}``;
``ATN_NATIVE_AGENTS=0|1`` overrides it. Without the
file a daemon that already hosts a native agent keeps the mode on (so an
upgrade hides nothing anyone used); a fresh one starts with it off.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FILE = "native_agents.json"


def _path(data_dir: Any) -> Path:
    return Path(data_dir) / FILE


def native_agents_enabled(runtime: Any) -> bool:
    env = os.environ.get("ATN_NATIVE_AGENTS", "").strip()
    if env in ("0", "1"):
        return env == "1"
    data_dir = getattr(getattr(runtime, "_config", None), "data_dir", None)
    if data_dir is not None:
        try:
            data = json.loads(_path(data_dir).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "enabled" in data:
                return bool(data["enabled"])
        except (OSError, ValueError):
            pass
    from .provider_identity import is_external
    agents = getattr(getattr(runtime, "registry", None), "_agents", {}) or {}
    return any(not is_external(defn) for defn in agents.values())


def set_native_agents(runtime: Any, enabled: bool) -> bool:
    data_dir = getattr(runtime._config, "data_dir")
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"enabled": bool(enabled)}), encoding="utf-8")
    os.replace(tmp, path)
    return bool(enabled)


REFUSAL = ("native agents are off on this daemon: it hosts external agents "
           "(provider 'external'). The owner can turn native agents on in "
           "the app.")
