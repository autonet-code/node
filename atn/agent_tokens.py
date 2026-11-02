"""Per-agent tokens for the agent API (atn/agent_api.py).

Calls under ``/agents/<id>/`` carry ``Authorization: Bearer <token>``, so
another process on the machine cannot act as an agent and the Activity log
can be trusted. Only a hash is kept, in ``<data_dir>/agent_tokens.json``;
the token itself is shown once, when it is issued. Read on every check, so
a reset (``atn agent-token <id>``) takes effect at once.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path

FILE = "agent_tokens.json"
_lock = threading.Lock()


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / FILE


def _load(data_dir: Path) -> dict[str, str]:
    try:
        data = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def has_token(data_dir: Path, agent_id: str) -> bool:
    return agent_id in _load(data_dir)


def issue(data_dir: Path, agent_id: str) -> str:
    """A new token for ``agent_id``, replacing any earlier one."""
    token = "atn_" + secrets.token_urlsafe(32)
    with _lock:
        data = _load(data_dir)
        data[agent_id] = _hash(token)
        path = _path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return token


def verify(data_dir: Path, agent_id: str, token: str) -> bool:
    want = _load(data_dir).get(agent_id)
    return bool(want and token) and hmac.compare_digest(want, _hash(token))
