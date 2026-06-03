"""Surface-owned credit store: per-user rolling request credits for a chat
Surface's input seam.

This is the gating STATE the chat Surface owns — distinct from autonet's
provider token budgets (atn/credit_budget.py, which meters AI spend). Here a
"credit" is one inbound request; users get a rolling-window allowance so a
shared support agent isn't swamped. Operators bypass it entirely (see
CreditPolicy).

JSON-persisted under the data dir, matching CreditBudgetStore's convention.
Self-contained: no platform or deployment dependency.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("atn.chat.credits")

_DEFAULT_WINDOW_SECS = 86_400   # 24h rolling
_DEFAULT_LIMIT = 5              # requests per window for an unconfigured user


class CreditStore:
    """Rolling-window per-user request credits.

    Usage:
        store = CreditStore(data_dir)
        store.remaining(uid)        # credits left in the window
        store.consume(uid)          # record one request
        store.is_blocked(uid)       # explicitly denied?
        store.set_user(uid, limit=20)   # configure a user's allowance
        store.block(uid) / store.unblock(uid)

    State persists to <data_dir>/chat_credits.json:
      {
        "window_secs": 86400,
        "default_limit": 5,
        "users": { "<uid>": {"limit": int, "blocked": bool} },
        "log":   { "<uid>": ["<iso ts>", ...] }    # request timestamps
      }
    """

    def __init__(self, data_dir: Path, *, window_secs: int | None = None,
                 default_limit: int | None = None) -> None:
        self._path = Path(data_dir) / "chat_credits.json"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._window = window_secs or _DEFAULT_WINDOW_SECS
        self._default_limit = default_limit if default_limit is not None else _DEFAULT_LIMIT
        self._users: dict[str, dict] = {}
        self._log: dict[str, list[str]] = {}
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._window = int(data.get("window_secs", self._window))
                self._default_limit = int(data.get("default_limit", self._default_limit))
                self._users = data.get("users", {}) or {}
                self._log = data.get("log", {}) or {}
        except Exception:
            log.warning("could not load chat credits from %s", self._path, exc_info=True)

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({
                "window_secs": self._window,
                "default_limit": self._default_limit,
                "users": self._users,
                "log": self._log,
            }, indent=2), encoding="utf-8")
        except Exception:
            log.warning("could not save chat credits to %s", self._path, exc_info=True)

    # -- helpers -------------------------------------------------------------

    def _cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(seconds=self._window)

    def _prune(self, uid: str) -> list[str]:
        """Drop log entries older than the window; return the live ones."""
        cutoff = self._cutoff()
        live: list[str] = []
        for ts in self._log.get(uid, []):
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    live.append(ts)
            except ValueError:
                continue
        if live:
            self._log[uid] = live
        else:
            self._log.pop(uid, None)
        return live

    # -- user config ---------------------------------------------------------

    def set_user(self, uid: str, *, limit: int) -> None:
        u = self._users.setdefault(uid, {})
        u["limit"] = int(limit)
        u.setdefault("blocked", False)
        self._save()

    def block(self, uid: str) -> None:
        self._users.setdefault(uid, {})["blocked"] = True
        self._save()

    def unblock(self, uid: str) -> None:
        if uid in self._users:
            self._users[uid]["blocked"] = False
            self._save()

    def is_blocked(self, uid: str) -> bool:
        return bool(self._users.get(uid, {}).get("blocked", False))

    def limit(self, uid: str) -> int:
        u = self._users.get(uid)
        if u is None:
            return self._default_limit
        if u.get("blocked"):
            return 0
        return int(u.get("limit", self._default_limit))

    # -- credits -------------------------------------------------------------

    def used(self, uid: str) -> int:
        return len(self._prune(uid))

    def remaining(self, uid: str) -> int:
        return max(0, self.limit(uid) - self.used(uid))

    def next_return(self, uid: str) -> datetime | None:
        """When the oldest in-window request falls out, freeing a credit."""
        live = self._prune(uid)
        if not live:
            return None
        oldest = min(datetime.fromisoformat(ts) for ts in live)
        return oldest + timedelta(seconds=self._window)

    def consume(self, uid: str) -> None:
        self._log.setdefault(uid, []).append(datetime.now(timezone.utc).isoformat())
        self._save()
