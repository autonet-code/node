"""Credit budget tracking for AI providers.

Tracks per-provider token budgets so the planning loop can:
  1. Know how much budget remains in the current period
  2. Proactively allocate unused budget to goal-aligned tasks
  3. Reset counters at period boundaries

Persisted as JSON at ``data_dir/budgets.json``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import CreditBudget

log = logging.getLogger(__name__)

_PERIOD_SECONDS = {
    "daily": 86_400,
    "weekly": 604_800,
    "monthly": 2_592_000,   # 30 days
}


class CreditBudgetStore:
    """Manages per-provider credit budgets with JSON persistence.

    Usage:
        store = CreditBudgetStore(data_dir)
        store.set_budget("claude_max", token_limit=500_000, period="monthly")
        store.record_usage("claude_max", 1200)
        util = store.get_utilization()  # {"claude_max": 0.0024}
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "budgets.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._budgets: dict[str, CreditBudget] = {}
        self._load()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_budget(self, provider: str) -> CreditBudget | None:
        """Get budget for a provider, or None if not configured."""
        self._check_period_resets()
        return self._budgets.get(provider)

    def get_all(self) -> list[CreditBudget]:
        """Return all configured budgets."""
        self._check_period_resets()
        return list(self._budgets.values())

    def get_utilization(self) -> dict[str, float]:
        """Return utilization percentage (0.0–1.0) per provider."""
        self._check_period_resets()
        result: dict[str, float] = {}
        for pid, b in self._budgets.items():
            if b.token_limit > 0:
                result[pid] = min(1.0, b.tokens_used / b.token_limit)
            else:
                result[pid] = 0.0
        return result

    def remaining_budget(self) -> dict[str, int]:
        """Return remaining tokens per provider (limit - used)."""
        self._check_period_resets()
        result: dict[str, int] = {}
        for pid, b in self._budgets.items():
            if b.token_limit > 0:
                result[pid] = max(0, b.token_limit - b.tokens_used)
            else:
                result[pid] = -1  # unlimited
        return result

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set_budget(
        self,
        provider: str,
        token_limit: int,
        period: str = "monthly",
        auto_allocate: bool = True,
    ) -> CreditBudget:
        """Configure (or update) the budget for a provider."""
        existing = self._budgets.get(provider)
        if existing:
            existing.token_limit = token_limit
            existing.period = period
            existing.auto_allocate = auto_allocate
        else:
            self._budgets[provider] = CreditBudget(
                provider=provider,
                period=period,
                token_limit=token_limit,
                auto_allocate=auto_allocate,
            )
        self._save()
        return self._budgets[provider]

    def remove_budget(self, provider: str) -> bool:
        """Remove budget for a provider.  Returns True if it existed."""
        if provider in self._budgets:
            del self._budgets[provider]
            self._save()
            return True
        return False

    def record_usage(self, provider: str, tokens: int) -> None:
        """Record token usage for a provider.  No-op if no budget configured."""
        self._check_period_resets()
        b = self._budgets.get(provider)
        if b is None:
            return
        b.tokens_used += tokens
        self._save()

    def reset_period(self, provider: str) -> None:
        """Manually reset the period counter for a provider."""
        b = self._budgets.get(provider)
        if b is None:
            return
        b.tokens_used = 0
        b.period_start = datetime.now(timezone.utc)
        self._save()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> dict[str, Any]:
        """Lightweight summary for the snapshot."""
        self._check_period_resets()
        result: dict[str, Any] = {}
        for pid, b in self._budgets.items():
            result[pid] = {
                "period": b.period,
                "limit": b.token_limit,
                "used": b.tokens_used,
                "utilization": round(b.tokens_used / b.token_limit, 4) if b.token_limit > 0 else 0.0,
                "auto_allocate": b.auto_allocate,
            }
        return result

    # ------------------------------------------------------------------
    # Period management
    # ------------------------------------------------------------------

    def _check_period_resets(self) -> None:
        """Auto-reset counters if a period boundary has passed."""
        now = datetime.now(timezone.utc)
        changed = False
        for b in self._budgets.values():
            period_secs = _PERIOD_SECONDS.get(b.period, _PERIOD_SECONDS["monthly"])
            if (now - b.period_start).total_seconds() >= period_secs:
                b.tokens_used = 0
                b.period_start = now
                changed = True
                log.info("Credit budget period reset for %s (%s)", b.provider, b.period)
        if changed:
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for pid, d in raw.items():
                ps = d.get("period_start")
                self._budgets[pid] = CreditBudget(
                    provider=pid,
                    period=d.get("period", "monthly"),
                    token_limit=d.get("token_limit", 0),
                    tokens_used=d.get("tokens_used", 0),
                    period_start=datetime.fromisoformat(ps) if ps else datetime.now(timezone.utc),
                    auto_allocate=d.get("auto_allocate", True),
                )
        except Exception:
            log.warning("Failed to load budgets from %s", self._path, exc_info=True)

    def _save(self) -> None:
        data: dict[str, Any] = {}
        for pid, b in self._budgets.items():
            data[pid] = {
                "period": b.period,
                "token_limit": b.token_limit,
                "tokens_used": b.tokens_used,
                "period_start": b.period_start.isoformat(),
                "auto_allocate": b.auto_allocate,
            }
        try:
            self._path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except Exception:
            log.exception("Failed to save budgets to %s", self._path)
