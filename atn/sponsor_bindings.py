"""Sponsor → dependent binding store ("work AI").

A *sponsor* agent supplies LLM inference to a *dependent* agent running on a
different daemon. The sponsor is the resource owner, so it is the authority:
it holds, locally, the set of dependent addresses it is willing to serve and
the token budget each may consume.

A binding is keyed by the dependent agent's on-chain address (its 0x wallet).
Everything the dependent spawns presents as that same address and draws on the
same budget — the sponsor sees one single thread per dependent, not a tree.

Budget is denominated in **tokens** (input + output) in v1. ATN settlement is
layered on later with the payment work; nothing here touches the chain.

Persisted as JSON at ``data_dir/sponsor_bindings.json`` (mirrors
``CreditBudgetStore``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(address: str) -> str:
    """Normalize an address for use as a binding key (lowercase, stripped)."""
    return (address or "").strip().lower()


@dataclass
class SponsorBinding:
    """A sponsor's grant to one dependent agent.

    ``budget_tokens`` of 0 means *unlimited* (serve until removed).
    """

    dependent_address: str
    budget_tokens: int = 0
    spent_tokens: int = 0
    label: str = ""
    created_at: str = field(default_factory=_now_iso)

    @property
    def unlimited(self) -> bool:
        return self.budget_tokens <= 0

    def remaining(self) -> int:
        """Remaining tokens; -1 signals unlimited."""
        if self.unlimited:
            return -1
        return max(0, self.budget_tokens - self.spent_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependent_address": self.dependent_address,
            "budget_tokens": self.budget_tokens,
            "spent_tokens": self.spent_tokens,
            "label": self.label,
            "created_at": self.created_at,
            "remaining_tokens": self.remaining(),
            "unlimited": self.unlimited,
        }


class SponsorBindingStore:
    """Sponsor-local registry of dependent grants with JSON persistence.

    Usage:
        store = SponsorBindingStore(data_dir)
        store.add("0xDEAD...", budget_tokens=1_000_000, label="alice's agent")
        if store.is_authorized("0xdead..."):  # case-insensitive
            ...
        store.record_spend("0xDEAD...", 1200)
        store.remaining("0xDEAD...")  # 998800
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / "sponsor_bindings.json"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._bindings: dict[str, SponsorBinding] = {}
        self._load()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, dependent_address: str) -> SponsorBinding | None:
        return self._bindings.get(_norm(dependent_address))

    def list(self) -> list[SponsorBinding]:
        return list(self._bindings.values())

    def is_authorized(self, dependent_address: str) -> bool:
        """True iff there is a binding for this address with budget left."""
        b = self.get(dependent_address)
        if b is None:
            return False
        return b.unlimited or b.remaining() > 0

    def remaining(self, dependent_address: str) -> int:
        """Remaining tokens for a dependent; -1 unlimited, 0 if unbound."""
        b = self.get(dependent_address)
        if b is None:
            return 0
        return b.remaining()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        dependent_address: str,
        budget_tokens: int = 0,
        label: str = "",
    ) -> SponsorBinding:
        """Create or update a binding. Updating resets neither spend nor
        created_at — only budget/label change."""
        key = _norm(dependent_address)
        if not key:
            raise ValueError("dependent_address is required")
        existing = self._bindings.get(key)
        if existing:
            existing.budget_tokens = budget_tokens
            if label:
                existing.label = label
        else:
            self._bindings[key] = SponsorBinding(
                dependent_address=dependent_address,
                budget_tokens=budget_tokens,
                label=label,
            )
        self._save()
        return self._bindings[key]

    def update_budget(self, dependent_address: str, budget_tokens: int) -> SponsorBinding | None:
        b = self.get(dependent_address)
        if b is None:
            return None
        b.budget_tokens = budget_tokens
        self._save()
        return b

    def record_spend(self, dependent_address: str, tokens: int) -> None:
        """Add to a dependent's spend. No-op if unbound."""
        b = self.get(dependent_address)
        if b is None:
            return
        b.spent_tokens += max(0, int(tokens))
        self._save()

    def remove(self, dependent_address: str) -> bool:
        key = _norm(dependent_address)
        if key in self._bindings:
            del self._bindings[key]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def to_summary_list(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in self._bindings.values()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for key, d in raw.items():
                self._bindings[key] = SponsorBinding(
                    dependent_address=d.get("dependent_address", key),
                    budget_tokens=d.get("budget_tokens", 0),
                    spent_tokens=d.get("spent_tokens", 0),
                    label=d.get("label", ""),
                    created_at=d.get("created_at", _now_iso()),
                )
        except Exception:
            log.warning("Failed to load sponsor bindings from %s", self._path, exc_info=True)

    def _save(self) -> None:
        data: dict[str, Any] = {}
        for key, b in self._bindings.items():
            data[key] = {
                "dependent_address": b.dependent_address,
                "budget_tokens": b.budget_tokens,
                "spent_tokens": b.spent_tokens,
                "label": b.label,
                "created_at": b.created_at,
            }
        try:
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed to save sponsor bindings to %s", self._path)
