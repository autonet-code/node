"""Concrete input-seam policies for the chat Surface.

The InputPolicy / PolicyDecision contract is the Surface-level gating seam (see
atn/surface.py) — re-exported here for convenience. This module provides the
concrete gates the chat Surface ships with:

  - AllowAll     — no gating (single-user / dev / stub).
  - OperatorGate — only configured operator user(s) may reach the agent.

Richer gates (credit meters, rep-weighted consensus) slot in as additional
InputPolicy implementations — supplied by the deployment to start_chat(), or
living in a connector the policy consults. autonet core stays generic; gating
lives at the Surface input seam, never in a connector's outbound tools.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Canonical contract lives with the Surface entity.
from ..surface import InputPolicy, PolicyDecision

if TYPE_CHECKING:
    from .protocol import Author
    from .credits import CreditStore

__all__ = ["InputPolicy", "PolicyDecision", "AllowAll", "OperatorGate", "CreditPolicy"]


class AllowAll:
    """No gating -- single-user / dev / stub use."""

    def evaluate(self, author: Any, agent_id: str, content: str) -> PolicyDecision:
        return PolicyDecision(allow=True)


class OperatorGate:
    """Only the configured operator user(s) may reach the agent.

    Whitelabel: no default operator — pass the deployment's operator ids. An
    empty set declines everyone (fail-closed), so configure it explicitly.
    """

    DENY_MESSAGE = (
        "Sorry — this is operator-gated right now. "
        "Ask the operator to relay your request."
    )

    def __init__(self, operator_ids: frozenset[str] | set[str] | None = None) -> None:
        self.operator_ids = frozenset(operator_ids or set())

    def evaluate(self, author: Any, agent_id: str, content: str) -> PolicyDecision:
        if getattr(author, "is_bot", False):
            return PolicyDecision(allow=False, reason="(bot author ignored)")
        if getattr(author, "id", None) in self.operator_ids:
            return PolicyDecision(allow=True)
        return PolicyDecision(allow=False, reason=self.DENY_MESSAGE)


class CreditPolicy:
    """Rolling-credit gate backed by a Surface-owned CreditStore.

    Encodes the whole member-vs-operator distinction as credits — there is no
    separate operator flag:
      - operator(s): unlimited (never blocked, never consumes a credit),
      - explicitly blocked users: denied,
      - everyone else: a rolling-window allowance; allowed (one credit consumed)
        while credits remain, denied with a "credits return at…" hint when out.
    """

    def __init__(self, store: "CreditStore",
                 operator_ids: frozenset[str] | set[str] | None = None) -> None:
        self.store = store
        self.operator_ids = frozenset(operator_ids or set())

    def evaluate(self, author: Any, agent_id: str, content: str) -> PolicyDecision:
        if getattr(author, "is_bot", False):
            return PolicyDecision(allow=False, reason="(bot author ignored)")
        uid = str(getattr(author, "id", "") or "")
        if not uid:
            return PolicyDecision(allow=False, reason="(no author id)")
        # Operator: unlimited.
        if uid in self.operator_ids:
            return PolicyDecision(allow=True)
        if self.store.is_blocked(uid):
            return PolicyDecision(
                allow=False,
                reason="Sorry — you don't have access to this assistant right now.")
        if self.store.remaining(uid) <= 0:
            when = self.store.next_return(uid)
            hint = f" Your next credit frees up around {when:%H:%M UTC}." if when else ""
            return PolicyDecision(
                allow=False, reason=f"You're out of credits for now.{hint}")
        self.store.consume(uid)
        return PolicyDecision(allow=True)
