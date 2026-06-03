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

__all__ = ["InputPolicy", "PolicyDecision", "AllowAll", "OperatorGate"]


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
