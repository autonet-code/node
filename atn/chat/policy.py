"""The input seam policy -- the one dОrg-specific layer.

Per the design doc (§2.4): access control, credits, whitelist, AND eventual
rep-weighted distributed input ALL live here, at the point where a platform
message becomes a send_agent_message. autonet core and the orchestrator model
stay pristine; this is the only place the dОrg deployment differs from stock
single-user autonet.

Phase 1 is just an operator gate (mirrors fleet_bot's OPERATOR_USER_ID): one
authorized human may direct the fleet; everyone else is politely declined.
Credits and rep-weighted consensus are deferred -- but they slot in HERE, as
richer InputPolicy implementations, without touching the ChatService or
autonet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .protocol import Author


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of evaluating an inbound message at the input seam."""

    allow: bool
    reason: str = ""          # human-readable; shown to the author when denied


class InputPolicy(Protocol):
    """Decides whether an inbound platform message may reach an agent.

    Implementations: operator-gate (now), credit-metered (member agents),
    rep-weighted consensus (the dОrg distributed-input endgame).
    """

    def evaluate(self, author: "Author", agent_id: str, content: str) -> PolicyDecision:
        ...


class AllowAll(InputPolicy):
    """No gating -- single-user / dev / stub use."""

    def evaluate(self, author: "Author", agent_id: str, content: str) -> PolicyDecision:
        return PolicyDecision(allow=True)


class OperatorGate(InputPolicy):
    """Only the configured operator user(s) may command the fleet.

    Mirrors fleet_bot.py's gate. The default operator id is Eight Rice's
    Discord id (938049028757807135); pass operator_ids to override.
    """

    DEFAULT_OPERATOR_IDS = frozenset({"938049028757807135"})
    DENY_MESSAGE = (
        "Sorry — directing the fleet is operator-gated right now. "
        "Ask the operator to relay your request."
    )

    def __init__(self, operator_ids: frozenset[str] | set[str] | None = None) -> None:
        self.operator_ids = frozenset(operator_ids) if operator_ids else self.DEFAULT_OPERATOR_IDS

    def evaluate(self, author: "Author", agent_id: str, content: str) -> PolicyDecision:
        if author.is_bot:
            return PolicyDecision(allow=False, reason="(bot author ignored)")
        if author.id in self.operator_ids:
            return PolicyDecision(allow=True)
        return PolicyDecision(allow=False, reason=self.DENY_MESSAGE)
