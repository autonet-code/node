"""Uniform budget pathway — one function answering "what are this agent's
effective limits" regardless of where the limit comes from.

An agent with an explicit per-agent budget and an agent with NONE must surface
their limits through the SAME rail and the SAME verbiage. For a budgeted agent
the limits are the per-agent caps (from ``AgentRegistry.get_budget_info``). For
an UNBUDGETED agent the effective "budget" is the daemon-wide ceiling that
actually binds it:

  * metered API provider → the daemon-wide DOLLAR cap (metering service), and
  * subscription provider → the inferred subscription remaining allocation.

Both cases return the same ``EffectiveLimits`` shape so the caller (the
``get_my_budget_status`` tool, the prompt/context surface) renders one verbiage
with no branch on "has budget vs not".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Provider names that are subscription-billed (no per-token dollar price).
# Kept in sync with metering.SUBSCRIPTION_PROVIDERS.
_SUBSCRIPTION_PROVIDERS = frozenset({"claude_max", "codex_max"})

# Known explicit provider names — a defn.provider that is one of these is used
# verbatim; anything else is treated as a model id and resolved to its channel.
_KNOWN_PROVIDER_NAMES = frozenset({
    "claude_max", "codex_max", "anthropic", "openai", "gemini",
    "deepseek", "ollama", "rpb", "substrate",
})


@dataclass
class EffectiveLimits:
    """The limits that actually bind an agent, from whatever source.

    ``source`` is "agent_budget" (per-agent cap declared on the agent) or
    "daemon_provider" (no per-agent budget; the daemon-wide provider ceiling
    binds instead). ``entries`` is a list of per-provider limit descriptors,
    each with a uniform set of fields so the same verbiage renders either way.
    """
    agent_id: str
    source: str = "daemon_provider"
    provider: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "source": self.source,
            "provider": self.provider,
            "limits": self.entries,
        }


def humanize_magnitude(n: float) -> str:
    """Short human-readable magnitude: 512 -> '512', 38_200 -> '38.2k',
    1_200_000 -> '1.2M'. Best-effort; never raises."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0"
    neg = n < 0
    n = abs(n)
    if n < 1000:
        s = str(int(round(n)))
    elif n < 1_000_000:
        s = f"{n / 1000:.1f}k"
    else:
        s = f"{n / 1_000_000:.1f}M"
    return f"-{s}" if neg else s


def format_budget_line(
    limits: "EffectiveLimits",
    last_turn_tokens: int | None = None,
) -> str:
    """Compose the one-line budget stamp injected into the newest incoming
    message so an agent sees its economic state at every decision point.

    Returns ``[Budget: unlimited]`` when no bounded budget binds the agent, else
    ``[Budget: <remaining> credits remaining | last turn: ~<n>]``. The last-turn
    segment is dropped when there is no prior metered turn (or the budget is
    unlimited). Pure — safe to unit-test in isolation.
    """
    entry = limits.entries[0] if getattr(limits, "entries", None) else None
    remaining_part = "unlimited"
    if entry is not None:
        limit = entry.get("limit")
        remaining = entry.get("remaining")
        unit = entry.get("unit", "tokens")
        if (limit is not None and limit > 0
                and remaining is not None and remaining >= 0):
            if unit == "usd":
                remaining_part = f"${remaining:,.2f} remaining"
            else:
                remaining_part = f"{humanize_magnitude(remaining)} credits remaining"

    line = f"[Budget: {remaining_part}"
    # Only surface last-turn cost alongside a bounded budget — an unlimited
    # agent's stamp stays the terse ``[Budget: unlimited]`` per spec.
    if remaining_part != "unlimited" and last_turn_tokens:
        line += f" | last turn: ~{humanize_magnitude(last_turn_tokens)}"
    return line + "]"


def _agent_provider_name(defn: Any, config: Any) -> str:
    """Best-effort resolve the provider NAME an agent runs on, WITHOUT
    instantiating a provider. Uses the explicit provider field when it names a
    known provider, else resolves the model id to its default channel."""
    provider = getattr(defn, "provider", "") or ""
    if isinstance(provider, list):
        provider = provider[0] if provider else ""
    if isinstance(provider, str) and provider in _KNOWN_PROVIDER_NAMES:
        return provider
    # Otherwise `provider` is (or falls back to) a model id → resolve channel.
    model = (
        (isinstance(provider, str) and provider)
        or getattr(defn, "cognitive_model", "")
        or (getattr(getattr(config, "orchestrator", None), "model", "") or "")
        or "claude-sonnet-4-6"
    )
    try:
        from .model_specs import resolve as _resolve_model
        spec = _resolve_model(model)
        if spec.default_channel:
            return spec.default_channel
    except Exception:
        pass
    return "claude_max"


def compute_effective_limits(
    agent_id: str,
    *,
    registry: Any,
    metering: Any = None,
    config: Any = None,
) -> EffectiveLimits:
    """Return the effective limits for an agent through the uniform rail.

    If the agent declares a per-agent budget, report those caps (source =
    "agent_budget"). Otherwise report the daemon-wide provider ceiling that
    binds it (source = "daemon_provider"): a dollar cap for a metered provider,
    or the inferred subscription remaining for a subscription provider.

    Pure over the injected ``registry`` / ``metering`` / ``config`` — the
    caller passes the runtime's instances. Never raises; degrades to an empty
    ``entries`` list if a source is unavailable.
    """
    defn = registry.get_agent(agent_id) if registry else None

    # --- Case 1: the agent has its own budget → report those caps. ----------
    own = {}
    try:
        own = registry.get_budget_info(agent_id) if registry else {}
    except Exception:
        own = {}
    if own:
        entries: list[dict[str, Any]] = []
        for key, info in own.items():
            entries.append({
                "key": key,
                "provider": info.get("provider", key),
                "scope": info.get("scope", "provider"),
                "unit": info.get("unit", "tokens"),
                "limit": info.get("limit"),
                "used": info.get("used"),
                "remaining": info.get("remaining"),
                "period": info.get("period"),
                "origin": "agent_budget",
            })
        return EffectiveLimits(
            agent_id=agent_id, source="agent_budget",
            provider=(entries[0]["provider"] if entries else ""),
            entries=entries,
        )

    # --- Case 2: no per-agent budget → the daemon-wide ceiling binds. -------
    provider = _agent_provider_name(defn, config) if defn else "claude_max"
    entries = []

    if provider in _SUBSCRIPTION_PROVIDERS:
        # Subscription provider: the inferred quota's remaining allocation is
        # this agent's effective budget. Expressed in tokens (the same unit a
        # budgeted agent's caps use) so the verbiage is uniform.
        est = None
        if metering is not None:
            try:
                est = metering.estimate_quota(provider)
            except Exception:
                est = None
        limit = getattr(est, "quota_tokens", None) if est else None
        remaining = getattr(est, "remaining_tokens", None) if est else None
        used = None
        if limit is not None and remaining is not None:
            used = max(0.0, limit - remaining)
        entries.append({
            "key": provider,
            "provider": provider,
            "scope": "provider",
            "unit": "tokens",
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "period": "subscription_window",
            "origin": "daemon_subscription",
            "confidence": (round(getattr(est, "confidence", 0.0), 4)
                           if est else 0.0),
            "current_utilization": (getattr(est, "current_utilization", None)
                                    if est else None),
        })
    else:
        # Metered API provider: the daemon-wide DOLLAR cap binds. Expressed in
        # USD (a dollar cap has no token figure), but through the same fields.
        posture = None
        if metering is not None:
            try:
                posture = metering.provider_spend(provider)
            except Exception:
                posture = None
        limit = posture.get("limit_usd") if posture else None
        entries.append({
            "key": provider,
            "provider": provider,
            "scope": "provider",
            "unit": "usd",
            "limit": (limit if limit and limit > 0 else None),
            "used": posture.get("spent_usd") if posture else None,
            "remaining": posture.get("remaining_usd") if posture else None,
            "period": posture.get("period") if posture else "none",
            "origin": "daemon_dollar_cap",
        })

    return EffectiveLimits(
        agent_id=agent_id, source="daemon_provider",
        provider=provider, entries=entries,
    )
