"""Which provider an agent actually runs on, resolved WITHOUT instantiating it.

Budgets are keyed by provider id, so the budget key for an agent must be the
provider its runs are actually booked against. That is the agent's own
``provider`` pin, except that a fallback chain names several providers and a
marketplace binding overrides both. This module mirrors
``ProviderManager.resolve_provider_with_fallback`` as a pure function so the
snapshot, ``get_agent``, the effective-limits rail and the budget-key
migration all name the same provider the execution engine books against.

There is no daemon-wide default: every cognitive agent names its provider and
model on purpose (decision 2026-10-06). An agent without both is an error,
never silently routed.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

# Provider ids the daemon knows how to build (ProviderManager._KNOWN_PROVIDERS).
KNOWN_PROVIDER_IDS = frozenset({
    "claude_max", "codex_max", "anthropic", "gemini", "openai", "deepseek",
    "ollama", "rpb", "service", "substrate",
})

# Providers that can only be built with an API key on file.
_API_KEY_PROVIDERS = frozenset({"anthropic", "gemini", "openai", "deepseek"})

# The provider id older clients wrote as a stand-in for "whatever the daemon
# picks". Only the budget-key migration below may reference it.
LEGACY_DEFAULT_BUDGET_KEY = "claude_max"


def _usable(pid: str, has_api_key: Callable[[str], bool],
            custom: frozenset[str]) -> bool:
    """Would ``_resolve_provider_by_name(pid)`` build without raising?"""
    if pid in _API_KEY_PROVIDERS:
        return has_api_key(pid)
    return pid in KNOWN_PROVIDER_IDS or pid in custom


def routing_error(
    provider: Any, model: Any, *, custom_providers: Iterable[str] = (),
    service_bound: bool = False,
) -> str | None:
    """Why a cognitive agent with this provider/model cannot run, or None.

    Both must be chosen deliberately: a non-empty provider the daemon knows
    (a built-in id, a custom provider id, or a non-empty fallback chain of
    them) and a non-empty model. Used at create/update time and again at
    resolve time, so a hand-edited agent.yaml fails loud instead of being
    routed somewhere nobody chose.

    Two choices name the model by themselves: a per-agent marketplace binding
    (``service_bound``) and the ``service`` provider. The seller declares the
    served model, so neither needs a model of its own.
    """
    if service_bound:
        return None
    custom = frozenset(custom_providers or ())
    if isinstance(provider, list):
        names = [p for p in provider if isinstance(p, str) and p]
        if not names:
            return "provider is required: choose the provider this agent runs on"
        unknown = [p for p in names
                   if p not in KNOWN_PROVIDER_IDS and p not in custom]
        if unknown:
            return f"unknown provider(s): {', '.join(unknown)}"
    elif not (isinstance(provider, str) and provider.strip()):
        return "provider is required: choose the provider this agent runs on"
    elif provider not in KNOWN_PROVIDER_IDS and provider not in custom:
        return (f"unknown provider {provider!r}: use a built-in provider id "
                "or a configured custom provider")
    if provider == "service":
        return None
    if not (isinstance(model, str) and model.strip()):
        return "model is required: choose the model this agent runs on"
    return None


def effective_provider_id(
    defn: Any,
    *,
    has_api_key: Callable[[str], bool] = lambda _pid: False,
    custom_providers: Iterable[str] = (),
) -> str:
    """The provider id an agent's runs are booked against ("" when the agent
    names none, which is a routing error the engine reports at run time)."""
    custom = frozenset(custom_providers or ())
    binding = getattr(defn, "service_provider", None)
    if isinstance(binding, dict) and binding:
        return "service"
    raw = getattr(defn, "provider", None)
    if isinstance(raw, list):
        names = [p for p in raw if isinstance(p, str) and p]
        for pid in names:
            if _usable(pid, has_api_key, custom):
                return pid
        return names[0] if names else ""
    if isinstance(raw, str) and raw and (raw in KNOWN_PROVIDER_IDS
                                         or raw in custom):
        return raw
    return ""


def explicit_provider_ids(defn: Any) -> set[str]:
    """Provider ids the agent names on purpose (pin or fallback chain)."""
    raw = getattr(defn, "provider", None)
    if isinstance(raw, list):
        return {p for p in raw if isinstance(p, str) and p in KNOWN_PROVIDER_IDS}
    if isinstance(raw, str) and raw in KNOWN_PROVIDER_IDS:
        return {raw}
    return set()


def migrate_legacy_budget_keys(defn: Any, effective: str) -> dict[str, str]:
    """Move budgets keyed under the legacy default-provider stand-in to the
    provider the agent actually runs on.

    Older clients keyed an unpinned agent's budget under ``claude_max`` no
    matter which provider the daemon routed it to, so the cap never bound.
    Moves ``claude_max`` -> ``effective`` and ``claude_max:<model>`` ->
    ``effective:<model>`` when the agent does not run on (or name) claude_max
    and the target key is free. Never drops a limit: a key whose target is
    already taken stays where it is. Mutates ``defn.budgets`` in place and
    returns ``{old_key: new_key}`` for every key moved.
    """
    budgets = getattr(defn, "budgets", None)
    legacy = LEGACY_DEFAULT_BUDGET_KEY
    if not isinstance(budgets, dict) or not budgets or not effective:
        return {}
    if effective == legacy or legacy in explicit_provider_ids(defn):
        return {}
    moved: dict[str, str] = {}
    for key in list(budgets.keys()):
        if key == legacy:
            new_key = effective
        elif isinstance(key, str) and key.startswith(legacy + ":"):
            new_key = effective + key[len(legacy):]
        else:
            continue
        if new_key in budgets:
            continue
        budgets[new_key] = budgets.pop(key)
        moved[key] = new_key
    return moved
