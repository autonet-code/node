"""Which provider an agent actually runs on, resolved WITHOUT instantiating it.

Budgets are keyed by provider id, so the budget key for an agent must be the
provider its runs are actually booked against. That is not always the
agent's ``provider`` field: an unpinned agent stores its creation-time MODEL
there (a routing hint), a fallback chain names several providers, and a
marketplace binding overrides both. This module mirrors
``ProviderManager.resolve_provider_with_fallback`` as a pure function so the
snapshot, ``get_agent``, the effective-limits rail and the budget-key
migration all name the same provider the execution engine books against.

No call site may substitute a fixed provider name for "the default provider";
the default is whatever this resolver returns for an unpinned agent.
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

# Short aliases the Claude routes accept (mirrors _resolve_provider_for_model).
_CLAUDE_ALIASES = ("sonnet", "opus", "haiku", "fable", "mythos")

# resolve_provider_with_fallback's model when neither the agent nor the
# daemon names one.
FALLBACK_MODEL = "claude-sonnet-4-6"

# The provider id older clients wrote as a stand-in for "whatever the daemon
# default is". Only the migration below may reference it.
LEGACY_DEFAULT_BUDGET_KEY = "claude_max"


def provider_for_model(model: str, *, has_api_key: Callable[[str], bool]) -> str:
    """The provider a bare model id routes to (``_resolve_provider_for_model``).

    An unknown prefix can only succeed on ollama (anything else fails loud at
    run time), so it maps to ``ollama`` without probing the local server.
    """
    m = (model or "").lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude") or m.startswith(_CLAUDE_ALIASES):
        return "anthropic" if has_api_key("anthropic") else "claude_max"
    return "ollama"


def _usable(pid: str, has_api_key: Callable[[str], bool],
            custom: frozenset[str]) -> bool:
    """Would ``_resolve_provider_by_name(pid)`` build without raising?"""
    if pid in _API_KEY_PROVIDERS:
        return has_api_key(pid)
    return pid in KNOWN_PROVIDER_IDS or pid in custom


def effective_provider_id(
    defn: Any,
    *,
    default_model: str = "",
    has_api_key: Callable[[str], bool] = lambda _pid: False,
    custom_providers: Iterable[str] = (),
) -> str:
    """The provider id an agent's runs are booked against."""
    custom = frozenset(custom_providers or ())
    binding = getattr(defn, "service_provider", None)
    if isinstance(binding, dict) and binding:
        return "service"
    raw = getattr(defn, "provider", None)
    model = (getattr(defn, "cognitive_model", "") or default_model
             or FALLBACK_MODEL)
    if isinstance(raw, list):
        names = [p for p in raw if isinstance(p, str) and p]
        for pid in names:
            if _usable(pid, has_api_key, custom):
                return pid
        if names:
            return names[0]
        return provider_for_model(model, has_api_key=has_api_key)
    if isinstance(raw, str) and raw:
        if raw in KNOWN_PROVIDER_IDS or raw in custom:
            return raw
        # Model-shaped routing hint: the engine routes by cognitive_model.
        return provider_for_model(
            getattr(defn, "cognitive_model", "") or raw,
            has_api_key=has_api_key)
    return provider_for_model(model, has_api_key=has_api_key)


def unpinned_provider_id(
    defn: Any,
    *,
    default_model: str = "",
    has_api_key: Callable[[str], bool] = lambda _pid: False,
) -> str:
    """The provider this agent would run on with its pin cleared (the
    "daemon default" choice in the UI): routed by its model."""
    model = (getattr(defn, "cognitive_model", "") or default_model
             or FALLBACK_MODEL)
    return provider_for_model(model, has_api_key=has_api_key)


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
