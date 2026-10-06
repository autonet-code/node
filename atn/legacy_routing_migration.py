"""One-time migration: pin every cognitive agent to an explicit provider+model.

Decision 2026-10-06: model selection is DELIBERATE. There is no daemon-wide
default provider/model, no inheritance from a parent and no hardcoded
fallback model; an agent without a provider or a model cannot run.

Agents written before that decision could leave either field empty (or hold a
model id in ``provider`` as a routing hint) and were routed at run time by the
old rules: ``model = cognitive_model or defaults.model or claude-sonnet-4-6``,
provider guessed from the model prefix. So nothing changes under the user,
this migration computes what each such agent resolves to TODAY under those
old rules, ONCE, and persists it into the agent's YAML.

It consumes the retired config.yaml ``defaults:`` section (which the
orchestrator migration in config.py produced from an even older
``orchestrator:`` section) and then deletes it. A stamp file in data_dir makes
it run once per install; afterwards an unpinned agent (e.g. a hand-edited
YAML) is an error, never silently routed.

This module is the ONLY place the old routing rules survive.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .models import AgentMode
from .provider_identity import KNOWN_PROVIDER_IDS, routing_error

if TYPE_CHECKING:
    from .config import ATNConfig

log = logging.getLogger(__name__)

STAMP_NAME = ".agent_routing_pinned"

# The old rules' last-resort model when neither the agent nor the daemon's
# ``defaults:`` section named one. Only this migration may use it.
_LEGACY_FALLBACK_MODEL = "claude-sonnet-4-6"
# Short aliases the old model-prefix routing treated as Claude models.
_LEGACY_CLAUDE_ALIASES = ("sonnet", "opus", "haiku", "fable", "mythos")


def _legacy_provider_for_model(
    model: str, *, has_api_key: Callable[[str], bool],
) -> str:
    """The provider the old rules routed a bare model id to."""
    m = (model or "").lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude") or m.startswith(_LEGACY_CLAUDE_ALIASES):
        return "anthropic" if has_api_key("anthropic") else "claude_max"
    return "ollama"


def legacy_resolved_route(
    defn: Any,
    *,
    legacy_model: str = "",
    has_api_key: Callable[[str], bool] = lambda _pid: False,
    custom_providers: Iterable[str] = (),
) -> tuple[Any, str]:
    """``(provider, model)`` an agent resolved to under the pre-2026-10-06
    rules (ProviderManager.resolve_provider_with_fallback as it was)."""
    custom = frozenset(custom_providers or ())
    own_model = getattr(defn, "cognitive_model", "") or ""
    model = own_model or legacy_model or _LEGACY_FALLBACK_MODEL
    raw = getattr(defn, "provider", None)
    if isinstance(raw, list):
        if [p for p in raw if isinstance(p, str) and p]:
            return raw, model
        return _legacy_provider_for_model(model, has_api_key=has_api_key), model
    if isinstance(raw, str) and raw:
        if raw in KNOWN_PROVIDER_IDS or raw in custom:
            return raw, model
        # Model-shaped routing hint: the old engine routed by cognitive_model
        # (else the hint itself) and ran on that same model.
        route_model = own_model or raw
        return (_legacy_provider_for_model(route_model, has_api_key=has_api_key),
                route_model)
    return _legacy_provider_for_model(model, has_api_key=has_api_key), model


def _drop_legacy_defaults_section(path: Path | None) -> bool:
    """Delete the top-level ``defaults:`` section from config.yaml. Text-level
    so the rest of the file keeps its comments and layout. Returns True when
    the file was rewritten."""
    if path is None or not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    dropped = False
    while i < len(lines):
        line = lines[i]
        if re.match(r"^defaults:", line):
            dropped = True
            i += 1
            # Block form: swallow the indented body (and blank lines inside it).
            while i < len(lines) and (
                    lines[i][:1] in (" ", "\t") or not lines[i].strip()):
                i += 1
            continue
        out.append(line)
        i += 1
    if not dropped:
        return False
    try:
        path.write_text("".join(out), encoding="utf-8")
    except OSError as exc:
        log.warning("config: could not remove the retired defaults: section "
                    "from %s: %s", path, exc)
        return False
    log.info("config: removed the retired defaults: section from %s", path)
    return True


def pin_unpinned_agents_from_legacy_defaults(
    agents: Iterable[Any], config: "ATNConfig", providers: Any = None,
) -> list[str]:
    """One-time: give every cognitive agent that lacks a provider or model the
    route it resolved to under the old rules, persist it, then drop the
    retired ``defaults:`` config section. Call at boot with the on-disk
    agents BEFORE they are registered (so budget-key migration sees the
    pinned provider). Returns the ids pinned. No-op once stamped."""
    stamp = Path(config.data_dir) / STAMP_NAME
    if stamp.exists():
        return []

    if providers is not None and callable(getattr(providers, "_key_probe", None)):
        has_api_key = providers._key_probe()
    else:
        has_api_key = lambda _pid: False  # noqa: E731
    custom = frozenset(getattr(providers, "_custom_providers", None) or ())
    legacy_model = (config.legacy_agent_routing or {}).get("model", "") or ""

    from .loader import save_agent

    pinned: list[str] = []
    save_failed = False
    for defn in agents:
        if getattr(defn, "mode", None) != AgentMode.COGNITIVE:
            continue
        binding = getattr(defn, "service_provider", None)
        if isinstance(binding, dict) and binding:
            continue
        if routing_error(defn.provider, defn.cognitive_model,
                         custom_providers=custom) is None:
            continue
        provider, model = legacy_resolved_route(
            defn, legacy_model=legacy_model, has_api_key=has_api_key,
            custom_providers=custom)
        if routing_error(provider, model, custom_providers=custom) is not None:
            log.warning("Agent %s: could not pin a route under the old rules "
                        "(provider=%r model=%r); it must be fixed by hand",
                        defn.id, provider, model)
            continue
        defn.provider = provider
        defn.cognitive_model = model
        try:
            save_agent(defn, config.agents_dir)
        except Exception:
            save_failed = True
            log.warning("Agent %s: pinned in memory but YAML save failed",
                        defn.id, exc_info=True)
        pinned.append(defn.id)
        log.info("Agent %s pinned to provider=%s model=%s (what it already "
                 "ran on)", defn.id, provider, model)

    if save_failed:
        # Keep the defaults: section and the stamp unset so the next boot
        # can finish the job from the same inputs.
        return pinned
    _drop_legacy_defaults_section(getattr(config, "source_path", None))
    config.legacy_agent_routing = {}
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(f"pinned {len(pinned)}\n", encoding="utf-8")
    except OSError:
        log.warning("Could not write %s", stamp, exc_info=True)
    return pinned
