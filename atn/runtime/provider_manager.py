"""LLM provider lifecycle & resolution."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..config import ATNConfig, save_provider_to_config, remove_provider_from_config
from ..events import EventBus
from ..models import AgentDefinition, StepType, TokenUsage
from ..providers.anthropic import AnthropicProvider
from ..providers.base import ProviderError
from ..providers.bridge import BridgeProvider
from ..providers.codex_bridge import CodexBridgeProvider
from ..providers.ollama import OllamaProvider
from ..providers.openai_compat import OpenAICompatibleProvider
from ..steps.cognitive import CognitiveStepExecutor

if TYPE_CHECKING:
    from ..credentials import CredentialStore
    from ..store import ExecutionLog

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model capability tiers
# ---------------------------------------------------------------------------
# Tier 4: orchestrator — multi-agent coordination, complex planning, 20+ tools
# Tier 3: autonomous   — independent multi-step execution, 10+ tools, self-correction
# Tier 2: tool-use     — reliable tool calling, structured instructions, single-task
# Tier 1: conversational — text in/out, unreliable with tools

TIER_LABELS: dict[int, str] = {
    4: "orchestrator",
    3: "autonomous",
    2: "tool-use",
    1: "conversational",
}

# Known model → tier mappings.  Prefix matching is used: the longest prefix
# that matches wins.  Unknown models default to tier 2.
_MODEL_TIERS: dict[str, int] = {
    # Anthropic
    "claude-opus-4": 4,
    "claude-sonnet-4": 3,
    "claude-haiku-4": 2,
    "claude-sonnet-3.5": 3,
    "claude-haiku-3.5": 2,
    # OpenAI
    "o3": 4,
    "o4-mini": 3,
    "o1": 3,
    "gpt-4.1": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 3,
    "gpt-4": 3,
    "gpt-3.5": 2,
    # Google
    "gemini-3-pro": 4,
    "gemini-3-flash": 3,
    "gemini-2.5-pro": 4,
    "gemini-2.5-flash": 3,
    "gemini-2.0-flash": 3,
    "gemini-2.0-pro": 4,
    "gemini-1.5-pro": 3,
    "gemini-1.5-flash": 2,
}

_DEFAULT_TIER = 2


def get_model_tier(model_id: str) -> int:
    """Return the capability tier (1-4) for a model ID.

    Uses longest-prefix matching against _MODEL_TIERS.
    Unknown models default to tier 2 (tool-use).
    """
    if not model_id:
        return _DEFAULT_TIER
    lower = model_id.lower()
    best_match = 0
    best_tier = _DEFAULT_TIER
    for prefix, tier in _MODEL_TIERS.items():
        if lower.startswith(prefix.lower()) and len(prefix) > best_match:
            best_match = len(prefix)
            best_tier = tier
    return best_tier


def get_tier_label(tier: int) -> str:
    """Return the human-readable label for a tier number."""
    return TIER_LABELS.get(tier, "unknown")


class ProviderManager:
    """Provider configuration, resolution, probing, and lifecycle."""

    # Registry of all known provider slots
    _KNOWN_PROVIDERS: dict[str, dict[str, Any]] = {
        "claude_max": {
            "name": "Claude Max",
            "description": "Claude via Claude Code bridge (requires Claude Max subscription)",
            "auth_type": "bridge",
            "orchestrator_capable": True,
        },
        "codex_max": {
            "name": "Codex (OpenAI)",
            "description": "OpenAI Codex via Codex CLI bridge (requires Codex/OpenAI subscription)",
            "auth_type": "bridge",
            "orchestrator_capable": True,
        },
        "anthropic": {
            "name": "Anthropic API",
            "description": "Claude models via Anthropic API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Gemini models via Google AI API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT models via OpenAI API (requires API key)",
            "auth_type": "api_key",
            "orchestrator_capable": True,
        },
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Local models via Ollama — no API key needed",
            "auth_type": "local",
            "orchestrator_capable": False,
        },
        "rpb": {
            "name": "RPB Network",
            "description": "Decentralized inference via RPB peer-to-peer network",
            "auth_type": "rpb",
            "orchestrator_capable": False,  # Not yet — will be True when mature
            "models": [],  # Dynamically discovered
        },
    }

    _PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_model": "gemini-2.0-flash",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "",
        },
    }

    def __init__(
        self,
        config: ATNConfig,
        credential_store: "CredentialStore",
        executors: dict,
        events: EventBus,
    ) -> None:
        self._config = config
        self.credential_store = credential_store
        self._executors = executors
        self.events = events

        self._custom_providers: set[str] = set()
        self._active_providers: dict[str, Any] = {}
        self._p2p_host: Any = None  # Set by runtime when P2P is available
        # Cache session stats when providers are removed, so the frontend
        # can still fetch stats after execution completes.
        self._cached_session_stats: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    def _resolve_api_key(self, provider_name: str) -> str:
        creds = self.credential_store.load(f"provider_{provider_name}")
        if creds.get("api_key"):
            return creds["api_key"]
        pconfig = self._config.providers.get(provider_name)
        if pconfig and pconfig.api_key:
            return pconfig.api_key
        return ""

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def resolve_provider_with_fallback(self, defn: AgentDefinition) -> Any:
        providers = defn.provider
        model = defn.cognitive_model or self._config.orchestrator.model or "claude-sonnet-4-6"

        if isinstance(providers, list):
            for provider_name in providers:
                try:
                    return self._resolve_provider_by_name(provider_name, model, defn.id)
                except Exception:
                    log.info("Provider '%s' not available for %s, trying next", provider_name, defn.id)
            first = providers[0] if providers else "claude_max"
            return self._resolve_provider_by_name(first, model, defn.id)
        elif providers:
            if providers in self._KNOWN_PROVIDERS or providers in self._custom_providers:
                return self._resolve_provider_by_name(providers, model, defn.id)
            return self._resolve_provider_for_model(providers, defn.id)
        else:
            return self._resolve_provider_for_model(model, defn.id)

    def _resolve_provider_by_name(self, provider_name: str, model: str, agent_id: str) -> Any:
        if provider_name == "claude_max":
            return BridgeProvider(model=model)
        if provider_name == "codex_max":
            api_key = self._resolve_api_key("codex_max") or self._resolve_api_key("openai")
            return CodexBridgeProvider(model=model, api_key=api_key)
        if provider_name == "anthropic":
            api_key = self._resolve_api_key("anthropic")
            if not api_key:
                raise ProviderError("No Anthropic API key configured")
            return AnthropicProvider(api_key=api_key, default_model=model, base_url="")
        if provider_name == "gemini":
            api_key = self._resolve_api_key("gemini")
            if not api_key:
                raise ProviderError("No Gemini API key configured")
            defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
            return OpenAICompatibleProvider(
                name=f"gemini-{agent_id}",
                base_url=defaults.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                api_key=api_key,
                default_model=model,
            )
        if provider_name == "openai":
            api_key = self._resolve_api_key("openai")
            if not api_key:
                raise ProviderError("No OpenAI API key configured")
            return OpenAICompatibleProvider(
                name=f"openai-{agent_id}",
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                default_model=model,
            )
        if provider_name == "ollama":
            defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
            pconfig = self._config.providers.get("ollama")
            return OllamaProvider(
                base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", "http://localhost:11434"),
                default_model=model,
            )
        if provider_name == "rpb":
            from ..providers.rpb import RPBNetworkProvider
            return RPBNetworkProvider(
                agent_id=agent_id,
                model=model,
                p2p_host=self._p2p_host,
            )
        if provider_name in self._custom_providers:
            pconfig = self._config.providers.get(provider_name)
            if pconfig and pconfig.base_url:
                api_key = self._resolve_api_key(provider_name)
                return OpenAICompatibleProvider(
                    name=provider_name,
                    base_url=pconfig.base_url,
                    api_key=api_key,
                    default_model=model,
                )
        raise ProviderError(f"Unknown provider: {provider_name}")

    def _resolve_provider_for_model(self, model_name: str, agent_id: str) -> Any:
        model_lower = model_name.lower()
        if model_lower.startswith("gemini"):
            defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
            api_key = self._resolve_api_key("gemini")
            if not api_key:
                log.warning("No Gemini API key found, falling back to BridgeProvider")
                return BridgeProvider(model=model_name)
            return OpenAICompatibleProvider(
                name=f"gemini-{agent_id}",
                base_url=defaults.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai"),
                api_key=api_key,
                default_model=model_name,
            )
        if model_lower.startswith(("gpt", "o1", "o3", "o4")):
            defaults = self._PROVIDER_DEFAULTS.get("openai", {})
            api_key = self._resolve_api_key("openai")
            if not api_key:
                log.warning("No OpenAI API key found, falling back to BridgeProvider")
                return BridgeProvider(model=model_name)
            return OpenAICompatibleProvider(
                name=f"openai-{agent_id}",
                base_url=defaults.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
                default_model=model_name,
            )
        # Default for claude-* models: prefer Anthropic API if key available
        if model_lower.startswith("claude"):
            api_key = self._resolve_api_key("anthropic")
            if api_key:
                return AnthropicProvider(api_key=api_key, default_model=model_name)
        return BridgeProvider(model=model_name)

    def get_bridge_provider(self, agent_id: str | None = None) -> Any:
        from ..orchestrator import ORCHESTRATOR_ID
        target = agent_id or ORCHESTRATOR_ID
        provider = self._active_providers.get(target)
        return provider if isinstance(provider, BridgeProvider) else None

    # ------------------------------------------------------------------
    # Setup and auto-detect
    # ------------------------------------------------------------------

    def setup_providers(self, cognitive: CognitiveStepExecutor) -> None:
        for name, pconfig in self._config.providers.items():
            try:
                if name in ("claude_max", "ollama"):
                    continue
                elif name == "anthropic":
                    api_key = self._resolve_api_key("anthropic")
                    if not api_key:
                        log.warning("Provider 'anthropic' has no API key — skipping")
                        continue
                    provider = AnthropicProvider(
                        api_key=api_key,
                        default_model=pconfig.default_model,
                        base_url=pconfig.base_url,
                    )
                elif name in ("gemini", "openai") or pconfig.base_url:
                    defaults = self._PROVIDER_DEFAULTS.get(name, {})
                    base_url = pconfig.base_url or defaults.get("base_url", "")
                    if not base_url:
                        log.warning("Provider '%s' has no base_url — skipping", name)
                        continue
                    api_key = self._resolve_api_key(name)
                    if not api_key:
                        log.warning("Provider '%s' has no API key — skipping", name)
                        continue
                    provider = OpenAICompatibleProvider(
                        name=name,
                        base_url=base_url,
                        api_key=api_key,
                        default_model=pconfig.default_model or defaults.get("default_model", ""),
                    )
                else:
                    ptype = pconfig.extra.get("type", "")
                    base_url = pconfig.base_url
                    if ptype == "openai_compat" or base_url:
                        if not base_url:
                            log.warning("Custom provider '%s' has no base_url — skipping", name)
                            continue
                        api_key = self._resolve_api_key(name)
                        provider = OpenAICompatibleProvider(
                            name=name,
                            base_url=base_url,
                            api_key=api_key,
                            default_model=pconfig.default_model,
                        )
                        self._custom_providers.add(name)
                    else:
                        log.warning("Unknown provider '%s' — skipping", name)
                        continue

                cognitive.register_provider(provider)
                log.info("Registered provider: %s (model=%s)", name, pconfig.default_model or "(default)")

            except ProviderError as exc:
                log.warning("Failed to initialise provider '%s': %s", name, exc)

        # Credential-store providers not in config
        for pid in ("anthropic", "gemini", "openai"):
            if pid in self._config.providers:
                continue
            api_key = self._resolve_api_key(pid)
            if not api_key:
                continue
            try:
                if pid == "anthropic":
                    provider = AnthropicProvider(api_key=api_key, default_model="", base_url="")
                else:
                    defaults = self._PROVIDER_DEFAULTS.get(pid, {})
                    provider = OpenAICompatibleProvider(
                        name=pid,
                        base_url=defaults.get("base_url", ""),
                        api_key=api_key,
                        default_model=defaults.get("default_model", ""),
                    )
                cognitive.register_provider(provider)
                log.info("Registered credential-store provider: %s", pid)
            except Exception as exc:
                log.warning("Failed to register credential-store provider '%s': %s", pid, exc)

    async def auto_detect_providers(self) -> None:
        cognitive = self._executors.get(StepType.COGNITIVE)
        if not isinstance(cognitive, CognitiveStepExecutor):
            return

        if "claude_max" not in cognitive._providers:
            try:
                ready = await self.probe_bridge()
                if ready:
                    self._hot_register_provider("claude_max", "")
                    log.info("Auto-detected Claude Max bridge")
                else:
                    log.info("Claude Max bridge not available")
            except Exception as exc:
                log.warning("Claude Max bridge auto-detect failed: %s", exc)

        if "ollama" not in cognitive._providers:
            try:
                models = await self.probe_ollama()
                if models is not None:
                    defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
                    pconfig = self._config.providers.get("ollama")
                    provider = OllamaProvider(
                        base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                        default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                    )
                    cognitive.register_provider(provider)
                    log.info("Auto-detected Ollama (%d model(s))", len(models))
                else:
                    log.info("Ollama not available at localhost:11434")
            except Exception as exc:
                log.warning("Ollama auto-detect failed: %s", exc)

    # ------------------------------------------------------------------
    # Available models
    # ------------------------------------------------------------------

    def get_available_models(self, provider_name: str, *, require_active: bool = False) -> list[dict[str, str]]:
        if require_active:
            cognitive = self._executors.get(StepType.COGNITIVE)
            if isinstance(cognitive, CognitiveStepExecutor):
                if provider_name not in cognitive._providers:
                    return []
            else:
                return []

        _PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
            "claude_max": [
                {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
                {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
                {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"},
            ],
        }
        models = _PROVIDER_MODELS.get(provider_name, [])
        # Enrich each model entry with capability tier
        for m in models:
            tier = get_model_tier(m["id"])
            m["capability_tier"] = tier
            m["tier_label"] = get_tier_label(tier)
        return models

    # ------------------------------------------------------------------
    # Provider management (config UI)
    # ------------------------------------------------------------------

    async def provider_list(self) -> list[dict[str, Any]]:
        cognitive = self._executors.get(StepType.COGNITIVE)
        registered: set[str] = set()
        if isinstance(cognitive, CognitiveStepExecutor):
            registered = set(cognitive._providers.keys())

        usage_by_provider = self._aggregate_provider_usage()

        providers = []
        for pid, info in self._KNOWN_PROVIDERS.items():
            is_active = pid in registered
            if is_active:
                configured = True
            elif info["auth_type"] == "api_key":
                configured = bool(self._resolve_api_key(pid))
            else:
                configured = False

            if is_active:
                setup_hint = None
            elif info["auth_type"] == "bridge":
                setup_hint = "connect"
            elif info["auth_type"] == "api_key":
                setup_hint = "api_key"
            else:
                setup_hint = None

            models = self.get_available_models(pid)
            # Derive orchestrator_capable from model tiers (backward compat)
            max_tier = max((m.get("capability_tier", _DEFAULT_TIER) for m in models), default=_DEFAULT_TIER) if models else _DEFAULT_TIER
            # Fall back to the static flag for providers with no model list
            orch_capable = max_tier >= 4 if models else info.get("orchestrator_capable", False)

            entry: dict[str, Any] = {
                "id": pid,
                "name": info["name"],
                "description": info["description"],
                "auth_type": info["auth_type"],
                "configured": configured,
                "active": is_active,
                "orchestrator_capable": orch_capable,
                "max_capability_tier": max_tier,
                "max_tier_label": get_tier_label(max_tier),
                "models": models,
                "setup_hint": setup_hint,
            }

            usage = usage_by_provider.get(pid)
            if usage:
                entry["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "total_tokens": usage.total,
                }

            if pid == "ollama":
                entry["local_models"] = await self.probe_ollama()

            providers.append(entry)

        for pid in sorted(self._custom_providers):
            is_active = pid in registered
            pconfig = self._config.providers.get(pid)
            base_url = pconfig.base_url if pconfig else ""

            default_model = pconfig.default_model if pconfig else ""
            custom_tier = get_model_tier(default_model)

            entry = {
                "id": pid,
                "name": pid,
                "description": f"Custom OpenAI-compatible provider ({base_url})",
                "auth_type": "api_key",
                "configured": True,
                "active": is_active,
                "orchestrator_capable": custom_tier >= 4,
                "max_capability_tier": custom_tier,
                "max_tier_label": get_tier_label(custom_tier),
                "custom": True,
                "base_url": base_url,
                "default_model": default_model,
                "models": [],
                "setup_hint": None,
            }

            usage = usage_by_provider.get(pid)
            if usage:
                entry["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "total_tokens": usage.total,
                }

            providers.append(entry)

        return providers

    async def configure_provider(self, provider_id: str, api_key: str = "") -> dict[str, Any]:
        info = self._KNOWN_PROVIDERS.get(provider_id)
        if not info:
            raise ValueError(f"Unknown provider: {provider_id}")

        if info["auth_type"] == "api_key":
            if not api_key:
                raise ValueError(f"Provider '{provider_id}' requires an API key")
            await self._validate_api_key(provider_id, api_key)
            self.credential_store.save(f"provider_{provider_id}", {"api_key": api_key})
            self._hot_register_provider(provider_id, api_key)
            return {"status": "ok", "provider": provider_id}

        elif info["auth_type"] == "local":
            models = await self.probe_ollama()
            if models is None:
                raise ValueError("Cannot connect to Ollama at localhost:11434. Is it running?")
            self._hot_register_provider(provider_id, "")
            return {"status": "ok", "provider": provider_id, "models": models}

        elif info["auth_type"] == "bridge":
            # Auto-install bridge dependencies if missing
            await self._ensure_bridge_deps()
            auth = await self._check_claude_auth()
            if not auth["installed"]:
                raise ValueError(
                    "Bridge setup failed. Ensure 'bun' is installed "
                    "(npm i -g bun) and try again."
                )
            if not auth["logged_in"]:
                log.info("Claude Code not authenticated — launching browser login")
                login_ok = await self._trigger_claude_login()
                if not login_ok:
                    raise ValueError(
                        "Claude Code authentication failed or timed out. "
                        "Try running 'claude auth login' manually in a terminal."
                    )
                auth = await self._check_claude_auth()
                if not auth["logged_in"]:
                    raise ValueError(
                        "Authentication completed but status check still shows "
                        "not logged in. Try again or check 'claude auth status'."
                    )
            ready = await self.probe_bridge()
            if not ready:
                raise ValueError(
                    "Claude Code is authenticated but the bridge failed to respond. "
                    "Check that 'bun' is installed and bridge dependencies are set up."
                )
            self._hot_register_provider(provider_id, "")
            return {
                "status": "ok",
                "provider": provider_id,
                "email": auth.get("email", ""),
                "subscription": auth.get("subscription", ""),
            }

        return {"status": "ok", "provider": provider_id}

    async def remove_provider(self, provider_id: str) -> dict[str, str]:
        self.credential_store.delete(f"provider_{provider_id}")
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            cognitive._providers.pop(provider_id, None)
        log.info("Provider '%s' removed", provider_id)
        return {"status": "removed", "provider": provider_id}

    async def add_custom_provider(
        self,
        provider_id: str,
        name: str,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
    ) -> dict[str, Any]:
        if not provider_id:
            raise ValueError("Provider ID is required")
        if provider_id in self._KNOWN_PROVIDERS:
            raise ValueError(f"'{provider_id}' is a built-in provider — use configure_provider() instead")
        if not base_url:
            raise ValueError("Base URL is required for custom providers")

        import httpx
        try:
            test_url = base_url.rstrip("/")
            if not test_url.endswith("/models"):
                models_url = test_url.rstrip("/")
                if models_url.endswith("/chat/completions"):
                    models_url = models_url.rsplit("/chat/completions", 1)[0]
                models_url += "/models"
            else:
                models_url = test_url

            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(models_url, headers=headers)
                if resp.status_code in (401, 403) and api_key:
                    raise ValueError(f"API key rejected by {base_url} (HTTP {resp.status_code})")
        except httpx.ConnectError:
            raise ValueError(f"Cannot connect to {base_url} — check the URL")
        except httpx.TimeoutException:
            raise ValueError(f"Timeout connecting to {base_url}")
        except ValueError:
            raise
        except Exception:
            pass

        spec: dict[str, Any] = {
            "type": "openai_compat",
            "base_url": base_url,
            "default_model": default_model,
        }
        save_provider_to_config(provider_id, spec)

        if api_key:
            self.credential_store.save(f"provider_{provider_id}", {"api_key": api_key})

        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            provider = OpenAICompatibleProvider(
                name=provider_id,
                base_url=base_url,
                api_key=api_key,
                default_model=default_model,
            )
            cognitive.register_provider(provider)

        self._custom_providers.add(provider_id)
        log.info("Custom provider added: %s (%s)", provider_id, base_url)
        return {"status": "ok", "provider": provider_id, "name": name}

    async def remove_custom_provider(self, provider_id: str) -> dict[str, str]:
        if provider_id in self._KNOWN_PROVIDERS:
            raise ValueError(f"'{provider_id}' is a built-in provider — use remove_provider() instead")
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            cognitive._providers.pop(provider_id, None)
        remove_provider_from_config(provider_id)
        self.credential_store.delete(f"provider_{provider_id}")
        self._custom_providers.discard(provider_id)
        log.info("Custom provider removed: %s", provider_id)
        return {"status": "removed", "provider": provider_id}

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    def cache_session_stats(self, agent_id: str) -> None:
        """Snapshot session stats for an agent before its provider is removed.

        Called by the execution engine right before popping the provider from
        _active_providers, so that get_session_stats can still return data
        after execution completes.
        """
        provider = self._active_providers.get(agent_id)
        if provider is not None and hasattr(provider, 'session_stats'):
            self._cached_session_stats[agent_id] = provider.session_stats

    def get_session_stats(self, agent_id: str | None = None) -> dict[str, Any]:
        from ..orchestrator import ORCHESTRATOR_ID
        target = agent_id or ORCHESTRATOR_ID
        # Try active provider first (richest / live stats)
        active = self._active_providers.get(target)
        if active is not None and hasattr(active, 'session_stats'):
            return active.session_stats
        # Fall back to cached stats from last execution
        cached = self._cached_session_stats.get(target)
        if cached is not None:
            return cached
        return {"error": f"No active session for '{target}'"}

    async def get_session_context(self, agent_id: str | None = None) -> dict[str, Any]:
        provider = self.get_bridge_provider(agent_id)
        if provider is None:
            target = agent_id or "orchestrator"
            return {"error": f"No active bridge session for '{target}'"}
        return await provider.get_session_context()

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    async def probe_ollama(self) -> list[dict[str, str]] | None:
        import httpx
        pconfig = self._config.providers.get("ollama")
        defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
        base_url = (pconfig.base_url if pconfig else "") or defaults.get("base_url", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                if resp.status_code != 200:
                    return None
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    size = m.get("size", 0)
                    size_gb = f"{size / 1e9:.1f}GB" if size else ""
                    models.append({"id": name, "name": name, "size": size_gb})
                return models
        except Exception:
            return None

    async def probe_bridge(self) -> bool:
        try:
            provider = BridgeProvider(model="sonnet")
            resp = await asyncio.wait_for(provider._send_request({"type": "ping"}), timeout=15)
            ok = resp.get("ok", False)
            await provider.close()
            return bool(ok)
        except Exception as exc:
            log.debug("Bridge probe failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hot_register_provider(self, provider_id: str, api_key: str) -> None:
        cognitive = self._executors.get(StepType.COGNITIVE)
        if not isinstance(cognitive, CognitiveStepExecutor):
            return
        try:
            if provider_id == "claude_max":
                pconfig = self._config.providers.get("claude_max")
                bridge_script = pconfig.extra.get("bridge_script", "") if pconfig else ""
                default_model = (
                    (pconfig.default_model if pconfig else "")
                    or self._config.orchestrator.model
                    or "claude-sonnet-4-6"
                )
                provider = BridgeProvider(
                    model=default_model,
                    bridge_script=bridge_script if bridge_script else None,
                )
            elif provider_id == "anthropic":
                pconfig = self._config.providers.get("anthropic")
                provider = AnthropicProvider(
                    api_key=api_key,
                    default_model=pconfig.default_model if pconfig else "",
                    base_url=pconfig.base_url if pconfig else "",
                )
            elif provider_id in ("gemini", "openai"):
                defaults = self._PROVIDER_DEFAULTS.get(provider_id, {})
                pconfig = self._config.providers.get(provider_id)
                provider = OpenAICompatibleProvider(
                    name=provider_id,
                    base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                    api_key=api_key,
                    default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                )
            elif provider_id == "ollama":
                defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
                pconfig = self._config.providers.get("ollama")
                provider = OllamaProvider(
                    base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", ""),
                    default_model=(pconfig.default_model if pconfig else "") or defaults.get("default_model", ""),
                )
            else:
                return
            cognitive.register_provider(provider)
            log.info("Hot-registered provider: %s", provider_id)
        except ProviderError as exc:
            log.warning("Failed to hot-register provider '%s': %s", provider_id, exc)

    async def _validate_api_key(self, provider_id: str, api_key: str) -> None:
        import httpx
        try:
            if provider_id == "anthropic":
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code == 403:
                        raise ValueError("API key does not have access")
                    if resp.status_code not in (200, 429, 529):
                        raise ValueError(f"Unexpected response: {resp.status_code}")
            elif provider_id == "gemini":
                defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
                base_url = defaults.get("base_url", "")
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code not in (200, 429):
                        raise ValueError(f"Unexpected response: {resp.status_code}")
            elif provider_id == "openai":
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Invalid API key")
                    if resp.status_code not in (200, 429):
                        raise ValueError(f"Unexpected response: {resp.status_code}")
        except httpx.ConnectError:
            raise ValueError(f"Cannot reach {provider_id} API — check your network")
        except httpx.TimeoutException:
            raise ValueError(f"Timeout connecting to {provider_id} API")

    async def _ensure_bridge_deps(self) -> None:
        """Auto-install bridge dependencies (bun install) if missing."""
        try:
            import bridge as _bridge_pkg
            bridge_dir = Path(_bridge_pkg.__file__).resolve().parent
        except ImportError:
            bridge_dir = Path(__file__).resolve().parent.parent.parent / "bridge"
        pkg_json = bridge_dir / "package.json"
        node_modules = bridge_dir / "node_modules"
        cli_js = node_modules / "@anthropic-ai" / "claude-agent-sdk" / "cli.js"

        if cli_js.exists():
            return  # Already installed

        if not pkg_json.exists():
            log.warning("Bridge package.json not found at %s", bridge_dir)
            return

        # Check bun is available; auto-install if missing
        import shutil
        bun = shutil.which("bun")
        if not bun:
            bun = await self._auto_install_bun()
        if not bun:
            # Fall back to npm if available
            npm = shutil.which("npm")
            if npm:
                log.info("bun not available, falling back to npm install...")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        npm, "install",
                        cwd=str(bridge_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    if proc.returncode == 0:
                        log.info("Bridge dependencies installed via npm")
                    return
                except Exception as e:
                    log.warning("npm install failed: %s", e)
            log.warning("Neither bun nor npm found — cannot auto-install bridge dependencies. "
                        "Install bun: https://bun.sh/docs/installation")
            return

        # Run bun install in the bridge directory
        log.info("Installing bridge dependencies in %s ...", bridge_dir)
        try:
            proc = await asyncio.create_subprocess_exec(
                bun, "install",
                cwd=str(bridge_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                log.info("Bridge dependencies installed successfully")
            else:
                log.warning("bun install failed (exit %s): %s",
                            proc.returncode, stderr.decode(errors="replace")[:500])
        except asyncio.TimeoutError:
            log.warning("bun install timed out")
        except Exception as e:
            log.warning("Failed to install bridge dependencies: %s", e)

    async def _auto_install_bun(self) -> str | None:
        """Download and install bun without requiring npm/node."""
        import shutil
        import sys

        log.info("bun not found — attempting automatic install...")
        try:
            if sys.platform == "win32":
                # PowerShell one-liner from bun.sh
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-Command",
                    "irm bun.sh/install.ps1 | iex",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # Unix: curl | bash
                proc = await asyncio.create_subprocess_shell(
                    "curl -fsSL https://bun.sh/install | bash",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                # Bun installs to ~/.bun/bin — refresh PATH
                bun_bin = Path.home() / ".bun" / "bin"
                if bun_bin.exists():
                    os.environ["PATH"] = str(bun_bin) + os.pathsep + os.environ.get("PATH", "")
                bun = shutil.which("bun")
                if bun:
                    log.info("bun installed successfully: %s", bun)
                    return bun
            log.warning("bun install script exited with code %s", proc.returncode)
        except asyncio.TimeoutError:
            log.warning("bun install timed out")
        except Exception as e:
            log.warning("Failed to auto-install bun: %s", e)
        return None

    def _resolve_sdk_cli(self) -> Path | None:
        try:
            import bridge as _bridge_pkg
            bridge_dir = Path(_bridge_pkg.__file__).resolve().parent
        except ImportError:
            bridge_dir = Path(__file__).resolve().parent.parent.parent / "bridge"
        cli_js = bridge_dir / "node_modules" / "@anthropic-ai" / "claude-agent-sdk" / "cli.js"
        if cli_js.exists():
            return cli_js
        return None

    async def _check_claude_auth(self) -> dict[str, Any]:
        cli_js = self._resolve_sdk_cli()
        if cli_js is None:
            return {"installed": False, "logged_in": False}
        try:
            proc = await asyncio.create_subprocess_exec(
                "bun", str(cli_js), "auth", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            data = json.loads(stdout.decode("utf-8", errors="replace"))
            return {
                "installed": True,
                "logged_in": bool(data.get("loggedIn", False)),
                "email": data.get("email", ""),
                "subscription": data.get("subscriptionType", ""),
            }
        except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError) as exc:
            log.debug("Claude auth status check failed: %s", exc)
            return {"installed": True, "logged_in": False}
        except Exception as exc:
            log.debug("Claude auth status check error: %s", exc)
            return {"installed": False, "logged_in": False}

    async def _trigger_claude_login(self) -> bool:
        cli_js = self._resolve_sdk_cli()
        if cli_js is None:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "bun", str(cli_js), "auth", "login",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
            return proc.returncode == 0
        except asyncio.TimeoutError:
            log.warning("Claude login timed out after 5 minutes")
            if proc.returncode is None:
                proc.terminate()
            return False
        except Exception as exc:
            log.debug("Claude login failed: %s", exc)
            return False

    def _aggregate_provider_usage(self) -> dict[str, TokenUsage]:
        totals: dict[str, TokenUsage] = {}
        # Access execution_log via the runtime reference that will be set
        execution_log = self._execution_log
        for agent_id in list(execution_log._logs.keys()):
            for rec in execution_log._logs[agent_id]:
                for provider, usage in rec.token_usage.items():
                    if provider not in totals:
                        totals[provider] = TokenUsage(provider=provider)
                    totals[provider].input_tokens += usage.input_tokens
                    totals[provider].output_tokens += usage.output_tokens
                    totals[provider].cache_read_tokens += usage.cache_read_tokens
                    totals[provider].cache_creation_tokens += usage.cache_creation_tokens
        return totals
