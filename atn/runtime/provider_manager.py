"""LLM provider lifecycle & resolution."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
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


def _sdk_platform_tag() -> str:
    """The @anthropic-ai/claude-code optional-dep platform suffix, e.g.
    'win32-x64', 'darwin-arm64', 'linux-x64'. Used to locate the bundled
    claude binary in node_modules when `claude` isn't on PATH."""
    import platform as _plat
    plat = sys.platform  # 'win32' | 'darwin' | 'linux'
    machine = _plat.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return f"{plat}-{arch}"


# ---------------------------------------------------------------------------
# Model capability tiers
# ---------------------------------------------------------------------------
# Tier 4: coordinating  — multi-agent coordination, complex planning, 20+ tools
# Tier 3: autonomous    — independent multi-step execution, 10+ tools, self-correction
# Tier 2: tool-use      — reliable tool calling, structured instructions, single-task
# Tier 1: conversational — text in/out, unreliable with tools
#
# Tier 4 was once named after a fixed root-agent role that no longer
# exists. The tier describes what a MODEL can sustain, not a position in
# the fleet, so the label now names the capability.

TIER_LABELS: dict[int, str] = {
    4: "coordinating",
    3: "autonomous",
    2: "tool-use",
    1: "conversational",
}

# Known model → tier mappings.  Prefix matching is used: the longest prefix
# that matches wins.  Unknown models default to tier 2.
_MODEL_TIERS: dict[str, int] = {
    # Anthropic
    "claude-fable-5": 4,
    "claude-mythos-5": 4,
    "claude-opus-4-8": 4,
    "claude-opus-4-7": 4,
    "claude-opus-4": 4,
    "claude-sonnet-5": 3,
    "claude-sonnet-4-7": 3,
    "claude-sonnet-4-6": 3,
    "claude-sonnet-4": 3,
    "claude-haiku-4": 2,
    "claude-sonnet-3.5": 3,
    "claude-haiku-3.5": 2,
    # OpenAI
    "gpt-5.5": 4,
    "gpt-5": 4,
    "o3": 4,
    "o4-mini": 3,
    "o1": 3,
    "gpt-4.1": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 3,
    "gpt-4": 3,
    "gpt-3.5": 2,
    # DeepSeek
    "deepseek-reasoner": 4,
    "deepseek-v4-pro": 4,
    "deepseek-chat": 3,
    "deepseek-v4-flash": 3,
    "deepseek-coder": 3,
    "deepseek": 3,
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


# Curated per-provider model lists.  Keys must match the provider IDs in
# ProviderManager._KNOWN_PROVIDERS.  Each entry is the canonical model ID
# the provider expects in API requests, plus a human-readable name.  Tier
# and context window are looked up from _MODEL_TIERS and model_specs.
_PROVIDER_MODELS: dict[str, list[dict[str, str]]] = {
    "claude_max": [
        {"id": "claude-fable-5",    "name": "Claude Fable 5"},
        {"id": "claude-opus-4-8",   "name": "Claude Opus 4.8"},
        {"id": "claude-opus-4-7",   "name": "Claude Opus 4.7"},
        {"id": "claude-sonnet-5",   "name": "Claude Sonnet 5"},
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
        {"id": "claude-opus-4-6",   "name": "Claude Opus 4.6"},
        {"id": "claude-haiku-4-5",  "name": "Claude Haiku 4.5"},
    ],
    # ChatGPT-account Codex auth accepts ONLY current Codex models —
    # verified empirically 2026-08 (retired ids get a 400 from the API).
    "codex_max": [
        {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra"},
        {"id": "gpt-5.5",       "name": "GPT-5.5"},
    ],
    "anthropic": [
        {"id": "claude-fable-5",    "name": "Claude Fable 5"},
        {"id": "claude-opus-4-8",   "name": "Claude Opus 4.8"},
        {"id": "claude-opus-4-7",   "name": "Claude Opus 4.7"},
        {"id": "claude-sonnet-5",   "name": "Claude Sonnet 5"},
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
        {"id": "claude-opus-4-6",   "name": "Claude Opus 4.6"},
        {"id": "claude-haiku-4-5",  "name": "Claude Haiku 4.5"},
    ],
    "openai": [
        {"id": "gpt-5.5",     "name": "GPT-5.5"},
        {"id": "gpt-5",       "name": "GPT-5"},
        {"id": "gpt-4.1",     "name": "GPT-4.1"},
        {"id": "gpt-4o",      "name": "GPT-4o"},
        {"id": "o3",          "name": "o3"},
        {"id": "o4-mini",     "name": "o4-mini"},
    ],
    "gemini": [
        {"id": "gemini-3-pro",     "name": "Gemini 3 Pro"},
        {"id": "gemini-3-flash",   "name": "Gemini 3 Flash"},
        {"id": "gemini-2.5-pro",   "name": "Gemini 2.5 Pro"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    ],
    "deepseek": [
        {"id": "deepseek-reasoner", "name": "DeepSeek V4-Pro (Reasoner)"},
        {"id": "deepseek-chat",     "name": "DeepSeek V4-Flash"},
    ],
}


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
        },
        "codex_max": {
            "name": "Codex (OpenAI)",
            "description": "OpenAI Codex via Codex CLI bridge (requires Codex/OpenAI subscription)",
            "auth_type": "bridge",
        },
        "anthropic": {
            "name": "Anthropic API",
            "description": "Claude models via Anthropic API (requires API key)",
            "auth_type": "api_key",
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Gemini models via Google AI API (requires API key)",
            "auth_type": "api_key",
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT models via OpenAI API (requires API key)",
            "auth_type": "api_key",
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "DeepSeek V4 models via DeepSeek API (requires API key)",
            "auth_type": "api_key",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Local models via Ollama — no API key needed",
            "auth_type": "local",
        },
        "rpb": {
            "name": "RPB Network",
            "description": "Decentralized inference via RPB peer-to-peer network",
            "auth_type": "rpb",
            "models": [],  # Dynamically discovered
        },
        "service": {
            "name": "Marketplace Service",
            "description": (
                "Inference bought off the services market: pay-per-call in ATN "
                "to another agent's daemon (docs/services_market.md)."
            ),
            "auth_type": "service",
            "models": [],  # The served model is the seller's declaration
        },
        "substrate": {
            "name": "World-Model Substrate",
            "description": "Substrate retrieval + local LLM render. Phase 6 of native integration.",
            "auth_type": "local",
            "models": [],
        },
    }

    _PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_model": "gemini-2.5-flash",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
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
        # Phase 6.4: substrate provider needs the daemon's WorldService
        # to probe. Two ways to attach it:
        #   - direct reference: ``self._world_service = svc``
        #   - lazy resolver: ``self._world_service_resolver = lambda: svc``
        # The resolver path lets the autonet AutonetBridge wire the
        # service even though autonet's WorldService is constructed
        # inside an executor thread (start() blocks). Resolver is
        # checked first; falls back to the direct reference.
        self._world_service: Any = None
        self._world_service_resolver: Any = None
        # Per-agent service binding (docs/services_market.md, 2026-07-26):
        # a bound child pays for its own inference, so building its provider
        # needs the child's OWN wallet key. The manager has no registry
        # reference (it is constructed before/alongside one), so the runtime
        # injects a resolver — same lazy-injection pattern as _world_service.
        # Signature: (agent_id) -> private_key_hex | "" ; and
        # (agent_id) -> 0x address | "" for the client address.
        self._agent_key_resolver: Any = None
        self._agent_address_resolver: Any = None
        # Cache session stats when providers are removed, so the frontend
        # can still fetch stats after execution completes.
        self._cached_session_stats: dict[str, dict[str, Any]] = {}

        # §10: cache the ollama /api/tags probe so unknown-model routing can
        # check "is this installed locally?" without a network call per resolve.
        # ``_ollama_tags_cache`` is the set of installed model ids (names, with
        # and without their :tag), refreshed at most every ~60s.
        self._ollama_tags_cache: set[str] | None = None
        self._ollama_tags_cache_ts: float = 0.0
        self._ollama_tags_ttl: float = 60.0

        # Live model-catalog cache for API providers that expose a /models
        # endpoint (anthropic, openai, gemini). Maps provider_id -> list of
        # {"id", "name"} dicts fetched from the provider, merged (union by id)
        # with the curated _PROVIDER_MODELS list in get_available_models.
        # In-memory, ~1h TTL. Empty/absent → curated list only.
        self._catalog_cache: dict[str, list[dict[str, str]]] = {}
        self._catalog_cache_ts: dict[str, float] = {}
        self._catalog_ttl: float = 3600.0

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
        model = defn.cognitive_model or self._config.default_model or "claude-sonnet-4-6"

        # Per-agent marketplace binding (docs/services_market.md, ratified
        # 2026-07-26: employer-chooses-the-tool). A binding is a PARENT's
        # decision about which substrate this agent thinks on, so it OVERRIDES
        # whatever provider/model the definition otherwise names — it is not
        # one candidate in a fallback chain. There is deliberately no fallback
        # off a binding either: silently falling back to a local provider would
        # hand the agent a substrate its parent did not buy, and silently
        # falling back to the daemon-level purchase would spend the OWNER's
        # wallet instead of the child's.
        # Require a real dict: the field is normalized to
        # {provider_address, spec_digest} on every write path, and insisting on
        # the type keeps a stray truthy value (a test double, a half-migrated
        # definition) from silently rerouting an agent's cognition.
        binding = getattr(defn, "service_provider", None)
        if isinstance(binding, dict) and binding:
            return self._build_service_provider(
                model, binding=binding, agent_id=defn.id)

        # For the rpb (dependent) provider, the agent routes inference to a
        # sponsor on another daemon.
        #
        # The dependent is the OWNER WALLET — one 0x per daemon, not per agent
        # (ratified 2026-07-25, docs/sponsored_inference.md). The sponsor binds
        # that one address and does not care how many agents sit behind it, or
        # whether any of them is registered on-chain. Both values are daemon
        # config, not agent state: there is no per-agent sponsor setting.
        agent_address = (getattr(self._config.autonet, "owner_wallet", "") or "").strip()
        sponsor_address = (
            getattr(self._config.autonet, "sponsor_address", "") or "").strip()

        if isinstance(providers, list):
            for provider_name in providers:
                try:
                    return self._resolve_provider_by_name(
                        provider_name, model, defn.id,
                        agent_address=agent_address, sponsor_address=sponsor_address)
                except Exception:
                    log.info("Provider '%s' not available for %s, trying next", provider_name, defn.id)
            first = providers[0] if providers else "claude_max"
            return self._resolve_provider_by_name(
                first, model, defn.id,
                agent_address=agent_address, sponsor_address=sponsor_address)
        elif providers:
            if providers in self._KNOWN_PROVIDERS or providers in self._custom_providers:
                return self._resolve_provider_by_name(
                    providers, model, defn.id,
                    agent_address=agent_address, sponsor_address=sponsor_address)
            return self._resolve_provider_for_model(providers, defn.id)
        else:
            return self._resolve_provider_for_model(model, defn.id)

    def _resolve_provider_by_name(
        self, provider_name: str, model: str, agent_id: str,
        *, agent_address: str = "", sponsor_address: str = "",
    ) -> Any:
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
        if provider_name == "deepseek":
            api_key = self._resolve_api_key("deepseek")
            if not api_key:
                raise ProviderError("No DeepSeek API key configured")
            defaults = self._PROVIDER_DEFAULTS.get("deepseek", {})
            return OpenAICompatibleProvider(
                name=f"deepseek-{agent_id}",
                base_url=defaults.get("base_url", "https://api.deepseek.com/v1"),
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
                agent_address=agent_address,
                model=model,
                p2p_host=self._p2p_host,
                sponsor_address=sponsor_address,
            )
        if provider_name == "service":
            return self._build_service_provider(model)
        if provider_name == "substrate":
            return self._build_substrate_provider(model, agent_id)
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

    def _build_service_provider(
        self, model: str, *,
        binding: dict[str, Any] | None = None,
        agent_id: str = "",
    ) -> Any:
        """Build the ServiceProvider for a purchased marketplace service.

        Consumer side of docs/services_market.md (Decision 2026-07-26): a
        purchased marketplace inference service backs an agent's provider.
        Two sources for the same two facts, differing in WHOSE KEY PAYS:

        **Daemon-level (owner) purchase** — ``binding=None``. Read from
        config.yaml; the payment is signed with the daemon OWNER key
        (``autonet.private_key``), matching the sponsor pipe's
        dependent-is-the-owner-wallet doctrine:

            providers:
              service:
                provider_address: "0x..."       # the serving agent
                spec_digest: "<sha256 hex>"     # the service spec bought
                default_model: "..."            # optional display label

        **Per-agent binding** — ``binding={provider_address, spec_digest}``
        off ``AgentDefinition.service_provider``, set only by the agent's
        PARENT. The payment is signed with the BOUND AGENT'S OWN wallet key,
        resolved through ``_agent_key_resolver``: a parent may buy a service
        and provision a child on it, and the child spends its own funded
        wallet. Spend authority is literal token custody — the parent controls
        the tap by controlling the refills, and never has to trust the child
        to honour a limit.
        """
        from ..providers.service import ServiceProvider

        pconfig = self._config.providers.get("service")
        extra = (pconfig.extra if pconfig else {}) or {}

        if binding:
            provider_address = str(binding.get("provider_address") or "").strip()
            spec_digest = str(binding.get("spec_digest") or "").strip()
            if not provider_address or not spec_digest:
                raise ProviderError(
                    f"agent '{agent_id}' has a malformed service_provider "
                    f"binding: both 'provider_address' and 'spec_digest' are "
                    f"required (got {binding!r})."
                )
            signing_key = self._resolve_agent_key(agent_id)
            client_address = self._resolve_agent_address(agent_id)
            return ServiceProvider(
                config=self._config.rpb,
                provider_address=provider_address,
                spec_digest=spec_digest,
                # A binding carries no model label of its own: the seller
                # declares the served model, and an empty value is legitimate.
                model=model or "",
                signing_key=signing_key,
                client_address=client_address,
                timeout=float(extra.get("timeout") or 0.0),
                payer_kind="agent",
                payer_id=agent_id,
            )

        provider_address = str(
            extra.get("provider_address")
            or extra.get("provider")
            or ""
        ).strip()
        spec_digest = str(
            extra.get("spec_digest") or extra.get("service_digest") or ""
        ).strip()
        if not provider_address or not spec_digest:
            raise ProviderError(
                "service provider needs both 'provider_address' and "
                "'spec_digest' under providers.service in config.yaml — "
                "it names the specific purchased service to buy inference from."
            )
        autonet = self._config.autonet
        return ServiceProvider(
            config=self._config.rpb,
            provider_address=provider_address,
            spec_digest=spec_digest,
            model=model or (pconfig.default_model if pconfig else ""),
            owner_private_key=(getattr(autonet, "private_key", "") or "").strip(),
            client_address=(getattr(autonet, "owner_wallet", "") or "").strip(),
            timeout=float(extra.get("timeout") or 0.0),
            payer_kind="owner",
        )

    def _resolve_agent_key(self, agent_id: str) -> str:
        """The bound agent's own wallet key, via the runtime-injected resolver.

        Returns "" when unavailable — the provider then fails loud at the
        payment step (with chain configured) or degrades open (without), which
        is the same honest-degradation contract the owner path has.
        """
        if not agent_id or self._agent_key_resolver is None:
            return ""
        try:
            return str(self._agent_key_resolver(agent_id) or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("agent key resolver failed for %s: %s", agent_id, exc)
            return ""

    def _resolve_agent_address(self, agent_id: str) -> str:
        """The bound agent's own 0x address (the payer label on the wire)."""
        if not agent_id or self._agent_address_resolver is None:
            return ""
        try:
            return str(self._agent_address_resolver(agent_id) or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("agent address resolver failed for %s: %s", agent_id, exc)
            return ""

    def _build_substrate_provider(self, model: str, agent_id: str) -> Any:
        """Build the SubstrateProvider composite.

        Phase 6.4 wiring:
          - The runtime must have attached a WorldService at
            ``self._world_service`` (or a resolver) so the substrate
            provider can probe it.
          - The renderer is ollama configured for this daemon; the
            substrate provider feeds the located region into ollama.
          - ``model`` passed in is the *renderer model* (e.g.
            "qwen3:4b") — the actual LLM call goes to ollama with
            this id.
        """
        ws = self._resolve_world_service()
        if ws is None:
            raise ProviderError(
                "substrate provider requires a WorldService; "
                "the runtime hasn't attached one yet"
            )
        from ..providers.substrate import SubstrateProvider
        ollama_defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
        ollama_pconfig = self._config.providers.get("ollama")
        renderer = OllamaProvider(
            base_url=(
                (ollama_pconfig.base_url if ollama_pconfig else "")
                or ollama_defaults.get("base_url", "http://localhost:11434")
            ),
            default_model=model,
        )
        return SubstrateProvider(
            world_service=ws,
            renderer=renderer,
            renderer_model=model,
        )

    def _resolve_world_service(self) -> Any:
        """Resolve the WorldService via direct reference or lazy
        resolver. Resolver wins if set."""
        if self._world_service_resolver is not None:
            try:
                ws = self._world_service_resolver()
                if ws is not None:
                    return ws
            except Exception as e:
                log.warning("world_service_resolver failed: %s", e)
        return self._world_service

    def _ollama_tags_cached(self) -> set[str]:
        """Return the set of locally-installed ollama model ids, cached ~60s.

        Sync (called from the sync resolve path). Tolerates ollama being down —
        returns an empty set on any error and caches that so we don't hammer a
        dead endpoint. Ids are stored both with and without their ``:tag`` so a
        bare name like ``qwen3.5`` also matches an installed ``qwen3.5:4b``.
        """
        import time
        now = time.monotonic()
        if (
            self._ollama_tags_cache is not None
            and (now - self._ollama_tags_cache_ts) < self._ollama_tags_ttl
        ):
            return self._ollama_tags_cache

        tags: set[str] = set()
        try:
            import httpx
            pconfig = self._config.providers.get("ollama")
            defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
            base_url = (
                (pconfig.base_url if pconfig else "")
                or defaults.get("base_url", "http://localhost:11434")
            ).rstrip("/")
            resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = (m.get("name") or "").lower()
                    if name:
                        tags.add(name)
                        if ":" in name:
                            tags.add(name.split(":", 1)[0])
        except Exception as exc:
            log.debug("ollama tags probe failed (treating as not installed): %s", exc)

        self._ollama_tags_cache = tags
        self._ollama_tags_cache_ts = now
        return tags

    def _model_installed_in_ollama(self, model_name: str) -> bool:
        lower = model_name.lower()
        tags = self._ollama_tags_cached()
        if lower in tags:
            return True
        # Also match on the bare base name (``qwen3.5`` matches ``qwen3.5:4b``).
        base = lower.split(":", 1)[0]
        return base in tags

    def _resolve_provider_for_model(self, model_name: str, agent_id: str) -> Any:
        model_lower = model_name.lower()
        if model_lower.startswith("gemini"):
            defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
            api_key = self._resolve_api_key("gemini")
            if not api_key:
                # §10: no silent BridgeProvider fallback onto the user's
                # subscription. A keyless gemini model is a routing error.
                raise ProviderError(
                    f"No Gemini API key configured for model '{model_name}': "
                    f"set a key or specify provider explicitly"
                )
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
                # §10: fail loud rather than silently billing the bridge.
                raise ProviderError(
                    f"No OpenAI API key configured for model '{model_name}': "
                    f"set a key or specify provider explicitly"
                )
            return OpenAICompatibleProvider(
                name=f"openai-{agent_id}",
                base_url=defaults.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
                default_model=model_name,
            )
        # Default for claude models: prefer Anthropic API if key available, else
        # the Claude Max bridge (an explicit, expected route for claude ids).
        # Match the "claude" prefix AND the canonical short aliases the bridge
        # accepts (sonnet/opus/haiku/fable/mythos) — these are Claude models, not
        # unknown ids, so they must NOT hit the fail-loud path below.
        _CLAUDE_ALIASES = ("sonnet", "opus", "haiku", "fable", "mythos")
        if model_lower.startswith("claude") or model_lower.startswith(_CLAUDE_ALIASES):
            api_key = self._resolve_api_key("anthropic")
            if api_key:
                return AnthropicProvider(api_key=api_key, default_model=model_name)
            return BridgeProvider(model=model_name)
        # §10: unknown prefix. Probe the ollama tags cache — if the model is
        # installed locally, route to ollama. Otherwise FAIL LOUD instead of
        # falling onto the bridge (the qwen3.5:4b-burned-45k-subscription bug,
        # live finding 1).
        if self._model_installed_in_ollama(model_name):
            defaults = self._PROVIDER_DEFAULTS.get("ollama", {})
            pconfig = self._config.providers.get("ollama")
            return OllamaProvider(
                base_url=(pconfig.base_url if pconfig else "") or defaults.get("base_url", "http://localhost:11434"),
                default_model=model_name,
            )
        raise ProviderError(
            f"unknown model '{model_name}': not a known cloud model and not "
            f"installed in ollama — set provider explicitly"
        )

    def get_bridge_provider(self, agent_id: str | None = None) -> Any:
        # Callers (the Runtime facade) resolve the default agent id (fleet
        # root); None here simply finds no provider.
        provider = self._active_providers.get(agent_id) if agent_id else None
        return provider if isinstance(provider, BridgeProvider) else None

    def get_registered_provider(self, provider_id: str) -> Any:
        """The executor-registered provider instance for ``provider_id``.

        Unlike ``_active_providers`` (keyed by agent_id, populated only while
        an agent is executing), this is the boot-time registry — present on a
        rootless fleet with no runs yet. None if not registered.
        """
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            return cognitive._providers.get(provider_id)
        return None

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
                elif name == "service":
                    # Marketplace inference (docs/services_market.md). Config
                    # is provider_address + spec_digest, no key/base_url — a
                    # misconfigured entry raises ProviderError and is skipped
                    # by the handler below like any other bad provider.
                    provider = self._build_service_provider(
                        pconfig.default_model)
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

        if "codex_max" not in cognitive._providers and self._check_codex_auth():
            try:
                ready = await self.probe_codex_bridge()
                if ready:
                    self._hot_register_provider("codex_max", "")
                    log.info("Auto-detected Codex bridge")
                else:
                    log.info("Codex auth present but bridge not available")
            except Exception as exc:
                log.warning("Codex bridge auto-detect failed: %s", exc)

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

    def get_available_models(
        self,
        provider_name: str | None = None,
        *,
        require_active: bool = False,
        min_tier: int = 0,
    ) -> list[dict[str, Any]]:
        """Return models, enriched with provider, tier, and context window.

        Args:
            provider_name: Filter to one provider, or None for all configured.
            require_active: Only include providers currently registered with
                the cognitive executor (i.e. successfully connected).
            min_tier: Drop entries below this capability tier.  0 = no filter.
        """
        from ..model_specs import context_window, max_output_tokens

        cognitive = self._executors.get(StepType.COGNITIVE)
        active_providers: set[str] = set()
        if isinstance(cognitive, CognitiveStepExecutor):
            active_providers = set(cognitive._providers.keys())

        # Decide which provider IDs to include.
        if provider_name is not None:
            if require_active and provider_name not in active_providers:
                return []
            target_ids = [provider_name]
        else:
            target_ids = list(_PROVIDER_MODELS.keys())
            for pid in sorted(self._custom_providers):
                if pid not in target_ids:
                    target_ids.append(pid)
            if require_active:
                target_ids = [p for p in target_ids if p in active_providers]

        out: list[dict[str, Any]] = []
        for pid in target_ids:
            # Curated list unioned with any cached live-fetched catalog (the
            # cache is primed asynchronously by refresh_model_catalogs /
            # provider_list; get_available_models stays sync).
            entries = self._merged_provider_models(pid)
            # For custom providers, fall back to user-supplied list on the
            # provider config (set via the custom-provider form).
            if not entries and pid in self._custom_providers:
                pconfig = self._config.providers.get(pid)
                custom_models = (pconfig.models if pconfig else None) or []
                for cm in custom_models:
                    if isinstance(cm, str):
                        entries.append({"id": cm, "name": cm})
                    elif isinstance(cm, dict) and cm.get("id"):
                        entries.append({"id": cm["id"], "name": cm.get("name", cm["id"])})
                if not entries:
                    default_model = pconfig.default_model if pconfig else ""
                    if default_model:
                        entries.append({"id": default_model, "name": default_model})
            for m in entries:
                tier = get_model_tier(m["id"])
                if tier < min_tier:
                    continue
                out.append({
                    "id": m["id"],
                    "name": m["name"],
                    "provider": pid,
                    "capability_tier": tier,
                    "tier_label": get_tier_label(tier),
                    "context_window": context_window(m["id"]),
                    "max_output_tokens": max_output_tokens(m["id"]),
                })
        return out

    # ------------------------------------------------------------------
    # Provider management (config UI)
    # ------------------------------------------------------------------

    async def provider_list(self) -> list[dict[str, Any]]:
        # Prime the live model-catalog cache (best-effort, ~1h TTL) so the
        # per-provider model lists below reflect newly-released models beyond
        # the curated set. Network failure → curated list only, no error.
        await self.refresh_model_catalogs()

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
            elif info["auth_type"] == "service":
                # Configured means the purchase is named: which agent serves
                # it, and which spec was bought.
                _sp = self._config.providers.get(pid)
                _extra = (_sp.extra if _sp else {}) or {}
                configured = bool(
                    str(_extra.get("provider_address") or "").strip()
                    and str(_extra.get("spec_digest") or "").strip()
                )
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
            # Loop capability is a per-MODEL property (model_specs), so the
            # provider only reports the best tier it can reach. The old
            # per-provider loop-capable boolean was dropped: it
            # squashed a per-model fact into a provider-wide claim that was
            # wrong in both directions (a capable provider serving a haiku
            # model, or ollama flagged incapable regardless of model).
            max_tier = max((m.get("capability_tier", _DEFAULT_TIER) for m in models), default=_DEFAULT_TIER) if models else _DEFAULT_TIER

            entry: dict[str, Any] = {
                "id": pid,
                "name": info["name"],
                "description": info["description"],
                "auth_type": info["auth_type"],
                "configured": configured,
                "active": is_active,
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

    def default_model_for(self, provider_id: str) -> str:
        """The model a new agent should get when the user picked a provider
        but no model. Precedence: per-provider config default → the daemon
        default IF it belongs to this provider → curated-list head →
        provider-defaults table. Never returns a model from a DIFFERENT
        provider (the create-form bug: picking codex_max still landed agents
        on the claude default)."""
        pconfig = self._config.providers.get(provider_id)
        if pconfig and pconfig.default_model:
            return pconfig.default_model
        daemon_default = self._config.default_model
        if daemon_default:
            try:
                from ..model_specs import resolve as _resolve
                if _resolve(daemon_default).default_channel == provider_id:
                    return daemon_default
            except Exception:
                pass
        curated = _PROVIDER_MODELS.get(provider_id)
        if curated:
            return curated[0]["id"]
        return self._PROVIDER_DEFAULTS.get(provider_id, {}).get("default_model", "")

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

        elif info["auth_type"] == "service":
            # No credential to store: the purchase is named in config.yaml
            # (providers.service.provider_address / .spec_digest). Build it so
            # a misconfiguration surfaces here rather than at first completion.
            provider = self._build_service_provider("")
            cognitive = self._executors.get(StepType.COGNITIVE)
            if isinstance(cognitive, CognitiveStepExecutor):
                cognitive.register_provider(provider)
            return {
                "status": "ok",
                "provider": provider_id,
                "provider_address": provider.provider_address,
                "spec_digest": provider.spec_digest,
            }

        elif info["auth_type"] == "bridge":
            # Auto-install bridge dependencies if missing
            await self._ensure_bridge_deps()

            if provider_id == "codex_max":
                # Codex has its own auth (ChatGPT login via 'codex login' →
                # ~/.codex/auth.json) — the Claude checks below don't apply.
                if not self._check_codex_auth():
                    raise ValueError(
                        "Codex is not logged in. Run 'codex login' in a "
                        "terminal (ChatGPT account), then connect again."
                    )
                ready = await self.probe_codex_bridge()
                if not ready:
                    raise ValueError(
                        "Codex auth found but the bridge failed to respond. "
                        "Check that 'bun' is installed and bridge dependencies "
                        "are set up (cd bridge && npm install)."
                    )
                self._hot_register_provider(provider_id, "")
                return {"status": "ok", "provider": provider_id}

            auth = await self._check_claude_auth()
            if not auth["installed"]:
                raise ValueError(
                    "Bridge setup failed. Ensure 'bun' is installed "
                    "(npm i -g bun) and try again."
                )
            # Guard: only launch the browser login when we DEFINITIVELY know
            # the user is not logged in. If the status probe couldn't verify
            # (timeout / unexpected output), surface a retry error instead of
            # popping a login — a valid token may already exist, and an
            # unverified probe must never trigger a re-auth.
            if not auth["logged_in"] and not auth.get("verified", False):
                raise ValueError(
                    "Couldn't verify Claude Code auth status (the check timed "
                    "out or returned unexpected output). Your existing login is "
                    "likely still valid — try connecting again. If it persists, "
                    "run 'claude auth status' in a terminal to check."
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
        models: list[Any] | None = None,
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
        # Normalize the model list — accept either bare strings or {id, name} dicts.
        normalized_models: list[Any] = []
        for entry in (models or []):
            if isinstance(entry, str) and entry.strip():
                normalized_models.append(entry.strip())
            elif isinstance(entry, dict) and entry.get("id"):
                normalized_models.append({"id": entry["id"], "name": entry.get("name", entry["id"])})
        if normalized_models:
            spec["models"] = normalized_models

        save_provider_to_config(provider_id, spec)

        # Mirror onto the in-memory ProviderConfig so the running daemon sees it.
        from ..config import ProviderConfig as _PC
        existing = self._config.providers.get(provider_id)
        if existing is None:
            self._config.providers[provider_id] = _PC(
                name=provider_id,
                base_url=base_url,
                default_model=default_model,
                models=normalized_models,
            )
        else:
            existing.base_url = base_url
            existing.default_model = default_model
            existing.models = normalized_models

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
        # Callers (the Runtime facade) resolve the default agent id (fleet
        # root); an unresolvable target falls through to the no-session error.
        target = agent_id or ""
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
            target = agent_id or "(no agent)"
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

    @staticmethod
    def _check_codex_auth() -> bool:
        """Codex CLI login state: 'codex login' writes ~/.codex/auth.json."""
        try:
            auth_file = Path.home() / ".codex" / "auth.json"
            return auth_file.is_file() and auth_file.stat().st_size > 2
        except Exception:
            return False

    async def probe_codex_bridge(self) -> bool:
        """Spawn the codex bridge and ping it — no model call, just proves
        the subprocess + SDK import + pipe protocol work."""
        try:
            provider = CodexBridgeProvider()
            resp = await asyncio.wait_for(provider._send_request({"type": "ping"}), timeout=20)
            ok = resp.get("ok", False)
            await provider.close()
            return bool(ok)
        except Exception as exc:
            log.debug("Codex bridge probe failed: %s", exc)
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
                    or self._config.default_model
                    or "claude-sonnet-4-6"
                )
                provider = BridgeProvider(
                    model=default_model,
                    bridge_script=bridge_script if bridge_script else None,
                )
            elif provider_id == "codex_max":
                pconfig = self._config.providers.get("codex_max")
                codex_key = api_key or self._resolve_api_key("codex_max") \
                    or self._resolve_api_key("openai") or ""
                provider = CodexBridgeProvider(
                    model=(pconfig.default_model if pconfig else "") or "gpt-5.6-terra",
                    api_key=codex_key,
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

    # ------------------------------------------------------------------
    # Live model-catalog fetch (anthropic / openai / gemini)
    # ------------------------------------------------------------------

    # Providers whose catalog we fetch live. claude_max / codex_max are curated
    # only (no catalog API); ollama has its own /api/tags probe.
    _CATALOG_PROVIDERS = ("anthropic", "openai", "gemini")

    @staticmethod
    def _openai_chat_capable(model_id: str) -> bool:
        """Filter OpenAI /v1/models noise down to chat-capable families.

        /v1/models returns embeddings, whisper, tts, dall-e, moderation, etc.
        Keep gpt-* and reasoning o-series (o1/o3/o4-...); drop the rest."""
        mid = model_id.lower()
        _NOISE = ("embed", "whisper", "tts", "dall-e", "dalle", "moderation",
                  "audio", "realtime", "image", "search", "transcribe", "davinci",
                  "babbage", "instruct-0", "codex")
        if any(tok in mid for tok in _NOISE):
            return False
        if mid.startswith("gpt-"):
            return True
        # Reasoning o-series: o1, o3, o4-mini, ... but not "o1-embedding" (caught
        # above). Require o + digit.
        if len(mid) >= 2 and mid[0] == "o" and mid[1].isdigit():
            return True
        return False

    async def _fetch_provider_catalog(self, provider_id: str) -> list[dict[str, str]]:
        """Fetch the live model list for an API provider. Returns a list of
        {"id","name"} dicts, or [] on any failure (network, auth, parse).
        Cached in-memory with ~1h TTL. No key → []."""
        import time as _time
        if provider_id not in self._CATALOG_PROVIDERS:
            return []
        now = _time.monotonic()
        cached = self._catalog_cache.get(provider_id)
        if cached is not None and (now - self._catalog_cache_ts.get(provider_id, 0.0)) < self._catalog_ttl:
            return cached

        api_key = self._resolve_api_key(provider_id)
        if not api_key:
            return []

        import httpx
        try:
            ids: list[str] = []
            # 5s, not 15: this primes the provider-card UI; a slow catalog
            # endpoint must not stall the first provider_list open.
            async with httpx.AsyncClient(timeout=5) as client:
                if provider_id == "anthropic":
                    resp = await client.get(
                        "https://api.anthropic.com/v1/models",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                        },
                    )
                    if resp.status_code != 200:
                        return self._catalog_cache.get(provider_id, [])
                    data = resp.json().get("data", [])
                    ids = [m.get("id", "") for m in data if m.get("id")]
                elif provider_id == "openai":
                    resp = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code != 200:
                        return self._catalog_cache.get(provider_id, [])
                    data = resp.json().get("data", [])
                    ids = [m.get("id", "") for m in data
                           if m.get("id") and self._openai_chat_capable(m["id"])]
                elif provider_id == "gemini":
                    defaults = self._PROVIDER_DEFAULTS.get("gemini", {})
                    base_url = defaults.get("base_url", "")
                    resp = await client.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code != 200:
                        return self._catalog_cache.get(provider_id, [])
                    # OpenAI-compat shape: {"data": [{"id": "..."}]}. Strip the
                    # "models/" prefix the native Gemini API uses if present.
                    data = resp.json().get("data", [])
                    for m in data:
                        mid = m.get("id", "")
                        if not mid:
                            continue
                        mid = mid.split("/", 1)[1] if mid.startswith("models/") else mid
                        if mid.startswith("gemini-"):
                            ids.append(mid)
            catalog = [{"id": mid, "name": mid} for mid in dict.fromkeys(ids)]
            self._catalog_cache[provider_id] = catalog
            self._catalog_cache_ts[provider_id] = now
            return catalog
        except Exception as exc:  # noqa: BLE001 — network failure → curated only
            log.debug("Model-catalog fetch for %s failed: %s", provider_id, exc)
            # Serve stale cache if we have one; else empty (curated only).
            return self._catalog_cache.get(provider_id, [])

    async def refresh_model_catalogs(self) -> None:
        """Prime the live-catalog cache for all API providers that have a key.
        Call before building provider entries so get_available_models can merge
        fetched ids. Best-effort — never raises."""
        import asyncio as _asyncio
        tasks = []
        for pid in self._CATALOG_PROVIDERS:
            if self._resolve_api_key(pid):
                tasks.append(self._fetch_provider_catalog(pid))
        if tasks:
            await _asyncio.gather(*tasks, return_exceptions=True)

    def _merged_provider_models(self, provider_id: str) -> list[dict[str, str]]:
        """Curated list for ``provider_id`` unioned with any cached live-fetch
        ids. Curated entries keep their order/metadata; fetched-only ids are
        appended with the id as the display name. No cache → curated only."""
        curated = list(_PROVIDER_MODELS.get(provider_id, []))
        fetched = self._catalog_cache.get(provider_id)
        if not fetched:
            return curated
        seen = {e["id"] for e in curated}
        merged = list(curated)
        for e in fetched:
            if e["id"] not in seen:
                merged.append({"id": e["id"], "name": e.get("name", e["id"])})
                seen.add(e["id"])
        return merged

    async def _ensure_bridge_deps(self) -> None:
        """Stage the bridge sources into ``~/.atn/bridge`` and install its
        dependencies there (site-packages is often read-only). Loud failure:
        raises ``ValueError`` — surfaced to the frontend provider card — when
        neither bun nor npm is available, or the install fails.

        Dev fast-path (repo checkout with node_modules already next to the
        packaged sources) is used in place with no copy/install.
        """
        from ..providers.bridge import stage_bridge_sources, _node_modules_ready

        try:
            bridge_dir, needs_install = stage_bridge_sources()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Claude Max bridge sources could not be staged into "
                f"~/.atn/bridge: {exc}"
            )

        pkg_json = bridge_dir / "package.json"
        if not pkg_json.exists():
            raise ValueError(
                f"Claude Max bridge is missing package.json (looked in "
                f"{bridge_dir}). The install is corrupt — reinstall the atn "
                "package."
            )

        if not needs_install and _node_modules_ready(bridge_dir):
            return  # Already installed and sources unchanged.

        import shutil
        # Prefer bun; auto-install it if absent (its failure is now loud too).
        bun = shutil.which("bun")
        if not bun:
            bun = await self._auto_install_bun()

        install_err = ""
        if bun:
            log.info("Installing bridge dependencies with bun in %s ...", bridge_dir)
            try:
                proc = await asyncio.create_subprocess_exec(
                    bun, "install",
                    cwd=str(bridge_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                if proc.returncode == 0 and _node_modules_ready(bridge_dir):
                    log.info("Bridge dependencies installed successfully (bun)")
                    return
                install_err = (stderr.decode(errors="replace")[:400]
                               or f"bun install exit {proc.returncode}")
            except asyncio.TimeoutError:
                install_err = "bun install timed out after 180s"
            except Exception as e:  # noqa: BLE001
                install_err = f"bun install error: {e}"

        # Fall back to npm.
        npm = shutil.which("npm")
        if npm:
            log.info("Installing bridge dependencies with npm in %s ...", bridge_dir)
            try:
                proc = await asyncio.create_subprocess_exec(
                    npm, "install",
                    cwd=str(bridge_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
                if proc.returncode == 0 and _node_modules_ready(bridge_dir):
                    log.info("Bridge dependencies installed successfully (npm)")
                    return
                install_err = (stderr.decode(errors="replace")[:400]
                               or f"npm install exit {proc.returncode}") or install_err
            except asyncio.TimeoutError:
                install_err = "npm install timed out after 300s"
            except Exception as e:  # noqa: BLE001
                install_err = f"npm install error: {e}"

        # If we get here, install did not succeed. Fail LOUD.
        if not bun and not npm:
            raise ValueError(
                "Claude Max bridge dependencies could not be installed: neither "
                "'bun' nor 'npm' was found (and bun auto-install failed). "
                "Install bun from https://bun.sh and reconnect the provider."
            )
        raise ValueError(
            "Claude Max bridge dependencies failed to install "
            f"(in {bridge_dir}): {install_err or 'unknown error'}. "
            "Install bun from https://bun.sh and reconnect the provider."
        )

    async def _auto_install_bun(self) -> str | None:
        """Download and install bun without requiring npm/node.

        Returns the bun path on success, ``None`` on failure. Failure is
        non-fatal here (the caller falls back to npm, then fails loud), but
        it is logged clearly rather than buried."""
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
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                # Bun installs to ~/.bun/bin — refresh PATH
                bun_bin = Path.home() / ".bun" / "bin"
                if bun_bin.exists():
                    os.environ["PATH"] = str(bun_bin) + os.pathsep + os.environ.get("PATH", "")
                bun = shutil.which("bun")
                if bun:
                    log.info("bun installed successfully: %s", bun)
                    return bun
            log.warning(
                "bun auto-install failed (exit %s): %s",
                proc.returncode, stderr.decode(errors="replace")[:300],
            )
        except asyncio.TimeoutError:
            log.warning("bun auto-install timed out after 120s")
        except Exception as e:
            log.warning("Failed to auto-install bun: %s", e)
        return None

    def _resolve_sdk_cli(self) -> Path | None:
        """Locate a Claude Code executable for auth-status / login flows.

        SDK 0.2.119+ no longer ships a bundled cli.js. Prefer the standalone
        @anthropic-ai/claude-code binary (the `claude` command); fall back to
        the legacy bundled cli.js for older SDK installs.
        """
        import shutil
        from ..providers.bridge import resolve_bridge_dir
        # node_modules now lives in the runtime dir (~/.atn/bridge) unless the
        # dev fast-path put it next to the packaged sources — resolve_bridge_dir
        # returns whichever is authoritative.
        bridge_dir = resolve_bridge_dir()
        nm = bridge_dir / "node_modules" / "@anthropic-ai"

        # Prefer a TRUE executable (.exe / native binary). On Windows the npm
        # global `claude` is a .CMD shim, and spawning a .cmd via
        # create_subprocess_exec from the daemon's running asyncio loop hangs
        # (no shell to interpret it). The bundled claude-code binary is a real
        # .exe and is always present after `bun install`, regardless of PATH.
        bin_name = "claude.exe" if sys.platform == "win32" else "claude"
        for candidate in (
            nm / "claude-code" / "bin" / bin_name,
            nm / f"claude-code-{_sdk_platform_tag()}" / bin_name,
        ):
            if candidate.exists():
                return candidate

        # PATH lookup. On Windows, skip shell-shim extensions (.cmd/.bat/.ps1)
        # — those can't be exec'd directly via create_subprocess_exec /
        # subprocess.run; the bundled .exe above is preferred anyway.
        claude_bin = shutil.which("claude")
        if claude_bin:
            p = Path(claude_bin)
            if sys.platform != "win32" or p.suffix.lower() not in (".cmd", ".bat", ".ps1"):
                return p

        # Legacy fallback: bundled cli.js (pre-0.2.119 SDK installs).
        cli_js = nm / "claude-agent-sdk" / "cli.js"
        if cli_js.exists():
            return cli_js
        return None

    def _build_claude_cmd(self, cli: Path, *args: str) -> tuple[str, ...]:
        """Build the argv tuple for invoking the claude CLI.

        - Binary path (e.g. ~/AppData/Roaming/npm/claude): invoked directly.
          On Windows this resolves to a .cmd shim; the cmd extension is added
          automatically by the OS for asyncio subprocess.
        - Legacy cli.js: invoked via `bun cli.js ...`.
        """
        if cli.suffix == ".js":
            return ("bun", str(cli), *args)
        return (str(cli), *args)

    async def _check_claude_auth(self) -> dict[str, Any]:
        cli = self._resolve_sdk_cli()
        if cli is None:
            return {"installed": False, "logged_in": False}
        try:
            argv = self._build_claude_cmd(cli, "auth", "status")
            # Run the CLI in a worker thread with synchronous subprocess.run.
            # On Windows the Proactor event loop's create_subprocess_exec +
            # communicate() can hang for these child processes; a plain
            # subprocess.run off the loop is reliable and never blocks it.
            import subprocess
            def _run_sync():
                return subprocess.run(
                    list(argv), capture_output=True, timeout=15,
                    # CRITICAL: close stdin. The claude CLI blocks waiting on
                    # stdin when it inherits a non-TTY pipe (the daemon's
                    # stdin), hanging `auth status` indefinitely.
                    stdin=subprocess.DEVNULL,
                )
            completed = await asyncio.to_thread(_run_sync)
            text = completed.stdout.decode("utf-8", errors="replace").strip()
            # Newer claude CLI emits human-readable output by default; older
            # cli.js emits JSON. Try JSON first, fall back to keyword sniffing.
            # `verified` distinguishes a definitive answer (we parsed real auth
            # status) from a probe that couldn't complete. Callers must only
            # launch the browser login when verified AND not logged in —
            # never on a failure-to-verify (that would re-pop a login even
            # though a valid token may already exist).
            try:
                data = json.loads(text)
                return {
                    "installed": True,
                    "logged_in": bool(data.get("loggedIn", False)),
                    "verified": True,
                    "email": data.get("email", ""),
                    "subscription": data.get("subscriptionType", ""),
                }
            except json.JSONDecodeError:
                lower = text.lower()
                logged_in = (
                    "logged in" in lower
                    or "authenticated" in lower
                    or "subscription" in lower
                ) and "not " not in lower.split("logged in")[0][-10:]
                return {
                    "installed": True,
                    "logged_in": logged_in,
                    "verified": True,
                    "email": "",
                    "subscription": "",
                }
        except FileNotFoundError as exc:
            log.debug("Claude CLI not found: %s", exc)
            return {"installed": False, "logged_in": False, "verified": False}
        except Exception as exc:
            import subprocess
            if isinstance(exc, (asyncio.TimeoutError, subprocess.TimeoutExpired)):
                log.warning("Claude auth status timed out: %s", exc)
                return {"installed": True, "logged_in": False, "verified": False}
            log.debug("Claude auth status error (%s): %s", type(exc).__name__, exc)
            return {"installed": False, "logged_in": False, "verified": False}

    async def _trigger_claude_login(self) -> bool:
        cli = self._resolve_sdk_cli()
        if cli is None:
            return False
        try:
            argv = self._build_claude_cmd(cli, "auth", "login")
            proc = await asyncio.create_subprocess_exec(
                *argv,
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
