"""Orchestrator-specific initialization and model switching."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .provider_manager import ProviderManager
    from .session_manager import SessionManager
    from ..config import ATNConfig
    from ..user_profile import UserProfileStore

log = logging.getLogger(__name__)


def _infer_provider_for_model(model: str) -> str | None:
    """Infer the provider name from a model ID prefix.

    Returns the provider slot best suited for the model, or None if unknown.
    Prefers bridge providers (claude_max, codex_max) over API-key providers.
    """
    m = model.lower()
    if m.startswith("claude-"):
        return "claude_max"
    if m.startswith("gpt-") or m.startswith("o3") or m.startswith("o4") or m.startswith("o1"):
        return "codex_max"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("deepseek"):
        return "deepseek"
    return None


class OrchestratorSetup:
    """Setup and model switching for the orchestrator agent."""

    def __init__(
        self,
        registry: "AgentRegistry",
        provider_manager: "ProviderManager",
        session_manager: "SessionManager",
        config: "ATNConfig",
        user_profile: "UserProfileStore",
    ) -> None:
        self.registry = registry
        self.provider_manager = provider_manager
        self.session_manager = session_manager
        self._config = config
        self.user_profile = user_profile

    async def setup_orchestrator(self, **kwargs: Any) -> str:
        from ..loader import load_agent_file
        from ..orchestrator import ORCHESTRATOR_ID, build_system_prompt_with_context, create_orchestrator_agent

        # Load saved orchestrator config (if the user edited fields via UI)
        saved_defn = None
        saved_yaml = self._config.agents_dir / ORCHESTRATOR_ID / "agent.yaml"
        if saved_yaml.exists():
            saved_defn, _ = load_agent_file(saved_yaml)
            if saved_defn:
                log.info("Loaded saved orchestrator config from %s", saved_yaml)

        system_prompt = build_system_prompt_with_context(
            data_dir=self._config.data_dir,
        )

        if "system_prompt" not in kwargs:
            # Prefer saved user-edited prompt; fall back to generated default
            if saved_defn and saved_defn.system_prompt:
                kwargs["system_prompt"] = saved_defn.system_prompt
            else:
                kwargs["system_prompt"] = system_prompt

        # Merge other user-editable fields from saved config
        if saved_defn:
            if "max_turns" not in kwargs and saved_defn.max_turns != 50:
                kwargs["max_turns"] = saved_defn.max_turns

        defn = create_orchestrator_agent(self._config.orchestrator, **kwargs)

        # Restore heartbeat config from saved definition
        if saved_defn and saved_defn.heartbeat and not defn.heartbeat:
            defn.heartbeat = saved_defn.heartbeat

        await self.registry.register_agent(defn)
        await self.registry.activate_agent(defn.id)
        log.info("Orchestrator registered and activated (provider=%s)", defn.provider)

        if self.session_manager.conversation.turn_count() == 0:
            self.session_manager._inject_status_briefing()

        return defn.id

    async def set_orchestrator_model(self, model: str) -> str:
        """Legacy wrapper — prefer set_agent_model()."""
        from ..orchestrator import ORCHESTRATOR_ID
        return await self.set_agent_model(ORCHESTRATOR_ID, model)

    async def set_agent_model(self, agent_id: str, model: str) -> str:
        """Change the cognitive model for any agent.

        Preserves the agent's system prompt, max_turns, heartbeat, and other
        config.  The caller is responsible for killing running executions first.
        """
        from ..orchestrator import ORCHESTRATOR_ID
        from ..models import AgentMode

        old_defn = self.registry._agents.get(agent_id)
        if old_defn is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        if old_defn.mode != AgentMode.COGNITIVE:
            raise ValueError(f"Agent '{agent_id}' is not a cognitive agent")

        # Root agent: restrict to tier-4 models on orchestrator-capable providers.
        # The provider is inferred from the model's prefix via the tier table.
        if agent_id == ORCHESTRATOR_ID:
            from .provider_manager import get_model_tier
            tier = get_model_tier(model)
            if tier < 4:
                raise ValueError(
                    f"Root agent requires a top-tier model (got {model!r}, tier {tier}). "
                    f"Pick an Opus/GPT-5/Gemini-Pro-class model."
                )
            # Determine which provider this model belongs to so the config persists correctly.
            new_provider = _infer_provider_for_model(model)
            if new_provider is None:
                raise ValueError(
                    f"Could not determine provider for model {model!r}. "
                    f"Supported: claude-*, gpt-*, o3, o4-*, gemini-*."
                )
            return await self._set_orchestrator_model_impl(model, old_defn, provider=new_provider)

        # Validate model against the agent's provider
        raw_provider = old_defn.provider or ""
        primary_provider = raw_provider[0] if isinstance(raw_provider, list) else raw_provider
        if primary_provider:
            available = self.provider_manager.get_available_models(primary_provider)
            available_ids = [m["id"] for m in available]
            if available_ids and model not in available_ids:
                raise ValueError(
                    f"Invalid model {model!r}. Available: {', '.join(available_ids)}"
                )

        # --- Generic agent model change ---
        # Build an updated definition preserving all existing config
        from dataclasses import replace
        new_defn = replace(old_defn, cognitive_model=model)

        # Re-register: unregister old, register new
        is_force = agent_id == ORCHESTRATOR_ID
        await self.registry.unregister_agent(agent_id, _force=is_force)
        await self.registry.register_agent(new_defn)

        # Reactivate if it was active/running before
        await self.registry.activate_agent(agent_id)

        # Close old provider so next execution gets a fresh one with new model
        old_provider = self.provider_manager._active_providers.pop(agent_id, None)
        if old_provider is not None:
            try:
                await old_provider.close()
            except Exception:
                pass

        self._stamp_active_model(agent_id, model)

        log.info("Agent '%s' model changed to '%s'", agent_id, model)
        return agent_id

    def _stamp_active_model(self, agent_id: str, model: str) -> None:
        """Sync the stats surfaces to a just-changed model.

        session_stats' ``active_model`` reports the model that LAST RAN
        (live provider, cached, or persisted stats). The UI treats it as
        backend truth and snaps its picker back to it — so a model change
        with no run in between would visually revert until the next
        execution. Stamp the new model forward; the next real run
        overwrites it with whatever actually executed.
        """
        cached = self.provider_manager._cached_session_stats.get(agent_id)
        if cached and cached.get("active_model"):
            cached["active_model"] = model
        try:
            convo = self.session_manager.get_agent_conversation_store(agent_id)
            stats = convo.get_session_stats()
            if stats.get("active_model"):
                stats["active_model"] = model
                convo.save_session_stats(stats)
        except Exception:
            log.debug("Could not stamp active_model for '%s'", agent_id, exc_info=True)

    async def _set_orchestrator_model_impl(
        self, model: str, old_defn: Any, *, provider: str | None = None,
    ) -> str:
        """Orchestrator-specific model change (persists to config file)."""
        from ..orchestrator import ORCHESTRATOR_ID, create_orchestrator_agent
        from ..config import save_orchestrator_model_to_config, save_orchestrator_provider_to_config

        self._config.orchestrator.model = model
        save_orchestrator_model_to_config(model)
        if provider is not None:
            self._config.orchestrator.provider = provider
            save_orchestrator_provider_to_config(provider)

        if ORCHESTRATOR_ID in self.registry._agents:
            await self.registry.unregister_agent(ORCHESTRATOR_ID, _force=True)

        create_kwargs: dict[str, Any] = {}
        if old_defn and old_defn.system_prompt:
            create_kwargs["system_prompt"] = old_defn.system_prompt
        if old_defn and old_defn.max_turns != 50:
            create_kwargs["max_turns"] = old_defn.max_turns
        defn = create_orchestrator_agent(self._config.orchestrator, **create_kwargs)
        if old_defn and old_defn.heartbeat:
            defn.heartbeat = old_defn.heartbeat

        await self.registry.register_agent(defn)
        await self.registry.activate_agent(defn.id)

        self.session_manager.clear_bridge_session()
        self._stamp_active_model(ORCHESTRATOR_ID, model)

        log.info("Orchestrator model changed to '%s'", model)
        return defn.id
