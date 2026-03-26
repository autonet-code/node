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
        from ..orchestrator import ORCHESTRATOR_ID, create_orchestrator_agent
        from ..config import save_orchestrator_model_to_config

        orch_defn = self.registry._agents.get(ORCHESTRATOR_ID)
        raw_provider = orch_defn.provider if orch_defn else ""
        primary_provider = raw_provider[0] if isinstance(raw_provider, list) else raw_provider
        available = self.provider_manager.get_available_models(primary_provider)
        available_ids = [m["id"] for m in available]
        if available_ids and model not in available_ids:
            raise ValueError(
                f"Invalid model {model!r}. Available: {', '.join(available_ids)}"
            )

        # Kill running orchestrator — caller must provide kill_agent
        # This is handled by the facade

        self._config.orchestrator.model = model
        save_orchestrator_model_to_config(model)

        # Preserve user-edited fields from the current definition
        old_defn = self.registry._agents.get(ORCHESTRATOR_ID)

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

        log.info("Orchestrator model changed to '%s'", model)
        return defn.id
