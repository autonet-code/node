"""Per-agent cognitive model switching."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .provider_manager import ProviderManager
    from .session_manager import SessionManager
    from ..config import ATNConfig

log = logging.getLogger(__name__)


class ModelSwitch:
    """Generic per-agent model switching."""

    def __init__(
        self,
        registry: "AgentRegistry",
        provider_manager: "ProviderManager",
        session_manager: "SessionManager",
        config: "ATNConfig",
    ) -> None:
        self.registry = registry
        self.provider_manager = provider_manager
        self.session_manager = session_manager
        self._config = config

    async def set_agent_model(self, agent_id: str, model: str) -> str:
        """Change the cognitive model for any agent.

        Preserves the agent's system prompt, max_turns, heartbeat, and other
        config.  The caller is responsible for killing running executions first.
        """
        from ..models import AgentMode

        old_defn = self.registry._agents.get(agent_id)
        if old_defn is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        if old_defn.mode != AgentMode.COGNITIVE:
            raise ValueError(f"Agent '{agent_id}' is not a cognitive agent")

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

        # Build an updated definition preserving all existing config
        from dataclasses import replace
        new_defn = replace(old_defn, cognitive_model=model)

        # Re-register: unregister old, register new
        await self.registry.unregister_agent(agent_id)
        await self.registry.register_agent(new_defn)

        # Persist to agent.yaml — without this the switch is RAM-only and
        # every daemon restart silently reverts the agent to whatever model
        # it was created with.
        try:
            from ..loader import save_agent
            save_agent(new_defn, self._config.agents_dir)
        except Exception as exc:
            log.warning("Model changed but YAML save failed for '%s': %s", agent_id, exc)

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

        # NOTE: the daemon-wide default deliberately does NOT follow a root
        # agent's model any more. That rule existed to keep the main agent's
        # model sticky across restarts — the YAML persistence above does that
        # properly — and in a multi-root fleet it silently rewrote the daemon
        # default whenever ANY root agent was retuned.

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
