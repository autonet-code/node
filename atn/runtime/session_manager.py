"""Conversation stores, message injection, delegate output persistence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..conversation import ConversationStore
from ..events import EventBus
from ..inbox import InboxManager
from ..models import (
    AgentMode,
    AgentStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
)

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager

log = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation stores, message routing, and delegate output."""

    def __init__(
        self,
        conversation: ConversationStore,
        registry: "AgentRegistry",
        provider_manager: "ProviderManager",
        engine: "ExecutionEngine",
        events: EventBus,
        inbox: InboxManager,
        config: Any,
    ) -> None:
        self.conversation = conversation
        self.registry = registry
        self.provider_manager = provider_manager
        self.engine = engine
        self.events = events
        self.inbox = inbox
        self._config = config

        self._agent_conversations: dict[str, ConversationStore] = {}
        self._delegate_output_dir = self._config.data_dir / "delegates"
        self._delegate_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Conversation reset
    # ------------------------------------------------------------------

    async def new_conversation(self) -> None:
        """Reset the orchestrator conversation (legacy wrapper)."""
        from ..orchestrator import ORCHESTRATOR_ID
        await self.reset_agent_conversation(ORCHESTRATOR_ID)

    async def reset_agent_conversation(self, agent_id: str) -> None:
        """Reset conversation history for any agent.

        Kills the provider session, archives conversation, and re-injects
        a status briefing for the root agent.  The caller is responsible
        for killing running executions first.
        """
        from ..orchestrator import ORCHESTRATOR_ID

        store = self.get_agent_conversation_store(agent_id)
        store.reset()

        old_provider = self.provider_manager._active_providers.pop(agent_id, None)
        if old_provider is not None:
            try:
                await old_provider.close()
            except Exception:
                pass

        # Root agent gets a fresh status briefing after reset
        if agent_id == ORCHESTRATOR_ID:
            self._inject_status_briefing()

        log.info("Conversation reset for %s", agent_id)

    # ------------------------------------------------------------------
    # Per-agent conversation stores
    # ------------------------------------------------------------------

    def get_agent_conversation_store(self, agent_id: str) -> ConversationStore:
        from ..orchestrator import ORCHESTRATOR_ID
        # The root agent's conversation store is the central one visible in the UI.
        if agent_id == ORCHESTRATOR_ID:
            return self.conversation
        if agent_id not in self._agent_conversations:
            store_dir = self._config.data_dir / "agents" / agent_id
            store_dir.mkdir(parents=True, exist_ok=True)
            self._agent_conversations[agent_id] = ConversationStore(store_dir)
        return self._agent_conversations[agent_id]

    async def send_agent_message(self, agent_id: str, text: str) -> dict:
        defn = self.registry.get_agent(agent_id)
        if defn is None:
            return {"error": f"Agent '{agent_id}' not found"}
        if defn.mode != AgentMode.COGNITIVE:
            return {"error": f"Agent '{agent_id}' is not a cognitive agent"}

        store = self.get_agent_conversation_store(agent_id)

        # Mid-session injection — only if actually running
        is_running = self.registry._running_count.get(agent_id, 0) > 0
        if is_running:
            provider = self.provider_manager._active_providers.get(agent_id)
            if provider is not None:
                # Record in conversation store here — the execution engine
                # won't see this message (it goes directly to the provider).
                store.add_user_turn(text)
                await provider.send_user_message(text)
                return {"status": "injected", "agent_id": agent_id}

        # Not running — post to inbox.  Do NOT add_user_turn here;
        # the execution engine records it when it drains the inbox
        # (avoiding duplicates in the conversation store).
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            type=MessageType.WORK,
            source="user",
            target=agent_id,
            priority=MessagePriority.HIGH,
            data={"instruction": text},
        )
        self.inbox.post(msg)

        # Re-activate completed agents (all agents keep their providers)
        status = self.registry.get_status(agent_id)
        if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
            self.registry._status[agent_id] = AgentStatus.ACTIVE
            log.info("Re-activated %s agent %s for follow-up message", status.value, agent_id)
            status = AgentStatus.ACTIVE
        if status == AgentStatus.ACTIVE:
            eid = await self.engine.trigger_run(agent_id, source="user")
            return {"status": "triggered", "agent_id": agent_id, "execution_id": eid}

        return {"status": "queued", "agent_id": agent_id}

    # ------------------------------------------------------------------
    # Delegate output persistence
    # ------------------------------------------------------------------

    def append_delegate_output(self, agent_id: str, text: str) -> None:
        path = self._delegate_output_dir / f"{agent_id}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    def get_delegate_output(self, agent_id: str) -> str:
        path = self._delegate_output_dir / f"{agent_id}.log"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def clear_delegate_output(self, agent_id: str) -> None:
        path = self._delegate_output_dir / f"{agent_id}.log"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Bridge session helpers
    # ------------------------------------------------------------------

    def clear_bridge_session(self) -> None:
        provider = self.provider_manager.get_bridge_provider()
        if provider is not None:
            provider._session_id = ""
            provider._sdk_num_turns = 0
            provider._cumulative_turns = 0
            provider._total_cost_usd = 0.0
            provider._cumulative_input_tokens = 0
            provider._cumulative_output_tokens = 0
            provider._cumulative_cache_read = 0
            provider._cumulative_cache_creation = 0
            provider._last_input_tokens = 0

    # ------------------------------------------------------------------
    # Status briefing
    # ------------------------------------------------------------------

    def _inject_status_briefing(self) -> None:
        from ..orchestrator import ORCHESTRATOR_ID
        from datetime import datetime, timezone

        agents = []
        for aid, defn in self.registry._agents.items():
            if aid == ORCHESTRATOR_ID:
                continue
            status = self.registry._status.get(aid, AgentStatus.REGISTERED).value
            running = self.registry._running_count.get(aid, 0)
            schedule = defn.schedule or None

            last = self.registry.execution_log.get_latest(aid)
            last_info = ""
            if last:
                elapsed = ""
                if last.completed_at:
                    ago = (datetime.now(timezone.utc) - last.completed_at).total_seconds()
                    if ago < 60:
                        elapsed = f"{int(ago)}s ago"
                    elif ago < 3600:
                        elapsed = f"{int(ago / 60)}m ago"
                    else:
                        elapsed = f"{ago / 3600:.1f}h ago"
                last_info = f", last run: {last.status.value}"
                if elapsed:
                    last_info += f" ({elapsed})"
                if last.error:
                    last_info += f" — {last.error[:80]}"

            line = f"- {defn.name} [{aid}] ({status}"
            if schedule:
                line += f", every {schedule}"
            if running:
                line += f", {running} running"
            line += f"{last_info})"
            agents.append(line)

        orch_defn = self.registry._agents.get(ORCHESTRATOR_ID)
        model = orch_defn.cognitive_model if orch_defn else ""

        now = datetime.now(timezone.utc)
        lines = [f"Current time: {now.strftime('%Y-%m-%dT%H:%M:%SZ')} ({now.strftime('%A, %B %d, %Y')})"]
        if model:
            lines.append(f"Model: {model}")
        if agents:
            lines.append(f"{len(agents)} agent(s) currently registered:")
            lines.extend(agents)
        else:
            lines.append("No agents registered. Clean slate.")

        briefing = "\n".join(lines)
        self.conversation.add_system_turn(briefing)
