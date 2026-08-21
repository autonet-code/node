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
    from ..input_arbiter import InputArbiter, SurfaceId

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
        arbiter: "InputArbiter | None" = None,
    ) -> None:
        self.conversation = conversation
        self.registry = registry
        self.provider_manager = provider_manager
        self.engine = engine
        self.events = events
        self.inbox = inbox
        self._config = config
        # Single-writer input gate (P3). May be None in bare/test construction;
        # a None arbiter means "no gating" (every message trusted).
        self._arbiter = arbiter

        self._agent_conversations: dict[str, ConversationStore] = {}
        self._delegate_output_dir = self._config.data_dir / "delegates"
        self._delegate_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Conversation reset
    # ------------------------------------------------------------------

    def _fleet_root_id(self) -> str | None:
        """The fleet root: first registered agent with no parent, or None."""
        for aid, defn in self.registry._agents.items():
            if not defn.parent_id:
                return aid
        return None

    async def new_conversation(self) -> None:
        """Reset the fleet root's conversation (legacy wrapper)."""
        root_id = self._fleet_root_id()
        if root_id is None:
            return
        await self.reset_agent_conversation(root_id)

    async def reset_agent_conversation(self, agent_id: str) -> None:
        """Reset conversation history for any agent.

        Kills the provider session, archives conversation, and re-injects
        a status briefing for root agents.  The caller is responsible
        for killing running executions first.
        """
        store = self.get_agent_conversation_store(agent_id)
        store.reset()

        old_provider = self.provider_manager._active_providers.pop(agent_id, None)
        if old_provider is not None:
            try:
                await old_provider.close()
            except Exception:
                pass

        # Root agents (falsy parent_id) hold user conversations — they get a
        # fresh status briefing after reset.
        defn = self.registry.get_agent(agent_id)
        if defn is not None and not defn.parent_id:
            self._inject_status_briefing(agent_id)

        log.info("Conversation reset for %s", agent_id)

    # ------------------------------------------------------------------
    # Per-agent conversation stores
    # ------------------------------------------------------------------

    def get_agent_conversation_store(self, agent_id: str) -> ConversationStore:
        # Every agent gets its own per-id store; the central self.conversation
        # store remains for the owner/UI surface only.
        if agent_id not in self._agent_conversations:
            store_dir = self._config.data_dir / "agents" / agent_id
            store_dir.mkdir(parents=True, exist_ok=True)
            self._agent_conversations[agent_id] = ConversationStore(store_dir)
        return self._agent_conversations[agent_id]

    async def send_agent_message(
        self, agent_id: str, text: str, *, surface: "SurfaceId | None" = None,
    ) -> dict:
        # Single-writer gate (P3): a message that came through an input surface
        # is only delivered if that surface currently holds the mic. surface=None
        # => trusted internal caller (agent tool, scheduler), never gated.
        # ALWAYS returns a dict; callers must inspect result.get("error").
        if surface is not None and self._arbiter is not None \
                and not self._arbiter.is_active(surface):
            return {"error": "not the active input surface",
                    "code": "input_not_active",
                    "holder": self._arbiter.holder_token()}
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

    def _inject_status_briefing(self, agent_id: str | None = None) -> None:
        """Write a fleet status briefing.

        ``agent_id`` names the agent being briefed (skipped in its own
        listing); its conversation store receives the briefing. With no
        agent named, the central owner/UI store receives it.
        """
        from datetime import datetime, timezone

        agents = []
        for aid, defn in self.registry._agents.items():
            if aid == agent_id:
                continue  # skip the agent being briefed itself
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

        briefed_defn = self.registry._agents.get(agent_id) if agent_id else None
        model = briefed_defn.cognitive_model if briefed_defn else ""

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
        store = (
            self.get_agent_conversation_store(agent_id)
            if agent_id else self.conversation
        )
        store.add_system_turn(briefing)
