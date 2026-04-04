"""Agent lifecycle & hierarchy management.

Owns: _agents, _status, _running_count, _schedule_table, _heartbeat_table,
      _last_idle, _child_counters
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent_identity import generate_agent_identity
from ..agent_wallet import AgentWalletManager
from ..events import Event, EventBus, EventType
from ..inbox import InboxManager
from ..models import (
    AgentDefinition,
    AgentStatus,
    ExecutionRecord,
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
)
from ..store import ExecutionLog, OutputStore

log = logging.getLogger(__name__)


class AgentRegistry:
    """Agent registration, activation, hierarchy, and idle tracking."""

    def __init__(
        self,
        events: EventBus,
        execution_log: ExecutionLog,
        inbox: InboxManager,
        output_store: OutputStore,
        config: Any,
    ) -> None:
        self.events = events
        self.execution_log = execution_log
        self.inbox = inbox
        self.output_store = output_store
        self._config = config

        self._agents: dict[str, AgentDefinition] = {}
        self._status: dict[str, AgentStatus] = {}
        self._running_count: dict[str, int] = {}
        self._schedule_table: dict[str, float] = {}
        self._heartbeat_table: dict[str, float] = {}
        self._last_idle: dict[str, datetime] = {}
        self._child_counters: dict[str, int] = {}
        # Budget tracking: agent_id -> provider -> tokens_used (cumulative across all executions)
        self._budget_used: dict[str, dict[str, int]] = {}
        # Wallet manager for agent economic sovereignty
        self._wallet_manager = AgentWalletManager()
        # Agent private keys: agent_id -> private_key_hex (parent holds child's key)
        self._agent_keys: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Identity persistence
    # ------------------------------------------------------------------

    def _identity_path(self, agent_id: str) -> Path:
        """Path to the persisted identity file for an agent."""
        return self._config.agents_dir / agent_id / "identity.json"

    def _save_identity(self, agent_id: str, identity: "AgentIdentity", private_key: str) -> None:
        """Persist agent identity and private key to disk."""
        from ..models import AgentIdentity  # noqa: F811
        path = self._identity_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "public_key": identity.public_key,
            "address": identity.address,
            "lineage_hash": identity.lineage_hash,
            "private_key": private_key,
            "registered_on_chain": identity.registered_on_chain,
            "registration_tx": identity.registration_tx,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Persisted identity for %s to %s", agent_id, path)

    def persist_identity(self, agent_id: str) -> None:
        """Re-save the identity file for an agent (e.g. after registration flag changes)."""
        defn = self._agents.get(agent_id)
        key = self._agent_keys.get(agent_id, "")
        if defn and defn.identity and key:
            self._save_identity(agent_id, defn.identity, key)

    def _load_identity(self, agent_id: str) -> tuple["AgentIdentity", str] | None:
        """Load persisted identity from disk. Returns (identity, private_key) or None."""
        from ..models import AgentIdentity
        path = self._identity_path(agent_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            identity = AgentIdentity(
                public_key=data.get("public_key", ""),
                address=data.get("address", ""),
                lineage_hash=data.get("lineage_hash", ""),
                registered_on_chain=data.get("registered_on_chain", False),
                registration_tx=data.get("registration_tx"),
            )
            private_key = data.get("private_key", "")
            if not private_key or not identity.address:
                log.warning("Identity file for %s is incomplete, regenerating", agent_id)
                return None
            return identity, private_key
        except Exception:
            log.warning("Failed to load identity for %s from %s", agent_id, path, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_agent(self, defn: AgentDefinition) -> str:
        self._agents[defn.id] = defn
        self._status[defn.id] = AgentStatus.REGISTERED
        self._running_count[defn.id] = 0
        # Heartbeat and legacy schedule are mutually exclusive.
        if defn.heartbeat:
            self._heartbeat_table[defn.id] = parse_interval(defn.heartbeat.interval)
            self._schedule_table.pop(defn.id, None)
            self._last_idle.setdefault(defn.id, datetime.now(timezone.utc))
        elif defn.schedule:
            self._schedule_table[defn.id] = parse_interval(defn.schedule)
            self._heartbeat_table.pop(defn.id, None)
            self._last_idle[defn.id] = datetime.now(timezone.utc)
        # Load or generate identity for persistent agents
        if defn.identity is None and defn.system_prompt:
            loaded = self._load_identity(defn.id)
            if loaded:
                identity, private_key = loaded
                defn.identity = identity
                self._agent_keys[defn.id] = private_key
                log.debug("Loaded persisted identity for %s: %s", defn.id, identity.address)
            else:
                parent_identity = None
                if defn.parent_id and defn.parent_id in self._agents:
                    parent_defn = self._agents[defn.parent_id]
                    parent_identity = getattr(parent_defn, 'identity', None)

                identity, private_key = generate_agent_identity(
                    agent_id=defn.id,
                    system_prompt=defn.system_prompt,
                    parent_identity=parent_identity,
                )
                defn.identity = identity
                self._agent_keys[defn.id] = private_key
                self._save_identity(defn.id, identity, private_key)

            # Initialize wallet
            defn.wallet = self._wallet_manager.create_wallet(
                agent_id=defn.id,
                address=identity.address,
                sponsor_funded=defn.sponsor_agent_id is not None,
            )

        # Hydrate execution history
        n = self.execution_log.hydrate(defn.id)
        if n:
            log.debug("Hydrated %d execution record(s) for %s", n, defn.id)
        await self.events.emit(Event(
            type=EventType.AGENT_REGISTERED,
            source="runtime",
            data={"agent_id": defn.id, "name": defn.name,
                  "mode": defn.mode.value,
                  "steps": len(defn.steps), "schedule": defn.schedule,
                  "description": defn.description,
                  "model": defn.model,
                  "parent_id": defn.parent_id,
                  "concurrency": defn.concurrency,
                  "notify_parent": defn.notify_parent},
        ))
        return defn.id

    async def unregister_agent(self, agent_id: str, *, _force: bool = False) -> None:
        """Remove an agent from the registry.

        NOTE: Callers must handle killing executions, cleaning up providers,
        conversations, and files before calling this.
        """
        from ..orchestrator import ORCHESTRATOR_ID
        if agent_id == ORCHESTRATOR_ID and not _force:
            raise ValueError("The orchestrator cannot be unregistered")
        self._agents.pop(agent_id, None)
        self._status.pop(agent_id, None)
        self._running_count.pop(agent_id, None)
        self._schedule_table.pop(agent_id, None)
        self._heartbeat_table.pop(agent_id, None)
        self._last_idle.pop(agent_id, None)
        self._agent_keys.pop(agent_id, None)
        self.inbox.remove_agent(agent_id)
        self.output_store.remove(agent_id)
        self.execution_log.remove_agent(agent_id)
        await self.events.emit(Event(
            type=EventType.AGENT_UNREGISTERED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    async def activate_agent(self, agent_id: str) -> None:
        self._require_agent(agent_id)
        self._status[agent_id] = AgentStatus.ACTIVE
        await self.events.emit(Event(
            type=EventType.AGENT_ACTIVATED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    async def deactivate_agent(self, agent_id: str) -> None:
        self._require_agent(agent_id)
        self._status[agent_id] = AgentStatus.STOPPED
        await self.events.emit(Event(
            type=EventType.AGENT_DEACTIVATED,
            source="runtime",
            data={"agent_id": agent_id},
        ))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self._agents.get(agent_id)

    def get_status(self, agent_id: str) -> AgentStatus | None:
        return self._status.get(agent_id)

    def list_agents(self) -> list[tuple[AgentDefinition, AgentStatus]]:
        return [
            (defn, self._status[aid])
            for aid, defn in self._agents.items()
        ]

    def build_agent_advertisements(self) -> list[dict]:
        """Build p2p AgentAdvertisement dicts for all agents with identity."""
        ads = []
        for aid, defn in self._agents.items():
            identity = defn.identity
            if not identity or not identity.address:
                continue
            parent_addr = ""
            if defn.parent_id:
                parent = self._agents.get(defn.parent_id)
                if parent and parent.identity:
                    parent_addr = parent.identity.address
            ads.append({
                "address": identity.address,
                "name": defn.name,
                "description": defn.description or "",
                "agent_type": defn.agent_type or "",
                "model": defn.model or "",
                "is_root": defn.parent_id is None or defn.parent_id == "",
                "parent_address": parent_addr,
                "registered_on_chain": identity.registered_on_chain,
            })
        return ads

    def get_agent_key(self, agent_id: str) -> str | None:
        """Get the private key for an agent (parent holds child's key)."""
        return self._agent_keys.get(agent_id)

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def _resolve_parent_agent_id(self, parent_id: str) -> str:
        if parent_id in self._agents:
            return parent_id
        from ..orchestrator import ORCHESTRATOR_ID
        if parent_id == "orch" and ORCHESTRATOR_ID in self._agents:
            return ORCHESTRATOR_ID
        return parent_id

    def get_children(self, agent_id: str) -> list[AgentDefinition]:
        from ..orchestrator import ORCHESTRATOR_ID
        children = []
        for defn in self._agents.values():
            if defn.parent_id == agent_id:
                children.append(defn)
            elif (agent_id == ORCHESTRATOR_ID
                  and defn.parent_id == "orch"):
                children.append(defn)
        return children

    def get_descendants(self, agent_id: str) -> list[AgentDefinition]:
        descendants: list[AgentDefinition] = []
        queue = [agent_id]
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            for child in children:
                descendants.append(child)
                queue.append(child.id)
        return descendants

    def generate_child_id(self, parent_id: str) -> str:
        count = self._child_counters.get(parent_id, 0) + 1
        self._child_counters[parent_id] = count
        return f"{parent_id}.{count}"

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def record_token_usage(self, agent_id: str, provider: str, tokens: int) -> str | None:
        """Record token usage for an agent and roll up to ancestors.

        Returns the agent_id of the first ancestor that exceeded its budget, or None.

        BRIDGE POINT — on-chain inference accounting:
        This is where off-chain inference usage should be batched and
        submitted to the RPB contract via OnChainService.record_inference().
        A future implementation would:
        1. Accumulate usage in a per-agent buffer (_pending_inference dict).
        2. When a buffer exceeds a threshold (e.g. 1000 tokens or 60s),
           flush it by calling record_inference(requester, provider, units,
           token, cost) with the owner key from config.
        3. On flush failure, keep the buffer and retry next cycle.
        The agent's on-chain address comes from defn.identity.address; the
        provider address would come from a provider->address mapping in
        config or Registry.
        """
        current = agent_id
        exceeded_agent = None
        while current:
            if current not in self._budget_used:
                self._budget_used[current] = {}
            self._budget_used[current][provider] = self._budget_used[current].get(provider, 0) + tokens

            # Check if this agent exceeded its budget
            defn = self._agents.get(current)
            if defn and defn.budgets:
                limit = defn.budgets.get(provider, 0)
                if limit > 0 and self._budget_used[current].get(provider, 0) >= limit:
                    if exceeded_agent is None:
                        exceeded_agent = current

            # Walk up to parent
            if defn and defn.parent_id:
                current = self._resolve_parent_agent_id(defn.parent_id)
                if current not in self._agents:
                    break
            else:
                break

        return exceeded_agent

    def check_budget(self, agent_id: str, provider: str) -> tuple[bool, str | None]:
        """Check if agent or any ancestor is over budget.

        Returns (ok, blocking_agent_id). ok=True means execution can proceed.
        """
        current = agent_id
        while current:
            defn = self._agents.get(current)
            if defn and defn.budgets:
                limit = defn.budgets.get(provider, 0)
                if limit > 0:
                    used = self._budget_used.get(current, {}).get(provider, 0)
                    if used >= limit:
                        return False, current
            if defn and defn.parent_id:
                current = self._resolve_parent_agent_id(defn.parent_id)
                if current not in self._agents:
                    break
            else:
                break
        return True, None

    def get_budget_info(self, agent_id: str) -> dict[str, dict[str, int]]:
        """Return budget info: {provider: {"limit": N, "used": M, "remaining": R}}"""
        defn = self._agents.get(agent_id)
        if not defn:
            return {}
        result = {}
        for provider, limit in (defn.budgets or {}).items():
            used = self._budget_used.get(agent_id, {}).get(provider, 0)
            result[provider] = {
                "limit": limit,
                "used": used,
                "remaining": max(0, limit - used) if limit > 0 else -1,
            }
        return result

    # ------------------------------------------------------------------
    # Failure propagation
    # ------------------------------------------------------------------

    async def notify_parent_of_failure(
        self, agent_id: str, record: ExecutionRecord,
        active_providers: dict,
    ) -> None:
        """Failures ALWAYS notify parent regardless of notify_parent flag (safety).

        No bridge injection — inbox message only.
        """
        if record.status != ExecutionStatus.FAILED:
            return
        defn = self._agents.get(agent_id)
        if not defn or not defn.parent_id:
            return
        resolved_parent = self._resolve_parent_agent_id(defn.parent_id)
        if resolved_parent not in self._agents:
            return

        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=agent_id,
            target=resolved_parent,
            type=MessageType.ALERT,
            priority=MessagePriority.HIGH,
            data={
                "type": "child_error",
                "child_agent": agent_id,
                "child_name": defn.name,
                "error": record.error or "Unknown error",
                "execution_id": record.execution_id,
                "instruction": (
                    f"Your child agent '{defn.name}' ({agent_id}) failed: "
                    f"{record.error or 'Unknown error'}. "
                    f"Investigate and decide whether to retry, fix, or escalate."
                ),
            },
        )
        self.inbox.post(msg)
        log.info(
            "Failure propagation: posted child_error ALERT for %s -> parent %s",
            agent_id, resolved_parent,
        )

    async def on_agent_completed(
        self, agent_id: str, record: ExecutionRecord,
        active_providers: dict,
    ) -> None:
        """Handle agent completion: update status and optionally notify parent.

        No bridge injection (send_user_message) — inbox message only.
        When notify_parent=False, skip notification entirely (child uses
        post_message explicitly when it has something to report).
        """
        defn = self._agents.get(agent_id)
        if not defn:
            return
        if record.status == ExecutionStatus.COMPLETED:
            if defn.schedule or defn.heartbeat:
                self._status[agent_id] = AgentStatus.ACTIVE
            else:
                self._status[agent_id] = AgentStatus.COMPLETED
        self._last_idle[agent_id] = datetime.now(timezone.utc)

        # Check notify_parent flag — if False, skip notification entirely
        if not defn.notify_parent:
            return

        parent_id = defn.parent_id
        if not parent_id:
            return
        resolved_parent = self._resolve_parent_agent_id(parent_id)
        if resolved_parent not in self._agents:
            return

        # Lean notification — no 2000-char result preview blob.
        # Parent uses get_children_status / get_output / file tools for details.
        status_str = record.status.value
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=agent_id,
            target=resolved_parent,
            type=MessageType.WORK,
            priority=MessagePriority.HIGH,
            data={
                "type": "child_completed",
                "child_id": agent_id,
                "child_name": defn.name,
                "status": status_str,
                "error": record.error,
                "instruction": (
                    f"Your child agent '{defn.name}' ({agent_id}) has {status_str}. "
                    f"Use get_children_status or read its conversation file for details."
                ),
            },
        )
        self.inbox.post(msg)
        log.info("Completion notification: posted child_completed for %s -> parent %s", agent_id, resolved_parent)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_interval(schedule: str) -> float:
    m = re.match(r"^(\d+)\s*([smh])$", schedule.strip().lower())
    if not m:
        raise ValueError(f"Invalid schedule: {schedule!r}  (use e.g. '30s', '5m', '1h')")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600}[unit]
