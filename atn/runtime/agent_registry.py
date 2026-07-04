"""Agent lifecycle & hierarchy management.

Owns: _agents, _status, _running_count, _schedule_table, _heartbeat_table,
      _last_idle, _child_counters
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
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


# Default safeguards for autonomous long-horizon work. Permissive enough
# that normal use never hits them; tight enough that runaway shapes
# (one parent spawning hundreds of children, or unbounded recursive
# delegation) are caught before they exhaust resources. Override per-agent
# via AgentDefinition.max_children / max_depth_below.
DEFAULT_MAX_CHILDREN_PER_PARENT = 20
DEFAULT_MAX_DEPTH_BELOW_ROOT = 6


# Period rollover for per-agent budgets. "none" means lifetime / no rollover.
_PERIOD_SECONDS: dict[str, int] = {
    "hourly": 3_600,
    "daily": 86_400,
    "weekly": 604_800,
    "monthly": 2_592_000,  # 30-day approximation
}


def _normalize_period(raw: Any) -> str:
    period = str(raw or "none").lower()
    if period not in _PERIOD_SECONDS and period != "none":
        return "none"
    return period


def budget_key(provider: str, model_id: str = "") -> str:
    """Composite budget key: ``provider`` (provider-wide) or ``provider:model``.

    Per-model caps use the composite form so an agent can declare
    ``budgets["claude_max:claude-opus-4-7"]`` alongside a provider-wide
    ``budgets["claude_max"]`` for everything else.
    """
    if not model_id:
        return provider
    return f"{provider}:{model_id}"


def _parse_one_budget(raw: Any) -> tuple[float, str, str]:
    """Parse one budget entry value into (limit, period, unit)."""
    if isinstance(raw, dict):
        period = _normalize_period(raw.get("period"))
        of = str(raw.get("of", "")).lower()
        if "pct" in raw:
            pct = float(raw.get("pct", 0) or 0)
            if of == "parent":
                # Frozen at registration. If still seen here, treat as uncomputed.
                return 0.0, period, "tokens"
            if of in ("subscription_5h", "5h", "subscription"):
                return pct, period, "pct_subscription_5h"
            return 0.0, period, "tokens"
        if "limit" in raw:
            return float(raw.get("limit", 0) or 0), period, "tokens"
        return 0.0, period, "tokens"
    try:
        return float(raw), "none", "tokens"
    except (TypeError, ValueError):
        return 0.0, "none", "tokens"


def _resolve_budget(defn: Any, key: str) -> tuple[float, str, str]:
    """Return (limit, period, unit) for an agent and a budget key.

    ``key`` is either a provider name (``claude_max``) or a composite
    ``provider:model_id`` form. Both are looked up directly on
    ``defn.budgets``; no fallback walking happens here — the recorder
    handles which keys to consult.
    """
    if not defn or not getattr(defn, "budgets", None):
        return 0.0, "none", "tokens"
    raw = defn.budgets.get(key)
    if raw is None:
        return 0.0, "none", "tokens"
    return _parse_one_budget(raw)


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
        # Consecutive FAILED executions per agent (reset on success) — the
        # scheduler's crash-loop backoff keys off this.
        self._consec_failures: dict[str, int] = {}
        # Budget tracking: agent_id -> provider -> tokens_used (cumulative
        # across all executions, persisted to data_dir/budget_state.json).
        self._budget_used: dict[str, dict[str, int]] = {}
        # Per-(agent, provider) period start timestamps (ISO strings, UTC) for
        # period-based budget rollover. agent_id -> provider -> iso_string.
        self._budget_period_start: dict[str, dict[str, str]] = {}
        self._budget_state_path: Path = self._config.data_dir / "budget_state.json"
        self._load_budget_state()
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

            # Phase 12 sanity check: identity.address MUST be the
            # address derived from private_key. Older identity files
            # (pre-substrate) sometimes had address overwritten with
            # the connected wallet — that breaks daemon signing.
            try:
                from eth_account import Account
                pk = private_key if private_key.startswith("0x") else "0x" + private_key
                derived = Account.from_key(pk).address
                if derived.lower() != identity.address.lower():
                    log.warning(
                        "Identity for %s has address %s but key derives %s — "
                        "regenerating (was likely overwritten by an old "
                        "user-wallet-as-agent flow)",
                        agent_id, identity.address, derived,
                    )
                    return None
            except ImportError:
                # eth_account not installed (offline/local-only mode);
                # accept the file as-is.
                pass
            except Exception:
                log.warning("Could not verify key/address match for %s; "
                            "accepting file as-is", agent_id, exc_info=True)

            return identity, private_key
        except Exception:
            log.warning("Failed to load identity for %s from %s", agent_id, path, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Chain reconciliation
    # ------------------------------------------------------------------

    def reconcile_chain_registrations(self) -> dict[str, Any]:
        """Re-verify each agent's ``registered_on_chain`` flag against
        ``Substrate.sol``.

        Daemons cache registration state in each agent's ``identity.json``
        so they don't re-hit chain on every read. That cache survives
        contract redeploys, which can leave agents wrongly believing
        they're still registered. This pass runs once at daemon boot —
        after agents are loaded but before they're advertised — and uses
        ``Substrate.areRegistered([...])`` in a single call to flip stale
        flags back.

        Returns a small summary dict for logging. No-op (and returns an
        empty summary) when ``substrate_address`` or ``rpc_url`` is
        unset, since the daemon is then running offline.
        """
        # Substrate config lives under `self._config.rpb`, populated from
        # `registry.json` by the config loader.
        rpb = getattr(self._config, "rpb", None)
        substrate_addr = getattr(rpb, "substrate_address", "") or ""
        rpc_url = getattr(rpb, "rpc_url", "") or ""
        if not substrate_addr or not rpc_url:
            log.debug("reconcile_chain_registrations: chain unset, skipping")
            return {"checked": 0, "flipped": [], "skipped_reason": "chain_unset"}

        # Collect agents that currently *think* they're registered.
        cached: list[tuple[str, str]] = []  # (agent_id, address)
        for agent_id, defn in self._agents.items():
            identity = getattr(defn, "identity", None)
            if not identity or not identity.registered_on_chain:
                continue
            address = getattr(identity, "address", "") or ""
            if not address:
                continue
            cached.append((agent_id, address))

        if not cached:
            return {"checked": 0, "flipped": []}

        # Single batched chain call. Failure is non-fatal — the daemon
        # can still operate offline; reconciliation will retry next boot.
        try:
            from web3 import Web3
            from ..on_chain import SUBSTRATE_ABI
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            c = w3.eth.contract(
                address=Web3.to_checksum_address(substrate_addr),
                abi=SUBSTRATE_ABI,
            )
            addrs = [Web3.to_checksum_address(a) for _, a in cached]
            results = c.functions.areRegistered(addrs).call()
        except Exception:
            log.warning(
                "reconcile_chain_registrations: chain read failed; "
                "leaving cached registration flags untouched",
                exc_info=True,
            )
            return {"checked": len(cached), "flipped": [], "error": "rpc_failed"}

        flipped: list[str] = []
        for (agent_id, _addr), still_registered in zip(cached, results):
            if still_registered:
                continue
            defn = self._agents.get(agent_id)
            if defn is None or defn.identity is None:
                continue
            log.info(
                "reconcile_chain_registrations: %s no longer registered on "
                "chain (likely a contract redeploy); flipping flag",
                agent_id,
            )
            defn.identity.registered_on_chain = False
            defn.identity.registration_tx = None
            try:
                self.persist_identity(agent_id)
            except Exception:
                log.warning(
                    "reconcile_chain_registrations: could not persist "
                    "identity for %s; flag is updated in memory only",
                    agent_id, exc_info=True,
                )
            flipped.append(agent_id)

        return {"checked": len(cached), "flipped": flipped}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _depth_below_root(self, parent_id: str | None) -> int:
        """Compute how many ancestors away from root a *new child* of parent_id
        would be. Root agents return 0 (the new child would be at depth 1).
        """
        if parent_id is None:
            return 0
        depth = 1
        current = self._resolve_parent_agent_id(parent_id)
        while current and current in self._agents:
            current_defn = self._agents[current]
            if current_defn.parent_id is None:
                return depth
            depth += 1
            current = self._resolve_parent_agent_id(current_defn.parent_id)
            if depth > DEFAULT_MAX_DEPTH_BELOW_ROOT * 4:
                # Pathological — broken parent chain. Bail.
                log.warning("Parent chain depth runaway from %s; assuming detached", parent_id)
                return depth
        return depth

    def _enforce_spawn_limits(self, defn: AgentDefinition) -> None:
        """Check spawn-count and depth limits before registering a new agent.
        Raises ValueError if either limit is exceeded. Skipped for re-registration
        (agent already in the registry — model swaps, reload from disk, etc.).
        """
        if defn.id in self._agents:
            return
        parent_id = defn.parent_id
        if parent_id is None:
            return  # root agent or user-created top-level — no spawn-limit check
        canonical_parent = self._resolve_parent_agent_id(parent_id)
        if canonical_parent not in self._agents:
            return  # disconnected parent (e.g. parent removed) — let it through
        parent_defn = self._agents[canonical_parent]
        max_children = getattr(parent_defn, "max_children", None) or DEFAULT_MAX_CHILDREN_PER_PARENT
        existing_children = sum(
            1 for d in self._agents.values()
            if d.parent_id == parent_id or d.parent_id == canonical_parent
        )
        if existing_children >= max_children:
            raise ValueError(
                f"Spawn limit exceeded: parent '{canonical_parent}' already has "
                f"{existing_children} children (limit {max_children}). "
                f"Raise AgentDefinition.max_children on the parent if this is intentional."
            )
        # Depth check: walk the chain to root, take the strictest max_depth_below
        # set anywhere along the way (root's setting governs the subtree, but a
        # tighter cap on an intermediate ancestor still wins). The depth checked
        # is "how deep would the new child be relative to *that* ancestor."
        depth_to_root = self._depth_below_root(parent_id)
        cursor = canonical_parent
        depth_above_cursor = 0  # parent is depth 0 above itself
        while cursor and cursor in self._agents:
            cursor_defn = self._agents[cursor]
            cap = getattr(cursor_defn, "max_depth_below", None)
            if cap:
                # New child is `depth_above_cursor + 1` below this ancestor
                if depth_above_cursor + 1 > cap:
                    raise ValueError(
                        f"Depth limit exceeded: registering '{defn.id}' would place it "
                        f"{depth_above_cursor + 1} levels below '{cursor}' (limit {cap}). "
                        f"Raise AgentDefinition.max_depth_below on '{cursor}' if intentional."
                    )
            if cursor_defn.parent_id is None:
                break
            cursor = self._resolve_parent_agent_id(cursor_defn.parent_id)
            depth_above_cursor += 1
        # Default cap from root if no explicit override exists upstream.
        if depth_to_root > DEFAULT_MAX_DEPTH_BELOW_ROOT:
            raise ValueError(
                f"Depth limit exceeded: registering '{defn.id}' would make it depth "
                f"{depth_to_root} below root (default limit {DEFAULT_MAX_DEPTH_BELOW_ROOT}). "
                f"Set AgentDefinition.max_depth_below on an ancestor if this is intentional."
            )

    def _freeze_parent_pct_budgets(self, defn: AgentDefinition) -> None:
        """Convert ``{pct, of: parent}`` entries to absolute caps using the
        nearest capping ancestor's cap in the same unit. Mutates defn.budgets.

        Resolution rule per provider:
          • Parent has tokens cap → child's pct of that → tokens cap.
          • Parent has pct_subscription_5h cap → child's pct of that pct
            → pct_subscription_5h cap (still expressed in subscription %).
          • No capping ancestor → reject (a percentage of nothing is nothing).
        """
        if defn.id in self._agents or defn.parent_id is None or not defn.budgets:
            return
        for provider, raw in list(defn.budgets.items()):
            if not isinstance(raw, dict):
                continue
            if "pct" not in raw or str(raw.get("of", "")).lower() != "parent":
                continue
            child_pct = float(raw.get("pct", 0) or 0)
            if child_pct <= 0 or child_pct > 100:
                raise ValueError(
                    f"Agent '{defn.id}' invalid pct={child_pct} on '{provider}': "
                    f"must be in (0, 100]."
                )
            period = _normalize_period(raw.get("period"))
            # Find nearest capping ancestor for this provider.
            cursor = self._resolve_parent_agent_id(defn.parent_id)
            ancestor_limit = 0.0
            ancestor_unit = "tokens"
            while cursor and cursor in self._agents:
                ancestor = self._agents[cursor]
                a_limit, _a_period, a_unit = _resolve_budget(ancestor, provider)
                if a_limit > 0:
                    ancestor_limit = a_limit
                    ancestor_unit = a_unit
                    break
                if ancestor.parent_id is None:
                    break
                cursor = self._resolve_parent_agent_id(ancestor.parent_id)
            if ancestor_limit <= 0:
                raise ValueError(
                    f"Agent '{defn.id}' uses pct-of-parent on '{provider}' but "
                    f"no ancestor declares a cap for that provider. Either name a "
                    f"concrete limit or have an ancestor cap '{provider}' first."
                )
            if ancestor_unit == "tokens":
                resolved = {"limit": int(ancestor_limit * child_pct / 100.0),
                            "period": period}
            else:
                # pct_subscription_5h → child's pct is pct-of-pct, still in subscription %.
                resolved = {"pct": ancestor_limit * child_pct / 100.0,
                            "of": "subscription_5h", "period": period}
            defn.budgets[provider] = resolved
            log.info(
                "Froze pct-of-parent for %s/%s: %.1f%% of ancestor %.4g%s → %r",
                defn.id, provider, child_pct, ancestor_limit, ancestor_unit, resolved,
            )

    def _enforce_budget_required(self, defn: AgentDefinition) -> None:
        """Validate a cognitive agent's budget IF it declares one.

        A per-agent budget is OPTIONAL: a child without one is still bounded by
        its subtree's root cap (token usage rolls up the parent chain), so a
        missing budget can't burn unbounded. Requiring one here was a footgun —
        it blocked conversational spawning (create_agent's schema didn't even
        expose budgets) and pushed the LLM to invent caps, which tended to be
        too small and false-fail the child. So: no budget => fine. If a budget
        IS declared, it must have a positive limit (catch typos / zero caps).
        """
        if defn.id in self._agents:
            return
        from ..models import AgentMode
        if defn.mode != AgentMode.COGNITIVE:
            return
        if not defn.budgets:
            return  # bounded by the subtree root; per-agent cap optional
        # A declared budget must resolve to a positive limit somewhere.
        for provider in defn.budgets:
            limit, _period, _unit = _resolve_budget(defn, provider)
            if limit > 0:
                return
        raise ValueError(
            f"Agent '{defn.id}' declared budgets={defn.budgets!r} but no "
            f"entry has a positive limit. Use a positive int or "
            f"{{'limit': N, 'period': '...'}} per provider, or omit budgets "
            f"to inherit the subtree's cap."
        )

    def validate_budget_update(
        self, defn: AgentDefinition, new_budgets: dict,
    ) -> str | None:
        """Cascade re-check for a budget UPDATE (blessed semantics,
        docs/tool_substrate.md floor item 3: parent-updateable within
        the parent's own headroom).

        Registration-time enforcement skips re-registrations, so
        updates need their own pass. Returns an error string or None.
        """
        if defn.parent_id is None or not new_budgets:
            return None
        old = defn.budgets
        defn.budgets = new_budgets
        try:
            self._check_budget_cascade(defn)
        except ValueError as exc:
            return str(exc)
        finally:
            defn.budgets = old
        return None

    def _enforce_budget_cascade(self, defn: AgentDefinition) -> None:
        """A child's per-provider limit must fit in the parent's remaining headroom.

        Comparable units only — a child capped in tokens isn't constrained by
        an ancestor capped in pct_subscription_5h (the units don't compose
        deterministically until refresh time). For mixed-unit chains, the
        ancestor's runtime check still binds the subtree at execution time.

        Skipped for roots and re-registrations.
        """
        if defn.id in self._agents:
            return
        if defn.parent_id is None:
            return
        if not defn.budgets:
            return  # already handled by _enforce_budget_required for cognitive
        self._check_budget_cascade(defn)

    def _check_budget_cascade(self, defn: AgentDefinition) -> None:
        """The cascade walk itself (raises ValueError on violation)."""
        for provider in list(defn.budgets):
            child_limit, _, child_unit = _resolve_budget(defn, provider)
            if child_limit <= 0:
                continue
            cursor = self._resolve_parent_agent_id(defn.parent_id)
            while cursor and cursor in self._agents:
                ancestor = self._agents[cursor]
                a_limit, _a_period, a_unit = _resolve_budget(ancestor, provider)
                if a_limit > 0 and a_unit == child_unit:
                    used = self._budget_used.get(cursor, {}).get(provider, 0)
                    remaining = max(0.0, a_limit - used)
                    if child_limit > remaining:
                        raise ValueError(
                            f"Budget cascade violation: child '{defn.id}' declares "
                            f"limit={child_limit:g} {child_unit} for '{provider}', "
                            f"but ancestor '{cursor}' only has {remaining:g} {a_unit} "
                            f"remaining (cap={a_limit:g}, used={used:g}). Lower the "
                            f"child's limit or raise the ancestor's."
                        )
                    break  # nearest comparable capping ancestor wins
                if a_limit > 0 and a_unit != child_unit:
                    log.warning(
                        "Skipping cascade check for %s/%s: child unit=%s, "
                        "ancestor %s unit=%s (incomparable).",
                        defn.id, provider, child_unit, cursor, a_unit,
                    )
                if ancestor.parent_id is None:
                    break
                cursor = self._resolve_parent_agent_id(ancestor.parent_id)

    async def register_agent(
        self, defn: AgentDefinition, *, legacy: bool = False,
    ) -> str:
        """Register an agent.

        ``legacy=True`` skips the mandatory-budget check — used by the
        on-disk agent loader so existing agents from before #21 still hydrate.
        Fresh agent creation (orchestrator's create_agent tool, frontend
        register-agent) goes through the strict path.
        """
        self._enforce_spawn_limits(defn)
        self._freeze_parent_pct_budgets(defn)
        if not legacy:
            self._enforce_budget_required(defn)
        self._enforce_budget_cascade(defn)
        self._agents[defn.id] = defn
        self._status[defn.id] = AgentStatus.REGISTERED
        self._running_count[defn.id] = 0
        self._note_child_id(defn.id)
        # Heartbeat and legacy schedule are mutually exclusive.
        if defn.heartbeat:
            self._heartbeat_table[defn.id] = parse_interval(defn.heartbeat.interval)
            self._schedule_table.pop(defn.id, None)
            self._last_idle.setdefault(defn.id, datetime.now(timezone.utc))
        elif defn.schedule:
            self._schedule_table[defn.id] = parse_interval(defn.schedule)
            self._heartbeat_table.pop(defn.id, None)
            self._last_idle[defn.id] = datetime.now(timezone.utc)
        # Load or generate identity for persistent agents. Every agent gets a
        # keypair — the identity is its on-chain self, independent of whether it
        # has a task prompt yet. (The prompt only seasons the lineage hash, and
        # generate_agent_identity handles an empty one.) Gating identity on a
        # non-empty system_prompt used to leave prompt-less agents — e.g. ones
        # provisioned deterministically via the Add-Agent form — with no
        # identity, so they could never register on-chain.
        if defn.identity is None:
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

                # Parentless agents root their lineage in the OWNER's
                # wallet when known (rootless fleet ≠ rootless lineage;
                # fleets hang from a human 0x, never an installation).
                owner_root = ""
                if parent_identity is None:
                    owner_root = getattr(
                        getattr(self._config, "autonet", None),
                        "owner_wallet", "") or ""

                identity, private_key = generate_agent_identity(
                    agent_id=defn.id,
                    system_prompt=defn.system_prompt,
                    parent_identity=parent_identity,
                    owner_root=owner_root,
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
        # The root agent is only protected in legacy mode (orchestrator
        # auto-provisioning enabled). In a rootless fleet it is a normal
        # agent and the owner may remove it like any other.
        _orch_cfg = getattr(self._config, "orchestrator", None)
        if agent_id == ORCHESTRATOR_ID and not _force \
                and getattr(_orch_cfg, "enabled", False):
            raise ValueError("The orchestrator cannot be unregistered")
        self._agents.pop(agent_id, None)
        self._status.pop(agent_id, None)
        self._running_count.pop(agent_id, None)
        self._schedule_table.pop(agent_id, None)
        self._heartbeat_table.pop(agent_id, None)
        self._last_idle.pop(agent_id, None)
        self._agent_keys.pop(agent_id, None)
        if self._budget_used.pop(agent_id, None) is not None or \
           self._budget_period_start.pop(agent_id, None) is not None:
            self._save_budget_state()
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
        # Manual (re)activation is a fresh start for the crash-loop guard.
        self._consec_failures.pop(agent_id, None)
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
                "is_sponsor": getattr(defn, "is_sponsor", False),
            })
        return ads

    def get_agent_key(self, agent_id: str) -> str | None:
        """Get the private key for an agent (parent holds child's key)."""
        return self._agent_keys.get(agent_id)

    def get_agent_id_by_address(self, address: str) -> str | None:
        """Reverse lookup: on-chain/identity address -> agent id, or None.

        Case-insensitive (Ethereum addresses are checksummed but compared
        lowercase). The only existing direction is agent_id -> identity.address
        (build_agent_advertisements); this is the inverse, used to turn a
        signed-challenge signer address back into the agent it roots at.
        Linear scan — fleets are tens of agents, not thousands."""
        if not address:
            return None
        target = address.lower()
        for aid, defn in self._agents.items():
            ident = defn.identity
            if ident and ident.address and ident.address.lower() == target:
                return aid
        return None

    def get_subtree_ids(self, root_id: str) -> set[str]:
        """The set of agent ids in root_id's subtree: {root} ∪ descendants.

        Used for scoping a connection's view (snapshot + event stream) to an
        agent and everything beneath it. Cycle-guarded: a malformed parent
        cycle terminates instead of looping. Includes root_id even if it has
        no children; returns just {root_id} for an unknown id (caller decides
        whether that's an error)."""
        seen: set[str] = {root_id}
        queue = [root_id]
        while queue:
            current = queue.pop(0)
            for child in self.get_children(current):
                if child.id not in seen:
                    seen.add(child.id)
                    queue.append(child.id)
        return seen

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

    def _note_child_id(self, agent_id: str) -> None:
        """Bump the parent's child counter past a hierarchical id.

        Counters are in-memory only; without this, a daemon restart
        that re-registers persisted agents (orch.1, orch.2, …) would
        reset generation to orch.1 and the next spawned child would
        COLLIDE with (and silently shadow) an existing agent. The old
        DelegateRegistry rebuilt counters in load(); this is that
        behavior at the new owner.
        """
        parent, sep, tail = agent_id.rpartition(".")
        if not sep or not tail.isdigit():
            return
        n = int(tail)
        if n > self._child_counters.get(parent, 0):
            self._child_counters[parent] = n

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def _load_budget_state(self) -> None:
        """Load persisted per-agent budget usage from disk."""
        path = self._budget_state_path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to load budget state from %s", path, exc_info=True)
            return
        used = raw.get("used", {})
        starts = raw.get("period_start", {})
        if isinstance(used, dict):
            for aid, providers in used.items():
                if isinstance(providers, dict):
                    self._budget_used[aid] = {
                        p: int(v) for p, v in providers.items()
                        if isinstance(v, (int, float))
                    }
        if isinstance(starts, dict):
            for aid, providers in starts.items():
                if isinstance(providers, dict):
                    self._budget_period_start[aid] = {
                        p: str(v) for p, v in providers.items() if v
                    }
        log.info("Loaded budget state for %d agent(s) from %s",
                 len(self._budget_used), path)

    def _save_budget_state(self) -> None:
        """Persist budget state to disk atomically (write tmp + replace)."""
        path = self._budget_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "used": self._budget_used,
                "period_start": self._budget_period_start,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            data = json.dumps(payload, indent=2)
            # Atomic: write to tmp in same directory, then os.replace.
            fd, tmp_path = tempfile.mkstemp(
                prefix=".budget_state.", suffix=".tmp", dir=str(path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(tmp_path, path)
            except Exception:
                # Cleanup tmp on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            log.exception("Failed to save budget state to %s", path)

    def _maybe_reset_period(self, agent_id: str, provider: str, period: str) -> bool:
        """If the period for (agent, provider) has elapsed, zero the counter
        and stamp a fresh period_started_at. Returns True if a reset happened.
        """
        if period == "none" or period not in _PERIOD_SECONDS:
            return False
        now = datetime.now(timezone.utc)
        starts = self._budget_period_start.get(agent_id, {})
        start_iso = starts.get(provider)
        if not start_iso:
            # First time we're tracking this period — stamp now, no reset.
            self._budget_period_start.setdefault(agent_id, {})[provider] = now.isoformat()
            return False
        try:
            start_dt = datetime.fromisoformat(start_iso)
        except ValueError:
            self._budget_period_start.setdefault(agent_id, {})[provider] = now.isoformat()
            return False
        if (now - start_dt).total_seconds() < _PERIOD_SECONDS[period]:
            return False
        # Period elapsed — reset counter and re-stamp.
        if agent_id in self._budget_used and provider in self._budget_used[agent_id]:
            self._budget_used[agent_id][provider] = 0
        self._budget_period_start.setdefault(agent_id, {})[provider] = now.isoformat()
        log.info("Budget period reset for %s/%s (%s)", agent_id, provider, period)
        return True

    def record_token_usage(
        self,
        agent_id: str,
        provider: str,
        tokens: int,
        *,
        model_class: str = "",
        model_id: str = "",
        tokens_per_pct: dict[str, float] | None = None,
    ) -> str | None:
        """Record token usage for an agent and roll up to ancestors.

        Walks each ancestor and updates *every* budget key that applies to
        this turn — both the provider-wide key (``provider``) and the
        per-model key (``provider:model_id``) when ``model_id`` is supplied.
        Returns the agent_id of the first ancestor where any cap is exceeded.

        Per-cap unit handling:
          • ``tokens`` → increment by ``tokens``
          • ``pct_subscription_5h`` → increment by ``tokens / tokens_per_pct[class]``
        """
        keys = [provider]
        if model_id:
            keys.append(budget_key(provider, model_id))

        current = agent_id
        exceeded_agent: str | None = None
        while current:
            defn = self._agents.get(current)
            for key in keys:
                limit, period, unit = _resolve_budget(defn, key)
                self._maybe_reset_period(current, key, period)
                if current not in self._budget_used:
                    self._budget_used[current] = {}
                increment = self._increment_for_unit(
                    tokens, unit, model_class, tokens_per_pct,
                )
                if increment > 0:
                    self._budget_used[current][key] = (
                        self._budget_used[current].get(key, 0) + increment
                    )
                if limit > 0 and self._budget_used[current].get(key, 0) >= limit:
                    if exceeded_agent is None:
                        exceeded_agent = current

            if defn and defn.parent_id:
                current = self._resolve_parent_agent_id(defn.parent_id)
                if current not in self._agents:
                    break
            else:
                break

        self._save_budget_state()
        return exceeded_agent

    @staticmethod
    def _increment_for_unit(
        tokens: int,
        unit: str,
        model_class: str,
        tokens_per_pct: dict[str, float] | None,
    ) -> float:
        """Convert a per-turn token count into the cap unit's increment."""
        if unit == "tokens":
            return float(tokens)
        if unit == "pct_subscription_5h":
            if not tokens_per_pct or not model_class:
                return 0.0  # estimator not ready or model unknown — skip enforcement increment
            rate = tokens_per_pct.get(model_class)
            if not rate or rate <= 0:
                return 0.0
            return tokens / rate
        return 0.0

    def check_budget(
        self,
        agent_id: str,
        provider: str,
        *,
        model_id: str = "",
    ) -> tuple[bool, str | None]:
        """Check if agent or any ancestor is over budget for the given keys.

        Examines both the provider-wide cap (``provider``) and the per-model
        cap (``provider:model_id``) when ``model_id`` is supplied. The first
        cap to fail wins.
        """
        keys = [provider]
        if model_id:
            keys.append(budget_key(provider, model_id))

        current = agent_id
        any_reset = False
        while current:
            defn = self._agents.get(current)
            for key in keys:
                limit, period, _unit = _resolve_budget(defn, key)
                if self._maybe_reset_period(current, key, period):
                    any_reset = True
                if limit > 0:
                    used = self._budget_used.get(current, {}).get(key, 0)
                    if used >= limit:
                        if any_reset:
                            self._save_budget_state()
                        return False, current
            if defn and defn.parent_id:
                current = self._resolve_parent_agent_id(defn.parent_id)
                if current not in self._agents:
                    break
            else:
                break
        if any_reset:
            self._save_budget_state()
        return True, None

    def get_budget_info(self, agent_id: str) -> dict[str, dict[str, Any]]:
        """Return budget info per declared key.

        Keys are either ``provider`` (provider-wide cap) or ``provider:model_id``
        (per-model cap). Each entry includes ``scope`` ('provider' | 'model'),
        ``provider``, optional ``model_id``, plus the standard limit/used/period
        fields.
        """
        defn = self._agents.get(agent_id)
        if not defn:
            return {}
        result: dict[str, dict[str, Any]] = {}
        any_reset = False
        for key in (defn.budgets or {}).keys():
            limit, period, unit = _resolve_budget(defn, key)
            if self._maybe_reset_period(agent_id, key, period):
                any_reset = True
            used = self._budget_used.get(agent_id, {}).get(key, 0)
            if ":" in key:
                provider, model_id = key.split(":", 1)
                scope = "model"
            else:
                provider, model_id = key, ""
                scope = "provider"
            result[key] = {
                "scope": scope,
                "provider": provider,
                "model_id": model_id,
                "limit": limit,
                "used": used,
                "remaining": max(0.0, limit - used) if limit > 0 else -1,
                "period": period,
                "unit": unit,
                "period_started_at": self._budget_period_start.get(agent_id, {}).get(key),
            }
        if any_reset:
            self._save_budget_state()
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
