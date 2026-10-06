"""Dashboard state aggregation (read-only)."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..models import AgentStatus, StepType, TaskStatus

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .execution_engine import ExecutionEngine
    from .provider_manager import ProviderManager
    from .scheduler import Scheduler
    from .session_manager import SessionManager
    from ..agent_registry import DelegateRegistry
    from ..store import ExecutionLog, OutputStore
    from ..inbox import InboxManager
    from ..user_profile import UserProfileStore
    from ..credit_budget import CreditBudgetStore
    from ..connectors_manager import ConnectorManager
    from ..steps.cognitive import CognitiveStepExecutor


def _preview(obj: Any, max_len: int = 120) -> str:
    if obj is None:
        return ""
    s = str(obj)
    return s[:max_len] + "..." if len(s) > max_len else s


# Voice availability is resolved once and cached. Importing voice_service
# pulls in numpy/sounddevice at module top, which contends on the global
# import lock. Doing that lazily *inside* snapshot() (a synchronous call on
# the event-loop thread) could deadlock against a background thread that is
# mid-import of the same heavy deps — wedging the WS handshake's initial
# snapshot send. Resolve it here, eagerly, so the request path never imports.
try:
    from ..voice_service import VOICE_AVAILABLE as _VOICE_AVAILABLE
except Exception:
    _VOICE_AVAILABLE = False


def _voice_available() -> bool:
    return _VOICE_AVAILABLE


def _daemon_version() -> str:
    """The ATN daemon's version.

    Installed metadata is only authoritative when the RUNNING code is the
    pip install. A source-checkout run (dev machine) imports the tree, not
    site-packages — a stale pip install of autonet-computer must not win
    (observed live: daemon on 0.7.x source reported 0.3.0 from a fossil
    install). Source runs read pyproject.toml and are tagged "+src".
    """
    from pathlib import Path
    try:
        import atn as _atn
        pkg_dir = Path(_atn.__file__).resolve().parent
    except Exception:
        pkg_dir = Path(__file__).resolve().parent.parent
    in_site = any(p in ("site-packages", "dist-packages") for p in pkg_dir.parts)
    if in_site:
        try:
            from importlib.metadata import version
            return version("autonet-computer")
        except Exception:
            pass
    else:
        try:
            import tomllib
            pyproject = pkg_dir.parent / "pyproject.toml"
            if pyproject.exists():
                with open(pyproject, "rb") as f:
                    return str(tomllib.load(f)["project"]["version"]) + "+src"
        except Exception:
            pass
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "unknown"


class SnapshotBuilder:
    """Builds the full dashboard snapshot from all runtime modules."""

    def __init__(
        self,
        registry: "AgentRegistry",
        engine: "ExecutionEngine",
        provider_manager: "ProviderManager",
        scheduler: "Scheduler",
        session_manager: "SessionManager",
        delegate_registry: "DelegateRegistry",
        execution_log: "ExecutionLog",
        output_store: "OutputStore",
        inbox: "InboxManager",
        user_profile: "UserProfileStore",
        credit_budget: "CreditBudgetStore",
        connectors: "ConnectorManager",
        config: Any,
        voice_ref: Any,
        autonet_ref: Any,
        arbiter_ref: Any = None,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.provider_manager = provider_manager
        self.scheduler = scheduler
        self.session_manager = session_manager
        self.delegate_registry = delegate_registry
        self.execution_log = execution_log
        self.output_store = output_store
        self.inbox = inbox
        self.user_profile = user_profile
        self.credit_budget = credit_budget
        self.connectors = connectors
        self._config = config
        self._voice_ref = voice_ref
        self._autonet_ref = autonet_ref
        self._arbiter_ref = arbiter_ref
        # Agent-supervisor reference, refreshed by the Runtime on every
        # snapshot. Only set when process isolation is wired; None otherwise,
        # in which case no agent carries a pid.
        self._supervisor_ref: Any = None

    def snapshot(self, scope_ids: set[str] | None = None) -> dict:
        """Build the dashboard snapshot.

        ``scope_ids`` scopes the view to a subtree: when provided, only agents
        (and their executions / children counts) whose id is in the set are
        included. None (the default) = full fleet — preserving the localhost /
        full-fleet-root behavior byte-for-byte. Global daemon sections
        (providers, connectors, voice, autonet, planning) are not per-agent and
        are returned unchanged regardless of scope."""
        # Worker PIDs, so an owner can map an agent to an OS process from the
        # UI (the CLI `agents` view already prints these). Empty when process
        # isolation is off / the agent runs in-process.
        worker_pids: dict[str, int] = {}
        try:
            for _aid, _w in getattr(
                    self._supervisor_ref, "_workers", {}).items():
                pid = getattr(_w, "pid", None)
                if pid:
                    worker_pids[_aid] = int(pid)
        except Exception:  # noqa: BLE001 — display-only, degrade quietly
            worker_pids = {}

        # One key probe for the whole pass: resolving each agent's effective
        # provider may ask whether an API key is on file.
        _key_probe = self.provider_manager._key_probe()
        agents = {}
        for aid, defn in self.registry._agents.items():
            if scope_ids is not None and aid not in scope_ids:
                continue
            last_output = self.output_store.read(aid)
            _prov = defn.provider or ""
            if isinstance(_prov, list):
                _prov = _prov[0] if _prov else ""
            agent_info: dict = {
                "name": defn.name,
                "description": defn.description,
                "model": defn.model,
                # Pinned inference provider ("" = daemon default). Model
                # pickers use it to offer only that provider's models.
                "provider": _prov,
                # The provider its runs (and so its budget) are booked
                # against, resolved the way the engine routes it.
                "effective_provider": self._effective_provider(
                    defn, _key_probe),
                "mode": defn.mode.value,
                "notify_parent": defn.notify_parent,
                "status": self.registry._status[aid].value,
                "schedule": defn.schedule,
                "heartbeat": defn.heartbeat.interval if defn.heartbeat else None,
                "concurrency": defn.concurrency,
                "running": self.registry._running_count.get(aid, 0),
                "steps": len(defn.steps),
                "step_types": sorted({s.type.value for s in defn.steps}),
                "inbox": self.inbox.count(aid),
                "last_output": _preview(last_output.data) if last_output else None,
                "path": str(self._config.agents_dir / aid),
                # OS pid of this agent's isolated worker process, None when it
                # is not running in one.
                "pid": worker_pids.get(aid),
            }
            if defn.identity and defn.identity.address:
                agent_info["agent_address"] = defn.identity.address
            if defn.identity and defn.identity.registered_on_chain:
                agent_info["registered_on_chain"] = True
            # Marketplace inference binding (docs/services_market.md,
            # 2026-07-26): the substrate this agent thinks on, bought by its
            # parent and paid for out of the agent's OWN wallet. Surfaced next
            # to `model` because it OVERRIDES it — a bound agent's real
            # substrate is the seller's declaration, not this label.
            if getattr(defn, "service_provider", None):
                agent_info["service_provider"] = dict(defn.service_provider)
            if defn.expose_as_tool:
                agent_info["expose_as_tool"] = True
                agent_info["tool_name"] = f"pipeline_{aid}"
            if defn.connector_ids:
                agent_info["connector_ids"] = defn.connector_ids
            agent_type = getattr(defn, "agent_type", "") or ""
            if agent_type and agent_type != "general":
                agent_info["agent_type"] = agent_type
            if defn.parent_id:
                agent_info["parent_id"] = self.registry._resolve_parent_agent_id(defn.parent_id)
            if getattr(defn, "cloned_from", None):
                agent_info["cloned_from"] = defn.cloned_from
            # Tool grant spec (bundle ids / flags) — lets the Tools screen
            # build the reverse index (which agents hold which bundle).
            if defn.tools:
                agent_info["tools"] = list(defn.tools)
            children_count = sum(
                1 for cid, d in self.registry._agents.items()
                if d.parent_id
                and self.registry._resolve_parent_agent_id(d.parent_id) == aid
                and (scope_ids is None or cid in scope_ids)
            )
            if children_count:
                agent_info["children_count"] = children_count
            agents[aid] = agent_info

        executions = {}
        for eid, rec in self.engine._executions.items():
            if scope_ids is not None and rec.agent_id not in scope_ids:
                continue
            defn = self.registry._agents.get(rec.agent_id)
            step_label = ""
            if defn and rec.current_step < len(defn.steps):
                s = defn.steps[rec.current_step]
                step_label = f"[{rec.current_step}] {s.name} ({s.type.value})"
            executions[eid] = {
                "agent_id": rec.agent_id,
                "step": step_label,
                "trigger": rec.trigger_source,
                "started_at": rec.started_at.isoformat(),
            }

        # Connectors
        from ..connectors import get_bundled_specs
        from ..oauth import requires_oauth
        bundled_ids = set(get_bundled_specs().keys())
        running_ids = set(self.connectors.list_running())
        connectors = {}
        for cid in self.connectors.list_available():
            spec = self.connectors.get_spec(cid)
            c_info: dict[str, Any] = {
                "name": spec.name if spec else cid,
                "description": spec.description if spec else "",
                "mode": spec.mode if spec else "",
                "running": cid in running_ids,
                "bundled": cid in bundled_ids,
            }
            if requires_oauth(cid):
                c_info["requires_oauth"] = True
                c_info["authenticated"] = self.provider_manager.credential_store.exists(cid)
            if cid in running_ids:
                session = self.connectors._sessions.get(cid)
                c_info["tool_count"] = len(session.tools) if session else 0
            using_agents = [
                aid for aid, d in self.registry._agents.items()
                if d.connector_ids and cid in d.connector_ids
            ]
            if using_agents:
                c_info["used_by"] = using_agents
            connectors[cid] = c_info

        # Providers summary
        from ..steps.cognitive import CognitiveStepExecutor
        cognitive = self.engine._executors.get(StepType.COGNITIVE)
        registered_providers: set[str] = set()
        if isinstance(cognitive, CognitiveStepExecutor):
            registered_providers = set(cognitive._providers.keys())
        claude_max_rate_limits = self._aggregate_claude_max_rate_limits()
        providers_summary = {}
        for pid, info in self.provider_manager._KNOWN_PROVIDERS.items():
            is_active = pid in registered_providers
            if is_active:
                configured = True
            elif info["auth_type"] == "api_key":
                configured = bool(self.provider_manager._resolve_api_key(pid))
            else:
                configured = False
            entry: dict = {
                "name": info["name"],
                "auth_type": info["auth_type"],
                "configured": configured,
                "active": is_active,
            }
            if pid == "claude_max" and claude_max_rate_limits:
                entry["rate_limits"] = claude_max_rate_limits
            providers_summary[pid] = entry
        for pid in sorted(self.provider_manager._custom_providers):
            is_active = pid in registered_providers
            providers_summary[pid] = {
                "name": pid,
                "auth_type": "api_key",
                "configured": True,
                "active": is_active,
                "custom": True,
            }

        # Planning
        pending_task_count = sum(
            1 for t in self.scheduler.planning_tasks
            if t.status in (TaskStatus.PROPOSED, TaskStatus.APPROVED, TaskStatus.ACTIVE)
        )

        # Voice
        voice_status = self._voice_snapshot()

        return {
            "system": {
                "os": platform.system(),
                "version": platform.version(),       # OS version (legacy field)
                "daemon_version": _daemon_version(),  # the ATN daemon's version
                "arch": platform.machine(),
                "python": platform.python_version(),
                "shell": "powershell" if platform.system() == "Windows" else "bash",
            },
            "update": self._update_snapshot(),
            # The model catalog for pickers. Active-only: a model the daemon
            # has no registered provider for is not pickable, so unconfigured
            # providers stay out of the list.
            "available_models": self.provider_manager.get_available_models(require_active=True),
            "providers": providers_summary,
            # The provider an unpinned new agent runs on (the create form's
            # "(daemon default)" choice), so budgets key to it, not a guess.
            "default_provider": self._default_provider(_key_probe),
            "agents": agents,
            "executions": executions,
            "connectors": connectors,
            "user": self.user_profile.to_summary_dict(),
            "budget": self.credit_budget.to_summary_dict(),
            # LEGACY-WIRE: the periodic planning review is gone (2026-08-30);
            # review keys stay for old frontends and read as "never scheduled".
            "planning": {
                "last_review": None,
                "next_review": None,
                "interval_hours": 0,
                "pending_tasks": pending_task_count,
            },
            "delegates": self._delegates_snapshot(),
            "voice": voice_status,
            "autonet": self._autonet_ref.get_status() if self._autonet_ref else {},
            # Single-writer input arbitration (P3). NOT a secret section — every
            # session may see who currently holds the mic and which surfaces are
            # connected. Deliberately outside _SECRET_SECTIONS in ws_server.
            "input": self._arbiter_ref.state() if self._arbiter_ref else {},
        }

    def _effective_provider(self, defn: Any, key_probe: Any) -> str:
        try:
            out = self.provider_manager.effective_provider_id(
                defn, has_api_key=key_probe)
        except Exception:
            return ""
        return out if isinstance(out, str) else ""

    def _default_provider(self, key_probe: Any) -> str:
        try:
            out = self.provider_manager.default_provider_id(
                has_api_key=key_probe)
        except Exception:
            return ""
        return out if isinstance(out, str) else ""

    def _aggregate_claude_max_rate_limits(self) -> dict:
        """Merge rate-limit snapshots across all active claude_max bridges.

        Each BridgeProvider subprocess sees its own stream of rate_limit_event
        messages.  They all observe the same underlying subscription, so we
        pick the freshest entry per rateLimitType across all of them.
        """
        merged: dict[str, dict] = {}
        for provider in self.provider_manager._active_providers.values():
            if getattr(provider, "name", "") != "claude_max":
                continue
            snapshot = getattr(provider, "_rate_limits", None)
            if not snapshot:
                continue
            for key, entry in snapshot.items():
                existing = merged.get(key)
                if existing is None or entry.get("updatedAt", 0) > existing.get("updatedAt", 0):
                    merged[key] = dict(entry)
        return merged

    def _delegates_snapshot(self) -> dict:
        tree = self.delegate_registry.get_tree()
        for node in tree.get("nodes", []):
            if node.get("status") in ("pending", "running"):
                text = self.session_manager.get_delegate_output(node["agent_id"])
                if text:
                    node["output"] = text
        return tree

    def _voice_snapshot(self) -> dict:
        voice = self._voice_ref
        if voice:
            return voice.get_status()
        return {"running": False, "available": _voice_available()}

    def _update_snapshot(self) -> dict:
        """Auto-update status for the dashboard.

        Reads the staged-update marker straight from disk (ground truth)
        rather than holding a reference to the poll task, so it's accurate
        regardless of whether the poll task is wired into this builder.
        """
        au = getattr(self._config, "auto_update", None)
        enabled = bool(getattr(au, "enabled", False))
        current = _daemon_version()
        out: dict = {
            "enabled": enabled,
            "current_version": current,
            "staged_version": "",
            "pending": False,
        }
        try:
            import json
            data_dir = getattr(self._config, "data_dir", None)
            if data_dir is not None:
                marker = Path(data_dir) / "staged_update" / "pending.json"
                if marker.exists():
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    out["staged_version"] = payload.get("version", "")
                    out["pending"] = True
        except Exception:
            pass
        return out
