"""The Runtime — central brain of ATN.

This package is a thin facade that composes focused submodules:
  - agent_registry:     Agent lifecycle & hierarchy
  - execution_engine:   Pipeline + cognitive execution paths
  - execution_control:  Kill, interrupt, hooks
  - scheduler:          Background scheduler + inbox watcher loops
  - provider_manager:   LLM provider lifecycle & resolution
  - session_manager:    Conversation stores, message injection, delegate output
  - snapshot:           Dashboard state aggregation
  - orchestrator_setup: Orchestrator-specific init & model switching
  - config_helpers:     Status briefing, credential injection, planning tasks

All public methods are delegated from the ``Runtime`` class below so that
``from atn.runtime import Runtime`` continues to work unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from ..events import Event, EventBus, EventType
from ..inbox import InboxManager
from ..models import (
    AgentDefinition,
    AgentMode,
    AgentStatus,
    StepType,
)
from ..config import ATNConfig
from ..connectors_manager import ConnectorManager, ConnectorSpec
from ..conversation import ConversationStore
from ..credit_budget import CreditBudgetStore
from ..sponsor_bindings import SponsorBindingStore
from ..credentials import CredentialStore
from ..agent_registry import DelegateRegistry
from ..store import ExecutionLog, OutputStore
from ..user_profile import UserProfileStore

from ..steps.cognitive import CognitiveStepExecutor
from ..steps.collect import CollectStepExecutor
from ..steps.message import MessageStepExecutor
from ..steps.pull import PullStepExecutor
from ..steps.script import ScriptStepExecutor

from .agent_registry import AgentRegistry
from .execution_engine import ExecutionEngine, _preview, _accumulate_child_usage
from .execution_control import ExecutionControl
from .scheduler import Scheduler
from .provider_manager import ProviderManager
from .session_manager import SessionManager
from .snapshot import SnapshotBuilder
from .orchestrator_setup import OrchestratorSetup
from .config_helpers import inject_connector_credentials, load_planning_tasks, save_planning_tasks

log = logging.getLogger(__name__)


class Runtime:
    """Thin facade that composes all runtime modules.

    Every public method signature is preserved so callers (cli.py,
    ws_server.py, orchestrator/tools.py, tests) need no changes.
    """

    def __init__(self, event_bus: EventBus, data_dir: Path | None = None, config: ATNConfig | None = None) -> None:
        self.events = event_bus
        self.data_dir = data_dir
        self._config = config or ATNConfig()

        # Single-writer input arbitration (P3). Runtime-owned; a gate ABOVE
        # every per-surface InputPolicy. Surfaces register/release against it;
        # SessionManager.send_agent_message consults is_active() at the
        # chokepoint. See atn/input_arbiter.py.
        from ..input_arbiter import InputArbiter
        self.input_arbiter = InputArbiter(self.events)

        # --- Core stores (owned by Runtime, injected into modules) ---
        self.inbox = InboxManager()
        self.output_store = OutputStore()
        self.execution_log = ExecutionLog(agents_dir=self._config.agents_dir)
        self.conversation = ConversationStore(self._config.data_dir)
        self.credential_store = CredentialStore(self._config.data_dir)
        self.user_profile = UserProfileStore(self._config.data_dir)
        self.credit_budget = CreditBudgetStore(self._config.data_dir)
        self.sponsor_bindings = SponsorBindingStore(self._config.data_dir)

        # Shutdown event
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Planning tasks
        self._planning_tasks_path = self._config.data_dir / "planning_tasks.json"
        self.planning_tasks = load_planning_tasks(self._planning_tasks_path)

        # MCP connectors
        from ..connectors import get_bundled_specs
        connector_specs = get_bundled_specs()
        for name, cc in self._config.connectors.items():
            connector_specs[name] = ConnectorSpec(
                mode=cc.mode,
                package=cc.package,
                entry=cc.entry,
                command=cc.command or None,
                args=cc.args,
                env=cc.env,
                env_required=cc.env_required,
            )
        self.connectors = ConnectorManager(connector_specs)
        inject_connector_credentials(self.connectors, self.credential_store)

        # Delegate registry
        self.delegate_registry = DelegateRegistry(
            store_path=self._config.data_dir / "delegates.json",
        )

        # Step executors
        cognitive = CognitiveStepExecutor()
        self._executors = {
            StepType.SCRIPT: ScriptStepExecutor(),
            StepType.MESSAGE: MessageStepExecutor(),
            StepType.PULL: PullStepExecutor(),
            StepType.COLLECT: CollectStepExecutor(),
            StepType.COGNITIVE: cognitive,
        }

        # --- Module composition ---
        self.registry = AgentRegistry(
            events=self.events,
            execution_log=self.execution_log,
            inbox=self.inbox,
            output_store=self.output_store,
            config=self._config,
        )

        self.providers = ProviderManager(
            config=self._config,
            credential_store=self.credential_store,
            executors=self._executors,
            events=self.events,
        )
        # Provide execution_log reference for usage aggregation
        self.providers._execution_log = self.execution_log

        # Setup providers from config (sync, before event loop)
        self.providers.setup_providers(cognitive)

        self.engine = ExecutionEngine(
            registry=self.registry,
            provider_manager=self.providers,
            session_manager=None,  # set below after SessionManager init
            events=self.events,
            execution_log=self.execution_log,
            inbox=self.inbox,
            connectors=self.connectors,
            output_store=self.output_store,
            delegate_registry=self.delegate_registry,
            executors=self._executors,
            config=self._config,
        )
        # Give engine a reference back to the full Runtime for tool execution
        self.engine._runtime_ref = self

        # --- Agent process-isolation supervisor (P3, flag-gated) ---------
        # The single per-Runtime holder of worker PID handles + kill/monitor/
        # orphan logic. Constructed unconditionally (cheap, side-effect free)
        # so ExecutionControl can consult it; but it only acts against a real
        # process when ATN_WORKER_ISOLATION is ON *and* a worker is registered.
        # With the flag OFF nothing here touches an OS process. The two security
        # hooks are injected as NO-OPs here (P3) — the vault track fills them.
        # Daemon-held Seam A -> Seam B bridge for the secret-allowance track.
        # Maps child_agent_id -> daemon-computed enforced grant (JSON-safe list
        # of resolved vault service names; NEVER the _ALLOWANCE_ALL sentinel,
        # NEVER on the RPC/worker path). Written at allowance intersection
        # (Seam A, before child spawn) and popped once at PID->identity bind
        # (Seam B). Nothing consumes it yet in P0 — inert until P4.
        self._pending_grants: dict[str, list[str]] = {}

        # --- Live broker sessions (secret-allowance track, Phase 4) ----------
        # pid -> {"agent_id": str, "services": list[str]} for every worker that
        # actually received a broker session at Seam B. The revoke hook pops this
        # to release the session + drop the monitor exposure. Daemon-only; never
        # carries a value and never crosses the worker/RPC boundary.
        self._grants: dict[int, dict[str, Any]] = {}

        # --- Vault broker client (secret-allowance track, Phase 2) -----------
        # The daemon's privileged client to the kevin vault-broker (owner secret
        # from BROKER_OWNER_SECRET). Constructed unconditionally (cheap, no I/O
        # until an op is called); nothing consumes it yet in P2 (Seam B / P4 is
        # the first caller of mint_nonce/register; _reap / P4 calls
        # release_session; the value-push listener is armed once the monitor
        # exists in P3). It is daemon-only and NEVER handed to a worker.
        from .broker_client import DaemonBrokerClient, _SECRET_MARKERS_OK
        self._broker_client = DaemonBrokerClient()
        # STARTUP INVARIANTS (fail-loud, dev-time): (1) the owner-secret env var
        # name must contain a _SECRET_MARKERS substring so the worker env-scrub
        # (worker_manager._scrub_env) strips it — otherwise a spawned worker
        # could inherit the owner secret and mint its own grants. (2) VAULT_TOKEN
        # must be ABSENT from the daemon env — the broker's Vault token lives ONLY
        # in the vault-svc process; if it leaked into the daemon a worker could
        # inherit it (SECRET/TOKEN markers do scrub it, but its presence here is a
        # deployment error worth surfacing).
        _secret_env_ok = _SECRET_MARKERS_OK("BROKER_OWNER_SECRET")
        if not _secret_env_ok:
            log.error(
                "SECURITY: BROKER_OWNER_SECRET env name is not covered by the "
                "worker env-scrub markers; a worker could inherit it. Refusing "
                "to treat the broker client as usable.")
        if os.environ.get("VAULT_TOKEN"):
            log.error(
                "SECURITY: VAULT_TOKEN is present in the daemon environment; the "
                "broker's Vault token must live ONLY in the vault-svc process. "
                "This is a deployment error.")

        # --- Security tripwire (secret-allowance track, Phase 3) -------------
        # The daemon-side live-exposure monitor. Subscribes to STEP_OUTPUT and
        # scans every agent transcript chunk for a leaked secret VALUE or its
        # staged PATH, firing SECURITY_ALARM on a hit. Constructed + STARTED
        # unconditionally here (cheap; with an empty live-set every scan is a
        # no-op, so flag-off behaviour is externally byte-identical). It MUST
        # exist before the supervisor wires Seam B / revoke below so that P4 can
        # refuse to grant when the monitor is down (monitor-down => no tripwire
        # => no secrets), and so drop_exposure is reachable from the revoke hook.
        #
        # The broker value-push (P2) is armed here: each staged exposure the
        # broker pushes back over the owner-authenticated pipe is delivered
        # straight into the monitor's live-set (value scanned daemon-side ONLY,
        # never into model context, never logged). Arming is Windows-only and
        # best-effort; a failure to arm leaves value_push_armed False, which P4
        # will treat as monitor-down (fail-closed: refuse to grant).
        from .security_monitor import SecurityMonitor
        self.security_monitor = SecurityMonitor(
            self.events, data_dir=self._config.data_dir)
        self.security_monitor.start()

        def _value_push_sink(exposure: dict) -> None:
            # Owner-authenticated push frame -> live-set. The raw value lives
            # ONLY here (and the worker's staged file); it is dropped at revoke.
            try:
                self.security_monitor.add_exposure(
                    value=exposure.get("value"),
                    staged_path=exposure.get("staged_path"),
                    secret_name=exposure.get("secret_name") or "",
                    agent_id=exposure.get("agent_id") or "",
                    pid=exposure.get("pid"),
                )
            except Exception:  # noqa: BLE001 — a bad push must not kill anything
                log.debug("value-push -> add_exposure failed", exc_info=True)

        try:
            self._broker_client.arm_value_push(_value_push_sink)
        except Exception:  # noqa: BLE001
            log.debug("value-push listener failed to arm", exc_info=True)

        from .worker_manager import WorkerManager
        from .agent_supervisor import AgentSupervisor
        _wi_cfg = getattr(self._config, "worker_isolation", None)
        _wi_cap = int(getattr(_wi_cfg, "memory_cap_mb", 0) or 0)
        self.supervisor = AgentSupervisor(
            worker_manager=WorkerManager(memory_cap_mb=_wi_cap),
            finalize_fn=self.engine._finalize_execution,
            on_pid_bound=self._on_pid_bound,
            on_pid_revoked=self._on_pid_revoked,
            workers_dir=self._config.data_dir / "workers",
        )
        # Let the engine reach the supervisor (P4 registers workers there).
        self.engine.supervisor = self.supervisor

        self.control = ExecutionControl(
            engine=self.engine,
            provider_manager=self.providers,
            supervisor=self.supervisor,
        )

        self.sessions = SessionManager(
            conversation=self.conversation,
            registry=self.registry,
            provider_manager=self.providers,
            engine=self.engine,
            events=self.events,
            inbox=self.inbox,
            config=self._config,
            arbiter=self.input_arbiter,
        )
        # Wire session_manager into engine (circular dep resolved here)
        self.engine.session_manager = self.sessions

        self.scheduler = Scheduler(
            registry=self.registry,
            engine=self.engine,
            provider_manager=self.providers,
            events=self.events,
            inbox=self.inbox,
            user_profile=self.user_profile,
            credit_budget=self.credit_budget,
            config=self._config,
            planning_tasks=self.planning_tasks,
        )

        self._orch_setup = OrchestratorSetup(
            registry=self.registry,
            provider_manager=self.providers,
            session_manager=self.sessions,
            config=self._config,
            user_profile=self.user_profile,
        )

        # Voice service (lazy)
        self.voice = None  # type: Any

        # Chat service (lazy) — binds agent conversations to a chat platform
        # (Discord etc). Started by the deployment via start_chat(), which
        # supplies the platform adapter + input policy; autonet core stays free
        # of platform SDKs and deployment-specific policy.
        self.chat = None  # type: Any
        self._chat_task = None  # background task running the chat adapter's client

        # WebSocket bridge (set by register_surface() from cli.py once the
        # bridge's listeners are up). A first-class Surface like chat/voice:
        # long-lived, bidirectional, with an input seam (its .policy). The CLI
        # owns the bridge's construction + port-retry; the runtime owns its
        # Surface identity and input-seam policy.
        self.ws_bridge = None  # type: Any

        # Agent directory publisher (lazy) — writes this daemon's
        # browser-reachable wss:// endpoint into the Firestore agent directory
        # so a remote frontend can find it by an agent's 0x address. Started in
        # start() only when autonet.public_ws_endpoint is configured.
        self.agent_directory = None  # type: Any
        self.chat_credits = None  # type: Any  # CreditStore when policy=credit

        # Autonet service (Story 3.2: pass data_dir for training data feed)
        from ..autonet_service import AutonetBridge
        self.autonet = AutonetBridge(
            self._config.autonet,
            event_bus=self.events,
            data_dir=str(self._config.data_dir),
        )
        # Give engine access to the autonet bridge for constitutional injection
        self.engine._autonet_bridge = self.autonet
        # Wire runtime reference so sponsor handler can resolve providers
        self.autonet._runtime = self
        # Wire agent registry into autonet bridge for p2p advertisement
        self.autonet.set_agent_registry(self.registry)

        # Trace logger (Phase B Step 1 — Agent Trace Collection)
        from ..trace_logger import TraceLogger, TraceLoggingConfig as _TLConfig
        _tl_cfg = self._config.trace_logging
        _trace_dir = (
            Path(_tl_cfg.trace_dir).expanduser()
            if _tl_cfg.trace_dir
            else self._config.data_dir / "traces"
        )
        self.trace_logger = TraceLogger(
            config=_TLConfig(
                enabled=_tl_cfg.enabled,
                trace_dir=str(_trace_dir),
                include_user_data=_tl_cfg.include_user_data,
                min_turns=_tl_cfg.min_turns,
            ),
            trace_dir=_trace_dir,
        )
        # Wire trace_logger into the execution engine
        self.engine.trace_logger = self.trace_logger

        # Tool registry
        from ..tool_registry import ToolRegistry
        self.tool_registry = ToolRegistry(self)

        # Snapshot builder (created lazily to capture voice/autonet refs)
        self._snapshot_builder: SnapshotBuilder | None = None

    def _get_snapshot_builder(self) -> SnapshotBuilder:
        if self._snapshot_builder is None:
            self._snapshot_builder = SnapshotBuilder(
                registry=self.registry,
                engine=self.engine,
                provider_manager=self.providers,
                scheduler=self.scheduler,
                session_manager=self.sessions,
                delegate_registry=self.delegate_registry,
                execution_log=self.execution_log,
                output_store=self.output_store,
                inbox=self.inbox,
                user_profile=self.user_profile,
                credit_budget=self.credit_budget,
                connectors=self.connectors,
                config=self._config,
                voice_ref=self.voice,
                autonet_ref=self.autonet,
                arbiter_ref=self.input_arbiter,
            )
        # Update mutable refs
        self._snapshot_builder._voice_ref = self.voice
        self._snapshot_builder._autonet_ref = self.autonet
        return self._snapshot_builder

    # ==================================================================
    # Lifecycle
    # ==================================================================

    # ------------------------------------------------------------------
    # Worker-isolation security seam (P3 no-op hooks).
    # Injected into AgentSupervisor so agent_supervisor.py never imports vault
    # code. Fired _on_pid_bound after spawn+ready (BEFORE "go") and _on_pid_revoked
    # in _reap. The vault/secrets track fills these later; they are pure no-ops now.
    # ------------------------------------------------------------------

    def _on_pid_bound(self, agent_id: str, pid: int, parent_agent_id: str | None) -> None:
        """Fired when a worker PID is bound to an agent identity, AFTER the worker
        is ready but BEFORE it is told "go" (so the grant is in place before the
        worker can run any tool). Mints a broker session for the daemon-computed
        allowance stashed at Seam A and binds it to this PID.

        FAIL-CLOSED at every gate — the default outcome is *no session* (deny):
          1. monitor-down (not subscribed/sweeper dead) OR value-push not armed
             => no tripwire => refuse to grant.
          2. no stashed grant / empty grant => nothing to authorize.
          3. no service maps to a broker policy => nothing to authorize.
          4. mint_nonce / register returns not-ok (broker down, bad reply) => no
             session.
        Only on a fully successful mint+register do we record the live session so
        the revoke hook can tear it down. The stash is always popped (consumed
        once) whether or not a session is minted, so it can't leak into a later
        PID rebind."""
        # (1) Monitor-down => no secrets. Both the scan (monitor subscribed +
        # sweeper alive) and the delivery channel (value-push armed) must be up,
        # else a staged value would never reach the scanner: a silent blind spot.
        try:
            monitor_up = self.security_monitor.is_healthy()
        except Exception:  # noqa: BLE001
            monitor_up = False
        push_up = False
        try:
            push_up = bool(self._broker_client.value_push_armed)
        except Exception:  # noqa: BLE001
            push_up = False
        if not (monitor_up and push_up):
            # Consume the stash so it can't be picked up by a future rebind.
            self._pending_grants.pop(agent_id, None)
            if not monitor_up or not push_up:
                log.warning(
                    "secret-grant refused for agent=%s pid=%s: monitor_up=%s "
                    "push_armed=%s (fail-closed)", agent_id, pid, monitor_up, push_up)
            return None

        # (2) The daemon-computed enforced grant (Seam A). Popped exactly once.
        l_child = self._pending_grants.pop(agent_id, [])
        if not l_child:
            return None

        # (3) Resolve service NAMES -> broker policy names (unmapped dropped).
        from .broker_client import _services_to_policies
        policies = _services_to_policies(l_child)
        if not policies:
            log.warning(
                "secret-grant: agent=%s pid=%s had grant %s but no service maps "
                "to a broker policy; no session", agent_id, pid, l_child)
            return None

        # (4) Mint a one-time nonce (owner-gated) then bind it to the PID. Any
        # not-ok reply (broker down, empty owner secret, bad frame) => no session.
        try:
            r = self._broker_client.mint_nonce(policies, agent_id=agent_id)
        except Exception:  # noqa: BLE001
            log.warning("secret-grant: mint_nonce raised for agent=%s pid=%s",
                        agent_id, pid, exc_info=True)
            return None
        if not (isinstance(r, dict) and r.get("ok") and r.get("nonce")):
            log.warning("secret-grant: mint_nonce not-ok for agent=%s pid=%s: %s",
                        agent_id, pid, (r or {}).get("error"))
            return None

        try:
            r2 = self._broker_client.register(pid, r["nonce"])
        except Exception:  # noqa: BLE001
            log.warning("secret-grant: register raised for agent=%s pid=%s",
                        agent_id, pid, exc_info=True)
            return None
        if not (isinstance(r2, dict) and r2.get("ok")):
            log.warning("secret-grant: register not-ok for agent=%s pid=%s: %s",
                        agent_id, pid, (r2 or {}).get("error"))
            return None

        # Session is live. Record it (services = the resolved vault NAMES, never a
        # value) so revoke can release it. The broker value-push (P2/P3) now feeds
        # the monitor as the worker stages each value.
        self._grants[int(pid)] = {"agent_id": agent_id, "services": list(l_child)}
        log.info("secret-grant: session minted for agent=%s pid=%s services=%s",
                 agent_id, pid, l_child)
        return None

    def _on_pid_revoked(self, pid: int, agent_id: str | None) -> None:
        """Fired when a worker PID is reaped. Tears down any broker session and
        drops the monitor exposure for the PID, and drains the stale grant stash.
        Each teardown is independent (best-effort): a failure in one must not
        block the others, and the broker's TTL is the backstop if release fails."""
        # Ownership guard (pid-reuse + M3-reconcile safe): peek first, and only
        # release a grant that belongs to the agent we're revoking for. A recycled
        # pid may now hold a DIFFERENT worker's live grant (M5 eviction, or the M3
        # off-loop bind reconcile) — tearing that down would DoS the live worker.
        grant = self._grants.get(int(pid)) if pid is not None else None
        if grant is not None:
            owner = grant.get("agent_id")
            if agent_id and owner not in (None, agent_id):
                grant = None  # not ours — leave it live
            else:
                self._grants.pop(int(pid), None)
        if grant is not None:
            # (a) Release the broker session (owner-gated). Independent try; the
            # broker also TTLs the session, so a failure here is not fatal.
            try:
                self._broker_client.release_session(int(pid))
            except Exception:  # noqa: BLE001
                log.debug("secret-revoke: release_session failed for pid=%s",
                          pid, exc_info=True)
            # (b) Drop the live exposure(s) for this PID — drops the raw value(s)
            # from the monitor live-set and clears the agent's tail buffer.
            try:
                self.security_monitor.drop_exposure(
                    pid=int(pid), agent_id=agent_id or grant.get("agent_id"))
            except Exception:  # noqa: BLE001
                log.debug("secret-revoke: drop_exposure failed for pid=%s",
                          pid, exc_info=True)

        # (c) Drain any stale stash keyed by this agent (e.g. a grant was stashed
        # at Seam A but the worker died before Seam B popped it). Uses the
        # pipe-bound agent_id the supervisor passed.
        if agent_id:
            self._pending_grants.pop(agent_id, None)
        return None

    async def start(self) -> None:
        # Recover crashed executions
        recovered = self.execution_log.recover_running()
        if recovered:
            log.warning("Recovered %d crashed execution(s) from previous run", len(recovered))

        # Worker-isolation orphan recovery (P3). Reap any worker processes left
        # behind by a previously-crashed daemon (verified via the
        # (pid, create_time, cmdline) triple to defeat PID reuse), then start the
        # monitor loop when the flag is ON. recover_orphans() ONLY reaps OS
        # processes + pid files; the crashed executions themselves are already
        # marked FAILED above by recover_running() (which owns the terminal
        # event) — the two paths are coordinated so exactly one terminal event
        # fires per execution. With the flag OFF there are no pid files, so this
        # is a no-op and the monitor never starts.
        try:
            orphans = self.supervisor.recover_orphans()
            if orphans:
                log.warning("Reaped %d orphan worker(s) from previous daemon: %s",
                            len(orphans), orphans)
        except Exception:
            log.debug("supervisor orphan recovery failed", exc_info=True)
        if self._config.worker_isolation.enabled:
            self.supervisor.start_monitor()

        self.delegate_registry.cleanup_orphans()
        self.delegate_registry.save()

        # Auto-detect providers (async probes)
        await self.providers.auto_detect_providers()

        # Start scheduler loops
        self.scheduler.start()

        # Start voice service if configured
        if self._config.voice.enabled:
            await self.start_voice()

        # Start chat service if configured (Discord etc)
        _chat_cfg = getattr(self._config, "chat", None)
        if _chat_cfg and _chat_cfg.enabled:
            res = await self.start_chat()
            log.info("chat service: %s", res)

        # Phase 12: autonet (training + libp2p) is registration-driven,
        # not boot-driven. The framework stays fully local until at
        # least one agent is registered on chain. Two ways autonet
        # starts:
        #   (a) explicit ``autonet.enabled = true`` in config — for
        #       users who want eager start (bootstrap nodes, services
        #       that always participate),
        #   (b) the user clicks "register on chain" on the first agent;
        #       the registration handler starts autonet just-in-time
        #       so it can read the libp2p PeerId before signing the tx.
        if self._config.autonet.enabled:
            await self.autonet.start()
            # Wire P2P host into provider manager for substrate
            # provider discovery (only meaningful once p2p is up).
            if self.autonet._p2p_host:
                self.providers._p2p_host = self.autonet._p2p_host
        # Note: a third trigger — "any locally-loaded agent has
        # registered_on_chain = true" — runs after agents load. See
        # ``maybe_autostart_autonet_for_registered_agents``.

        # Load constitution text (needed for prompt injection on registered agents)
        await self.autonet.load_constitution()

        # Attach trace logger to event bus (subscribes to AGENT_TOOL_USE_* events)
        self.trace_logger.attach(self.events)

        # Publish browser-reachable reachability to the Firestore agent
        # directory (so a remote frontend can find this daemon by an agent's 0x
        # address). Only when an endpoint is configured; best-effort — never
        # blocks boot.
        endpoint = getattr(self._config.autonet, "public_ws_endpoint", "")
        if endpoint:
            try:
                from ..agent_directory import AgentDirectoryPublisher
                self.agent_directory = AgentDirectoryPublisher(
                    self, endpoint=endpoint,
                    project=getattr(self._config.autonet, "firestore_project", ""))
                await self.agent_directory.start()
            except Exception:
                log.debug("Agent directory publisher failed to start", exc_info=True)
                self.agent_directory = None

        await self.events.emit(Event(
            type=EventType.RUNTIME_STARTED,
            source="runtime",
        ))

    async def maybe_autostart_autonet_for_registered_agents(self) -> bool:
        """Auto-start autonet if any locally-loaded agent has
        ``registered_on_chain = True``.

        Phase 12: registration is the signal of intent to participate.
        Once an agent is registered, the daemon should always run
        autonet on subsequent boots — otherwise the user's network
        identity sits dormant and consensus operations don't fire.

        Returns True if autonet was started by this call.
        """
        if self.autonet.state.status.value in ("running", "paused"):
            return False  # Already up.
        # Walk the agent registry; bail early on the first registered.
        for defn, _ in self.list_agents():
            if defn.identity and defn.identity.registered_on_chain:
                log.info("Auto-starting autonet: agent %s is registered on chain",
                         defn.id)
                result = await self.autonet.start()
                if result.get("status") not in ("started", "already_running"):
                    log.warning("Auto-start of autonet failed: %s", result)
                    return False
                if self.autonet._p2p_host:
                    self.providers._p2p_host = self.autonet._p2p_host
                return True
        return False

    async def stop(self) -> None:
        self._shutdown_event.set()
        await self.control.kill_all()

        # Stop the supervisor monitor loop (idempotent; no-op if never started).
        try:
            await self.supervisor.stop_monitor()
        except Exception:
            log.debug("supervisor stop_monitor failed", exc_info=True)

        # Tear down the security tripwire: disarm the value-push listener (drops
        # any live raw values held for scanning) and stop the monitor (unsubscribe
        # + halt the TTL sweeper). Idempotent; safe when never armed.
        try:
            self._broker_client.disarm_value_push()
        except Exception:
            log.debug("value-push disarm failed", exc_info=True)
        try:
            self.security_monitor.stop()
        except Exception:
            log.debug("security_monitor stop failed", exc_info=True)

        if self.agent_directory is not None:
            await self.agent_directory.stop()
            self.agent_directory = None

        if self.voice is not None:
            await self.voice.stop()
            self.voice = None

        await self.scheduler.stop()

        if self.autonet.state.status.value in ("running", "paused"):
            await self.autonet.stop()

        await self.connectors.stop_all()

        # Close active cognitive-mode providers
        for provider in list(self.providers._active_providers.values()):
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception:
                    pass
        self.providers._active_providers.clear()

        # Close pipeline-registered providers
        cognitive = self._executors.get(StepType.COGNITIVE)
        if isinstance(cognitive, CognitiveStepExecutor):
            for provider in cognitive._providers.values():
                if hasattr(provider, "close"):
                    try:
                        await provider.close()
                    except Exception:
                        pass

        await self.events.emit(Event(
            type=EventType.RUNTIME_STOPPED,
            source="runtime",
        ))

    # ==================================================================
    # Voice Service
    # ==================================================================

    async def start_voice(self, *, policy: Any = None) -> dict:
        if self.voice is not None:
            return {"status": "already_running"}
        try:
            from ..voice_service import VOICE_AVAILABLE, VoiceService
            if not VOICE_AVAILABLE:
                return {
                    "status": "unavailable",
                    "error": "Voice extras not installed.  "
                             "Install with: pip install atn[voice]",
                }
            # Voice is a Surface; it gates input at the same seam chat does.
            # No policy supplied → AllowAll (today's pass-through behavior).
            self.voice = VoiceService(
                self.events, self, self._config.voice, policy=policy)
            await self.voice.start()
            return {"status": "started", "backend": self._config.voice.backend}
        except Exception as exc:
            log.warning("Failed to start voice service: %s", exc)
            self.voice = None
            return {"status": "failed", "error": str(exc)}

    async def stop_voice(self) -> dict:
        if self.voice is None:
            return {"status": "not_running"}
        await self.voice.stop()
        self.voice = None
        return {"status": "stopped"}

    # ==================================================================
    # Chat service — bind agent conversations to a chat platform
    # ==================================================================

    async def start_chat(
        self,
        client: Any = None,
        channel_id: str = "",
        *,
        bound_agent: str | None = None,
        policy: Any = None,
        orchestrator_label: str | None = None,
        excluded_agents: set[str] | None = None,
    ) -> dict:
        """Start the chat service, binding agents to a chat platform.

        With no args, builds the adapter + policy from ``config.chat`` (the
        normal daemon-boot path). A caller may instead pass a prebuilt
        ``client`` (MessagingClient) / ``policy`` to override — e.g. a custom
        deployment policy (credits/consensus) or a test stub. The bot token is
        read from the env var named in config (secrets stay out of the file).
        """
        if self.chat is not None:
            return {"status": "already_running"}
        cfg = getattr(self._config, "chat", None)
        self._chat_task = None
        try:
            from ..chat import ChatService

            # Resolve adapter + binding from config unless explicitly supplied.
            if client is None:
                client, channel_id = self._build_chat_adapter(cfg, channel_id)
            if policy is None and cfg is not None:
                policy = self._build_chat_policy(cfg)
            if orchestrator_label is None:
                orchestrator_label = (cfg.orchestrator_label if cfg else "K3V|N")
            if excluded_agents is None and cfg is not None and cfg.excluded_agents:
                excluded_agents = set(cfg.excluded_agents)
            if bound_agent is None:
                bound_agent = (cfg.bound_agent if cfg else "orchestrator")

            # If the adapter owns a connection (Discord), bring it online first.
            if hasattr(client, "start") and hasattr(client, "wait_ready"):
                self._chat_task = asyncio.create_task(client.start())
                await client.wait_ready()

            self.chat = ChatService(
                self.events, self, client, channel_id,
                bound_agent=bound_agent,
                policy=policy,
                orchestrator_label=orchestrator_label,
                excluded_agents=excluded_agents,
            )
            # Wire inbound platform messages into the service's input seam.
            if hasattr(client, "on_message"):
                client.on_message = self.chat.handle_inbound
            await self.chat.start()
            return {"status": "started", "channel_id": channel_id}
        except Exception as exc:
            log.warning("Failed to start chat service: %s", exc, exc_info=True)
            self.chat = None
            if getattr(self, "_chat_task", None):
                self._chat_task.cancel()
                self._chat_task = None
            return {"status": "failed", "error": str(exc)}

    def _build_chat_adapter(self, cfg: Any, channel_id: str) -> tuple[Any, str]:
        """Construct the platform adapter from config. Discord for now."""
        if cfg is None:
            raise RuntimeError("no chat config")
        channel_id = channel_id or cfg.channel_id
        if not channel_id:
            raise RuntimeError("chat.channel_id not set")
        if cfg.platform == "discord":
            import os
            from ..chat.adapters.discord_adapter import DiscordAdapter
            token = os.environ.get(cfg.token_env, "")
            if not token:
                raise RuntimeError(f"chat token env {cfg.token_env} not set")
            return DiscordAdapter(token, allowed_channel_ids={channel_id}), channel_id
        raise RuntimeError(f"unknown chat platform: {cfg.platform}")

    def _build_chat_policy(self, cfg: Any) -> Any:
        """Construct the input-seam policy from config: open / operator / credit."""
        from ..chat.policy import AllowAll, OperatorGate, CreditPolicy
        ops = frozenset(getattr(cfg, "operator_ids", []) or [])
        kind = getattr(cfg, "policy", "open")
        if kind == "operator":
            return OperatorGate(ops)
        if kind == "credit":
            from ..chat.credits import CreditStore
            store = CreditStore(
                self._config.data_dir,
                window_secs=getattr(cfg, "credit_window_secs", None),
                default_limit=getattr(cfg, "credit_default_limit", None),
            )
            # Expose the store so operators can manage allowances at runtime.
            self.chat_credits = store
            return CreditPolicy(store, ops)
        return AllowAll()

    def register_surface(self, ws_bridge: Any) -> None:
        """Register the WebSocket bridge as a first-class Surface.

        Called by the CLI once the bridge's listeners are up. Records it as
        self.ws_bridge and wires its input-seam policy (default AllowAll). The
        single-writer *decision* isn't a policy — it belongs to the
        runtime-owned InputArbiter (P3), a gate above every surface's policy;
        this only supplies the per-surface seam object the Surface contract
        expects. No general registry: active_surfaces() keeps an explicit
        (chat, voice, ws_bridge) tuple."""
        self.ws_bridge = ws_bridge
        ws_bridge.policy = self._make_ws_input_policy()

    def _make_ws_input_policy(self) -> Any:
        """Build the WS surface's input-seam policy from config, mirroring the
        chat-policy switch. "allow" => AllowAll (today's default). "single_writer"
        is accepted but resolves to AllowAll here: single-writer is enforced by
        the InputArbiter gate (P3), not by a per-surface policy object."""
        from ..surface import AllowAll
        kind = getattr(self._config.autonet, "ws_input_policy", "allow")
        if kind == "single_writer":
            # Seam reserved for the arbiter gate (P3); no per-surface gating today.
            return AllowAll()
        return AllowAll()

    def active_surfaces(self) -> list:
        """Currently-running Surfaces (chat, voice, ws_bridge). Used by the
        execution engine to collect surface-contributed tools for an agent and
        to route surface tool calls back. Explicit tuple, not a registry."""
        return [s for s in (self.chat, self.voice, self.ws_bridge) if s is not None]

    async def stop_chat(self) -> dict:
        if self.chat is None:
            return {"status": "not_running"}
        await self.chat.stop()
        self.chat = None
        task = getattr(self, "_chat_task", None)
        if task is not None:
            try:
                client = getattr(self.chat, "client", None)
                if client is not None and hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass
            task.cancel()
            self._chat_task = None
        return {"status": "stopped"}

    # ==================================================================
    # Agent Management — delegate to registry
    # ==================================================================

    async def register_agent(
        self, defn: AgentDefinition, *, legacy: bool = False,
    ) -> str:
        return await self.registry.register_agent(defn, legacy=legacy)

    async def unregister_agent(self, agent_id: str, *, _force: bool = False) -> None:
        from ..orchestrator import ORCHESTRATOR_ID
        if agent_id == ORCHESTRATOR_ID and not _force:
            raise ValueError("The orchestrator cannot be unregistered")
        await self.control.kill_agent(agent_id)
        # Clean up the persisted provider (closes session / subprocess)
        old_provider = self.providers._active_providers.pop(agent_id, None)
        if old_provider is not None:
            try:
                await old_provider.close()
            except Exception:
                pass
        self.providers._cached_session_stats.pop(agent_id, None)
        # Clean up persistent files before unregistering from registry
        self.sessions._agent_conversations.pop(agent_id, None)
        agent_data_dir = self._config.data_dir / "agents" / agent_id
        if agent_data_dir.exists():
            shutil.rmtree(agent_data_dir, ignore_errors=True)
        delegate_log = self._config.data_dir / "delegates" / f"{agent_id}.log"
        if delegate_log.exists():
            delegate_log.unlink(missing_ok=True)
        agent_yaml = self._config.agents_dir / f"{agent_id}.yaml"
        if agent_yaml.exists():
            agent_yaml.unlink(missing_ok=True)
        await self.registry.unregister_agent(agent_id, _force=_force)

    async def activate_agent(self, agent_id: str) -> None:
        return await self.registry.activate_agent(agent_id)

    async def deactivate_agent(self, agent_id: str) -> None:
        return await self.registry.deactivate_agent(agent_id)

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self.registry.get_agent(agent_id)

    def get_status(self, agent_id: str) -> AgentStatus | None:
        return self.registry.get_status(agent_id)

    def list_agents(self) -> list[tuple[AgentDefinition, AgentStatus]]:
        return self.registry.list_agents()

    def reconcile_chain_registrations(self) -> dict[str, Any]:
        return self.registry.reconcile_chain_registrations()

    def get_children(self, agent_id: str) -> list[AgentDefinition]:
        return self.registry.get_children(agent_id)

    def get_descendants(self, agent_id: str) -> list[AgentDefinition]:
        return self.registry.get_descendants(agent_id)

    def generate_child_id(self, parent_id: str) -> str:
        return self.registry.generate_child_id(parent_id)

    # ==================================================================
    # Execution — delegate to engine
    # ==================================================================

    async def trigger_run(self, agent_id: str, source: str = "user") -> str | None:
        return await self.engine.trigger_run(agent_id, source)

    async def route_tool_call(self, name: str, tool_input: dict, agent_id: str) -> dict:
        return await self.engine.route_tool_call(name, tool_input, agent_id)

    # ==================================================================
    # Kill switches — delegate to control
    # ==================================================================

    async def kill_execution(self, execution_id: str) -> bool:
        return await self.control.kill_execution(execution_id)

    async def kill_agent(self, agent_id: str) -> int:
        return await self.control.kill_agent(agent_id)

    async def kill_all(self) -> int:
        return await self.control.kill_all()

    def register_interrupt_hook(self, execution_id: str, hook: Callable) -> None:
        self.control.register_interrupt_hook(execution_id, hook)

    def unregister_interrupt_hook(self, execution_id: str) -> None:
        self.control.unregister_interrupt_hook(execution_id)

    async def send_delegate_message(self, agent_id: str, content: str) -> bool:
        return await self.control.send_delegate_message(agent_id, content)

    async def interrupt_delegate(self, agent_id: str) -> bool:
        return await self.control.interrupt_delegate(agent_id)

    async def interrupt_orchestrator(self) -> bool:
        return await self.control.interrupt_orchestrator()

    # ==================================================================
    # Session management — delegate to sessions
    # ==================================================================

    async def new_conversation(self) -> None:
        from ..orchestrator import ORCHESTRATOR_ID
        await self.reset_agent_conversation(ORCHESTRATOR_ID)

    async def reset_agent_conversation(self, agent_id: str) -> None:
        """Reset conversation history for any agent."""
        await self.control.kill_agent(agent_id)
        await self.sessions.reset_agent_conversation(agent_id)

    def get_agent_conversation_store(self, agent_id: str) -> ConversationStore:
        return self.sessions.get_agent_conversation_store(agent_id)

    async def send_agent_message(
        self, agent_id: str, text: str, *, surface: "Any | None" = None,
    ) -> dict:
        """Deliver a user message to an agent. ``surface`` is the SurfaceId of
        the input surface, if this came through one; the SessionManager gates it
        against the InputArbiter (single-writer). surface=None => trusted
        internal caller (orchestrator tool, scheduler) — never mic-gated.
        ALWAYS returns a dict; callers must branch on result.get('error')."""
        return await self.sessions.send_agent_message(agent_id, text, surface=surface)

    def append_delegate_output(self, agent_id: str, text: str) -> None:
        self.sessions.append_delegate_output(agent_id, text)

    def get_delegate_output(self, agent_id: str) -> str:
        return self.sessions.get_delegate_output(agent_id)

    def clear_delegate_output(self, agent_id: str) -> None:
        self.sessions.clear_delegate_output(agent_id)

    def _clear_bridge_session(self) -> None:
        self.sessions.clear_bridge_session()

    def _inject_status_briefing(self) -> None:
        self.sessions._inject_status_briefing()

    # ==================================================================
    # Provider management — delegate to providers
    # ==================================================================

    def _resolve_provider_for_model(self, model_name: str, agent_id: str) -> Any:
        return self.providers._resolve_provider_for_model(model_name, agent_id)

    def _get_bridge_provider(self, agent_id: str | None = None) -> Any:
        return self.providers.get_bridge_provider(agent_id)

    def get_session_stats(self, agent_id: str | None = None) -> dict[str, Any]:
        from ..orchestrator import ORCHESTRATOR_ID
        target = agent_id or ORCHESTRATOR_ID

        # 1. Live provider (best — real-time data)
        stats = self.providers.get_session_stats(target)
        if "error" not in stats:
            return stats

        # 2. Persisted stats from conversation store (survives restarts)
        convo = self.sessions.get_agent_conversation_store(target)
        persisted = convo.get_session_stats()
        if persisted:
            return persisted

        # 3. Genuinely no data
        return stats

    async def get_session_context(self, agent_id: str | None = None) -> dict[str, Any]:
        return await self.providers.get_session_context(agent_id)

    def _get_available_models(self, provider_name: str, **kwargs) -> list[dict[str, str]]:
        return self.providers.get_available_models(provider_name, **kwargs)

    async def provider_list(self) -> list[dict[str, Any]]:
        return await self.providers.provider_list()

    async def configure_provider(self, provider_id: str, api_key: str = "") -> dict[str, Any]:
        return await self.providers.configure_provider(provider_id, api_key)

    async def remove_provider(self, provider_id: str) -> dict[str, str]:
        return await self.providers.remove_provider(provider_id)

    async def add_custom_provider(self, provider_id: str, name: str, base_url: str, api_key: str = "", default_model: str = "", models: list[Any] | None = None) -> dict[str, Any]:
        return await self.providers.add_custom_provider(provider_id, name, base_url, api_key, default_model, models)

    async def remove_custom_provider(self, provider_id: str) -> dict[str, str]:
        return await self.providers.remove_custom_provider(provider_id)

    # ==================================================================
    # Orchestrator setup — delegate to _orch_setup
    # ==================================================================

    async def setup_orchestrator(self, **kwargs: Any) -> str:
        return await self._orch_setup.setup_orchestrator(**kwargs)

    async def set_orchestrator_model(self, model: str) -> str:
        from ..orchestrator import ORCHESTRATOR_ID
        return await self.set_agent_model(ORCHESTRATOR_ID, model)

    async def set_agent_model(self, agent_id: str, model: str) -> str:
        """Change the cognitive model for any agent."""
        await self.control.kill_agent(agent_id)
        return await self._orch_setup.set_agent_model(agent_id, model)

    # ==================================================================
    # Snapshot — delegate to snapshot builder
    # ==================================================================

    def snapshot(self, scope_ids: set[str] | None = None) -> dict:
        return self._get_snapshot_builder().snapshot(scope_ids)

    # ==================================================================
    # Planning tasks
    # ==================================================================

    def _load_planning_tasks(self) -> None:
        self.planning_tasks = load_planning_tasks(self._planning_tasks_path)

    def _save_planning_tasks(self) -> None:
        save_planning_tasks(self.planning_tasks, self._planning_tasks_path)

    # ==================================================================
    # Connector credentials
    # ==================================================================

    def _inject_connector_credentials(self) -> None:
        inject_connector_credentials(self.connectors, self.credential_store)

    # ==================================================================
    # Backward-compatible attribute access
    # ==================================================================

    @property
    def _agents(self) -> dict:
        """Backward compat: tools.py and others access runtime._agents directly."""
        return self.registry._agents

    @property
    def _status(self) -> dict:
        return self.registry._status

    @property
    def _running_count(self) -> dict:
        return self.registry._running_count

    @property
    def _executions(self) -> dict:
        return self.engine._executions

    @property
    def _tasks(self) -> dict:
        return self.engine._tasks

    @property
    def _cancels(self) -> dict:
        return self.engine._cancels

    @property
    def _active_providers(self) -> dict:
        return self.providers._active_providers

    @property
    def _custom_providers(self) -> set:
        return self.providers._custom_providers

    @property
    def _delegate_done(self) -> dict:
        return self.engine._delegate_done

    @property
    def _completion_callbacks(self) -> dict:
        return self.engine._completion_callbacks

    @property
    def _child_counters(self) -> dict:
        return self.registry._child_counters

    @property
    def _interrupt_hooks(self) -> dict:
        return self.engine._interrupt_hooks

    @property
    def _schedule_table(self) -> dict:
        return self.registry._schedule_table

    @property
    def _heartbeat_table(self) -> dict:
        return self.registry._heartbeat_table

    @property
    def _last_idle(self) -> dict:
        return self.registry._last_idle

    @property
    def _last_planning_review(self):
        return self.scheduler._last_planning_review

    @_last_planning_review.setter
    def _last_planning_review(self, value):
        self.scheduler._last_planning_review = value

    @property
    def _planning_interval(self):
        return self.scheduler._planning_interval

    @_planning_interval.setter
    def _planning_interval(self, value):
        self.scheduler._planning_interval = value

    @property
    def _running(self):
        return self.scheduler._running

    # Expose _KNOWN_PROVIDERS and _PROVIDER_DEFAULTS for tools that reference them
    _KNOWN_PROVIDERS = ProviderManager._KNOWN_PROVIDERS
    _PROVIDER_DEFAULTS = ProviderManager._PROVIDER_DEFAULTS

    # Static/class methods that external code might reference
    @staticmethod
    def _parse_interval(schedule: str) -> float:
        from .agent_registry import parse_interval
        return parse_interval(schedule)

    @staticmethod
    def _append_history_to_prompt(system_prompt: str, prior_turns: list) -> str:
        return ExecutionEngine._append_history_to_prompt(system_prompt, prior_turns)

    # Helper used by external code
    def _require_agent(self, agent_id: str) -> None:
        self.registry._require_agent(agent_id)

    def _resolve_api_key(self, provider_name: str) -> str:
        return self.providers._resolve_api_key(provider_name)

    def _resolve_provider_by_name(self, provider_name: str, model: str, agent_id: str) -> Any:
        return self.providers._resolve_provider_by_name(provider_name, model, agent_id)

    def _resolve_provider_with_fallback(self, defn: AgentDefinition) -> Any:
        return self.providers.resolve_provider_with_fallback(defn)

    def _hot_register_provider(self, provider_id: str, api_key: str) -> None:
        self.providers._hot_register_provider(provider_id, api_key)

    async def _auto_detect_providers(self) -> None:
        await self.providers.auto_detect_providers()

    async def _probe_ollama(self) -> list[dict[str, str]] | None:
        return await self.providers.probe_ollama()

    async def _probe_bridge(self) -> bool:
        return await self.providers.probe_bridge()

    def _delegates_snapshot(self) -> dict:
        return self._get_snapshot_builder()._delegates_snapshot()

    def _voice_snapshot(self) -> dict:
        return self._get_snapshot_builder()._voice_snapshot()

    def _resolve_parent_agent_id(self, parent_id: str) -> str:
        return self.registry._resolve_parent_agent_id(parent_id)

    async def _notify_parent_of_failure(self, agent_id: str, record) -> None:
        await self.registry.notify_parent_of_failure(
            agent_id, record, self.providers._active_providers,
        )

    async def _on_agent_completed(self, agent_id: str, record) -> None:
        await self.registry.on_agent_completed(
            agent_id, record, self.providers._active_providers,
        )

    # Delegate output dir (for tests/tools that access it)
    @property
    def _delegate_output_dir(self):
        return self.sessions._delegate_output_dir

    # Agent conversations (for tests/tools that access it)
    @property
    def _agent_conversations(self):
        return self.sessions._agent_conversations
